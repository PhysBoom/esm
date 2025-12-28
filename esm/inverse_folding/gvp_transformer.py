# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from scipy.spatial import transform

from esm.data import Alphabet

from .features import DihedralFeatures
from .gvp_encoder import GVPEncoder
from .gvp_utils import unflatten_graph
from .gvp_transformer_encoder import GVPTransformerEncoder
from .transformer_decoder import TransformerDecoder
from .util import rotate, CoordBatchConverter 


class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )
        decoder_embed_tokens = self.build_embedding(
            args, alphabet, args.decoder_embed_dim, 
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        decoder = self.build_decoder(args, alphabet, decoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra
    
    def sample(self, coords, partial_seq=None, temperature=1.0, confidence=None, device=None):
        if device is None and isinstance(coords, torch.Tensor):
            device = coords.device

        if isinstance(coords, torch.Tensor):
            assert coords.dim() == 4 and coords.size(-2) == 3 and coords.size(-1) == 3
            B, L = coords.shape[:2]
            coords_list = coords.detach().cpu().tolist()
            if isinstance(confidence, torch.Tensor):
                conf_list = confidence.detach().cpu().tolist()
            else:
                conf_list = confidence
            items = [(coords_list[b], (conf_list[b] if conf_list is not None else None), None) for b in range(B)]
        else:
            B = len(coords)
            L = len(coords[0]) if B > 0 else 0
            items = [(coords[b], (confidence[b] if confidence is not None else None), None) for b in range(B)]

        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence_batch, _, _, padding_mask = batch_converter(items, device=device)

        d = self.decoder.dictionary
        mask_idx = d.get_idx("<mask>")
        cath_idx = d.get_idx("<cath>")

        sampled_tokens = torch.full((B, 1 + L), mask_idx, dtype=torch.long, device=device)
        sampled_tokens[:, 0] = cath_idx

        if partial_seq is not None:
            if isinstance(partial_seq[0], str):
                for b in range(B):
                    s = partial_seq[b]
                    for i in range(min(L, len(s))):
                        c = s[i]
                        if c not in {"?", "*", "-", "_", " "}:
                            sampled_tokens[b, i + 1] = d.get_idx(c)
            else:
                for b in range(B):
                    row = partial_seq[b]
                    for i in range(min(L, len(row))):
                        c = row[i]
                        if c is None:
                            continue
                        if isinstance(c, str) and c in {"<mask>", "?", "*", "-", "_", " "}:
                            continue
                        sampled_tokens[b, i + 1] = d.get_idx(c)

        encoder_out = self.encoder(batch_coords, padding_mask, confidence_batch)
        incremental_state = {}

        for i in range(1, L + 1):
            need = (sampled_tokens[:, i] == mask_idx)
            if not torch.any(need):
                continue

            decoder_in = sampled_tokens[:, :1] if i == 1 else sampled_tokens[:, i - 1:i]
            logits, _ = self.decoder(decoder_in, encoder_out, incremental_state=incremental_state)

            step_logits = logits[:, :, -1] / float(temperature)  # B x V
            step_probs = F.softmax(step_logits, dim=-1)

            sampled = torch.multinomial(step_probs[need], 1).squeeze(-1)
            sampled_tokens[need, i] = sampled

        out = []
        for b in range(B):
            toks = sampled_tokens[b, 1:].tolist()
            out.append("".join(d.get_tok(t) for t in toks))
        return out