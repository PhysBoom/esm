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

    def sample(self, coords, B: int, partial_seq=None, temperature=1.0, confidence=None, device=None):

        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)

        # CoordBatchConverter expects python/numpy coords in most ESM IF1 builds
        if torch.is_tensor(coords):
            L = coords.shape[0]
            coords_in = coords.detach().cpu().numpy()
        else:
            L = len(coords)
            coords_in = coords

        d = self.decoder.dictionary

        # Build the set of allowed output token ids = standard amino acids
        # (This is the key fix that avoids sampling invalid / special ids.)
        aa_letters = list("ACDEFGHIKLMNPQRSTVWY")
        allowed_ids = torch.tensor([d.get_idx(a) for a in aa_letters], device=device, dtype=torch.long)

        # Optional: allow X if present in dict
        try:
            x_id = d.get_idx("X")
            if x_id is not None and x_id >= 0:
                allowed_ids = torch.cat([allowed_ids, torch.tensor([x_id], device=device)])
        except Exception:
            pass

        mask_idx = d.get_idx("<mask>")
        cath_idx = d.get_idx("<cath>")

        # Prepare a true batch for the encoder: B copies of the same coords
        batch_converter = CoordBatchConverter(d)
        items = [(coords_in, confidence, None)] * B
        batch_coords, confidence_b, _, _, padding_mask = batch_converter(items, device=device)

        # Tokens on GPU
        sampled_tokens = torch.full((B, 1 + L), mask_idx, device=device, dtype=torch.long)
        sampled_tokens[:, 0] = cath_idx

        # Handle partial_seq (same partial for all B)
        fixed = [False] * L
        if partial_seq is not None:
            for j, c in enumerate(partial_seq):
                if c is not None and c != "<mask>":
                    fixed[j] = True
                    sampled_tokens[:, j + 1] = d.get_idx(c)

        with torch.inference_mode():
            encoder_out = self.encoder(batch_coords, padding_mask, confidence_b)

            incremental_state = dict()

            for i in range(1, L + 1):
                if fixed[i - 1]:
                    continue

                logits, _ = self.decoder(
                    sampled_tokens[:, :i],
                    encoder_out,
                    incremental_state=incremental_state,
                )

                if logits.dim() != 3:
                    raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

                # Normalize logits to [T, B, Vlogits]
                if logits.size(1) == B:          # [T, B, V]
                    logits_tbv = logits
                elif logits.size(0) == B:        # [B, T, V] -> [T, B, V]
                    logits_tbv = logits.permute(1, 0, 2).contiguous()
                else:
                    raise RuntimeError(f"Can't infer batch dim: logits={tuple(logits.shape)} B={B}")

                step_logits_full = logits_tbv[-1]  # [B, Vlogits]

                # Restrict to allowed amino-acid ids
                step_logits = step_logits_full.index_select(dim=-1, index=allowed_ids)  # [B, |AA|]

                # Sample within allowed set
                step_logits = torch.nan_to_num(step_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                probs = F.softmax(step_logits / temperature, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                choice = torch.multinomial(probs, 1).squeeze(-1)         # [B] in 0..|AA|-1
                next_tok = allowed_ids.index_select(0, choice)           # map back to real token ids

                sampled_tokens[:, i] = next_tok

        ids = sampled_tokens[:, 1:].detach().cpu().tolist()
        return ["".join(d.get_tok(t) for t in row) for row in ids]




