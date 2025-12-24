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
    
    def _expand_encoder_out(self, encoder_out, B: int):
        # ESM inverse folding encoder_out is typically a dict with tensors shaped [T, 1, C] or similar.
        # We expand any singleton batch dim to B.
        if torch.is_tensor(encoder_out):
            x = encoder_out
            if x.dim() >= 2 and x.size(1) == 1:
                return x.expand(x.size(0), B, *x.shape[2:]).contiguous()
            if x.dim() >= 1 and x.size(0) == 1:
                return x.expand(B, *x.shape[1:]).contiguous()
            return x

        if isinstance(encoder_out, dict):
            return {k: self._expand_encoder_out(v, B) for k, v in encoder_out.items()}
        if isinstance(encoder_out, (list, tuple)):
            return type(encoder_out)(self._expand_encoder_out(v, B) for v in encoder_out)
        return encoder_out

    def sample(self, coords, B: int, partial_seq=None, temperature=1.0, confidence=None, device=None):
        """
        Batched sampling: returns B sequences for the same coords backbone.

        coords: backbone coords in the same format expected by CoordBatchConverter
            (typically a python/numpy structure; if you have a torch tensor,
                pass coords.detach().cpu().numpy()).
        """
        import torch
        import torch.nn.functional as F

        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)

        # IMPORTANT: CoordBatchConverter expects non-CUDA python/numpy coords in most ESM builds.
        # If you pass a torch Tensor here and get garbage outputs, convert it:
        # if torch.is_tensor(coords): coords = coords.detach().cpu().numpy()

        L = len(coords)

        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )

        d = self.decoder.dictionary
        mask_idx = d.get_idx("<mask>")
        cath_idx = d.get_idx("<cath>")

        # Precompute fixed positions on CPU to avoid Python branching on CUDA tensors
        fixed = [False] * L
        if partial_seq is not None:
            for j, c in enumerate(partial_seq):
                if c is not None and c != "<mask>":
                    fixed[j] = True

        # Tokens on GPU
        sampled_tokens = torch.full((B, 1 + L), mask_idx, device=device, dtype=torch.long)
        sampled_tokens[:, 0] = cath_idx

        # Fill known tokens (same partial_seq for all B)
        if partial_seq is not None:
            for j, c in enumerate(partial_seq):
                if fixed[j]:
                    sampled_tokens[:, j + 1] = d.get_idx(c)

        # Encoder once (batch=1), then expand encoder outputs to batch=B
        with torch.inference_mode():
            encoder_out_1 = self.encoder(batch_coords, padding_mask, confidence)  # batch=1
            encoder_out = self._expand_encoder_out(encoder_out_1, B)

            # Correctness-first: disable incremental_state (avoid cache shape pitfalls when expanding)
            for i in range(1, L + 1):
                if fixed[i - 1]:
                    continue

                logits, _ = self.decoder(
                    sampled_tokens[:, :i],
                    encoder_out,
                    incremental_state=None,
                )

                # Normalize logits to [T, B, V]
                if logits.dim() != 3:
                    raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

                # If [T, B, V]
                if logits.size(1) == B:
                    logits_tbv = logits
                # If [B, T, V]
                elif logits.size(0) == B:
                    logits_tbv = logits.permute(1, 0, 2).contiguous()
                else:
                    raise RuntimeError(
                        f"Can't infer batch dim from logits shape {tuple(logits.shape)} with B={B}"
                    )

                # Take logits for the newest position (0-based: i-1)
                step_logits = logits_tbv[i - 1]  # [B, V]
                probs = F.softmax(step_logits / temperature, dim=-1)
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)

        # Convert once per sequence (avoid per-token CUDA sync)
        ids = sampled_tokens[:, 1:].detach().cpu().tolist()  # (B, L)
        return ["".join(d.get_tok(t) for t in row) for row in ids]

