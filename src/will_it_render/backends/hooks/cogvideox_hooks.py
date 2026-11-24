# Portions of this file (the attention-output computation reproduced to keep generation
# bit-exact) are adapted from the 🤗 diffusers ``CogVideoXAttnProcessor2_0`` and are licensed
# under the Apache License, Version 2.0:
#
#     Copyright 2024 The HuggingFace Team. All rights reserved.
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#         http://www.apache.org/licenses/LICENSE-2.0
#
# See THIRD_PARTY.md. The surrounding AttnSteer capture logic is MIT-licensed (see LICENSE).
"""AttnSteer capture hook for CogVideoX joint self-attention.

Unlike LTX/Wan (which have a dedicated cross-attention module), CogVideoX concatenates the
226 T5 text tokens with the video tokens into ONE sequence and runs full self-attention
over it (``CogVideoXAttnProcessor2_0``): ``hidden = cat([text, video]); sdpa(q, k, v)``.
There is no attention-weight matrix to read from the fused SDPA kernel, and the "attention
toward the target word" we probe is the *video-query × text-key* sub-block of that joint
attention.

``CogVideoXCrossAttnProbeContext`` swaps ``attn1.processor`` on the selected transformer
blocks for a processor that (a) reproduces the real CogVideoX attention output bit-for-bit
(so generation is unchanged) and (b) additionally materializes
``softmax(Q_video Kᵀ_text / √d)`` and records a ``BlockCapture`` with the exact shape the
LTX/Wan backends produce (``attn_map`` ``[B, H, Lq, Lk]``, ``text_key`` ``[B, Lk, H, D]``),
so the generic aggregator / token-mapper / storage are reused unchanged.

Text keys carry NO rotary embedding in CogVideoX (RoPE is applied to video queries/keys
only), so the captured text-key sub-block is position-free — the same regime the LTX/Wan
probes were trained in.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from will_it_render.common.probe_types import BlockCapture, CaptureType

_MISSING = object()


class _CogVideoXCaptureProcessor:
    """Drop-in for ``CogVideoXAttnProcessor2_0`` that also records the video→text block.

    Body is copied verbatim from diffusers' ``CogVideoXAttnProcessor2_0.__call__`` (RoPE on
    the video span only, SDPA, text/video re-split) with a single capture insert after the
    q/k projections + RoPE, before SDPA — so the returned tensors are identical to the
    stock processor.
    """

    def __init__(self, ctx: "CogVideoXCrossAttnProbeContext", order: int) -> None:
        self._ctx = ctx
        self._order = order  # ascending position among hooked blocks (for reference)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple] = None,
    ):
        from diffusers.models.embeddings import apply_rotary_emb

        ctx = self._ctx
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # RoPE on the video span only (text keys/queries stay position-free).
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # ---- CAPTURE: video-query × text-key sub-block of the joint attention ----
        # Gated by ctx.active (cond pass only) and capped at the hooked-block count so a
        # steering recompute cannot duplicate captures. Blocks fire in ascending index, so
        # ctx.captures ends up in hooked-block order — matching the Wan/LTX convention.
        if ctx.active and len(ctx.captures) < ctx._n_blocks:
            # Extraction (detach=True) computes the capture in fp32 for a faithful stored
            # feature. Steering (detach=False) keeps the autograd graph of every hooked
            # block's softmax resident simultaneously (the probe loss reads them); at
            # Lq=17550 that is ~16 GB of fp32 softmaxes + an equal backward graph, which
            # OOMs a 94-96 GB card at the first steered step. Compute the capture in the
            # model's native dtype (bf16) for steering — ~2x smaller, and the steering
            # gradient is tolerant of the reduced precision.
            _cap_dtype = torch.float32 if ctx.detach else query.dtype
            q_v = query[:, :, text_seq_length:, :].to(_cap_dtype)  # [B, H, Lq, D]
            k_t = key[:, :, :text_seq_length, :].to(_cap_dtype)    # [B, H, Lk, D]

            def _compute_capture(q_v, k_t):
                # matmul → softmax over Lk → (steering) pool Lq to the probe's grid.
                attn_w = torch.softmax(q_v @ k_t.transpose(-2, -1) * (head_dim ** -0.5), dim=-1)  # [B,H,Lq,Lk]
                if ctx.steer_pool is not None:
                    # Pool Lq to the probe's grid NOW (steering), so only the reduced map is kept
                    # on-GPU with grad. Reuse the aggregator's own _spatial_downsample (operating
                    # on [rows, Lq]) so the pooled grid is identical to the extraction convention;
                    # it commutes with the later target-token mean over Lk.
                    from will_it_render.common.aggregator import _spatial_downsample
                    f, h_lat, w_lat = ctx.steer_pool
                    B_, H_, Lq_, Lk_ = attn_w.shape
                    flat = attn_w.permute(0, 1, 3, 2).reshape(B_ * H_ * Lk_, Lq_)  # [(B·H·Lk), Lq]
                    flat = _spatial_downsample(flat, h_lat, w_lat, f)              # [(B·H·Lk), Lq/f²]
                    attn_w = flat.reshape(B_, H_, Lk_, -1).permute(0, 1, 3, 2).contiguous()
                return attn_w

            # Steering: gradient-checkpoint the capture so the FULL pre-pool softmax
            # [B,H,Lq,Lk] (≈380 MB bf16 × 21 hooked blocks ≈ 8 GB) is NOT retained for
            # backward — it is recomputed from q_v/k_t (already resident as slices of the
            # block's SDPA query/key), leaving only the small pooled map on-GPU. This is
            # what lets a 94 GB card fit (post-steer_pool peak was ~99 GB). Numerically
            # identical (no dropout). Extraction (detach=True, no grad) runs it directly.
            if not ctx.detach and torch.is_grad_enabled():
                import torch.utils.checkpoint as _cp
                attn_w = _cp.checkpoint(_compute_capture, q_v, k_t, use_reentrant=False)
            else:
                attn_w = _compute_capture(q_v, k_t)
            ctx.captures.append(
                BlockCapture(
                    attn_map=ctx._store(attn_w),                   # [B, H, Lq, Lk]
                    text_key=ctx._store(k_t.permute(0, 2, 1, 3)),  # [B, Lk, H, D]
                    capture_type=CaptureType.VIDEO_CA,
                )
            )

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogVideoXCrossAttnProbeContext:
    """Context manager that swaps ``attn1.processor`` on CogVideoX transformer blocks.

    Each active denoising step produces one ``BlockCapture`` per hooked block (ascending
    block index), accessible via ``self.captures``. The driver clears ``self.captures``
    between steps and toggles ``self.active`` to gate the cond vs. uncond forward (run as
    two separate model calls, like Wan).

    ``capture_blocks`` selects which blocks to hook (0-based); ``None`` hooks all.
    ``detach=True`` (extraction) detaches captures; ``detach=False`` (steering) keeps them
    connected to the latent for backprop.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        capture_blocks: Optional[list[int]] = None,
        detach: bool = True,
        steer_pool: Optional[tuple[int, int, int]] = None,
    ) -> None:
        self.detach = detach
        # Steering only: (factor, h_lat, w_lat). When set, the video-query (Lq) spatial grid
        # of each captured softmax is average-pooled to the probe's resolution AT CAPTURE
        # TIME, so we retain [B,H,Lq/f²,Lk] instead of the full map (≈4× less VRAM + graph).
        # The downstream steering aggregation must then run with spatial_downsample=1.
        self.steer_pool = steer_pool
        self._blocks = transformer.transformer_blocks
        self.n_blocks_total = len(self._blocks)
        if capture_blocks is None:
            self._hook_indices = list(range(self.n_blocks_total))
        else:
            self._hook_indices = sorted(set(int(b) for b in capture_blocks))
            bad = [b for b in self._hook_indices if not (0 <= b < self.n_blocks_total)]
            if bad:
                raise ValueError(
                    f"capture_blocks {bad} out of range for model with "
                    f"{self.n_blocks_total} transformer blocks"
                )
        self._originals: dict[int, object] = {}
        self.captures: list[BlockCapture] = []
        self.active: bool = True
        self._n_blocks = len(self._hook_indices)

    def _store(self, t: torch.Tensor) -> torch.Tensor:
        # Extraction (detach=True): offload to CPU immediately. The fp32 video→text maps
        # are ~0.76 GB each (21 hooked blocks ≈ 16 GB/step); keeping them on GPU exhausts
        # an 80 GB card and, held past the loop, OOMs the VAE decode. Steering (detach=False)
        # must keep them on-GPU and connected to the graph for backprop.
        return t if not self.detach else t.detach().to("cpu")

    def __enter__(self) -> "CogVideoXCrossAttnProbeContext":
        self.captures = []
        for order, i in enumerate(self._hook_indices):
            attn = self._blocks[i].attn1
            self._originals[i] = attn.get_processor()
            attn.set_processor(_CogVideoXCaptureProcessor(self, order))
        return self

    def __exit__(self, *_) -> None:
        for i, proc in self._originals.items():
            self._blocks[i].attn1.set_processor(proc)
        self._originals = {}
