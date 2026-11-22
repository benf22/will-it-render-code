"""3D-CNN ResNet probe for per-layer attention features.

Operates on a single transformer block's cross-attention map, with heads as the
channel dimension: input ``[B, n_heads, T, h, w]`` (e.g. ``[B, 32, 13, 16, 16]``).
Intended for ``per_layer: true`` runs, where ProbeDataset slices one block and
``trainer.py`` passes ``n_blocks_heads = n_heads``.

Design: the residual stack uses spatial-only (2D) convolutions — kernel
``(1, 3, 3)`` — so each frame is processed independently. The temporal axis is
mixed only in the final residual block (kernel ``(3, 3, 3)``); ``AdaptiveAvgPool3d``
then completes the temporal/spatial collapse. Kept tiny (~80k params at the
defaults) for a few-thousand-example regime — or ~30k with
``temporal_spatial=False`` (pure (3,1,1) final block).

Optional ``layer_agg`` front-end (``LayerAgg``): instead of feeding a single
already-reduced block (per-layer slice or ``avg_channels`` mean), the probe can take the
captured blocks *stacked* (``avg_mode="stack"``, input ``[B, L*H, T, h, w]``) and *learn*
how to combine the ``L`` layers — a slim aggregator (``scalar`` | ``per_head`` | ``conv_k``,
tens-to-hundreds of params) that collapses the layer axis before the stem.

Optional ``perframe`` output: because the clip head is a *mean* over ``(T, h, w)`` followed
by a *linear* 1×1×1 conv, the clip logit already equals the mean of per-frame logits. Setting
``perframe=True`` exposes those per-latent-frame logits via ``forward(..., return_frames=True)``
— with ``perframe_head="shared"`` at zero parameter cost and with an unchanged state dict.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_groups(groups: int, ch: int) -> int:
    """Largest valid GroupNorm group count ≤ ``groups`` that divides ``ch``."""
    g = min(groups, ch)
    while ch % g != 0:
        g -= 1
    return max(1, g)


def _make_norm(norm: str, ch: int, groups: int) -> nn.Module:
    """Build the per-block normalization layer.

    ``"group"``    → ``GroupNorm`` over ``_safe_groups(groups, ch)`` groups (default).
    ``"instance"`` → ``InstanceNorm3d(affine=True)`` — normalizes each channel per sample
    across ``(T, h, w)``, removing per-head/per-channel magnitude and emphasizing the spatial
    *pattern* of attention. ``groups`` is ignored in this branch.
    """
    if norm == "group":
        return nn.GroupNorm(_safe_groups(groups, ch), ch)
    if norm == "instance":
        return nn.InstanceNorm3d(ch, affine=True)
    raise ValueError(f"Unknown norm {norm!r}. Choose 'group' or 'instance'.")


class DropBlock3d(nn.Module):
    """Spatial-only DropBlock: drops contiguous ``block_size × block_size`` h/w regions.

    Unlike ``Dropout3d`` (which zeros whole channel maps) or plain dropout (independent
    pixels), DropBlock removes contiguous spatial squares — forcing the probe to rely on
    spatial context rather than a single hot location. The temporal axis is left intact
    (kernel ``(1, bs, bs)``), so each frame is masked independently. No-op at eval or when
    ``drop_prob == 0``.
    """

    def __init__(self, drop_prob: float, block_size: int) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        _, _, _, h, w = x.shape
        bs = min(self.block_size, h, w)
        # gamma: per-seed Bernoulli rate so the expected fraction of zeroed units ≈ drop_prob.
        gamma = self.drop_prob / (bs * bs) * (h * w) / ((h - bs + 1) * (w - bs + 1))
        seeds = torch.bernoulli(torch.full_like(x, gamma))
        block_mask = F.max_pool3d(
            seeds, kernel_size=(1, bs, bs), stride=1, padding=(0, bs // 2, bs // 2)
        )
        # max_pool with even bs can return a +1 spatial size; crop back to the input grid.
        block_mask = block_mask[:, :, :, :h, :w]
        keep = 1.0 - block_mask
        count = keep.numel()
        kept = keep.sum()
        if kept == 0:
            return torch.zeros_like(x)
        return x * keep * (count / kept)


class LayerAgg(nn.Module):
    """Slim learned aggregation over the transformer-block (layer) axis.

    Replaces ``avg_channels=layers``' uniform mean over the captured blocks with a
    *learned* combine, keeping the heads as channels for the downstream ResNet. Input is
    the stacked per-block map ``[B, L, H, T, h, w]`` (the dataset's ``avg_mode="stack"``);
    output collapses the ``L`` axis.

    Modes (extra params for L blocks, H heads). The first three are the original linear
    aggregators; the last three are the *capacity-rebalancing* options — richer combines
    that still output ``H`` channels (so the downstream ResNet stem stays lean, unlike
    ``conv_k`` which widens it), letting you move parameter budget out of the backbone and
    into the aggregation while keeping the total roughly constant.

      ``scalar``      → one weight per layer, shared across heads → ``[B, H, …]``   (L)
      ``per_head``    → per-(layer, head) weight                  → ``[B, H, …]``   (L·H)
      ``conv_k``      → grouped 1×1 mix L→K per head; the K views are concatenated on the
                        channel axis (widens the stem)            → ``[B, K·H, …]`` (L·K·H)
      ``conv_k_proj`` → conv_k's K views, then a per-head ``K→1`` projection back to H, so
                        the stem stays lean                       → ``[B, H, …]``   (L·K·H + K·H)
      ``mlp``         → per-head nonlinear MLP over the layer axis (``L→hidden→1``), added as
                        a residual on top of the uniform mean     → ``[B, H, …]``
                        (H·(L·hidden + hidden + hidden + 1))
      ``attn``        → content-based softmax over L per head (per-sample adaptive weights);
                        layers compete on a simplex               → ``[B, H, …]``   (H + L·H)

    All modes initialize to the uniform mean (linear weights = ``1/L``; the ``mlp`` residual
    and ``attn`` logits init to 0), so an untrained module reproduces the baseline and
    training departs from there. ``out_channels`` tells the caller how wide to size the stem.
    """

    def __init__(self, n_blocks: int, n_heads: int, mode: str, k: int = 1, hidden: int = 16,
                 step_embed_dim: int = 0, init_blocks: list[int] | None = None,
                 init_random: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.k = k
        self.hidden = hidden
        # cache of the last forward's per-(b,l,h) softmax weights, for entropy regularization
        # (attn / attn_step only); set in forward, read by reg_loss.
        self._last_attn_w: torch.Tensor | None = None
        init = 1.0 / n_blocks
        if mode == "scalar":
            self.weight = nn.Parameter(torch.full((n_blocks,), init))
            self.out_channels = n_heads
        elif mode == "per_head":
            self.weight = nn.Parameter(torch.full((n_blocks, n_heads), init))
            self.out_channels = n_heads
        elif mode == "conv_k":
            self.weight = nn.Parameter(torch.full((n_blocks, k, n_heads), init))
            self.out_channels = k * n_heads
        elif mode == "conv_k_proj":
            # K learned views (each init to the mean) then a per-head K→1 projection
            # (init 1/K) → output averages the K views == uniform mean. Stem stays at H.
            self.weight = nn.Parameter(torch.full((n_blocks, k, n_heads), init))
            self.proj = nn.Parameter(torch.full((n_heads, k), 1.0 / k))
            self.out_channels = n_heads
        elif mode == "mlp":
            # Per-head MLP over the layer axis: L → hidden → 1, applied at every (T,h,w)
            # position with per-head weights, added as a residual to the uniform mean.
            # Last layer zero-init → at init the residual is 0 and the output == mean.
            self.w1 = nn.Parameter(torch.empty(n_heads, n_blocks, hidden))
            nn.init.normal_(self.w1, std=0.02)
            self.b1 = nn.Parameter(torch.zeros(n_heads, hidden))
            self.w2 = nn.Parameter(torch.zeros(n_heads, hidden))  # zero-init → mean at start
            self.b2 = nn.Parameter(torch.zeros(n_heads))
            self.act = nn.SiLU(inplace=False)
            self.out_channels = n_heads
        elif mode == "attn":
            # Content-based softmax over L per head: score = scale_h * pool(x) + bias_{l,h},
            # weights = softmax_l(score). scale=0, bias=0 → uniform softmax == mean at init.
            self.scale = nn.Parameter(torch.zeros(n_heads))
            self.bias = nn.Parameter(torch.zeros(n_blocks, n_heads))
            self.out_channels = n_heads
        elif mode == "attn_step":
            # STEP-CONDITIONED attention: the per-(layer,head) softmax logits get an extra bias
            # that is a function of the denoising-step embedding, so the LEARNED LAYER COMBINE
            # itself varies with the step (early steps weight different transformer blocks than
            # late ones). step_proj is zero-init, so at init the step term is 0 and this reduces
            # to `attn` → uniform mean. Requires step conditioning (the probe passes the step
            # embedding into forward); without it, falls back to plain attn.
            self.scale = nn.Parameter(torch.zeros(n_heads))
            self.bias = nn.Parameter(torch.zeros(n_blocks, n_heads))
            if step_embed_dim <= 0:
                raise ValueError("layer_agg='attn_step' requires step_embed_dim > 0 (set step_cond).")
            self.step_proj = nn.Linear(step_embed_dim, n_blocks * n_heads)
            nn.init.zeros_(self.step_proj.weight)
            nn.init.zeros_(self.step_proj.bias)
            self.out_channels = n_heads
        else:
            raise ValueError(
                f"Unknown layer_agg mode {mode!r}. Choose 'scalar', 'per_head', 'conv_k', "
                f"'conv_k_proj', 'mlp', 'attn', or 'attn_step'."
            )

        # Init strategy for the layer combine (mutually exclusive; warm-start wins if both set):
        #   default     → uniform 1/L (untrained == uniform-mean baseline)
        #   init_blocks → warm-start: concentrate on a known-good subset
        #   init_random → break the uniform symmetry without a subset prior (control)
        if init_blocks:
            self._apply_warmstart(init_blocks)
        elif init_random:
            self._apply_random_init()

    def _apply_random_init(self) -> None:
        """Random per-layer init instead of the uniform 1/L, to break the symmetry that may trap
        the combine near-uniform — without giving the good-subset prior that warm-start does.
        Untrained output no longer equals the mean baseline (intentional for this control)."""
        with torch.no_grad():
            if self.mode in ("scalar", "per_head", "conv_k", "conv_k_proj"):
                self.weight.uniform_(0.0, 2.0 / self.n_blocks)   # mean ~1/L, varied per layer
            elif self.mode in ("attn", "attn_step"):
                self.bias.normal_(0.0, 1.0)                       # asymmetric softmax logits
            else:  # mlp aggregates via a hardcoded uniform mean — no per-layer weight to randomize.
                import warnings
                warnings.warn(f"layer_agg random init ignored for mode={self.mode!r}")

    def _apply_warmstart(self, init_blocks: list[int]) -> None:
        """Warm-start the layer combine to favour a known-good subset of blocks instead of the
        uniform ``1/L``. Linear modes get all mass on the selected blocks (others 0); attention
        modes get a positive logit bias on the selected blocks (softmax concentrates there). The
        combine can still move all 48 layers during training — this only changes where it starts.
        """
        sel = sorted(set(int(b) for b in init_blocks if 0 <= int(b) < self.n_blocks))
        if not sel:
            return
        with torch.no_grad():
            if self.mode in ("scalar", "per_head", "conv_k", "conv_k_proj"):
                # weight leading dim is the layer axis L; concentrate mass on `sel`.
                self.weight.zero_()
                self.weight[sel] = 1.0 / len(sel)
            elif self.mode in ("attn", "attn_step"):
                # bias is [L, H]; a +logit on selected blocks → softmax mass concentrates there.
                self.bias.zero_()
                self.bias[sel] = 4.0
            else:  # mlp aggregates via a hardcoded uniform mean — no per-layer weight to seed.
                import warnings
                warnings.warn(f"layer_agg warm-start ignored for mode={self.mode!r}")

    def reg_loss(self, l1: float = 0.0, entropy: float = 0.0) -> torch.Tensor:
        """Aggregator-targeted regularization, added to the training loss.

        ``l1``      — mean |layer weight|, pushing the linear combine toward selecting few
                      layers (group/instance norm downstream absorbs scale, so this drives
                      sparsity rather than mere shrinkage). Applies to weight-based modes.
        ``entropy`` — mean entropy of the per-sample attention softmax over L, penalised to make
                      the ``attn`` / ``attn_step`` weights peaky (concentrate on few layers).
        """
        dev = next(self.parameters()).device
        total = torch.zeros((), device=dev)
        if l1 and hasattr(self, "weight"):
            total = total + l1 * self.weight.abs().mean()
        if entropy and self.mode in ("attn", "attn_step") and self._last_attn_w is not None:
            w = self._last_attn_w.clamp_min(1e-9)            # [B, L, H]
            ent = -(w * w.log()).sum(dim=1).mean()           # mean over B, H
            total = total + entropy * ent
        return total

    def forward(self, x: torch.Tensor, step_emb: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, H, T, h, w]; einsum contracts the L (layer) axis with the weights.
        # step_emb (optional, [B, step_embed_dim]) is used only by the step-conditioned mode.
        if self.mode == "attn_step":
            pooled = x.mean(dim=(3, 4, 5))                    # [B, L, H]
            score = pooled * self.scale[None, None, :] + self.bias[None, :, :]
            if step_emb is not None:
                score = score + self.step_proj(step_emb).view(-1, self.n_blocks, self.n_heads)
            w = torch.softmax(score, dim=1)
            self._last_attn_w = w
            return torch.einsum("blh,blhtxy->bhtxy", w, x)
        if self.mode == "scalar":
            return torch.einsum("l,blhtxy->bhtxy", self.weight, x)
        if self.mode == "per_head":
            return torch.einsum("lh,blhtxy->bhtxy", self.weight, x)
        if self.mode == "conv_k":
            # [B, L, H, T, h, w] → [B, K, H, T, h, w], then merge (K, H) into channels.
            out = torch.einsum("lkh,blhtxy->bkhtxy", self.weight, x)
            B, K, H, T, hh, ww = out.shape
            return out.reshape(B, K * H, T, hh, ww)
        if self.mode == "conv_k_proj":
            # K views [B, K, H, …] then per-head K→1 projection → [B, H, …]; lean stem.
            views = torch.einsum("lkh,blhtxy->bkhtxy", self.weight, x)
            return torch.einsum("bkhtxy,hk->bhtxy", views, self.proj)
        if self.mode == "mlp":
            # Uniform mean over layers + per-head MLP(L→hidden→1) residual.
            mean = x.mean(dim=1)  # [B, H, T, h, w]
            hid = self.act(torch.einsum("blhtxy,hld->bhtxyd", x, self.w1)
                           + self.b1[None, :, None, None, None, :])
            resid = (torch.einsum("bhtxyd,hd->bhtxy", hid, self.w2)
                     + self.b2[None, :, None, None, None])
            return mean + resid
        # attn: per-(b,l,h) score from the layer-head map's spatial mean, softmax over L.
        pooled = x.mean(dim=(3, 4, 5))                       # [B, L, H]
        score = pooled * self.scale[None, None, :] + self.bias[None, :, :]
        w = torch.softmax(score, dim=1)                      # [B, L, H] (competes over L)
        self._last_attn_w = w
        return torch.einsum("blh,blhtxy->bhtxy", w, x)


class ResBlock3d(nn.Module):
    """Pre-activation (GroupNorm → SiLU → Conv) residual block.

    The kernel is built from two independent axes:

    ``temporal`` → time-axis tap: ``False`` → 1, ``True`` → 3.
    ``spatial``  → h/w-axis tap:  ``True`` (default) → 3, ``False`` → 1.

    So ``temporal=False, spatial=True`` → ``(1, 3, 3)`` (spatial only, default);
    ``temporal=True, spatial=True``  → ``(3, 3, 3)`` (full 3D, mixes time + space);
    ``temporal=True, spatial=False`` → ``(3, 1, 1)`` (pure temporal mixing, ~9× cheaper
    than ``(3, 3, 3)`` — spatial mixing already done by upstream blocks).
    """

    def __init__(
        self,
        ch: int,
        groups: int = 8,
        dropout: float = 0.3,
        temporal: bool = False,
        spatial: bool = True,
        norm: str = "group",
        dropblock_size: int = 0,
    ) -> None:
        super().__init__()
        kt = 3 if temporal else 1
        ks = 3 if spatial else 1
        k = (kt, ks, ks)
        pad = (kt // 2, ks // 2, ks // 2)
        self.norm1 = _make_norm(norm, ch, groups)
        self.conv1 = nn.Conv3d(ch, ch, kernel_size=k, padding=pad, bias=False)
        self.norm2 = _make_norm(norm, ch, groups)
        # dropblock_size > 0 → drop contiguous spatial regions; else whole channel maps.
        self.dropout: nn.Module = (
            DropBlock3d(dropout, dropblock_size) if dropblock_size > 0 else nn.Dropout3d(dropout)
        )
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=k, padding=pad, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return x + h


class StepFiLM(nn.Module):
    """Denoising-step positional encoding → per-channel FiLM modulation.

    For *multi-step pooled* training: one model serves cross-attention features from
    several denoising steps, conditioned on which step produced them. The (integer) step
    value is sinusoidally embedded (diffusion-timestep style), passed through a small
    trunk, and projected to per-channel ``(gamma, beta)`` applied as
    ``x * (1 + gamma) + beta`` at one or more points in the backbone.

    The projection heads are **zero-initialized**, so at init ``gamma = beta = 0`` (identity)
    and the conditioned model reproduces the un-conditioned probe — training a single pooled
    step is then equivalent to the per-step probe, and conditioning departs from there.
    """

    def __init__(self, embed_dim: int, channels: list[int]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.half = embed_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(2 * self.half, embed_dim), nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Linear(embed_dim, 2 * c) for c in channels])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _sinusoid(self, step: torch.Tensor) -> torch.Tensor:
        # step: [B] float → [B, 2*half] sinusoidal features.
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(self.half, device=step.device, dtype=torch.float32)
            / max(1, self.half - 1)
        )
        ang = step.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def embed(self, step: torch.Tensor) -> torch.Tensor:
        return self.trunk(self._sinusoid(step))

    def apply_film(self, idx: int, emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gb = self.heads[idx](emb)                 # [B, 2C]
        c = x.shape[1]
        gamma = gb[:, :c, None, None, None]       # broadcast over T, h, w
        beta = gb[:, c:, None, None, None]
        return x * (1 + gamma) + beta


class ResNetProbe(nn.Module):
    """Spatial-only ResNet stack + temporal-mixing final block → logit.

    Pipeline (defaults, input [B, 32, 13, 16, 16]):
        1. view flat/shaped input → [B, n_blocks_heads, T, h, w]
        2. stem Conv3d (1,3,3): n_blocks_heads → base_ch         (spatial)
        3. n_res_blocks × ResBlock3d (spatial only)
        4. downsample Conv3d (1,3,3) stride (1,2,2): base_ch → 2*base_ch  (h,w halved)
        5. final ResBlock3d (temporal=True): mixes the time axis
        6. GroupNorm → SiLU → AdaptiveAvgPool3d(1) → 1×1×1 Conv3d → [B, 1]
    """

    def __init__(
        self,
        n_blocks_heads: int,
        latent_thw: tuple[int, int, int],  # (T, h, w) of the stored latent
        base_ch: int = 16,
        n_res_blocks: int = 3,
        groups: int = 8,
        dropout: float = 0.3,
        temporal_spatial: bool = True,
        head_dropout: float = 0.0,
        dropblock_size: int = 0,
        norm: str = "group",
        layer_agg: str = "none",
        n_blocks: int = 0,
        layer_agg_k: int = 1,
        layer_agg_hidden: int = 16,
        layer_agg_init_blocks: list[int] | None = None,
        layer_agg_init_random: bool = False,
        step_cond: bool = False,
        step_embed_dim: int = 64,
        multitask: bool = False,
        aux_dim: int = 1,
        perframe: bool = False,
        perframe_head: str = "shared",
        perframe_tap: str = "post_temporal",
        input_norm: str = "none",
        classhead: bool = False,
        n_classes: int = 0,
        class_head_arch: str = "pooled",
        class_mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.n_blocks_heads = n_blocks_heads
        self.latent_thw = latent_thw  # stored for checkpoint / repr
        self.temporal_spatial = temporal_spatial
        # Multi-task: a second, parallel head off the same pooled backbone features. Off by
        # default so forward(x) returns the single class logit exactly as before (every existing
        # inference/steering caller is unchanged). When on, forward(..., return_aux=True) also
        # returns an aux prediction [B, aux_dim] (e.g. the video's aesthetic score) — a shared
        # trunk with two linear heads, the standard hard-parameter-sharing multi-task setup.
        self.multitask = multitask
        self.aux_dim = aux_dim
        # Per-latent-frame presence output. The clip head is a MEAN over (T, h, w) followed
        # by a LINEAR 1×1×1 conv, and a linear map over a mean commutes, so
        #     clip_logit == mean_t [ conv_out( mean_{h,w} feat[:, :, t] ) ]
        # i.e. the per-frame logits are already implicit in every trained checkpoint. With
        # perframe_head="shared" we simply apply the SAME head per frame: zero new
        # parameters, an unchanged state dict, and frame_logits.mean(1) == the clip logit by
        # construction. "separate" gives the frame head its own 1×1×1 conv (initialised from
        # conv_out, so it starts identical) for arms that want the two to diverge.
        # Off by default → forward(x) returns the single clip logit exactly as before.
        self.perframe = perframe
        if perframe_head not in ("shared", "separate"):
            raise ValueError(
                f"Unknown perframe_head {perframe_head!r}. Choose 'shared' or 'separate'."
            )
        self.perframe_head = perframe_head
        # Where the per-frame head TAPS the backbone:
        #   "post_temporal" (default) — reads the temporal_block output, so each frame logit has
        #       a 5-latent-frame receptive field (±2) and, under "shared", the clip-head readout.
        #   "pre_temporal" — a SEPARATE frame branch tapped BEFORE the temporal_block (off the
        #       downsample output, receptive field 1 in time), mirroring the clip head's
        #       post-downsample stack but with NO temporal mixing (a spatial-only ResBlock3d +
        #       its own norm + its own 1×1×1 head). Each frame logit then depends only on that
        #       frame's attention map ⇒ frame-independent by construction. The clip head is left
        #       exactly as-is. Built below (needs out_ch, so after it is defined).
        if perframe_tap not in ("post_temporal", "pre_temporal"):
            raise ValueError(
                f"Unknown perframe_tap {perframe_tap!r}. Choose 'post_temporal' or 'pre_temporal'."
            )
        self.perframe_tap = perframe_tap

        out_ch = 2 * base_ch

        # Learned layer aggregation (optional front-end). When enabled, the input arrives
        # STACKED — [B, n_blocks_heads, T, h, w] with n_blocks_heads = L*H (the dataset's
        # avg_mode="stack") — and a slim LayerAgg collapses the L (block) axis, keeping the
        # H heads as channels. When "none" the legacy contract holds: the input has already
        # been reduced upstream (per_layer slice or avg_channels), so n_blocks_heads IS the
        # stem's channel count and no LayerAgg is built.
        self.layer_agg_mode = layer_agg
        self.layer_n_blocks = n_blocks
        if layer_agg != "none":
            if n_blocks <= 0 or n_blocks_heads % n_blocks != 0:
                raise ValueError(
                    f"layer_agg={layer_agg!r} requires n_blocks > 0 dividing n_blocks_heads "
                    f"(got n_blocks={n_blocks}, n_blocks_heads={n_blocks_heads})."
                )
            self.layer_n_heads = n_blocks_heads // n_blocks
            self.layer_agg: nn.Module | None = LayerAgg(
                n_blocks=n_blocks, n_heads=self.layer_n_heads, mode=layer_agg,
                k=layer_agg_k, hidden=layer_agg_hidden,
                step_embed_dim=(step_embed_dim if step_cond else 0),
                init_blocks=layer_agg_init_blocks,
                init_random=layer_agg_init_random,
            )
            stem_in = self.layer_agg.out_channels
        else:
            self.layer_n_heads = n_blocks_heads
            self.layer_agg = None
            stem_in = n_blocks_heads

        # Head dropout: Dropout3d on the raw input zeros whole head channels (per-layer mode,
        # where each channel is a head; in multi-block mode it drops block-head pairs).
        self.head_drop = nn.Dropout3d(head_dropout)
        # Input mass normalization. The measured box attack is overwhelmingly *inflation*:
        # on the 2026-08-13 run KS(clean, adversarial) was ~1.0 on total attention mass but
        # only ~0.18 on peak. Renormalising each (channel, frame) map to unit spatial mass
        # makes the probe INVARIANT to that direction, removing the attack's main lever by
        # construction. The cost is that absolute attention magnitude may itself carry
        # presence signal, so this can trade clean accuracy for robustness — which is exactly
        # what the sweep measures. "none" keeps the legacy behaviour bit-for-bit.
        # col_frame / col_video (2026-10-11): normalise every RAW input column — one
        # (block, head) attention map of the target word — BEFORE the layer aggregation, so
        # prompts of any length share one input scale (a longer prompt spreads each row's
        # softmax over more text tokens and shrinks the target column ~2.5x on WAN 21f).
        #   col_frame → divide each (block, head, frame) h×w map by its own spatial mean
        #   col_video → divide each (block, head) T×h×w map by its own mean (keeps the
        #               relative temporal profile)
        # Dividing by the mean (not the sum) = sum-normalisation × #positions, keeping values O(1).
        if input_norm not in ("none", "mass", "col_frame", "col_video"):
            raise ValueError(f"Unknown input_norm {input_norm!r}. "
                             "Choose 'none', 'mass', 'col_frame' or 'col_video'.")
        self.input_norm = input_norm

        # Spatial stem: per-frame conv, no temporal mixing. stem_in is the post-aggregation
        # channel count (H or K·H when layer_agg is on; n_blocks_heads otherwise).
        self.stem = nn.Conv3d(stem_in, base_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)

        # Spatial-only residual stack (DropBlock if dropblock_size > 0).
        self.res_blocks = nn.ModuleList(
            ResBlock3d(
                base_ch, groups=groups, dropout=dropout, temporal=False,
                norm=norm, dropblock_size=dropblock_size,
            )
            for _ in range(n_res_blocks)
        )

        # Spatial downsample (h, w → h/2, w/2), channels base_ch → 2*base_ch.
        self.downsample = nn.Conv3d(
            base_ch, out_ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False
        )

        # Final block: this is where the temporal axis is aggregated. With
        # temporal_spatial=False it uses a pure-temporal (3,1,1) kernel (~9× fewer
        # params than (3,3,3)); spatial mixing is already handled by the blocks above.
        # Spatial extent is already halved here, so it keeps plain Dropout3d (no DropBlock).
        self.temporal_block = ResBlock3d(
            out_ch, groups=groups, dropout=dropout, temporal=True, spatial=temporal_spatial,
            norm=norm,
        )

        # Optional denoising-step conditioning (multi-step pooled training). FiLM is
        # applied after the stem (base_ch) and after the spatial downsample (out_ch).
        # Off by default → forward(x) is unchanged; identity-init so it's a no-op until
        # trained even when enabled.
        self.step_cond = step_cond
        self.step_film: StepFiLM | None = (
            StepFiLM(step_embed_dim, channels=[base_ch, out_ch]) if step_cond else None
        )

        # Head: norm → SiLU → avg pool → 1×1×1 conv → logit.
        self.norm_out = _make_norm(norm, out_ch, groups)
        self.act = nn.SiLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_out = nn.Conv3d(out_ch, 1, kernel_size=1)
        # Auxiliary (multi-task) head — same 1×1×1 conv on the pooled features.
        self.conv_out_aux: nn.Module | None = (
            nn.Conv3d(out_ch, aux_dim, kernel_size=1) if multitask else None
        )
        # Class-identity head: predicts WHICH object class (dog / cat / …) the attention map
        # belongs to. Two architectures (class_head_arch):
        #   "pooled" (default) — a 1×1×1 conv on the FULLY-POOLED backbone features, i.e. a linear
        #       readout of the 32-dim (2*base_ch) bottleneck. Cheap, but taps at the very end.
        #   "branch" — a SEPARATE (non-shared) encoder off the AGGREGATED input features (x_in),
        #       mirroring the per-frame "pre_temporal" branch but WITH temporal mixing: its
        #       res-blocks and final block use full (3,3,3) kernels (space+time), a spatial
        #       downsample, then global pool → a fully-connected MLP classification head. Own
        #       fresh weights; the backbone stays frozen. Taps earlier + has real capacity, so it
        #       is not limited to a linear map of the 32-dim pooled vector.
        self.classhead = classhead
        self.n_classes = n_classes
        if class_head_arch not in ("pooled", "branch"):
            raise ValueError(f"Unknown class_head_arch {class_head_arch!r}. Choose 'pooled' or 'branch'.")
        self.class_head_arch = class_head_arch
        if classhead and n_classes < 2:
            raise ValueError(f"classhead=True requires n_classes >= 2 (got {n_classes}).")
        self.conv_out_class: nn.Module | None = None
        self.class_stem: nn.Module | None = None
        self.class_res: nn.ModuleList | None = None
        self.class_down: nn.Module | None = None
        self.class_block: nn.Module | None = None
        self.class_norm: nn.Module | None = None
        self.class_fc: nn.Module | None = None
        if classhead and class_head_arch == "pooled":
            self.conv_out_class = nn.Conv3d(out_ch, n_classes, kernel_size=1)
        elif classhead and class_head_arch == "branch":
            # Own encoder off x_in [B, stem_in, T, h, w]; (3,3,3) blocks mix time+space.
            self.class_stem = nn.Conv3d(stem_in, base_ch, kernel_size=(1, 3, 3),
                                        padding=(0, 1, 1), bias=False)
            self.class_res = nn.ModuleList(
                ResBlock3d(base_ch, groups=groups, dropout=dropout, temporal=True,
                           spatial=True, norm=norm)
                for _ in range(n_res_blocks)
            )
            self.class_down = nn.Conv3d(base_ch, out_ch, kernel_size=(1, 3, 3),
                                        stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
            self.class_block = ResBlock3d(out_ch, groups=groups, dropout=dropout,
                                          temporal=True, spatial=True, norm=norm)
            self.class_norm = _make_norm(norm, out_ch, groups)
            self.class_fc = nn.Sequential(
                nn.Linear(out_ch, class_mlp_hidden), nn.SiLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(class_mlp_hidden, n_classes),
            )
        # Per-frame head. "shared" reuses conv_out (no module, no parameters); "separate"
        # allocates its own conv, copied from conv_out so an untrained probe reproduces the
        # shared behaviour and training departs from there.
        self.conv_out_frame: nn.Module | None = None
        if perframe and perframe_head == "separate" and perframe_tap == "post_temporal":
            self.conv_out_frame = nn.Conv3d(out_ch, 1, kernel_size=1)
            with torch.no_grad():
                self.conv_out_frame.weight.copy_(self.conv_out.weight)
                self.conv_out_frame.bias.copy_(self.conv_out.bias)

        # Frame-INDEPENDENT branch (perframe_tap="pre_temporal"): a SEPARATE spatial-only encoder
        # that mirrors the clip backbone (stem → res-blocks → downsample → block → norm → SiLU →
        # per-frame spatial pool → 1×1×1 head), run with the T axis FOLDED INTO THE BATCH so that
        # every conv AND every GroupNorm acts within a single frame. (A plain x_pre tap is NOT
        # frame-independent: GroupNorm normalizes over (C, T, H, W), pooling across frames — so
        # upstream norms and the block's own norm both leak between frames. Folding T into the
        # batch makes the norm statistics per-frame ⇒ frame_logit[t] depends only on frame t.)
        # Own freshly-initialised weights; the clip head/backbone are untouched and stay frozen
        # under perframe_freeze_backbone.
        self.frame_stem: nn.Module | None = None
        self.frame_res: nn.ModuleList | None = None
        self.frame_down: nn.Module | None = None
        self.frame_block: nn.Module | None = None
        self.frame_norm_out: nn.Module | None = None
        if perframe and perframe_tap == "pre_temporal":
            self.frame_stem = nn.Conv3d(stem_in, base_ch, kernel_size=(1, 3, 3),
                                        padding=(0, 1, 1), bias=False)
            self.frame_res = nn.ModuleList(
                ResBlock3d(base_ch, groups=groups, dropout=dropout, temporal=False,
                           norm=norm, dropblock_size=dropblock_size)
                for _ in range(n_res_blocks)
            )
            self.frame_down = nn.Conv3d(base_ch, out_ch, kernel_size=(1, 3, 3),
                                        stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
            self.frame_block = ResBlock3d(
                out_ch, groups=groups, dropout=dropout, temporal=False,
                spatial=temporal_spatial, norm=norm,
            )
            self.frame_norm_out = _make_norm(norm, out_ch, groups)
            self.conv_out_frame = nn.Conv3d(out_ch, 1, kernel_size=1)

    def frame_head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the head(s) that produce the logits — the only trainable set when
        the backbone is frozen for a head-only per-frame fine-tune.

        For ``perframe_tap="pre_temporal"`` the per-frame head is a fully separate branch
        (``frame_block`` + ``frame_norm_out`` + ``conv_out_frame``); the clip ``conv_out`` is NOT
        included, so a head-only fine-tune leaves the clip probe entirely frozen and trains only
        the new frame-independent readout."""
        if self.perframe_tap == "pre_temporal":
            mods = [self.frame_stem, self.frame_res, self.frame_down, self.frame_block,
                    self.frame_norm_out, self.conv_out_frame]
            return [p for m in mods for p in m.parameters()]
        params = list(self.conv_out.parameters())
        if self.conv_out_frame is not None:
            params += list(self.conv_out_frame.parameters())
        return params

    def class_head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the class-identity head — the only trainable set when the backbone is
        frozen for a head-only class fine-tune (``class_freeze_backbone``). The clip ``conv_out``
        and every backbone module are excluded, so training changes only the new class readout.
        For ``class_head_arch='branch'`` this is the whole separate encoder + FC head."""
        if not self.classhead:
            return []
        if self.class_head_arch == "branch":
            mods = [self.class_stem, self.class_res, self.class_down, self.class_block,
                    self.class_norm, self.class_fc]
            return [p for m in mods for p in m.parameters()]
        return list(self.conv_out_class.parameters()) if self.conv_out_class is not None else []

    def layer_agg_reg(self, l1: float = 0.0, entropy: float = 0.0) -> torch.Tensor:
        """Layer-aggregation regularization for the training loss (0 when no aggregator).

        Delegates to ``LayerAgg.reg_loss``; reads the weights cached by the most recent
        ``forward``, so call it after the forward pass of the same batch.
        """
        if self.layer_agg is None:
            return torch.zeros((), device=self.conv_out.weight.device)
        return self.layer_agg.reg_loss(l1, entropy)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                return_aux: bool = False, return_frames: bool = False,
                return_class: bool = False):
        """``cond`` is the per-sample denoising-step value ``[B]`` for step-conditioned
        (multi-step pooled) probes; ignored when ``step_cond`` is off, so single-step
        callers keep using ``model(x)`` unchanged.

        ``return_aux=True`` (multi-task only) additionally returns the aux head's prediction:
        the call yields ``(logit [B, 1], aux [B, aux_dim])`` instead of the bare logit. Default
        ``False`` → returns just the class logit, so every existing caller is unaffected.

        ``return_frames=True`` (per-frame only) additionally returns the per-latent-frame
        presence logits ``[B, T]``. Requested alone the call yields
        ``(logit, frame_logits)``; together with ``return_aux`` it yields
        ``(logit, aux, frame_logits)``. Under ``perframe_head="shared"``
        ``frame_logits.mean(1) == logit[:, 0]`` exactly (a linear head over a mean commutes),
        so the clip prediction is the frame-average and steering can target e.g.
        ``min_t frame_logits`` for "present in every frame".

        ``return_class=True`` (class head only) additionally returns the class-identity logits
        ``[B, n_classes]``, appended AFTER any aux / frame outputs; requested alone the call
        yields ``(logit, class_logits)``."""
        T, h, w = self.latent_thw
        B = x.shape[0]

        # [B, D] or [B, C, T, h, w] → [B, n_blocks_heads, T, h, w]
        x = x.reshape(B, self.n_blocks_heads, T, h, w)

        # Compute the step embedding up front so the (optional) step-conditioned aggregator
        # can use it too — not just the downstream FiLM.
        emb = self.step_film.embed(cond) if (self.step_film is not None and cond is not None) else None

        if self.input_norm == "col_frame":
            x = x / x.mean(dim=(3, 4), keepdim=True).clamp_min(1e-8)
        elif self.input_norm == "col_video":
            x = x / x.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)

        if self.layer_agg is not None:
            # Stacked input: split L*H back into [B, L, H, T, h, w] and learn the L combine.
            x = x.reshape(B, self.layer_n_blocks, self.layer_n_heads, T, h, w)
            x = self.layer_agg(x, emb)  # → [B, H, T, h, w] (step_emb used only by attn_step)

        if self.input_norm == "mass":
            # per (batch, channel, frame): divide the h×w map by its own sum
            B_, C_, T_, H_, W_ = x.shape
            flat = x.reshape(B_, C_, T_, H_ * W_)
            x = (flat / flat.sum(-1, keepdim=True).clamp_min(1e-8)).reshape(B_, C_, T_, H_, W_)
        x = self.head_drop(x)  # drop whole heads (no-op when head_dropout == 0)
        x_in = x               # post-aggregation input; the frame-independent branch re-encodes it
        x = self.stem(x)
        if emb is not None:
            x = self.step_film.apply_film(0, emb, x)
        for block in self.res_blocks:
            x = block(x)
        x = self.downsample(x)
        x_pre = x                                # pre-temporal tap (RF 1 in time)
        if emb is not None:
            x = self.step_film.apply_film(1, emb, x)
        x = self.temporal_block(x)

        xn = self.act(self.norm_out(x))          # [B, 2*base_ch, T, h/2, w/2]
        x = self.pool(xn)             # [B, 2*base_ch, 1, 1, 1]
        logit = self.conv_out(x).reshape(B, 1)   # class logit [B, 1]

        frame_logits = None
        if return_frames and self.perframe:
            if self.perframe_tap == "pre_temporal":
                # Frame-independent branch: fold T into the batch dim so every conv AND GroupNorm
                # is per-frame, then run the spatial-only encoder + own head. frame_logits[t]
                # depends only on frame t's attention map (verified by the perturbation test).
                Cin = x_in.shape[1]
                xf = x_in.permute(0, 2, 1, 3, 4).reshape(B * T, Cin, 1, h, w)  # [B*T, Cin, 1, h, w]
                xf = self.frame_stem(xf)
                for block in self.frame_res:
                    xf = block(xf)
                xf = self.frame_down(xf)
                xf = self.frame_block(xf)
                xf = self.act(self.frame_norm_out(xf))
                xf = xf.mean(dim=(3, 4), keepdim=True)               # per-frame spatial pool
                frame_logits = self.conv_out_frame(xf).reshape(B, T)
            else:
                # Spatial-only pool keeps the temporal axis, then the (shared or separate)
                # 1×1×1 head scores each latent frame (RF 5: through the temporal_block).
                frame_head = self.conv_out if self.conv_out_frame is None else self.conv_out_frame
                frame_feat = xn.mean(dim=(3, 4), keepdim=True)        # [B, C, T, 1, 1]
                frame_logits = frame_head(frame_feat).reshape(B, T)   # [B, T]

        # Assemble the output tuple in a FIXED order — (logit, [aux], [frame_logits],
        # [class_logits]) — so every existing caller's positional unpacking is preserved
        # (aux before frames, as before) and the class head is simply appended at the end.
        extras: list[torch.Tensor] = []
        if return_aux and self.conv_out_aux is not None:
            extras.append(self.conv_out_aux(x).reshape(B, self.aux_dim))  # [B, aux_dim]
        if frame_logits is not None:
            extras.append(frame_logits)                                    # [B, T]
        if return_class and self.classhead:
            if self.class_head_arch == "branch":
                # Separate encoder off the aggregated input x_in, mixing time+space, then FC.
                xc = self.class_stem(x_in)
                for blk in self.class_res:
                    xc = blk(xc)
                xc = self.class_down(xc)
                xc = self.class_block(xc)
                xc = self.act(self.class_norm(xc))
                xc = self.pool(xc).flatten(1)                 # [B, out_ch]
                extras.append(self.class_fc(xc))               # [B, n_classes]
            else:
                extras.append(self.conv_out_class(x).reshape(B, self.n_classes))  # [B, n_classes]
        if not extras:
            return logit
        return (logit, *extras)
