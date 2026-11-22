"""Steering models — the loss function that drives inference-time optimization.

A ``SteeringModel`` turns the live cross-attention captures of one denoising step
into a scalar loss whose gradient (w.r.t. the latent ``x_t``) nudges generation
toward / away from a target class.

This is a *wrapper* around the trained probes, deliberately decoupled from the probe
architecture so different "keying" strategies can be plugged in (which probe applies
at which denoising step / block). The per-denoising-step keyer
(``StepKeyedSteeringModel``) is implemented here; a per-(step, block) variant is a
future subclass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from will_it_render.common.attnsteer_factory import build_attnsteer
from will_it_render.common.steering_base import SteeringModel
from will_it_render.common.aggregator import aggregate_captures_to_vector

# ``SteeringModel`` ABC and ``build_attnsteer`` were hoisted into ``common`` to break the
# ``steering ↔ baselines`` import cycle and the ``steering → training.trainer`` heavy pull
# (WILL_IT_RENDER_PLAN.md §10b R1). Both are re-exported here (imported above) so existing
# ``from steering.steering_model import SteeringModel`` / ``build_attnsteer`` callers keep working.


def _build_probe_from_ckpt(
    ckpt: dict, device: torch.device, *, force_perframe: bool = False
) -> torch.nn.Module:
    """Reconstruct a probe from a training checkpoint and freeze it.

    The checkpoint (written by ``training/trainer.py``) carries every argument
    ``build_attnsteer`` needs, so this is loader-format-agnostic across probe types.

    ``force_perframe`` turns the per-latent-frame presence head on even when the
    checkpoint didn't record it — used by the timeline steerer, which needs
    ``forward(..., return_frames=True)``. Safe because ``perframe_head="shared"`` adds no
    parameters and keeps the state dict identical (resnet only; ignored otherwise).
    """
    if force_perframe and ckpt.get("probe_model") == "resnet":
        ckpt = {**ckpt, "resnet_perframe": True,
                "resnet_perframe_head": ckpt.get("resnet_perframe_head", "shared")}
    model = build_attnsteer(
        probe_model=ckpt["probe_model"],
        n_blocks_heads=ckpt["n_blocks_heads"],
        latent_thw=tuple(ckpt["latent_thw"]),
        bottleneck_ch=ckpt.get("bottleneck_ch", 128),
        conv_kernel_size=ckpt.get("conv_kernel_size", 3),
        mlp_hidden=ckpt.get("mlp_hidden", 128),
        mlp_hidden_dims=tuple(ckpt.get("mlp_hidden_dims", (512, 128))),
        n_blocks=ckpt.get("n_blocks", 0),
        n_heads=ckpt.get("n_heads", 0),
        attn_pool_conv_ch=ckpt.get("attn_pool_conv_ch", 32),
        attn_pool_mlp_hidden=ckpt.get("attn_pool_mlp_hidden", 64),
        resnet_base_ch=ckpt.get("resnet_base_ch", 16),
        resnet_n_blocks=ckpt.get("resnet_n_blocks", 3),
        resnet_groups=ckpt.get("resnet_groups", 8),
        resnet_temporal_spatial=ckpt.get("resnet_temporal_spatial", True),
        resnet_head_dropout=ckpt.get("resnet_head_dropout", 0.0),
        resnet_dropblock_size=ckpt.get("resnet_dropblock_size", 0),
        resnet_norm=ckpt.get("resnet_norm", "group"),
        resnet_layer_agg=ckpt.get("resnet_layer_agg", "none"),
        resnet_layer_agg_k=ckpt.get("resnet_layer_agg_k", 1),
        resnet_layer_agg_hidden=ckpt.get("resnet_layer_agg_hidden", 16),
        # Input normalisation must follow the checkpoint (mass / col_frame / col_video); before
        # 2026-10-11 it was dropped here, so a normalised probe would have run un-normalised.
        resnet_input_norm=ckpt.get("resnet_input_norm", "none"),
        # Per-frame presence head — needed so ``forward(..., return_frames=True)`` yields
        # the per-latent-frame logits the timeline steerer consumes. With
        # ``perframe_head="shared"`` this adds zero parameters and leaves the state dict
        # unchanged, so it also loads cleanly off a clip-only checkpoint (frame logits are
        # then the clip head applied per frame; ``mean_t == clip_logit``). Off by default,
        # so the per-clip steerer's ``model(feat) -> logit`` path is untouched.
        resnet_perframe=ckpt.get("resnet_perframe", False),
        resnet_perframe_head=ckpt.get("resnet_perframe_head", "shared"),
        resnet_perframe_tap=ckpt.get("resnet_perframe_tap", "post_temporal"),
        resnet_step_cond=ckpt.get("resnet_step_cond", False),
        resnet_step_embed_dim=ckpt.get("resnet_step_embed_dim", 64),
        transformer_d_model=ckpt.get("transformer_d_model", 256),
        transformer_nhead=ckpt.get("transformer_nhead", 4),
        transformer_num_layers=ckpt.get("transformer_num_layers", 3),
        transformer_dim_feedforward=ckpt.get("transformer_dim_feedforward", 512),
        transformer_head_attn_nheads=ckpt.get("transformer_head_attn_nheads", 4),
        transformer_layer_attn_nheads=ckpt.get("transformer_layer_attn_nheads", 4),
        transformer_token_hidden=ckpt.get("transformer_token_hidden", 0),
        transformer_token_mode=ckpt.get("transformer_token_mode", "linear"),
        transformer_token_conv_ch=ckpt.get("transformer_token_conv_ch", 32),
        transformer_mlp_hidden=ckpt.get("transformer_mlp_hidden", 128),
        transformer_pool=ckpt.get("transformer_pool", "query"),
        transformer_timestep_mode=ckpt.get("transformer_timestep_mode", "none"),
        transformer_num_timesteps=ckpt.get("transformer_num_timesteps", 0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class StepKeyedSteeringModel(SteeringModel):
    """One probe per denoising step (per-step, non per_layer probes).

    Loads ``probe_step_NNN.pt`` checkpoints from a directory and keys them by the
    denoising-step value they were trained on (which equals the sampler's
    ``step_idx``). At each steered step it aggregates the captures into the exact
    feature vector the probe was trained on (reusing ``aggregate_captures_to_vector``,
    kept differentiable), reshapes it the way ``ProbeDataset`` does, and computes
    ``BCE(probe_logit, target)`` — ``target=1.0`` induces the class, ``0.0`` suppresses.

    Assumes the probe was trained on a **full-block** extraction (``capture_blocks:
    null``): the steering loop hooks all transformer blocks, so the aggregated vector
    has ``n_blocks_total * n_heads`` block-head channels. Per-layer (per-(step, block))
    probes are rejected at load time — they are a future subclass.
    """

    def __init__(
        self,
        checkpoints_dir: str | Path,
        device: torch.device,
        agg_kwargs: dict,
        probe_config: dict,
        target: float = 1.0,
        weight: float = 1.0,
        balance_weight: float = 0.0,
        loss_type: str = "mean",
        steps_filter: Optional[list[int]] = None,
        nearest_probe: bool = False,
        n_iters_per_step: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.target = float(target)
        self.weight = float(weight)
        # How the per-object BCEs are combined into the step loss:
        #   "mean"     — average of the per-object BCEs (the original behaviour).
        #   "worst"    — only the single worst object (max BCE = the object furthest from
        #                target); focuses the whole gradient on the currently-hardest object.
        #   "inv_conf" — confidence-weighted average: weight_i = |target − score_i|
        #                (detached), so an object already at the target contributes ~0 and a
        #                far-from-target object contributes ~1. An object at prob 1.0 (target
        #                1) gets weight ~0, one at prob 0.1 gets weight ~0.9. Falls back to a
        #                uniform mean when every object is already satisfied (Σweight → 0).
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in ("mean", "worst", "inv_conf"):
            raise ValueError(
                f"Unknown loss_type {loss_type!r}; choose 'mean', 'worst' or 'inv_conf'.")
        # Penalises imbalance across multiple target objects: + balance_weight *
        # (max_i score_i − min_i score_i). Keeps every object steered, not just the
        # easiest. 0 = off; no effect with a single target word.
        self.balance_weight = float(balance_weight)
        # extract_keys / spatial_downsample / latent_hw — must match how the probe's
        # training features were produced (read from the model config's extraction:).
        self.agg_kwargs = dict(agg_kwargs)
        self.steps_filter = set(steps_filter) if steps_filter is not None else None
        # When True, a denoising step with no probe of its own borrows the closest
        # available probe (see _probe_step); lets a contiguous steps_to_steer range
        # steer every step even though probes were only captured at a sparse subset.
        self.nearest_probe = bool(nearest_probe)
        # Optional per-denoising-step gradient-iteration budget: {step_idx: n_iters}.
        # Steps not in the map fall back to the global n_iters (run_steering_optimization).
        # Lets a schedule spend more iterations on the steps that matter most (e.g. early).
        self.n_iters_per_step = (
            {int(k): int(v) for k, v in n_iters_per_step.items()} if n_iters_per_step else None
        )

        # From the extraction probe_config.yaml: which transformer blocks were captured
        # (the steering loop hooks exactly these) and the *pre-reduction* block/head
        # counts. The checkpoint's own n_blocks is post-reduction (1 for avg_mode
        # layers/all), so we must reshape the aggregated vector with these instead.
        cb = probe_config.get("capture_blocks")
        self.capture_blocks: Optional[list[int]] = sorted(int(b) for b in cb) if cb else None
        self.n_blocks_extracted = int(probe_config["n_blocks"])
        self.n_heads = int(probe_config["n_heads"])
        self.n_blocks_total = int(probe_config.get("n_blocks_total", self.n_blocks_extracted))

        self.probes: dict[int, dict] = {}
        ckpt_dir = Path(checkpoints_dir)
        ckpt_paths = sorted(ckpt_dir.glob("**/probe_step_*.pt"))
        # Prefer the canonical best-epoch checkpoint; skip the *_last.pt siblings.
        ckpt_paths = [p for p in ckpt_paths if not p.stem.endswith("_last")]
        if not ckpt_paths:
            raise FileNotFoundError(
                f"No probe_step_*.pt checkpoints found under {ckpt_dir}"
            )
        for path in ckpt_paths:
            ckpt = torch.load(str(path), map_location=device)
            if ckpt.get("block_index") is not None:
                print(
                    f"[steering] skipping per-layer probe {path.name} "
                    f"(block_index={ckpt['block_index']}); StepKeyedSteeringModel "
                    f"supports per-step probes only."
                )
                continue
            step = int(ckpt["step"])
            self.probes[step] = {
                "model": self._load_probe(ckpt),
                "avg_mode": ckpt.get("avg_mode", "none"),
                "expected_channels": int(ckpt["n_blocks_heads"]),  # post-reduction channels
                "latent_thw": tuple(ckpt["latent_thw"]),
            }
        if not self.probes:
            raise RuntimeError(
                f"No usable per-step probes loaded from {ckpt_dir} "
                f"(all checkpoints were per-layer?)."
            )
        self._probe_steps = sorted(self.probes)
        steered = sorted(s for s in self.probes if self.should_steer(s))
        near = (" [nearest-probe on: steps without a probe borrow the closest one]"
                if self.nearest_probe else "")
        print(f"[steering] loaded probes for steps {self._probe_steps}; "
              f"will steer steps {steered}{near} (target={self.target}, weight={self.weight})")

    def _load_probe(self, ckpt: dict) -> torch.nn.Module:
        """Reconstruct one per-step probe from its checkpoint. Overridable so a subclass
        can force the per-frame head on (see :class:`PerFrameTimelineSteeringModel`)."""
        return _build_probe_from_ckpt(ckpt, self.device)

    def _probe_step(self, step_idx: int) -> Optional[int]:
        """Denoising step → the probe-step key to score it with.

        Exact match when a probe exists at ``step_idx``. Otherwise, when
        ``nearest_probe`` is enabled, the closest available probe step (ties broken
        toward the earlier step); else ``None`` (steering is skipped at that step).
        """
        if step_idx in self.probes:
            return step_idx
        if not self.nearest_probe or not self._probe_steps:
            return None
        return min(self._probe_steps, key=lambda s: (abs(s - step_idx), s))

    def should_steer(self, step_idx: int) -> bool:
        if self.steps_filter is not None and step_idx not in self.steps_filter:
            return False
        return self._probe_step(step_idx) is not None

    def _format(self, vec: torch.Tensor, entry: dict) -> torch.Tensor:
        """Flat aggregated vector → ``[1, C, T, h, w]`` probe input.

        Reshapes the captured-block stack ``[n_blocks_extracted, n_heads, T, h, w]``
        then applies the checkpoint's ``avg_mode`` — mirroring
        ``ProbeDataset._load_feat`` (the captured-blocks / non-scatter path).
        """
        T, h, w = entry["latent_thw"]
        nbe, nh = self.n_blocks_extracted, self.n_heads
        expected = nbe * nh * T * h * w
        if vec.numel() != expected:
            raise ValueError(
                f"Steering feature size {vec.numel()} != expected {expected} "
                f"(= n_blocks_extracted {nbe} × n_heads {nh} × T {T} × h {h} × w {w}). "
                f"Check that the loop hooked capture_blocks={self.capture_blocks} and "
                f"that extract_keys/spatial_downsample match the extraction config."
            )
        feat = vec.view(nbe, nh, T, h, w)
        # TEMPORARY HACK (2026-10-11) — dataset-level feature renorm, see
        # _install_feature_renorm_HACK. No-op unless steering.model.feature_renorm is set.
        # REMOVE once the probe is retrained with a column normalisation.
        if entry.get("renorm_hack") is not None:
            a, b = entry["renorm_hack"]
            feat = feat * a.to(feat.device, feat.dtype) + b.to(feat.device, feat.dtype)
        if entry.get("renorm_prompt_hack") is not None:
            # per-PROMPT: this clip's own (block, head) mean/std over all T*h*w positions are
            # mapped to the training set's typical per-clip values (differentiable).
            mode, M, S = entry["renorm_prompt_hack"]
            M = M.to(feat.device, feat.dtype)
            m = feat.mean(dim=(2, 3, 4), keepdim=True)
            if mode == "prompt_scale":
                feat = feat * M / m.clamp_min(1e-12)
            else:  # prompt_affine
                sd = feat.std(dim=(2, 3, 4), keepdim=True, unbiased=False).clamp_min(1e-12)
                feat = (feat - m) / sd * S.to(feat.device, feat.dtype) + M
        avg_mode = entry["avg_mode"]
        if avg_mode == "layers":
            feat = feat.mean(dim=0)  # average over captured blocks → [n_heads, T, h, w]
        elif avg_mode == "all":
            feat = feat.reshape(nbe * nh, T, h, w).mean(dim=0, keepdim=True)  # [1, T, h, w]
        elif avg_mode in ("none", "stack"):
            # "none" (full-block) and "stack" (captured-blocks subset, layer_agg probes)
            # both feed the un-averaged block-major stack [n_blocks_extracted*n_heads, T, h, w];
            # mirrors ProbeDataset._load_feat's capture_blocks stack path.
            feat = feat.reshape(nbe * nh, T, h, w)
        else:
            raise ValueError(f"Unsupported avg_mode {avg_mode!r} for steering.")
        if feat.shape[0] != entry["expected_channels"]:
            raise ValueError(
                f"Formatted channels {feat.shape[0]} != probe's expected "
                f"{entry['expected_channels']} for avg_mode={avg_mode!r}. The per-step "
                f"steerer supports avg_mode layers/all on subset captures, or none on "
                f"full-block extractions; per-(step,block) probes are not supported here."
            )
        return feat.unsqueeze(0)  # add batch dim

    def loss(
        self,
        captures: list,
        step_idx: int,
        *,
        target_words_indices: dict[str, list[int]],
    ) -> tuple[torch.Tensor, dict]:
        probe_step = self._probe_step(step_idx)
        entry = self.probes[probe_step]
        if not target_words_indices:
            raise ValueError("target_words_indices is empty; nothing to steer toward.")
        # One probe score per word: slice + mean-pool this word's (subword) attention
        # exactly like training's aggregate_captures_per_word, format, then score. The
        # captures forward is shared across words; only the cheap probe pass repeats.
        # Differentiable: aggregate_captures_to_vector uses pure torch ops and the
        # captures are non-detached (probe context built with detach=False).
        per_word_bce: list[torch.Tensor] = []
        per_word_score: list[torch.Tensor] = []   # differentiable per-word probabilities
        per_word_prob: dict[str, float] = {}
        for word, indices in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
            # Probes are fp32; the aggregated feature inherits the capture dtype, which is
            # bf16 on the CogVideoX steering path (fp32 captures OOM a 94-96 GB card). Upcast
            # before the probe. No-op for the fp32-capture backends (LTX/WAN).
            feat = self._format(vec, entry).to(self.device).float()
            logit = entry["model"](feat)  # [1, 1]
            target = torch.full_like(logit, self.target)
            per_word_bce.append(F.binary_cross_entropy_with_logits(logit, target))
            score = torch.sigmoid(logit).mean()
            per_word_score.append(score)
            per_word_prob[word] = float(score.detach())
        # Combine the per-word BCEs (see loss_type doc in __init__), then scale by the
        # objective weight. The magnitude regularizer is added by the steering loop on top.
        bce_stack = torch.stack(per_word_bce)      # [n_words]
        score_stack = torch.stack(per_word_score)  # [n_words], differentiable probs
        bce_mean = bce_stack.mean()                # raw mean, always logged for comparability
        if self.loss_type == "worst" and len(per_word_bce) > 1:
            combined = bce_stack.max()             # only the single hardest object
        elif self.loss_type == "inv_conf" and len(per_word_bce) > 1:
            # weight_i = |target − score_i|, detached (a weighting, not part of the
            # objective's own derivative). Satisfied objects (score≈target) → weight≈0.
            w = (self.target - score_stack).abs().detach()
            wsum = w.sum()
            if float(wsum) < 1e-6:                 # everyone satisfied → nothing to focus on
                combined = bce_mean
            else:
                combined = (w * bce_stack).sum() / wsum
        else:
            combined = bce_mean
        loss = combined * self.weight
        # Balance term: penalise the spread between the best- and worst-scored objects,
        # so steering raises every object's score rather than just the easiest one.
        # The spread is always computed (for reporting/before-after analysis); it is only
        # *added to the loss* when balance_weight > 0. No-op for a single target word.
        balance = torch.zeros((), device=loss.device)
        if len(per_word_score) > 1:
            scores = torch.stack(per_word_score)
            balance = scores.max() - scores.min()
            if self.balance_weight:
                loss = loss + self.balance_weight * balance
        prob = sum(per_word_prob.values()) / len(per_word_prob)
        return loss, {
            "step": step_idx,
            "probe_step": probe_step,
            "prob": prob,
            "per_word_prob": per_word_prob,
            "balance": float(balance.detach()),
            "bce": float(bce_mean.detach()),   # raw mean BCE (pre-weight)
            "loss": float(loss.detach()),
        }

    def per_object_bce(self, captures, step_idx, *, target_words_indices):
        """Diagnostic: per-object BCE tensors (differentiable), one per target word.

        Same per-word computation as :meth:`loss` but returns each word's BCE separately,
        so a caller can ``autograd.grad`` each toward ``x_t`` and compare per-object
        gradient directions (multi-object interaction analysis). Not used by steering.
        """
        entry = self.probes[self._probe_step(step_idx)]
        out: dict[str, torch.Tensor] = {}
        for word, indices in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
            # Same bf16→fp32 upcast as loss(): cogx captures are bf16, probes are fp32.
            feat = self._format(vec, entry).to(self.device).float()
            logit = entry["model"](feat)
            target = torch.full_like(logit, self.target)
            out[word] = F.binary_cross_entropy_with_logits(logit, target)
        return out

    def per_object_prob_with(self, captures, probe_step, *, target_words_indices):
        """Per-object probability under an EXPLICITLY chosen probe, as a held-out judge.

        Steering optimises the probe that ``_probe_step`` selects for the current denoising
        step, and that probe saturates: by the end of a step BOTH present and absent objects
        read >0.99 (measured final gap 0.000-0.005), so its own score cannot say whether the
        gain is real or a shortcut into its blind spot. A probe trained at a DIFFERENT
        denoising step was never optimised against here, so its reading is not saturated by
        construction and acts as an independent judge of the same latent.

        Returns ``{word: p}`` (detached) or ``{}`` if that probe step is not loaded.
        """
        entry = self.probes.get(probe_step)
        if entry is None:
            return {}
        out: dict[str, float] = {}
        with torch.no_grad():
            for word, indices in target_words_indices.items():
                vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
                logit = entry["model"](self._format(vec, entry).to(self.device))
                out[word] = float(torch.sigmoid(logit.detach()).mean())
        return out

    def per_object_bce_with(self, captures, probe_step, *, target_words_indices):
        """Per-object BCE tensors under an EXPLICITLY chosen probe (differentiable).

        Same computation as :meth:`per_object_bce` but scores ``self.probes[probe_step]``
        instead of the step's nearest probe. Used by the PROBE_MIX steering variations
        (steering_core) to build the update from an alternate or a blended (own+neighbour)
        probe rather than only the step's own — the own probe saturates within a step, so a
        neighbour that was not optimised against provides an independent, non-saturated signal.
        Returns ``{word: bce_tensor}`` (grad flows to ``x_t`` via ``captures``).
        """
        entry = self.probes[probe_step]
        out: dict[str, torch.Tensor] = {}
        for word, indices in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
            feat = self._format(vec, entry).to(self.device)
            logit = entry["model"](feat)
            target = torch.full_like(logit, self.target)
            out[word] = F.binary_cross_entropy_with_logits(logit, target)
        return out

    def probe_neighbors(self, step_idx: int) -> list:
        """The probe steps ADJACENT to this step's own probe, in the sorted loaded-probe list.

        "Own" = the probe :meth:`_probe_step` selects for ``step_idx`` (its own step, or the
        nearest under ``nearest_probe``); the neighbours are the immediately lower and higher
        loaded probe steps. Interior probes have two neighbours, endpoints have one. This is the
        neighbour set the PROBE_MIX alternate/common-loss variations cycle over.
        """
        own = self._probe_step(step_idx)
        if own is None or own not in self._probe_steps:
            return []
        i = self._probe_steps.index(own)
        nbrs = []
        if i - 1 >= 0:
            nbrs.append(self._probe_steps[i - 1])
        if i + 1 < len(self._probe_steps):
            nbrs.append(self._probe_steps[i + 1])
        return nbrs

    @property
    def probe_steps(self) -> list:
        """Denoising steps that have a loaded probe (sorted)."""
        return sorted(self.probes)

    def per_object_attention(self, captures, step_idx, *, target_words_indices):
        """Diagnostic: per-object cross-attention map [T, h, w] (mean over blocks + heads).

        Same aggregation the probe/spatial_entropy consume — the target word's cross-attention
        toward its (subword) text tokens, averaged over the captured blocks and heads, kept at
        the latent (frame T, height h, width w) resolution. Detached; for visualization only.
        """
        T, h, w = self.probes[self._probe_step(step_idx)]["latent_thw"]
        nbe, nh = self.n_blocks_extracted, self.n_heads
        out: dict[str, torch.Tensor] = {}
        for word, indices in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
            out[word] = vec.view(nbe, nh, T, h, w).mean(dim=(0, 1)).detach()  # [T, h, w]
        return out

    def spatial_entropy(
        self,
        captures: list,
        step_idx: int,
        *,
        target_words_indices: dict[str, list[int]],
    ) -> torch.Tensor:
        """Mean per-frame **spatial entropy** of the target-word attention (↓ = more focused).

        Attention-space regularizer copied from TempoControl (Schiber et al., *Temporal
        Attention Guidance for Text-to-Video Models*, arXiv:2510.02226, eq. 8) — see
        ``baselines/tempocontrol/loss.py`` for the original port. Build ``A[T, h, w]`` as
        the target word's cross-attention averaged over blocks, heads and its subword
        tokens (the same aggregation the probe consumes), then per frame
        ``H_t = −Σ_{h,w} p·log p`` with ``p`` the frame's spatial distribution, averaged
        over **all** frames (the "present in every frame" control signal → all frames
        active). Lower entropy = attention concentrated on a compact region (a clean,
        localized object). Differentiable w.r.t. ``x_t`` via the non-detached captures;
        the steering loop scales this by ``entropy_reg`` and adds it to the probe BCE.
        """
        T, h, w = self.probes[self._probe_step(step_idx)]["latent_thw"]
        nbe, nh = self.n_blocks_extracted, self.n_heads
        per_word_H: list[torch.Tensor] = []
        for _word, indices in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
            A = vec.view(nbe, nh, T, h, w).mean(dim=(0, 1))     # [T, h, w] — mean over blocks+heads
            flat = A.reshape(T, h * w)
            p = flat / (flat.sum(dim=1, keepdim=True) + 1e-10)  # per-frame spatial distribution
            H_t = -(p * torch.log(p + 1e-10)).sum(dim=1)        # [T]
            per_word_H.append(H_t.mean())                       # mean over all frames
        return torch.stack(per_word_H).mean()                   # mean over target words


class PerFrameTimelineSteeringModel(StepKeyedSteeringModel):
    """Timeline steering: drive *when* each object is present via the per-frame head.

    Same per-step probe loading / capture-aggregation as
    :class:`StepKeyedSteeringModel`, but the objective is **per latent frame**: for each
    target word the probe emits per-frame presence logits
    (``ResNetProbe.forward(return_frames=True)`` → ``[1, T]``), and the loss is the BCE
    of those logits against a **user-supplied binary timeline** ``c ∈ {0,1}^T`` (which
    frames the object should exist). The per-clip steerer is the special case
    ``c = all-ones`` under ``frame_reduce="mean"`` (since ``mean_t frame_logit ==
    clip_logit``).

    Timelines are supplied per prompt via :meth:`set_prompt_timelines` (keyed by the same
    surface target word as ``target_words_indices``) and resampled to the probe's ``T``
    latent frames; a word with no timeline defaults to all-ones (induce everywhere).
    """

    def __init__(self, *args, frame_reduce: str = "balanced",
                 present_weight: float = 1.0, absent_weight: float = 1.0,
                 loss_kind: str = "bce", word_frame_order: str = "word_first",
                 huber_m: float = 2.0, rank_margin: float = 1.0,
                 anchor_weight: float = 1.0, frame_grad_mask: bool = False,
                 frame_mask_lo: float = 0.0, frame_mask_hi: float = 1.0,
                 frame_mask_ramp: int = 1,
                 class_select: str = "both",
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # class_select: WHICH half of the per-frame objective is optimized (balanced modes only).
        #   "both"         — present + absent, the established objective (default, unchanged).
        #   "absent_only"  — only the frames the object must be ABSENT from carry a gradient.
        #   "present_only" — only the frames it is REQUESTED in.
        #   "greedy_max"   — per optimization iteration, whichever class currently has the
        #                    LARGER loss is optimized alone; the choice can flip between
        #                    iterations as the leading class is paid down.
        # The two halves fight through a shared presence level (DEBUG.md §2, §6): "both" parks
        # at the equalized fixed point, the single-class arms show what each half can do with
        # the level free, and greedy_max alternates instead of averaging.
        self.class_select = str(class_select).lower()
        _CS = ("both", "absent_only", "present_only", "greedy_max")
        if self.class_select not in _CS:
            raise ValueError(f"Unknown class_select {class_select!r}; choose one of {_CS}.")
        # Per-word record of which class the last loss() actually optimized (for the trace).
        self._class_choice: list[str] = []
        # Route the two frame classes to DISJOINT latent frames (see per_frame_class_losses).
        # Read by the steering loop, which then builds the update from two backward passes
        # instead of one; a no-op unless loss_kind == "bce".
        self.frame_grad_mask = bool(frame_grad_mask)
        # Softened routing. The hard mask (lo=0, hi=1, ramp=1) sends each half's gradient ONLY to
        # its own frames, which makes the boundary frames a discontinuity: frame f is written
        # exclusively by the present half and frame f+1 exclusively by the absent half, even
        # though the object has to physically appear/disappear across that seam. lo/hi bound the
        # routing weight away from 0/1 so each half keeps a minority share everywhere, and ramp
        # is the width (in latent frames, odd) of the box-smoothing applied to the control before
        # it is rescaled — so the transition is spread over `ramp` frames instead of one step.
        # The absent half always gets (1 - m), so the per-frame weights still sum to 1 and the
        # step size is matched to the hard mask.
        self.frame_mask_lo = float(frame_mask_lo)
        self.frame_mask_hi = float(frame_mask_hi)
        self.frame_mask_ramp = int(frame_mask_ramp)
        if not 0.0 <= self.frame_mask_lo <= self.frame_mask_hi <= 1.0:
            raise ValueError(f"need 0 <= frame_mask_lo <= frame_mask_hi <= 1; got "
                             f"lo={self.frame_mask_lo}, hi={self.frame_mask_hi}.")
        if self.frame_mask_ramp < 1 or self.frame_mask_ramp % 2 == 0:
            raise ValueError(f"frame_mask_ramp must be a positive ODD frame count (it is a "
                             f"centered box width); got {self.frame_mask_ramp}.")
        self.rank_margin = float(rank_margin)     # logit margin for rank/margin loss_kinds
        self.anchor_weight = float(anchor_weight)  # clip-level anchor for centered_bce
        self.huber_m = float(huber_m)   # bce_huber ceiling on the per-frame gradient
        # word_frame_order: how per-word and per-frame axes are collapsed (multi-object only).
        #   "word_first" (default) — reduce each word's per-frame BCE to a scalar (frame_reduce)
        #                            then combine words (loss_type). The established path.
        #   "frame_first"          — mean the per-word per-frame BCE ACROSS WORDS at each frame,
        #                            then reduce over frames (plain mean over decided frames).
        self.word_frame_order = str(word_frame_order).lower()
        if self.word_frame_order not in ("word_first", "frame_first"):
            raise ValueError(f"Unknown word_frame_order {word_frame_order!r}; "
                             "choose 'word_first' or 'frame_first'.")
        # loss_kind: how the per-frame logits are compared to the timeline.
        #   "bce" — per-frame BCE toward c_t, reduced by frame_reduce (balanced|mean).
        #   "kl"  — temporal-distribution KL: q=softmax_t(frame_logits), r=control/Σcontrol,
        #           loss=KL(r‖q). A relative/shape objective — concentrates presence MASS on
        #           the requested frames (0-cost where c=0), unlike BCE's absolute per-frame
        #           targets. (Per-frame *Bernoulli* KL ≡ BCE for hard targets, so we use the
        #           frame-distribution form.)
        #   "pearson" (aka "variance"/"corr") — TempoControl-style: MAXIMIZE the Pearson
        #           correlation between the per-frame prediction profile and the control
        #           signal (loss = 1 − corr). Scale-/shift-invariant ⇒ rewards temporal
        #           CONTRAST, not global level; cannot be gamed by all-on/all-off presence.
        # The next three are the LEVEL-INVARIANT contrast objectives (see _contrast_loss):
        #   "rank"         — pairwise AUC surrogate over (absent, present) frame-logit pairs
        #   "margin"       — the same, through the class means only
        #   "centered_bce" — BCE on mean-removed frame logits (shape) + a clip-level anchor
        #                    (level), so shape and level are separate knobs
        self.loss_kind = str(loss_kind).lower()
        if self.loss_kind in ("variance", "corr"):
            self.loss_kind = "pearson"                # aliases → TempoControl-style corr
        _LK = ("bce", "kl", "pearson", "rank", "margin", "centered_bce")
        if self.loss_kind not in _LK:
            raise ValueError(f"Unknown loss_kind {loss_kind!r}; choose one of {_LK}.")
        self.frame_reduce = str(frame_reduce).lower()
        # *_max reduce over each class with the WORST frame (max BCE) → drives the MIN
        # per-frame logit up ⇒ "present in EVERY frame" (appear everywhere). *_min reduce
        # with the BEST frame (min BCE) → drives the MAX per-frame logit up ⇒ "present in
        # AT LEAST ONE frame". With an all-ones timeline both are pure per-clip presence
        # objectives routed through the per-frame head (no temporal target needed).
        _FR = ("balanced", "mean", "balanced_max", "balanced_min", "logsum_frames",
               "log_max", "log_min", "log_mean", "gb_mean", "gb_max", "gb_min",
               "detached_weight", "neg_sumlog_score", "bce_huber")
        if self.frame_reduce not in _FR:
            raise ValueError(f"Unknown frame_reduce {frame_reduce!r}; choose one of {_FR}.")
        # gb_* modes expose the (present, absent) per-class sub-losses to the loop's
        # optimization.grad_balance (unit-norm each gradient), so they REQUIRE grad_balance.
        self._frame_grad_balance = self.frame_reduce in ("gb_mean", "gb_max", "gb_min")
        # In "balanced" mode these weight the present-frame vs absent-frame mean (a ½/½
        # split by default) so a mostly-0 or mostly-1 timeline can't let the majority
        # class dominate the gradient — the loss-side mirror of presence-acc/absent-acc.
        self.present_weight = float(present_weight)
        self.absent_weight = float(absent_weight)
        # class_select rides on the same per-class split, so it only exists where that split
        # does: the contrast objectives (rank/margin/centered_bce) and the non-balanced reduces
        # never form the two scalars. Fail loudly rather than silently ignoring the knob.
        if self.class_select != "both" and (
                self.loss_kind != "bce" or self.frame_reduce not in ("balanced", "balanced_max")):
            raise ValueError(
                f"class_select={self.class_select!r} needs loss_kind='bce' and "
                f"frame_reduce in ('balanced','balanced_max'); got loss_kind={self.loss_kind!r}, "
                f"frame_reduce={self.frame_reduce!r}.")
        self._raw_timelines: dict[str, list[float]] = {}
        self._timeline_cache: dict[tuple[str, int], torch.Tensor] = {}

    # Rebuild each per-step probe with the per-frame head forced on, so
    # ``model(feat, return_frames=True)`` yields per-latent-frame logits even off a
    # clip-only checkpoint (shared head = zero new params, identical state dict).
    def _load_probe(self, ckpt: dict) -> torch.nn.Module:
        return _build_probe_from_ckpt(ckpt, self.device, force_perframe=True)

    def set_prompt_timelines(self, timelines: dict[str, list[float]]) -> None:
        """Per-prompt hook (mirrors the baseline ``set_prompt_context``): store the raw
        per-word binary timelines. Resampling to the probe's ``T`` happens lazily in
        :meth:`_timeline_for` and is cached per ``(word, T)``."""
        self._raw_timelines = dict(timelines or {})
        self._timeline_cache.clear()

    def _timeline_for(self, word: str, T: int) -> torch.Tensor:
        """The word's timeline as a ``[T]`` tensor on ``self.device`` (cached).

        Missing word → all-ones (induce in every frame = the per-clip objective) with a
        one-time warning. Any other length is resampled to ``T`` via
        ``_resolve_control_signal`` (``adaptive_max_pool1d`` — the TempoControl helper)."""
        key = (word, T)
        cached = self._timeline_cache.get(key)
        if cached is not None:
            return cached
        raw = self._raw_timelines.get(word)
        if raw is None:
            print(f"[timeline] no timeline for target word {word!r}; defaulting to "
                  f"all-ones (present in every frame)")
        # Reuse the baseline's resampler ('all_ones'/list → [T]); imported lazily to keep
        # this module independent of the baselines package.
        from will_it_render.evaluation.baselines.tempocontrol.loss import _resolve_control_signal

        sig = _resolve_control_signal(raw, T, self.device)
        self._timeline_cache[key] = sig
        return sig

    def _contrast_loss(self, frame_logits: torch.Tensor, clip_logit: torch.Tensor,
                       c: torch.Tensor, decided: torch.Tensor) -> torch.Tensor:
        """Level-INVARIANT timeline objectives, computed on the per-frame LOGITS.

        Every ``frame_reduce`` mode is a function of the ABSOLUTE per-frame probabilities,
        but a latent nudge mostly moves a global presence *level* shared by both classes:
        raise it and the present loss falls while the absent loss rises. The optimizer then
        parks at an equalized fixed point where neither class can improve without the other
        getting worse (measured on WAN oo/18 —
        ``experiment_reports/2026_08_26_STEER_EVAL_wan_temporal_control/DEBUG.md`` §2, §6).

        These three score the CONTRAST between the two classes instead, and the level
        cancels out of all of them, so both halves can fall together.
        """
        z, cc = frame_logits[decided], c[decided]
        pres, abst = z[cc >= 0.5], z[cc < 0.5]
        if pres.numel() == 0 or abst.numel() == 0:
            # A constant timeline has no contrast to score; fall back to plain BCE so an
            # all-present / all-absent word still contributes a gradient.
            return (F.binary_cross_entropy_with_logits(z, cc) if decided.any()
                    else frame_logits.sum() * 0.0)
        m = self.rank_margin
        if self.loss_kind == "rank":
            # Pairwise AUC surrogate: every (absent, present) pair should be ordered by at
            # least m. The level cancels exactly inside z_s − z_t, so only the ORDERING is
            # scored. [n_abs, n_pres] — at most ~T²/4 pairs (T=21 on WAN), so free.
            return F.softplus(abst[:, None] - pres[None, :] + m).mean()
        if self.loss_kind == "margin":
            # Same invariance through the class MEANS only: one scalar, coarser (it cannot
            # see individual mis-ordered frames) but with a much smoother gradient.
            return F.softplus(m - (pres.mean() - abst.mean()))
        # "centered_bce": BCE on the frame logits with their own mean removed, so the loss
        # sees only the SHAPE, plus a clip-level anchor that owns the LEVEL. Splitting them
        # means neither class pays for the other: the shape term can carve the timeline
        # while the anchor independently keeps the object in the clip at all.
        shape = F.binary_cross_entropy_with_logits(z - z.mean(), cc)
        anchor = F.binary_cross_entropy_with_logits(
            clip_logit.reshape(()), torch.full((), self.target, device=z.device))
        return shape + self.anchor_weight * anchor

    def _select_classes(self, pp, ap):
        """Apply ``class_select`` to the (present, absent) class scalars → the pair to combine.

        Dropping a class here rather than zeroing its weight keeps the surviving half at FULL
        scale (``_reduce_frames`` renormalizes by the weights it actually used), so
        "absent_only" is the absent loss itself, not half of it — the single-class arms are
        then step-size-matched to "both" rather than silently running at lr/2.

        ``greedy_max`` compares the WEIGHTED class losses (the terms "both" would average) and
        keeps the larger, re-deciding on every optimization iteration since the loss is rebuilt
        each time. When one class is empty the timeline is constant and there is nothing to
        select, so the surviving class is kept whatever the mode asks for.
        """
        if self.class_select == "both" or pp is None or ap is None:
            return pp, ap
        if self.class_select == "present_only":
            keep = "present"
        elif self.class_select == "absent_only":
            keep = "absent"
        else:   # greedy_max
            keep = ("present" if float((self.present_weight * pp).detach())
                    >= float((self.absent_weight * ap).detach()) else "absent")
        self._class_choice.append(keep)
        return (pp, None) if keep == "present" else (None, ap)

    def _reduce_frames(self, frame_bce: torch.Tensor, c: torch.Tensor, decided: torch.Tensor):
        """Per-frame BCE ``[T]`` + timeline ``c`` [T] + decided-mask ``[T]`` → (scalar, comps).

        ``comps`` is ``None`` except for the grad-balance frame modes (``gb_mean``/``gb_max``),
        where it is ``[present_part, absent_part]`` — the two sub-losses the steering loop's
        ``optimization.grad_balance`` unit-normalizes and sums (present-vs-absent balancing at
        the frame level, analogous to the cross-object balancing)."""
        present = decided & (c >= 0.5)
        absent = decided & (c < 0.5)
        zero = frame_bce.sum() * 0.0
        fr = self.frame_reduce

        def red(mask, how):  # per-class scalar (None if the class is empty)
            if not mask.any():
                return None
            if how == "max":
                return frame_bce[mask].max()
            if how == "min":
                return frame_bce[mask].min()
            return frame_bce[mask].mean()

        if fr == "bce_huber":
            # Capped mirror of log(BCE): Huber-on-BCE. grad = min(BCE, m)·(p−c) ⇒ self-annealing
            # near the target, log-steep when wrong but bounded (ceiling ≈ m). Balanced over
            # present/absent classes like "balanced".
            m = self.huber_m
            hub = torch.where(frame_bce <= m, 0.5 * frame_bce ** 2, m * (frame_bce - 0.5 * m))
            parts = [hub[msk].mean() for msk in (present, absent) if msk.any()]
            return (torch.stack(parts).mean() if parts else zero), None
        if fr == "mean":
            return (frame_bce[decided].mean() if decided.any() else zero), None
        if fr == "neg_sumlog_score":
            # −Σ_class log(mean_class score), score_t = exp(−BCE_t) (= correct-class prob). Logs the
            # SCORE not the loss ⇒ the p(1−p) factor makes the gradient VANISH at the target
            # (in-distribution), unlike Σlog(BCE). Sum inside the log ⇒ bounded.
            parts = []
            for mask in (present, absent):
                if mask.any():
                    parts.append(-torch.log(torch.exp(-frame_bce[mask]).mean().clamp_min(1e-6)))
            return (torch.stack(parts).sum() if parts else zero), None
        if fr == "detached_weight":
            # Σ_class stop_grad(w_class)·mean_class BCE, w_class = detached class-mean BCE normalized
            # (up-weight the worse class). Magnitude follows plain BCE (vanishes at target); the
            # detached weight only reprioritizes — "log's balancing without log's non-vanishing grad".
            live = [frame_bce[m].mean() for m in (present, absent) if m.any()]
            if not live:
                return zero, None
            stk = torch.stack(live)
            w = stk.detach(); w = w / w.sum().clamp_min(1e-6)
            return (w * stk).sum(), None
        if fr == "logsum_frames":
            # Σ_t log(BCE_t) over decided frames (= log Π BCE_t); clamp for stability.
            return (torch.log(frame_bce[decided].clamp_min(1e-6)).sum() if decided.any() else zero), None

        if fr in ("balanced_max", "log_max", "gb_max"):
            how = "max"
        elif fr in ("balanced_min", "log_min", "gb_min"):
            how = "min"
        else:
            how = "mean"
        pp, ap = red(present, how), red(absent, how)

        if fr in ("gb_mean", "gb_max", "gb_min"):
            comps = [x for x in (pp, ap) if x is not None]
            scalar = torch.stack(comps).sum() if comps else zero   # for logging / reg extraction
            return scalar, (comps if len(comps) > 1 else None)     # <2 parts → no grad-balance
        if fr in ("log_max", "log_min", "log_mean"):
            parts = [torch.log(x.clamp_min(1e-6)) for x in (pp, ap) if x is not None]
            return (torch.stack(parts).sum() if parts else zero), None
        # "balanced" / "balanced_max": weighted mean of the present- and absent-class scalars.
        pp, ap = self._select_classes(pp, ap)
        parts, wsum = [], 0.0
        if pp is not None:
            parts.append(self.present_weight * pp); wsum += self.present_weight
        if ap is not None:
            parts.append(self.absent_weight * ap); wsum += self.absent_weight
        return (torch.stack(parts).sum() / wsum if parts else zero), None

    def per_frame_class_losses(self, captures, step_idx, *, target_words_indices):
        """``{word: (L_present, L_absent, control[T])}`` — the objective split by control bit.

        ``L_present`` is the mean per-frame BCE over the frames the word is REQUESTED in
        (``c=1``), ``L_absent`` the mean over the frames it must be ABSENT from (``c=0``);
        either is ``None`` when its class is empty (a constant timeline). Both carry the
        ``present_weight``/``absent_weight`` normalization, so their SUM is exactly the
        ``frame_reduce="balanced"`` objective — the split is a pure decomposition, and a
        run with the mask off reproduces `balanced` bit-for-bit.

        They are returned separately (rather than summed) so the loop can backprop each one
        on its own and apply it only to the latent frames of its own class
        (:data:`steering_core.run_steering_optimization`, ``frame_grad_mask``). The
        motivation: one shared latent carries both halves, so the absent-frame gradient and
        the present-frame gradient land on *the same* elements and largely cancel — the
        equalized fixed point measured in
        ``experiment_reports/2026_08_26_STEER_EVAL_wan_temporal_control/DEBUG.md`` §2, §6.
        Masking makes the two halves structurally unable to fight over a frame.
        """
        entry = self.probes[self._probe_step(step_idx)]
        T = int(entry["latent_thw"][0])
        out: dict[str, tuple] = {}
        wsum = self.present_weight + self.absent_weight
        for word, indices in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
            feat = self._format(vec, entry).to(self.device).float()
            _clip, frame_logits = entry["model"](feat, return_frames=True)
            frame_logits = frame_logits.reshape(-1)          # [T]
            c = self._timeline_for(word, T)
            decided = ~torch.isnan(c)
            c_filled = torch.nan_to_num(c, nan=0.0)
            frame_bce = F.binary_cross_entropy_with_logits(
                frame_logits, c_filled, reduction="none")    # [T]
            pmask = decided & (c_filled >= 0.5)
            amask = decided & (c_filled < 0.5)
            lp = (self.present_weight / wsum) * frame_bce[pmask].mean() if pmask.any() else None
            la = (self.absent_weight / wsum) * frame_bce[amask].mean() if amask.any() else None
            out[word] = (lp, la, c_filled.detach())
        return out

    def loss(
        self,
        captures: list,
        step_idx: int,
        *,
        target_words_indices: dict[str, list[int]],
    ) -> tuple[torch.Tensor, dict]:
        entry = self.probes[self._probe_step(step_idx)]
        if not target_words_indices:
            raise ValueError("target_words_indices is empty; nothing to steer toward.")
        T = int(entry["latent_thw"][0])
        per_word_obj: list[torch.Tensor] = []           # one scalar loss per object
        per_word_score: list[torch.Tensor] = []         # differentiable mean frame prob
        per_word_prob: dict[str, float] = {}
        per_word_frame_prob: dict[str, list[float]] = {}
        per_word_target: dict[str, list[float]] = {}
        per_word_bce_present: dict[str, float] = {}     # mean per-frame BCE over control=1
        per_word_bce_absent: dict[str, float] = {}      # mean per-frame BCE over control=0
        per_word_frame_bce: list[torch.Tensor] = []     # [T] per word (bce loss_kind only)
        per_word_decided: list[torch.Tensor] = []       # [T] bool per word
        frame_comps: list[torch.Tensor] = []            # gb_* modes: [present,absent] sub-losses
        self._class_choice = []                         # filled by _select_classes, per word
        for word, indices in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
            feat = self._format(vec, entry).to(self.device).float()
            # [1, 1], [1, T] — clip logit + per-frame presence logits for this word.
            clip_logit, frame_logits = entry["model"](feat, return_frames=True)
            frame_logits = frame_logits.reshape(-1)     # [T]
            c = self._timeline_for(word, T)             # [T] in {0,1}
            decided = ~torch.isnan(c)
            c_filled = torch.nan_to_num(c, nan=0.0)
            if self.loss_kind == "kl" and float(c_filled.sum()) > 0:
                # Temporal-distribution KL(r‖q): push presence MASS onto the requested frames.
                logq = torch.log_softmax(frame_logits[decided], dim=0)          # [n_dec]
                r = c_filled[decided] / c_filled[decided].sum()                  # target dist
                per_word_obj.append((r * (torch.log(r + 1e-10) - logq)).sum())
            elif self.loss_kind == "pearson":
                # TempoControl-style (arXiv:2510.02226): match the per-frame prediction's
                # TEMPORAL PROFILE to the control signal by MAXIMIZING their Pearson
                # correlation (= normalized covariance). Scale-/shift-invariant, so unlike
                # BCE it rewards temporal CONTRAST, not global presence level — it cannot be
                # gamed by turning the object all-on or all-off. loss = 1 − corr ∈ [0, 2].
                p = frame_logits[decided]
                cc = c_filled[decided]
                pm = p - p.mean()
                cm = cc - cc.mean()
                denom = torch.sqrt((pm * pm).sum() * (cm * cm).sum()).clamp_min(1e-8)
                corr = (pm * cm).sum() / denom          # 0 if control has no variance
                per_word_obj.append(1.0 - corr)
            elif self.loss_kind in ("rank", "margin", "centered_bce"):
                per_word_obj.append(
                    self._contrast_loss(frame_logits, clip_logit, c_filled, decided))
            else:
                frame_bce = F.binary_cross_entropy_with_logits(
                    frame_logits, c_filled, reduction="none")  # [T]
                per_word_frame_bce.append(frame_bce)
                per_word_decided.append(decided)
                scalar, comps = self._reduce_frames(frame_bce, c_filled, decided)
                per_word_obj.append(scalar)
                if comps is not None:
                    frame_comps.extend(comps)
            # The two halves `frame_reduce` combines, logged for diagnostics: under
            # "balanced" the objective is exactly their (weighted) mean, so a trace can show
            # which class the optimizer is actually paying down. Computed (detached) for
            # EVERY loss_kind, so a contrast objective can be compared against plain BCE in
            # the same units even though it never forms these terms.
            with torch.no_grad():
                fb = F.binary_cross_entropy_with_logits(
                    frame_logits, c_filled, reduction="none")
            pmask, amask = decided & (c_filled >= 0.5), decided & (c_filled < 0.5)
            if pmask.any():
                per_word_bce_present[word] = float(fb[pmask].mean())
            if amask.any():
                per_word_bce_absent[word] = float(fb[amask].mean())
            frame_prob = torch.sigmoid(frame_logits)    # [T], differentiable
            per_word_score.append(frame_prob[decided].mean() if decided.any()
                                  else frame_prob.mean())
            per_word_prob[word] = float(per_word_score[-1].detach())
            per_word_frame_prob[word] = [float(x) for x in frame_prob.detach().cpu()]
            per_word_target[word] = [float(x) for x in c.detach().cpu()]
        # Combine the per-object frame-losses with the same rule as the per-clip steerer
        # (mean / worst / inv_conf), then scale by the objective weight.
        obj_stack = torch.stack(per_word_obj)           # [n_words]
        score_stack = torch.stack(per_word_score)       # [n_words], differentiable
        obj_mean = obj_stack.mean()
        multi = len(per_word_obj) > 1
        if self.word_frame_order == "frame_first" and len(per_word_frame_bce) > 1:
            # Mean the per-word per-frame BCE across words at each frame, then mean over the
            # frames decided by any word (order swapped vs the default word-then-frame).
            stacked = torch.stack(per_word_frame_bce)              # [n_words, T]
            dec = torch.stack(per_word_decided).any(dim=0)         # [T]
            per_frame = stacked.mean(dim=0)                        # [T]
            combined = per_frame[dec].mean() if dec.any() else per_frame.mean()
        elif self.loss_type == "worst" and multi:
            combined = obj_stack.max()
        elif self.loss_type == "inv_conf" and multi:
            # Weight objects by how far their mean frame prob is from the timeline's mean
            # target (detached) — an object already on-timeline contributes ~0.
            w = obj_stack.detach().clamp(min=0.0)
            wsum = w.sum()
            combined = obj_mean if float(wsum) < 1e-6 else (w * obj_stack).sum() / wsum
        else:
            combined = obj_mean
        loss = combined * self.weight
        # Sub-losses (× weight) exposed for gradient-norm balancing in the steering loop
        # (optimization.grad_balance): default = the per-object losses (cross-object
        # balancing); for the gb_* frame modes it is instead the per-class
        # [present, absent] sub-losses. Set for compatibility; a no-op for a single object.
        if self._frame_grad_balance and frame_comps:
            self._last_obj_losses = [o * self.weight for o in frame_comps]
        else:
            self._last_obj_losses = [o * self.weight for o in per_word_obj]
        balance = torch.zeros((), device=loss.device)
        if len(per_word_score) > 1:
            balance = score_stack.max() - score_stack.min()
            if self.balance_weight:
                loss = loss + self.balance_weight * balance
        prob = sum(per_word_prob.values()) / len(per_word_prob)
        # The single `prob` above mixes absent- and present-frame readings, so it can drop
        # simply because the absent frames were fixed. Split it by the control bit: the
        # objective wants prob_absent → 0 and prob_present → 1, so these two move in
        # OPPOSITE directions when steering is working (unlike their mean).
        prob_absent, prob_present = {}, {}
        for word, fp in per_word_frame_prob.items():
            tgt = per_word_target[word]
            a = [p for p, t in zip(fp, tgt) if t == 0.0]
            p_ = [p for p, t in zip(fp, tgt) if t == 1.0]
            if a:
                prob_absent[word] = sum(a) / len(a)
            if p_:
                prob_present[word] = sum(p_) / len(p_)
        return loss, {
            "step": step_idx,
            "prob": prob,                                # mean frame prob (cross-word)
            "per_word_prob": per_word_prob,
            "per_word_prob_absent": prob_absent,         # word → mean prob over control=0 frames
            "per_word_prob_present": prob_present,       # word → mean prob over control=1 frames
            "per_word_bce_absent": per_word_bce_absent,  # word → mean BCE over control=0 frames
            "per_word_bce_present": per_word_bce_present,
            "per_word_frame_prob": per_word_frame_prob,  # word → [T] presence probs
            "frame_target": per_word_target,             # word → [T] requested timeline
            "balance": float(balance.detach()),
            # class_select != "both": which half each word actually optimized on THIS
            # iteration — under greedy_max the sequence is the schedule the run chose.
            "class_choice": list(self._class_choice),
            "bce": float(obj_mean.detach()),             # raw mean per-object frame loss
            "loss": float(loss.detach()),
        }

    def readout_frames(self, captures, probe_step, *, target_words_indices):
        """Diagnostic: per-frame presence probs from ``captures`` using a FIXED meter probe
        (``probe_step``), regardless of the current denoising step — so the same instrument
        can read presence at every step of the trajectory. No grad, no loss."""
        entry = self.probes[probe_step]
        out = {}
        for word, indices in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
            feat = self._format(vec, entry).to(self.device)
            with torch.no_grad():
                _clip, frame_logits = entry["model"](feat, return_frames=True)
            out[word] = [float(x) for x in torch.sigmoid(frame_logits.reshape(-1)).cpu()]
        return out


# WHY THIS IS STILL HERE (2026-10-19, kept on purpose when night-v3-2026-09-25 was merged into
# feat/merge-ltx-wan-cogx): OFF by default — a no-op unless `steering.model.feature_renorm` is set —
# so it changes nothing in normal runs. It is kept (1) to reproduce the 10-11/10-12 renorm reports
# (dataset-level and per-prompt renorm: no help offline or in 5-prompt steering), and (2) because we may
# want it later: the off-distribution problem it targets is real — TempoControl-style prompts put
# 3-8x less attention mass on the object word than our training prompts. Until a probe retrained
# with per-column normalisation (`resnet_input_norm: col_frame|col_video`) is adopted, this is the
# only steering-time knob for that shift.
def _install_feature_renorm_HACK(model: SteeringModel, stats_path: Optional[str]) -> SteeringModel:
    """TEMPORARY HACK (2026-10-11) — REMOVE once the probe is retrained with a per-column
    normalisation (so prompts of any length share one input scale).

    Maps the probe input from the prompt set it is steered on (e.g. the TempoControl template)
    onto the probe's TRAINING distribution with fixed, dataset-level factors per
    (probe step, block, head):  x' = (x - mu_src) / sd_src * sd_train + mu_train.
    The factors are per DATASET, never per prompt; they come from
    _scratch/loss_quality/renorm_stats.py (json with steps -> mu_train/sd_train/mu_tc/sd_tc,
    each [n_blocks][n_heads]). Steps without a probe use their nearest probe's factors (the
    factors live on the probe entry). Offline on the 80 TempoControl prompts this did NOT help
    (clip AUROC 0.49-0.63 vs 0.54-0.67 un-renormed; report
    2026_10_11_STEER_EVAL_wan_temporal_control_lossquality21f, "dataset-level renorm").
    """
    if not stats_path:
        return model
    import json
    # str -> mode "dataset" (fixed affine, below). dict {mode, stats} also allows the per-PROMPT
    # modes "prompt_scale" (x * M/m_p) and "prompt_affine" ((x - m_p)/s_p * S + M), where m_p/s_p
    # are the clip's own (block, head) mean/std and M/S the training set's typical per-clip values
    # (json from _scratch/loss_quality/renorm_prompt_eval.py: steps -> M_train/S_train).
    mode = "dataset"
    if isinstance(stats_path, dict):
        mode, stats_path = str(stats_path.get("mode", "dataset")), stats_path["stats"]
    if mode not in ("dataset", "prompt_scale", "prompt_affine"):
        raise ValueError(f"feature_renorm mode {mode!r}: choose dataset | prompt_scale | prompt_affine")
    stats = json.load(open(stats_path))["steps"]
    probes = getattr(model, "probes", None)
    if not probes:
        raise ValueError("steering.model.feature_renorm needs a per-step probe steering model")
    if mode != "dataset":
        for step, entry in probes.items():
            st = stats.get(str(step))
            if st is None:
                raise ValueError(f"feature_renorm stats {stats_path} have no probe step {step}")
            M, S = (torch.tensor(st[k], dtype=torch.float32)[:, :, None, None, None] for k in ("M_train", "S_train"))
            entry["renorm_prompt_hack"] = (mode, M, S)
        print(f"[steering] TEMPORARY HACK feature_renorm ON, mode {mode} ({stats_path}) for probe steps {sorted(probes)}")
        return model
    for step, entry in probes.items():
        st = stats.get(str(step))
        if st is None:
            raise ValueError(f"feature_renorm stats {stats_path} have no probe step {step}")
        mu_t, sd_t, mu_s, sd_s = (torch.tensor(st[k], dtype=torch.float32)[:, :, None, None, None]
                                  for k in ("mu_train", "sd_train", "mu_tc", "sd_tc"))
        a = sd_t / sd_s.clamp_min(1e-12)
        entry["renorm_hack"] = (a, mu_t - mu_s * a)
    print(f"[steering] TEMPORARY HACK feature_renorm ON ({stats_path}): probe input mapped "
          f"source->train per (step, block, head) for probe steps {sorted(probes)}")
    return model


def build_steering_model(cfg: dict, device: torch.device, agg_kwargs: dict, probe_config: dict,
                         steps_filter: Optional[list[int]] = None) -> SteeringModel:
    """Factory: build the steering model from the ``steering.model`` config block."""
    model = _build_steering_model_impl(cfg, device, agg_kwargs, probe_config, steps_filter)
    # TEMPORARY HACK (2026-10-11): remove with _install_feature_renorm_HACK.
    return _install_feature_renorm_HACK(model, cfg.get("feature_renorm"))


def _build_steering_model_impl(
    cfg: dict,
    device: torch.device,
    agg_kwargs: dict,
    probe_config: dict,
    steps_filter: Optional[list[int]] = None,
) -> SteeringModel:
    """Factory: build the steering model from the ``steering.model`` config block."""
    # ``per_step`` (== ``per_clip``) is the clip-level steerer; ``per_frame`` is the
    # timeline steerer (control *when* each object is present). Anything else is a baseline.
    mode = str(cfg.get("mode", "per_step"))
    obj = cfg.get("objective", {})
    if mode in ("per_step", "per_clip"):
        return StepKeyedSteeringModel(
            checkpoints_dir=cfg["checkpoints_dir"],
            device=device,
            agg_kwargs=agg_kwargs,
            probe_config=probe_config,
            target=float(obj.get("target", 1.0)),
            weight=float(obj.get("weight", 1.0)),
            balance_weight=float(obj.get("balance_weight", 0.0)),
            loss_type=str(obj.get("loss_type", "mean")),
            steps_filter=steps_filter,
            nearest_probe=bool(cfg.get("nearest_probe", False)),
            n_iters_per_step=cfg.get("n_iters_per_step"),
        )
    if mode == "per_frame":
        return PerFrameTimelineSteeringModel(
            checkpoints_dir=cfg["checkpoints_dir"],
            device=device,
            agg_kwargs=agg_kwargs,
            probe_config=probe_config,
            target=float(obj.get("target", 1.0)),
            weight=float(obj.get("weight", 1.0)),
            balance_weight=float(obj.get("balance_weight", 0.0)),
            loss_type=str(obj.get("loss_type", "mean")),
            steps_filter=steps_filter,
            nearest_probe=bool(cfg.get("nearest_probe", False)),
            n_iters_per_step=cfg.get("n_iters_per_step"),
            frame_reduce=str(obj.get("frame_reduce", "balanced")),
            present_weight=float(obj.get("present_weight", 1.0)),
            absent_weight=float(obj.get("absent_weight", 1.0)),
            loss_kind=str(obj.get("loss_kind", "bce")),
            word_frame_order=str(obj.get("word_frame_order", "word_first")),
            huber_m=float(obj.get("huber_m", 2.0)),
            rank_margin=float(obj.get("rank_margin", 1.0)),
            anchor_weight=float(obj.get("anchor_weight", 1.0)),
            frame_grad_mask=bool(obj.get("frame_grad_mask", False)),
            frame_mask_lo=float(obj.get("frame_mask_lo", 0.0)),
            frame_mask_hi=float(obj.get("frame_mask_hi", 1.0)),
            frame_mask_ramp=int(obj.get("frame_mask_ramp", 1)),
            class_select=str(obj.get("class_select", "both")),
        )
    # Anything else is a baseline (see baselines/__init__.py) — imported lazily so this
    # module stays independent of the baselines package.
    from baselines import build_baseline_steering_model

    return build_baseline_steering_model(
        mode, cfg, device, agg_kwargs, probe_config, steps_filter
    )
