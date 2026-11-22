"""Attend-and-Excite's generative semantic nursing as a :class:`SteeringModel`.

Port of *Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image
Diffusion Models* (Chefer et al., SIGGRAPH 2023 — arXiv:2301.13826,
github.com/yuval-alaluf/Attend-and-Excite) onto LTX-2 video generation.

Their released code is a fork of a ``StableDiffusionPipeline`` (the guidance lives inside
``AttendAndExcitePipeline.__call__``, attention is captured by monkey-patched
``AttnProcessor``s keyed by the U-Net's 16×16 resolution), so — as with the TempoControl
port — we bring over the **objective** and let ``steering/steering_loop.py`` supply the
latent optimization. Every arm of the comparison therefore runs through the identical
generation path.

The objective, per denoising step, on the cross-attention of one *subject* word:

    Ā     = mean over heads and blocks of attn[:, :, text_tokens]   drop BOS/EOS/pad
    A     = softmax_over_text( 100 · Ā )                            sharpen, ×100
    A_s   = mean over subject s's subword columns of A              → [T, h, w]
    M_s   = max over patches of  gaussian_smooth(A_s)      (k=3, σ=0.5, reflect pad)
    loss  = max_s  max(0, 1 − M_s)                          the *most neglected* subject

The head/layer mean comes **before** the softmax, as in their ``aggregate_attention`` →
``_compute_max_attention_per_index`` order; sharpening per head and averaging afterwards
would be a different, smoother statistic.

``M_s`` reads "how strongly does subject *s* own its best patch". The outer ``max`` is
what makes this A&E rather than a generic attention-energy objective: only the currently
most-neglected subject receives gradient at each iteration.

**The video adaptation.** A&E maxes over a 16×16 image grid; LTX-2's token grid is
``13 × 16 × 16`` (T × h × w), so the spatial resolution is *identical* and the paper's
``k=3, σ=0.5`` kernel transfers with no rescaling — only the new frame axis needs a
reduction, chosen by ``frame_reduce``:

- ``global`` (default) — max over ``(t, h, w)``: one strong patch **somewhere in the
  clip**. The literal extension of A&E's single spatial max to a spatiotemporal volume,
  and the matched objective for a *per-clip* presence comparison.
- ``min`` — ``min_t max_{h,w}``: a strong patch in **every** frame. Composes A&E's own
  "most neglected element" principle over the frame axis too; stricter, and it optimizes
  a per-frame objective.
- ``mean`` — ``mean_t max_{h,w}``: the soft middle, reported only as an ablation.

Guidance schedule, step size and iterative refinement are the paper's, expressed through
the loop's generic hooks (:meth:`iters_for_step`, :meth:`lr_multiplier`,
:meth:`stop_iteration`) so no A&E-specific branch lives in the shared loop.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from will_it_render.common.steering_base import SteeringModel
from will_it_render.common.aggregator import _resolve_spatial, _spatial_downsample

# Their ``_compute_max_attention_per_index``: the attention slice is scaled by this before
# the text-axis softmax, which sharpens the distribution over prompt tokens.
_SOFTMAX_TEMP = 100.0
# Their ``_compute_loss`` target: a subject is "excited enough" at max-attention 1.0.
_MAX_ATTENTION_TARGET = 1.0


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """The paper's ``GaussianSmoothing`` kernel (2D, normalized to sum 1)."""
    grid = torch.arange(kernel_size, dtype=torch.float32, device=device)
    mean = (kernel_size - 1) / 2.0
    k1d = torch.exp(-(((grid - mean) / sigma) ** 2) / 2.0)
    k2d = torch.outer(k1d, k1d)
    k2d = k2d / k2d.sum()
    return k2d.to(dtype).view(1, 1, kernel_size, kernel_size)


def _smooth(maps: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Gaussian-smooth each frame's ``[h, w]`` map — their reflect-padded 2D smoothing.

    ``maps`` is ``[T, h, w]``; every frame is smoothed independently (the paper smooths a
    single image map), so the spatial semantics are unchanged frame to frame.
    """
    kernel = _gaussian_kernel_2d(kernel_size, sigma, maps.device, maps.dtype)
    pad = kernel_size // 2
    x = maps.unsqueeze(1)                                   # [T, 1, h, w]
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kernel).squeeze(1)                   # [T, h, w]


class AttendExciteSteeringModel(SteeringModel):
    """Attend-and-Excite guidance, keyed by denoising step (no trained probe involved).

    Hooks **all** transformer blocks (``capture_blocks = None``): LTX-2 is a DiT whose
    every block attends at the same ``13×16×16`` grid, so there is no resolution
    hierarchy to select from and "all layers at the chosen resolution" reads as all 48.
    Same backprop footprint as our conv_k48 steering → B200/H200/H100-NVL/RTX6000-Blackwell.
    """

    def __init__(
        self,
        device: torch.device,
        agg_kwargs: dict,
        *,
        sample_steps: int = 50,
        max_step: int = 25,
        thresholds: Optional[dict] = None,
        max_refinement_steps: int = 20,
        scale_range: Sequence[float] = (1.0, 0.5),
        frame_reduce: str = "global",
        word_reduce: str = "mean",
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        softmax_temp: float = _SOFTMAX_TEMP,
        capture_blocks: Optional[Sequence[int]] = None,
        steps_filter: Optional[list[int]] = None,
    ) -> None:
        self.device = device
        # steering/steering_core.py gates its per-step schedule (iters_for_step / lr_multiplier /
        # stop_iteration) on exactly this string, so no A&E-specific branch lives in the shared loop.
        self.mode = "attend_excite"
        # Kept as a LIVE reference (not copied into plain attrs): the CogVideoX backend's
        # capture-time steer_pool patch (model_backend.py::run_with_steering) mutates this
        # dict after construction (spatial_downsample→1, latent_hw→post-pool grid) so the
        # captures arrive already pooled; spatial_downsample/latent_hw are properties below.
        self.agg_kwargs = agg_kwargs
        # The steering loop reads this to decide which blocks to hook. None = all of them
        # (LTX: all 48 at one resolution). On WAN the un-subset 30-block grad forward at
        # Lq=32760 OOMs even a 180 GB B200, so the WAN config hooks a block subset (the same
        # 15 the probe uses) — A&E just means over whichever blocks are captured.
        self.capture_blocks: Optional[list[int]] = list(capture_blocks) if capture_blocks is not None else None

        self.sample_steps = int(sample_steps)
        # Their ``max_iter_to_alter``: guidance is applied to the first N steps only
        # (25 of 50 in the paper), after which denoising proceeds untouched.
        self.max_step = int(max_step)
        # Their ``thresholds``: {step: minimum required max-attention}. At these steps the
        # latent is refined repeatedly until the weakest subject clears the bar.
        self.thresholds = {int(k): float(v) for k, v in (thresholds or {0: 0.05, 10: 0.5, 20: 0.8}).items()}
        self.max_refinement_steps = int(max_refinement_steps)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))

        self.frame_reduce = str(frame_reduce).lower()
        if self.frame_reduce not in ("global", "min", "mean"):
            raise ValueError(
                f"Unknown frame_reduce {frame_reduce!r}; choose 'global' (max over the "
                f"whole clip), 'min' (worst frame) or 'mean'."
            )
        self.word_reduce = str(word_reduce).lower()
        if self.word_reduce not in ("mean", "max"):
            raise ValueError(f"Unknown word_reduce {word_reduce!r}; choose 'mean' or 'max'.")
        self.smooth_attentions = bool(smooth_attentions)
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.softmax_temp = float(softmax_temp)
        self.steps_filter = set(steps_filter) if steps_filter is not None else None

        # Positions of the real prompt tokens in the attention key axis, set per prompt by
        # the backend (see set_text_token_indices). None → every key position is used, with
        # a warning: padding would then dilute the text-axis softmax.
        self._text_token_indices: Optional[list[int]] = None
        self._warned_no_span = False
        # Loss of the current step's latest iteration, for the refinement stop test.
        self._last_loss: float = float("inf")

        steered = sorted(s for s in range(self.sample_steps) if self.should_steer(s))
        nblk = "all" if self.capture_blocks is None else f"{len(self.capture_blocks)} subset"
        print(
            f"[attend_excite] will steer steps {steered} (blocks={nblk}) "
            f"(frame_reduce={self.frame_reduce}, thresholds={self.thresholds}, "
            f"max_refinement_steps={self.max_refinement_steps}, "
            f"scale_range={self.scale_range}, smooth={self.smooth_attentions})"
        )

    # ------------------------------------------------------------------ prompt context

    @property
    def spatial_downsample(self) -> int:
        return int(self.agg_kwargs.get("spatial_downsample", 1) or 1)

    @property
    def latent_hw(self):
        return self.agg_kwargs.get("latent_hw")

    def set_text_token_indices(self, indices: Sequence[int]) -> None:
        """Key positions holding real prompt tokens (no BOS/EOS/padding).

        A&E re-normalizes the attention over the *text* axis
        (``attention_maps[:, :, 1:last_idx]`` — drop the start token and everything from
        the end token on). Our attention key axis is padded to the text encoder's max
        length, so the equivalent slice has to be supplied per prompt; the backend derives
        it from the tokenizer's offset mapping (special tokens have an empty offset span).
        """
        self._text_token_indices = [int(i) for i in indices]

    def set_prompt_context(self, prompt_idx: Optional[int] = None, prompt: Optional[str] = None) -> None:
        """Reset the per-prompt refinement state (called once per prompt by the driver)."""
        self._last_loss = float("inf")

    # ------------------------------------------------------------- schedule / loop hooks

    def should_steer(self, step_idx: int) -> bool:
        # Their ``i < max_iter_to_alter``: guidance on the first ``max_step`` steps.
        if step_idx >= self.max_step:
            return False
        return self.steps_filter is None or step_idx in self.steps_filter

    def iters_for_step(self, step_idx: int) -> int:
        """Iteration budget for this step — one plain update, plus refinement if applicable.

        Their loop does exactly one gradient update per guided step, except at the three
        threshold steps where ``_perform_iterative_refinement_step`` may run up to
        ``max_refinement_steps`` extra updates. :meth:`stop_iteration` ends the refinement
        early once the weakest subject clears the threshold.
        """
        if step_idx in self.thresholds:
            return 1 + self.max_refinement_steps
        return 1

    def lr_multiplier(self, step_idx: int, n_steps: Optional[int] = None) -> float:
        """Their ``step_size = scale_factor · √(scale_range[i])``.

        ``scale_range`` is linearly interpolated across all sampler steps, so the step size
        decays over the trajectory; the config's ``lr`` plays the role of ``scale_factor``.
        ``n_steps`` is supplied by the loop (which owns the sigma schedule), so the decay
        never depends on a config value duplicating the real sampler length.
        """
        lo, hi = self.scale_range
        total = int(n_steps or self.sample_steps)
        if total <= 1:
            return math.sqrt(lo)
        frac = min(max(step_idx / (total - 1), 0.0), 1.0)
        return math.sqrt(lo + (hi - lo) * frac)

    def stop_iteration(self, step_idx: int, metrics: dict, iter_idx: int) -> bool:
        """End this step's optimization early — their ``while loss > target_loss``.

        ``iter_idx`` is the index of the iteration whose loss was just computed and which
        is *about* to be applied. Iteration 0 is the unconditional update every guided step
        performs, so the test only bites from iteration 1 on, and only where a threshold is
        defined.
        """
        if iter_idx < 1 or step_idx not in self.thresholds:
            return False
        target_loss = max(0.0, _MAX_ATTENTION_TARGET - self.thresholds[step_idx])
        return float(metrics.get("ae_loss", self._last_loss)) <= target_loss

    # ------------------------------------------------------------------------ the loss

    def _pooled_attention(self, captures: list) -> tuple[torch.Tensor, dict[int, int]]:
        """Captures → the softmax-renormalized text map ``[Lq, n_text]`` for this step.

        Order matters and follows the reference exactly: their ``aggregate_attention``
        averages the stored maps over **heads and layers first**, and
        ``_compute_max_attention_per_index`` then applies ``×100`` + softmax to that single
        averaged map. Softmaxing per head and averaging afterwards would average a set of
        sharpened distributions instead of sharpening the average — a different (smoother)
        statistic. Returns the map plus the original-token-index → column lookup.
        """
        Lk = captures[0].attn_map.shape[-1]
        if self._text_token_indices is None:
            if not self._warned_no_span:
                print(
                    "[attend_excite] no text-token span set for this prompt; softmaxing "
                    "over ALL key positions, so padding will dilute the normalization. "
                    "The LTX backend sets the span via set_text_token_indices()."
                )
                self._warned_no_span = True
            idx = list(range(Lk))
        else:
            idx = [i for i in self._text_token_indices if i < Lk]
        if not idx:
            raise ValueError(
                f"No valid text token positions for this prompt (key length {Lk}); "
                "check the tokenizer offset mapping passed to set_text_token_indices()."
            )
        # Mean over heads within each block, then over blocks → [Lq, n_text].
        per_block = [cap.attn_map[0][:, :, idx].mean(dim=0) for cap in captures]
        pooled = torch.stack(per_block).mean(dim=0)
        pooled = torch.softmax(pooled * self.softmax_temp, dim=-1)
        return pooled, {orig: j for j, orig in enumerate(idx)}

    def _word_map(
        self,
        pooled: torch.Tensor,
        pos: dict[int, int],
        lq_raw: int,
        indices: Sequence[int],
    ) -> torch.Tensor:
        """Pooled text map + one word's subword indices → ``A_s [T, h, w]``.

        The text-axis softmax already happened in :meth:`_pooled_attention` (it is a
        normalization across tokens, so it cannot come after the target is selected). All
        that remains is picking the word's columns and pooling its subwords — the paper
        passes a single token index per subject; a mean over subwords matches how our own
        per-word features are built.
        """
        cols = [pos[i] for i in indices if i in pos]
        if not cols:
            raise ValueError(
                "Target word matched no token inside the prompt's text span; "
                "check build_token_index_map / set_text_token_indices."
            )
        w = pooled[:, cols]                                     # [Lq, n_sub]
        attn = w.max(dim=-1).values if self.word_reduce == "max" else w.mean(dim=-1)

        if self.latent_hw is None:
            raise ValueError(
                "latent_hw is unknown — the model config must define "
                "video_frame_height_width + latent_scale_factors so the flattened "
                "attention row can be reshaped to (T, h, w)."
            )
        if self.spatial_downsample > 1:
            attn = _spatial_downsample(
                attn.unsqueeze(0), self.latent_hw[0], self.latent_hw[1], self.spatial_downsample
            ).squeeze(0)
        n_frames, h_p, w_p = _resolve_spatial(lq_raw, self.latent_hw[0], self.latent_hw[1])
        ds = self.spatial_downsample
        return attn.view(n_frames, h_p // ds, w_p // ds)

    def _max_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """``A_s [T, h, w]`` → the scalar ``M_s`` the loss drives to 1.

        Their per-token statistic is ``max`` over the (smoothed) 16×16 map; the frame axis
        is reduced by ``frame_reduce`` (see the module docstring).
        """
        if self.smooth_attentions:
            attn = _smooth(attn, self.kernel_size, self.sigma)
        if self.frame_reduce == "global":
            return attn.max()
        per_frame = attn.reshape(attn.shape[0], -1).max(dim=1).values   # [T]
        return per_frame.min() if self.frame_reduce == "min" else per_frame.mean()

    def _per_word_hinge(
        self,
        captures: list,
        *,
        target_words_indices: dict[str, list[int]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Their per-subject ``max(0, 1 − M_s)``, one differentiable scalar per subject.

        Split out of :meth:`loss` so the per-subject components can also be handed to a
        gradient-aggregation rule (see :meth:`per_object_bce`) instead of being collapsed
        by the outer ``max``. Returns ``(hinges, per_word_max)``; every subject in
        ``target_words_indices`` appears in both.
        """
        if not target_words_indices:
            raise ValueError("target_words_indices is empty; nothing to steer toward.")
        if not captures:
            raise ValueError("no attention captures for this step; check capture_blocks.")

        # One head/layer-pooled, softmax-renormalized text map per step, shared by every
        # target word (it does not depend on the target).
        pooled, pos = self._pooled_attention(captures)
        lq_raw = captures[0].attn_map.shape[2]

        hinges: dict[str, torch.Tensor] = {}
        per_word_max: dict[str, float] = {}
        for word, indices in target_words_indices.items():
            attn = self._word_map(pooled, pos, lq_raw, indices).to(self.device)
            m = self._max_attention(attn)
            # Their ``max(0, 1. - curr_max)`` — a subject at/above the target is satisfied.
            hinges[word] = (_MAX_ATTENTION_TARGET - m).clamp(min=0.0)
            per_word_max[word] = float(m.detach())
        return hinges, per_word_max

    def per_object_bce(
        self,
        captures: list,
        step_idx: int,
        *,
        target_words_indices: dict[str, list[int]],
    ) -> dict[str, torch.Tensor]:
        """Per-subject losses for the GRAD_AGGREGATE path (CAGrad et al.).

        ``steering/steering_core.py`` enters its gradient-aggregation branch on
        ``hasattr(steering_model, "per_object_bce")`` and differentiates each returned
        scalar against ``x_t`` separately, so supplying the per-subject hinges here is the
        whole of "CAGrad over A&E": the rule replaces the outer ``max`` of :meth:`loss` as
        the choice of *which* subject the iteration serves.

        NAME. It is A&E's hinge, not a BCE — the method keeps the probe path's name because
        the loop dispatches on it. Note the consequence for rules that read ``scores``:
        ``_probs_from_scores`` maps a score to ``p = exp(−score)``, which is meaningful for
        a BCE but not for this hinge, so the ``prio``/``gate`` CAGrad families are NOT valid
        here yet (they need the separate per-object-scores hook). The plain families
        (``cagrad_{patch,global}[_nounit]_c*``) never touch ``scores`` and are fine.

        Subjects already at/above the target are DROPPED rather than returned with a zero
        hinge: their gradient is exactly zero, which would make the rules' ``g/(‖g‖+1e-12)``
        unit-norm garbage and put an all-zero row in the per-voxel Gram. Dropping them is
        also the behaviour the outer ``max`` has anyway (a satisfied subject never wins it).
        """
        hinges, _ = self._per_word_hinge(captures, target_words_indices=target_words_indices)
        active = {w: h for w, h in hinges.items() if float(h.detach()) > 0.0}
        # All subjects satisfied: hand back the single least-excited one so the caller still
        # gets a well-formed dict. Its gradient is zero, so the update is a no-op — the same
        # thing the unaggregated loss does at this point.
        if not active:
            w = max(hinges, key=lambda k: float(hinges[k].detach()))
            return {w: hinges[w]}
        return active

    def loss(
        self,
        captures: list,
        step_idx: int,
        *,
        target_words_indices: dict[str, list[int]],
    ) -> tuple[torch.Tensor, dict]:
        hinges, per_word_max = self._per_word_hinge(
            captures, target_words_indices=target_words_indices)

        # Their ``loss = max(losses)``: the whole gradient goes to the most neglected
        # subject at this iteration. This is the defining choice of the method — a mean
        # here would be a generic attention-energy objective.
        loss = torch.stack(list(hinges.values())).max()
        self._last_loss = float(loss.detach())

        max_attn = sum(per_word_max.values()) / len(per_word_max)
        return loss, {
            "step": step_idx,
            # "prob" is the scalar the sweep aggregation and before/after tooling track.
            # Here it is the mean per-subject max attention — a monotone "how excited"
            # signal in [0, 1], NOT a probability.
            "prob": max_attn,
            "per_word_prob": dict(per_word_max),
            "max_attention": max_attn,
            "min_max_attention": min(per_word_max.values()),   # the subject driving the loss
            "ae_loss": self._last_loss,                        # read by stop_iteration
            "loss": self._last_loss,
        }


def build_attend_excite_model(
    cfg: dict,
    device: torch.device,
    agg_kwargs: dict,
    probe_config: dict,
    steps_filter: Optional[list[int]] = None,
) -> AttendExciteSteeringModel:
    """Build from the ``steering.model`` block (+ ``steering.objective``).

    ``probe_config`` is unused — this baseline has no trained probe — but is accepted so
    the signature matches ``steering.steering_model.build_steering_model``.
    """
    obj = cfg.get("objective", {}) or {}
    thresholds = obj.get("thresholds")
    if isinstance(thresholds, dict):
        thresholds = {int(k): float(v) for k, v in thresholds.items()}
    return AttendExciteSteeringModel(
        device=device,
        agg_kwargs=agg_kwargs,
        sample_steps=int(cfg.get("sample_steps", obj.get("sample_steps", 50))),
        max_step=int(obj.get("max_step", 25)),
        thresholds=thresholds,
        max_refinement_steps=int(obj.get("max_refinement_steps", 20)),
        scale_range=obj.get("scale_range", (1.0, 0.5)),
        frame_reduce=str(obj.get("frame_reduce", "global")),
        word_reduce=str(obj.get("word_reduce", "mean")),
        smooth_attentions=bool(obj.get("smooth_attentions", True)),
        sigma=float(obj.get("sigma", 0.5)),
        kernel_size=int(obj.get("kernel_size", 3)),
        softmax_temp=float(obj.get("softmax_temp", _SOFTMAX_TEMP)),
        capture_blocks=obj.get("capture_blocks"),
        steps_filter=steps_filter,
    )
