"""TempoControl's temporal attention guidance as a :class:`SteeringModel`.

Port of *TempoControl: Temporal Attention Guidance for Text-to-Video Models*
(Schiber et al., CVPR 2026 — arXiv:2510.02226, github.com/Shira-Schiber/TempoControl)
onto LTX-2. Their released code is a hard fork of Wan2.1 (attention capture is surgery
inside ``wan/modules/model.py``, the guidance loop lives inside ``WanT2V.generate``, and
the token grid is hardcoded to ``reshape(21, 30, 52)``), so the *method* is ported rather
than the code: only the loss lives here, and ``steering/steering_loop.py`` provides the
same latent optimization our own probe steering uses.

The objective (their equations 4–8) over the layer/head/token-averaged cross-attention
map ``A[T, h, w]`` of one target word and a per-frame control signal ``c[T]``:

    E_t = Σ_{h,w} A[t]                             per-frame attention energy
    H_t = −Σ_{h,w} p·log p,  p = A[t] / Σ A[t]      per-frame spatial entropy

    loss = pearson_weight · (−corr(â, c))          (eq. 4)  temporal shape
         + energy_weight  · (−mean_{c>0.7} E_t
                             + mean_{c<0.7} E_t)   (eq. 5, 6)  presence / absence
         + entropy_weight · mean_{c>0.7} H_t       (eq. 8)  spatial focus
         + weakest_weight · (−min_{c>0.7} E_t)     ours — see below

**"Present in all frames"** is the degenerate control signal ``c = 1`` (the default
``control_signal: all_ones``). Note this makes the Pearson term *identically zero*:
``cov = mean((â−mean â)·(c−mean c))`` and ``c−mean c = 0``, so the term contributes no
gradient — and their correlation-based early stopping never fires. That is not a defect
of the port: it is exactly how TempoControl treats its own *static object*, whose
``control_signal2`` is all ones in ``data/two_objects.csv``. The all-frames baseline is
therefore energy + entropy, at the paper's weights.

``weakest_weight`` is the one term we add (default 0, i.e. off): the mean energy can rise
while a few frames stay empty, so the ablation pushes up the *weakest* frame instead of
the average. Reported separately, never as "TempoControl".
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from will_it_render.common.control_signal import resolve_control_signal as _resolve_control_signal
from will_it_render.common.steering_base import SteeringModel
from will_it_render.common.aggregator import _resolve_spatial, aggregate_captures_to_vector

# Frames whose control signal exceeds this count as "concept should be visible" — the
# threshold TempoControl uses throughout (equations 5, 6, 8).
_ACTIVE_THRESHOLD = 0.7


def _pearson_loss(attention_signal: torch.Tensor, control_signal: torch.Tensor) -> torch.Tensor:
    """``−corr(a, c)`` over the interior frames — mirrors their ``pearson_loss``.

    The first and last frames are dropped exactly as in the reference implementation.
    Returns 0 for a constant control signal (zero covariance), with no gradient.
    """
    a = attention_signal[1:-1]
    c = control_signal[1:-1]
    a_mean, c_mean = a.mean(), c.mean()
    cov = ((a - a_mean) * (c - c_mean)).mean()
    std_a = torch.sqrt(((a - a_mean) ** 2).mean() + 1e-6)
    std_c = torch.sqrt(((c - c_mean) ** 2).mean() + 1e-6)
    return -cov / (std_a * std_c)




class TempoControlSteeringModel(SteeringModel):
    """TempoControl guidance, keyed by denoising step (no trained probe involved).

    Hooks **all** transformer blocks (``capture_blocks = None``) because the method
    averages cross-attention across every layer — the same memory profile as our 48-layer
    conv_k steering, so dense prompts need a B200/H200.
    """

    def __init__(
        self,
        device: torch.device,
        agg_kwargs: dict,
        *,
        diffusion_steps_range: Sequence[int],
        control_signal: str | Sequence[float] | None = "all_ones",
        control_on_frames: Optional[int] = None,
        control_on_frac: Optional[float] = None,
        control_window_seed: int = 0,
        energy_weight: float = 0.3,
        entropy_weight: float = 10.0,
        pearson_weight: float = 0.0,
        weakest_weight: float = 0.0,
        steps_filter: Optional[list[int]] = None,
        correlation_based_steps: bool = True,
        pearson_threshold: float = 0.9,
    ) -> None:
        self.device = device
        # Keys are never needed here; keep only the attention-shaping flags so the
        # aggregated vector is exactly [n_blocks, n_heads, Lq].
        self.agg_kwargs = {**agg_kwargs, "extract_keys": False}
        self.spatial_downsample = int(self.agg_kwargs.get("spatial_downsample", 1) or 1)
        self.latent_hw = self.agg_kwargs.get("latent_hw")
        # The steering loop reads this to decide which blocks to hook. None = all of them.
        self.capture_blocks: Optional[list[int]] = None

        self.diffusion_steps_range = set(int(s) for s in diffusion_steps_range)
        self.steps_filter = set(steps_filter) if steps_filter is not None else None
        self.control_signal_cfg = control_signal
        # random_window mode (the presence-persistence Pareto knob): each prompt gets one
        # random contiguous window of `control_on_frames` (or round(control_on_frac*T))
        # active frames, placed deterministically from the prompt text so every prompt
        # gets a different area of the timeline reproducibly. All target words of a prompt
        # share the same window; the neg-energy term suppresses the object outside it.
        self.random_window = isinstance(control_signal, str) and control_signal.strip().lower() == "random_window"
        self.control_on_frames = int(control_on_frames) if control_on_frames is not None else None
        self.control_on_frac = float(control_on_frac) if control_on_frac is not None else None
        self.control_window_seed = int(control_window_seed)
        self._cur_key: str = ""          # per-prompt identity for window seeding
        self._window_cache: dict = {}     # (key, T) -> control tensor
        # Per-WORD control signals for the current prompt (TempoControl's two-object case:
        # the temp word gets control_signal1, the static word all-ones). Populated by
        # set_prompt_timelines from the prompts-CSV `timelines` column; empty ⇒ every word
        # shares the single configured control_signal (the one-object / all-ones case).
        self._per_word_control: dict[str, list[float]] = {}
        self.energy_weight = float(energy_weight)
        self.entropy_weight = float(entropy_weight)
        self.pearson_weight = float(pearson_weight)
        self.weakest_weight = float(weakest_weight)
        # Correlation-based early stopping (TempoControl §Correlation-based early stopping):
        # once the FIRST word's (control_signal1's) attention-vs-control Pearson correlation
        # reaches pearson_threshold, that denoising step stops optimizing early. The steering
        # loop reads these two attrs generically; the metric is the first word's correlation
        # (`pearson_primary`), matching their use of control_signal1 only. For an all-ones
        # control the correlation is identically 0 < 0.9, so this never fires — the existing
        # all-ones baseline is unchanged.
        self.correlation_based_steps = bool(correlation_based_steps)
        self.pearson_threshold = float(pearson_threshold)
        self.early_stop_metric = "pearson_primary"
        self.early_stop_threshold = self.pearson_threshold if self.correlation_based_steps else None
        # Cached per (n_frames) — the latent frame count is fixed for a run (non-random modes).
        self._control_signal: Optional[torch.Tensor] = None

        steered = sorted(s for s in self.diffusion_steps_range if self.should_steer(s))
        win = (f" on_frames={self.control_on_frames} on_frac={self.control_on_frac} "
               f"seed={self.control_window_seed}") if self.random_window else ""
        print(
            f"[tempocontrol] will steer steps {steered} "
            f"(control_signal={control_signal}{win}, energy={self.energy_weight}, "
            f"entropy={self.entropy_weight}, pearson={self.pearson_weight}, "
            f"weakest={self.weakest_weight})"
        )

    def set_prompt_context(self, prompt_idx: Optional[int] = None, prompt: Optional[str] = None) -> None:
        """Called once per prompt by the driver so random_window picks a fresh window.

        The window is seeded by the prompt TEXT (stable + globally unique, unlike the
        batch-local prompt_idx), so the placement is reproducible and independent of how
        prompts are batched across the queue.
        """
        self._cur_key = prompt if prompt else str(prompt_idx)

    def set_prompt_timelines(self, timelines: dict[str, list[float]]) -> None:
        """Receive per-word control signals for the next prompt (steer.py calls this).

        ``timelines`` maps target word → per-latent-frame 0/1 schedule (the prompts-CSV
        ``timelines`` column, e.g. ``cat:0000001111111;bird:1111111111111``). Each word is
        then steered toward its *own* signal in :meth:`loss` — the temp object toward
        ``control_signal1``, the static object toward the all-ones ``control_signal2`` —
        instead of every word sharing one control. Empty ⇒ fall back to the configured
        single ``control_signal`` for all words.
        """
        self._per_word_control = {w.lower(): list(c) for w, c in (timelines or {}).items()}

    def _control_for_word(self, word: str, n_frames: int, device: torch.device) -> torch.Tensor:
        """Per-word control tensor: the word's own timeline if supplied, else the global one."""
        sig = self._per_word_control.get(word.lower())
        if sig is None:
            return self._get_control(n_frames, device)
        return _resolve_control_signal(sig, n_frames, device)

    def _random_window(self, n_frames: int, device: torch.device) -> torch.Tensor:
        import hashlib
        import random as _random

        key = (self._cur_key, n_frames)
        if key in self._window_cache:
            return self._window_cache[key].to(device)
        if self.control_on_frames is not None:
            n_on = self.control_on_frames
        elif self.control_on_frac is not None:
            n_on = int(round(self.control_on_frac * n_frames))
        else:
            raise ValueError("random_window control needs control_on_frames or control_on_frac.")
        n_on = max(1, min(int(n_on), n_frames))
        h = int(hashlib.md5(
            f"{self.control_window_seed}|{self._cur_key}|{n_frames}|{n_on}".encode()
        ).hexdigest(), 16)
        start = _random.Random(h).randint(0, n_frames - n_on)
        sig = torch.zeros(n_frames, dtype=torch.float32)
        sig[start:start + n_on] = 1.0
        self._window_cache[key] = sig
        return sig.to(device)

    def current_control(self) -> Optional[dict]:
        """The window chosen for the current prompt (for logging / in-window fidelity).

        Returns ``None`` for non-random modes, or ``{on, start, n_on, n_frames}`` once a
        window has been materialized for the current prompt (i.e. after the first loss)."""
        if not self.random_window:
            return None
        for (key, T), sig in self._window_cache.items():
            if key == self._cur_key:
                on = [int(x) for x in (sig > 0.5).nonzero().flatten().tolist()]
                return {"n_frames": T, "n_on": len(on), "start": on[0] if on else None, "on": on}
        return None

    def _get_control(self, n_frames: int, device: torch.device) -> torch.Tensor:
        if self.random_window:
            return self._random_window(n_frames, device)
        if self._control_signal is None:
            self._control_signal = _resolve_control_signal(
                self.control_signal_cfg, n_frames, device
            )
        return self._control_signal.to(device)

    def should_steer(self, step_idx: int) -> bool:
        if step_idx not in self.diffusion_steps_range:
            return False
        return self.steps_filter is None or step_idx in self.steps_filter

    def _attention_map(self, captures: list, indices: list[int]) -> torch.Tensor:
        """Captures → ``A[T, h, w]``: mean over target tokens, heads and layers.

        Equivalent to TempoControl's ``get_attention_map``; the token mean and the
        spatial pooling are done by the shared, differentiable
        ``aggregate_captures_to_vector``.
        """
        vec = aggregate_captures_to_vector(captures, indices, **self.agg_kwargs)
        n_blocks = len(captures)
        n_heads = captures[0].attn_map.shape[1]
        if self.latent_hw is None:
            raise ValueError(
                "latent_hw is unknown — the model config must define "
                "video_frame_height_width + latent_scale_factors so the flattened "
                "attention row can be reshaped to (T, h, w)."
            )
        # Resolve the grid from the AGGREGATED vector length, which already reflects any
        # spatial pooling aggregate_captures_to_vector applied AND any pre-pooling the backend
        # did at capture time (CogVideoX stores attention already pooled to 15x22, Lq=4290).
        # Deriving from the raw attn_map length and then dividing by spatial_downsample
        # double-counts the pool for a pre-pooled backend (cogx -> 7x11 instead of 15x22);
        # for the ds=1 backends (LTX) per_head == lq_raw, so this is identical to the old path.
        per_head = vec.numel() // (n_blocks * n_heads)
        n_frames, h_eff, w_eff = _resolve_spatial(per_head, self.latent_hw[0], self.latent_hw[1])
        return vec.view(n_blocks, n_heads, n_frames, h_eff, w_eff).mean(dim=(0, 1))

    def _terms(self, attn: torch.Tensor, control: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Per-word loss + raw (pre-weight) term values for one ``A[T, h, w]``."""
        flat = attn.reshape(attn.shape[0], -1)          # [T, h*w]
        energies = flat.sum(dim=1)                       # E_t
        probs = flat / (flat.sum(dim=1, keepdim=True) + 1e-10)
        entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # H_t

        active = control > _ACTIVE_THRESHOLD
        inactive = control < _ACTIVE_THRESHOLD
        zero = torch.zeros((), device=attn.device, dtype=energies.dtype)

        # eq. 5 — amplify attention where the concept must be visible.
        energy_term = -energies[active].mean() if active.any() else zero
        # eq. 6 — suppress it elsewhere (empty, hence 0, for an all-ones signal).
        neg_energy_term = energies[inactive].mean() if inactive.any() else zero
        # eq. 8 — keep the visible frames spatially focused.
        entropy_term = entropies[active].mean() if active.any() else zero
        # eq. 4 — temporal shape; identically 0 for a constant control signal.
        signal = flat.mean(dim=1)
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-6)
        pearson_term = _pearson_loss(signal, control)
        # Ours (off by default): lift the weakest visible frame, not just the mean.
        weakest_term = -energies[active].min() if active.any() else zero

        loss = (
            self.pearson_weight * pearson_term
            + self.energy_weight * (energy_term + neg_energy_term)
            + self.entropy_weight * entropy_term
            + self.weakest_weight * weakest_term
        )
        raw = {
            "energy": float(energies[active].mean().detach()) if active.any() else 0.0,
            "min_energy": float(energies[active].min().detach()) if active.any() else 0.0,
            "neg_energy": float(neg_energy_term.detach()),
            "entropy": float(entropy_term.detach()),
            "pearson": float(-pearson_term.detach()),  # report the correlation itself
        }
        return loss, raw

    def loss(
        self,
        captures: list,
        step_idx: int,
        *,
        target_words_indices: dict[str, list[int]],
    ) -> tuple[torch.Tensor, dict]:
        if not target_words_indices:
            raise ValueError("target_words_indices is empty; nothing to steer toward.")
        if not captures:
            raise ValueError("no attention captures for this step; check capture_blocks.")

        probe_attn = self._attention_map(captures, next(iter(target_words_indices.values())))
        n_frames = probe_attn.shape[0]

        per_word_loss: list[torch.Tensor] = []
        per_word_raw: dict[str, dict] = {}
        for word, indices in target_words_indices.items():
            attn = self._attention_map(captures, indices).to(self.device)
            control = self._control_for_word(word, n_frames, attn.device)
            word_loss, raw = self._terms(attn, control)
            per_word_loss.append(word_loss)
            per_word_raw[word] = raw

        loss = torch.stack(per_word_loss).mean()

        # First word == control_signal1's object (temp word): its correlation is the
        # early-stopping signal, matching TempoControl (which early-stops on control_signal1
        # only, not the multi-word mean).
        first_word = next(iter(target_words_indices))
        pearson_primary = per_word_raw[first_word]["pearson"]

        def _mean(key: str) -> float:
            return sum(r[key] for r in per_word_raw.values()) / len(per_word_raw)

        # "prob" is what the steering loop / sweep aggregation log as the tracked scalar.
        # Here it is the mean per-frame attention energy — a monotone "how present"
        # signal, NOT a probability. Kept under this key so the existing before/after
        # tooling works unchanged.
        return loss, {
            "step": step_idx,
            "prob": _mean("energy"),
            "per_word_prob": {w: r["energy"] for w, r in per_word_raw.items()},
            "energy": _mean("energy"),
            "min_energy": _mean("min_energy"),
            "neg_energy": _mean("neg_energy"),
            "entropy": _mean("entropy"),
            "pearson": _mean("pearson"),
            "pearson_primary": pearson_primary,
            "loss": float(loss.detach()),
        }


def build_tempo_control_model(
    cfg: dict,
    device: torch.device,
    agg_kwargs: dict,
    probe_config: dict,
    steps_filter: Optional[list[int]] = None,
) -> TempoControlSteeringModel:
    """Build from the ``steering.model`` block (+ ``steering.objective``).

    ``probe_config`` is unused — this baseline has no trained probe — but is accepted so
    the signature matches ``steering.steering_model.build_steering_model``.
    """
    obj = cfg.get("objective", {}) or {}
    # Which steps the method may optimize: their `diffusion.diffusion_steps_range`.
    # Defaults to their published range; `steps_to_steer` (steps_filter) narrows it.
    steps_range = cfg.get("diffusion_steps_range", obj.get("diffusion_steps_range"))
    if steps_range is None:
        steps_range = steps_filter if steps_filter is not None else list(range(11))
    return TempoControlSteeringModel(
        device=device,
        agg_kwargs=agg_kwargs,
        diffusion_steps_range=steps_range,
        control_signal=obj.get("control_signal", "all_ones"),
        control_on_frames=obj.get("control_on_frames"),
        control_on_frac=obj.get("control_on_frac"),
        control_window_seed=int(obj.get("control_window_seed", 0)),
        energy_weight=float(obj.get("energy_weight", 0.3)),
        entropy_weight=float(obj.get("entropy_weight", 10.0)),
        pearson_weight=float(obj.get("pearson_weight", 0.0)),
        weakest_weight=float(obj.get("weakest_weight", 0.0)),
        steps_filter=steps_filter,
        correlation_based_steps=bool(obj.get("correlation_based_steps", True)),
        pearson_threshold=float(obj.get("pearson_threshold", 0.9)),
    )
