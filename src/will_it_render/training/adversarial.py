"""Adversarial (PGD) training on the probe's *input attention maps*.

Motivation — the inference-time exploit
---------------------------------------
At steering time an optimizer moves the latent to maximize the probe's score. It
routinely finds attention configurations that spike the score **without** putting the
object in the frame: the probe is a fixed function and the optimizer is free to walk
off the data manifold. Adversarial training simulates exactly that inner optimization
*during* probe training: for every batch we run a small PGD attack that pushes each
example's attention map toward the WRONG label, then train on the perturbed maps too.
The probe is forced to place its decision boundary where a small, *domain-legal*
rearrangement of attention cannot cross it.

What "domain-legal" means for these features (read before setting `projection`)
--------------------------------------------------------------------------------
The stored feature is **not** a full attention matrix. `aggregate_captures_to_vector`
takes `softmax_over_keys(Q·Kᵀ)[q, k]` and averages over the *target-word* key indices:

    x[block, head, q] = mean_{k ∈ target tokens} A[q, k]        with  Σ_k A[q, k] = 1

The key axis is marginalized away before storage, so there is **no key axis left in the
probe's input to renormalize** — "sum to 1 over the key dim" is unavailable here, and
the sum over the *query* axis is not conserved by anything. The constraints that are
actually implied by the data:

* **`box`** (default) — `x ∈ [0, domain_max]` elementwise, inside an ε-ball of the clean
  map. This is the only rigorously valid domain constraint, and it is also the one that
  matters: it *permits* the global mass-inflation attack ("make the word attend harder
  everywhere"), which is precisely the exploit the steering optimizer finds. A
  mass-conserving projection would train the probe only against *redistribution* and
  leave the inflation exploit unmodelled.
* **`simplex`** — exact Euclidean projection onto `{x ≥ 0, Σ_group x = mass}`, per group
  along `axis`, with `mass` taken from the clean map. Mass-conserving: models a pure
  *redistribution* attack (attention moves around the frame, total per-head mass fixed).
* **`softmax`** — perturbs in logit space, `x = mass · softmax(log x_clean + δ)` with
  `‖δ‖_∞ ≤ ε`. Also mass-conserving, and the most faithful to the generative process
  (the real map *is* a softmax, and the latent optimizer really does move the logits) —
  but ε is in logit units, not attention units, so it needs its own ε scale.

`box` is the default; `simplex` / `softmax` are there when you want the redistribution-only
threat model. Both mass-conserving modes also clamp to `[0, domain_max]` afterwards, which
makes the projection approximate at the (rare) points where the mass-preserving simplex
point would exceed `domain_max`.

Usage
-----
    cfg = AdvConfig.from_training_cfg(cfg.training)
    attack = AttentionPGD(cfg, feat_shape=(C, T, h, w))
    x_adv = attack.perturb(model, x, y, forward=lambda m, z: m(z))
    loss = (1 - cfg.alpha) * loss_clean + cfg.alpha * loss_adv

The attack never touches `p.grad` of the model (it differentiates w.r.t. the input only,
via `torch.autograd.grad`) and restores `requires_grad` / train-mode state on exit, so it
drops into an existing training loop without disturbing the outer optimizer.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

# Forward signature used by the attack: (model, x) -> logit [B, 1] or [B].
ForwardFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]

_TINY = 1e-12


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class AdvConfig:
    """Adversarial-training hyper-parameters (all `training.adv_*` config keys)."""

    enabled: bool = False
    alpha: float = 0.5              # loss interpolation: (1-α)·clean + α·adversarial
    steps: int = 5                  # PGD iterations
    epsilon: float = 0.05           # attack budget (attention units; logit units for softmax)
    step_size: Optional[float] = None   # None → 2.5·ε/steps (Madry's rule of thumb)
    norm: str = "linf"              # linf | l2
    projection: str = "box"         # box | simplex | softmax  (see module docstring)
    axis: str = "spatial"           # grouping for simplex/softmax: spatial | frame_spatial
    domain_max: float = 1.0         # elementwise upper bound of a valid attention weight
    random_start: bool = True       # uniform init inside the ε-ball (PGD, not FGSM)
    on: str = "all"                 # attack all | pos | neg examples
    objective: str = "clip"         # clip | perframe_min  (what the attacker maximizes)
    eval_mode: bool = True          # run the attack with the model in eval() (no dropout noise)
    warmup_epochs: int = 0          # clean-only epochs before the attack switches on
    eval_robust: bool = True        # also report robust val metrics on eval epochs
    eval_train: bool = True         # ALSO attack the train split (robust train/val gap)
    eval_train_max: int = 1024      # cap train examples in that pass (0 = the whole trainset)
    eval_val_max: int = 0           # cap val examples in the PER-EPOCH selection pass (0 = all).
                                    #   Only used when best_metric is adversarial; the eval-epoch
                                    #   reporting pass always covers the full val set.
    loss: str = "bce"               # bce | alp | trades — how the adversarial term enters the loss
    pair_on: str = "frame"          # frame | clip — which readout alp/trades tie together.
                                    #   Must match what the attack targets: under
                                    #   objective=perframe_min, pairing on "clip" hardens
                                    #   the wrong head (measured: clip adv AUROC 0.998 while
                                    #   min-frame stayed 0.000). "clip" reproduces the
                                    #   2026-08-13 runs.
    trades_beta: float = 1.0        # trades: weight on KL(adv‖clean); alp: weight on the logit-pair term
    save_examples: int = 50         # val examples whose adversarial maps are saved (0 = none)
    save_maps: str = "per_layer"    # pooled | per_layer | full — fidelity of the saved maps
    example_seed: int = 0           # seeds the saved-example attack so it can be regenerated
    init_ckpt_dir: Optional[str] = None   # fine-tune: warm-start from <dir>/<run_label>/probe_<run_label>.pt

    # ---- construction ------------------------------------------------------------
    @classmethod
    def from_training_cfg(cls, t) -> "AdvConfig":
        """Build from a `cfg.training` section (DictConfig or dict) via `adv_*` keys."""
        g = t.get
        return cls(
            enabled=bool(g("adv_training", False)),
            alpha=float(g("adv_alpha", 0.5)),
            steps=int(g("adv_pgd_steps", 5)),
            epsilon=float(g("adv_epsilon", 0.05)),
            step_size=(float(g("adv_step_size")) if g("adv_step_size", None) is not None else None),
            norm=str(g("adv_norm", "linf")),
            projection=str(g("adv_projection", "box")),
            axis=str(g("adv_axis", "spatial")),
            domain_max=float(g("adv_domain_max", 1.0)),
            random_start=bool(g("adv_random_start", True)),
            on=str(g("adv_on", "all")),
            objective=str(g("adv_objective", "clip")),
            eval_mode=bool(g("adv_eval_mode", True)),
            warmup_epochs=int(g("adv_warmup_epochs", 0)),
            eval_robust=bool(g("adv_eval_robust", True)),
            eval_train=bool(g("adv_eval_train", True)),
            eval_train_max=int(g("adv_eval_train_max", 1024)),
            eval_val_max=int(g("adv_eval_val_max", 0)),
            loss=str(g("adv_loss", "bce")),
            pair_on=str(g("adv_pair_on", "frame")),
            trades_beta=float(g("adv_trades_beta", 1.0)),
            save_examples=int(g("adv_save_examples", 50)),
            save_maps=str(g("adv_save_maps", "per_layer")),
            example_seed=int(g("adv_example_seed", 0)),
            init_ckpt_dir=(str(g("adv_init_ckpt_dir")) if g("adv_init_ckpt_dir", None) is not None else None),
        )

    @classmethod
    def coerce(cls, adv) -> "AdvConfig":
        """Accept `None` / dict / AdvConfig (queue tasks round-trip through JSON)."""
        if adv is None:
            return cls()
        if isinstance(adv, cls):
            return adv
        return cls(**{k: v for k, v in dict(adv).items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- derived / validation ----------------------------------------------------
    @property
    def resolved_step_size(self) -> float:
        if self.step_size is not None:
            return self.step_size
        return 2.5 * self.epsilon / max(1, self.steps)

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.norm not in ("linf", "l2"):
            raise ValueError(f"adv_norm must be 'linf' or 'l2' (got {self.norm!r}).")
        if self.projection not in ("box", "simplex", "softmax"):
            raise ValueError(
                f"adv_projection must be 'box', 'simplex' or 'softmax' (got {self.projection!r})."
            )
        if self.axis not in ("spatial", "frame_spatial"):
            raise ValueError(
                f"adv_axis must be 'spatial' or 'frame_spatial' (got {self.axis!r})."
            )
        if self.on not in ("all", "pos", "neg"):
            raise ValueError(f"adv_on must be 'all', 'pos' or 'neg' (got {self.on!r}).")
        if self.objective not in ("clip", "perframe_min"):
            raise ValueError(
                f"adv_objective must be 'clip' or 'perframe_min' (got {self.objective!r})."
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"adv_alpha must be in [0, 1] (got {self.alpha}).")
        if self.steps < 1:
            raise ValueError(f"adv_pgd_steps must be >= 1 (got {self.steps}).")
        if self.epsilon <= 0:
            raise ValueError(f"adv_epsilon must be > 0 (got {self.epsilon}).")
        if self.projection == "softmax" and self.norm != "linf":
            raise ValueError("adv_projection='softmax' only supports adv_norm='linf' "
                             "(the budget is an L∞ ball in logit space).")
        if self.save_maps not in ("pooled", "per_layer", "full"):
            raise ValueError(
                f"adv_save_maps must be 'pooled', 'per_layer' or 'full' (got {self.save_maps!r})."
            )
        if self.loss not in ("bce", "alp", "trades"):
            raise ValueError(f"adv_loss must be 'bce', 'alp' or 'trades' (got {self.loss!r}).")
        if self.pair_on not in ("frame", "clip"):
            raise ValueError(f"adv_pair_on must be 'frame' or 'clip' (got {self.pair_on!r}).")

    def summary(self) -> str:
        s = (f"PGD adv training: α={self.alpha} steps={self.steps} ε={self.epsilon} "
             f"(step={self.resolved_step_size:.4g}) norm={self.norm} "
             f"proj={self.projection}")
        if self.projection in ("simplex", "softmax"):
            s += f"/{self.axis}"
        s += f" on={self.on} obj={self.objective}"
        if self.warmup_epochs:
            s += f" warmup={self.warmup_epochs}ep"
        return s


# --------------------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------------------
def project_simplex(v: torch.Tensor, mass: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact Euclidean projection of `v` onto `{x ≥ 0, Σ_dim x = mass}` (Duchi et al. 2008).

    `mass` broadcasts against `v` with `dim` of size 1. Vectorized over every other axis,
    so one sort handles the whole batch.
    """
    v = v.movedim(dim, -1)
    mass = mass.movedim(dim, -1)
    n = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    cssv = u.cumsum(dim=-1) - mass
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    # For a descending-sorted u the condition holds exactly on a prefix, so counting the
    # True entries gives ρ = max{j : u_j - (Σ_{i≤j} u_i - mass)/j > 0}.
    cond = u - cssv / ind > 0
    rho = cond.sum(dim=-1, keepdim=True).clamp(min=1)
    theta = cssv.gather(-1, rho - 1) / rho.to(v.dtype)
    return (v - theta).clamp_(min=0).movedim(-1, dim)


class AttentionPGD:
    """Multi-step PGD on attention maps, projected back onto the valid-attention set.

    `feat_shape` is the probe's unflattened input `(C, T, h, w)`; it is only needed to
    know how to group elements for the mass-conserving projections. Flat `[B, D]` inputs
    are handled transparently.
    """

    def __init__(self, cfg: AdvConfig, feat_shape: tuple[int, int, int, int],
                 n_blocks: int = 0) -> None:
        cfg.validate()
        self.cfg = cfg
        self.feat_shape = tuple(int(s) for s in feat_shape)
        C, T, h, w = self.feat_shape
        # n_blocks (L) is only needed to save maps per LAYER; C = L*H under avg_mode=stack.
        self.n_blocks = int(n_blocks) if n_blocks and C % int(n_blocks) == 0 else 0
        # Group size along which mass is conserved / softmax is taken.
        self.group = h * w if cfg.axis == "spatial" else T * h * w
        # BCE with no pos_weight and no label smoothing: the attacker's objective is the
        # raw "flip this example", not the (class-balanced) training objective.
        self.attack_criterion = nn.BCEWithLogitsLoss()

    # ---- example selection --------------------------------------------------------
    def select(self, y: torch.Tensor) -> torch.Tensor:
        """Rows to attack: valid labels, filtered by `adv_on`. `y` is `[B, 1]` or `[B]`."""
        y = y.reshape(-1)
        valid = y >= 0                      # -1 marks an undecided label — never attack it
        if self.cfg.on == "pos":
            return valid & (y >= 0.5)
        if self.cfg.on == "neg":
            return valid & (y < 0.5)
        return valid

    # ---- the attack ---------------------------------------------------------------
    def perturb(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        forward: Optional[ForwardFn] = None,
        return_trace: bool = False,
        seed: Optional[int] = None,
    ):
        """Return adversarial attention maps for `x` (detached, same shape/dtype).

        Targeted: each example is pushed toward the **flipped** label `1 - round(y)`, i.e.
        negatives are perturbed to look present and positives to look absent — the two
        failure modes a steering optimizer exploits.

        `return_trace=True` also returns the **attack-convergence trace**: the mean
        `P(flipped label)` after each PGD iteration (index 0 = before the first step). A
        trace that is still climbing at the last index means `adv_pgd_steps` is too small
        and the training signal is a weaker attack than the one steering will actually run.

        `seed` makes the run **reproducible**: PGD is already deterministic given
        (weights, input, config), so the random start is the only stochastic part, and
        seeding it pins the whole attack. That is what lets a saved example be regenerated
        exactly instead of stored (see `regenerate_adv_maps`). Training leaves it `None`,
        so each batch gets a fresh random start.
        """
        cfg = self.cfg
        if x.numel() == 0:
            return (x.detach(), []) if return_trace else x.detach()
        forward = forward or (lambda m, z: m(z))
        x_clean = x.detach()
        y_flip = (1.0 - (y.detach().reshape(-1, 1) >= 0.5).to(x_clean.dtype))
        trace: list[float] = [] if return_trace else None
        gen = None
        if seed is not None:
            gen = torch.Generator(device=x_clean.device).manual_seed(int(seed))

        with _attack_mode(model, eval_mode=cfg.eval_mode), torch.enable_grad():
            if cfg.projection == "softmax":
                x_adv = self._pgd_softmax(model, forward, x_clean, y_flip, trace, gen)
            else:
                x_adv = self._pgd_additive(model, forward, x_clean, y_flip, trace, gen)
        return (x_adv.detach(), trace) if return_trace else x_adv.detach()

    def _record(self, trace, logit, y_flip) -> None:
        """Append mean `P(flipped label)` — 1.0 means the attack has fully won."""
        if trace is None:
            return
        p = torch.sigmoid(logit.detach())
        trace.append(float(torch.where(y_flip > 0.5, p, 1.0 - p).mean()))

    # ---- additive PGD in attention space (box / simplex) ---------------------------
    def _pgd_additive(self, model, forward, x_clean, y_flip, trace=None, gen=None) -> torch.Tensor:
        cfg = self.cfg
        step = cfg.resolved_step_size
        x_adv = x_clean.clone()
        if cfg.random_start:
            if cfg.norm == "linf":
                noise = torch.empty_like(x_adv).uniform_(-cfg.epsilon, cfg.epsilon, generator=gen)
                x_adv = x_adv + noise
            else:
                noise = torch.empty_like(x_adv).normal_(generator=gen)
                x_adv = x_adv + cfg.epsilon * _l2_normalize(noise)
            x_adv = self._project(x_adv, x_clean)

        for _ in range(cfg.steps):
            x_adv.requires_grad_(True)
            logit = self._attack_logit(model, forward, x_adv)
            self._record(trace, logit, y_flip)
            loss = self.attack_criterion(logit, y_flip)
            # grad w.r.t. the input ONLY — never populates p.grad, so the outer optimizer's
            # state is untouched and no gradient buffers are allocated for the weights.
            (grad,) = torch.autograd.grad(loss, x_adv)
            del loss, logit
            with torch.no_grad():
                # In-place descent step into grad's storage, then reuse it as the scratch
                # buffer for the projection — one full-batch allocation per PGD step
                # instead of four (see _project).
                if cfg.norm == "linf":
                    grad.sign_()
                else:
                    grad.copy_(_l2_normalize(grad))
                grad.mul_(-step).add_(x_adv.detach())
                x_adv = self._project(grad, x_clean)
            x_adv = x_adv.detach()
        if trace is not None:                 # score of the map actually returned
            with torch.no_grad():
                self._record(trace, self._attack_logit(model, forward, x_adv), y_flip)
        return x_adv

    # ---- PGD in softmax-logit space -------------------------------------------------
    def _pgd_softmax(self, model, forward, x_clean, y_flip, trace=None, gen=None) -> torch.Tensor:
        """`x = mass · softmax(log x_clean + δ)`, `‖δ‖_∞ ≤ ε` — exactly normalized by
        construction, so no projection back onto the simplex is ever needed."""
        cfg = self.cfg
        step = cfg.resolved_step_size
        shaped = self._grouped(x_clean)
        base_logit = shaped.clamp_min(_TINY).log()
        mass = shaped.sum(dim=-1, keepdim=True)
        delta = torch.zeros_like(base_logit)
        if cfg.random_start:
            delta.uniform_(-cfg.epsilon, cfg.epsilon, generator=gen)

        for _ in range(cfg.steps):
            delta.requires_grad_(True)
            x_adv = self._ungrouped(mass * torch.softmax(base_logit + delta, dim=-1), x_clean)
            logit = self._attack_logit(model, forward, x_adv)
            self._record(trace, logit, y_flip)
            loss = self.attack_criterion(logit, y_flip)
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta = (delta.detach() - step * grad.sign()).clamp_(-cfg.epsilon, cfg.epsilon)
            delta = delta.detach()
        with torch.no_grad():
            x_adv = self._ungrouped(mass * torch.softmax(base_logit + delta, dim=-1), x_clean)
            if trace is not None:
                self._record(trace, self._attack_logit(model, forward, x_adv), y_flip)
        return x_adv

    # ---- helpers --------------------------------------------------------------------
    def _attack_logit(self, model, forward, x: torch.Tensor) -> torch.Tensor:
        """Scalar-per-example logit the attacker moves: the clip logit, or the
        min-over-frames logit when the probe carries a per-frame presence head (that is
        what an 'object present in every frame' steering objective actually optimizes)."""
        out = forward(model, x)
        if self.cfg.objective == "perframe_min":
            _, frame_logits = out
            return frame_logits.min(dim=1, keepdim=True).values
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(-1, 1)

    def _grouped(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, ...] → [B, n_groups, group]` for the mass-conserving projections."""
        return x.reshape(x.shape[0], -1, self.group)

    def _ungrouped(self, g: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return g.reshape(like.shape)

    def _project(self, x_adv: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """ε-ball first, then the attention domain — the domain constraint wins, since a
        map outside `[0, domain_max]` (or off the mass simplex) is not a thing the DiT can
        produce, while a slightly-too-large step merely makes the attack stronger.

        Written with in-place ops on a single scratch tensor: at the champion's shape one
        full-batch copy is 2.6 GB, so the naive version's 3–4 temporaries are what actually
        exhausts VRAM. `x_adv` is always detached here, so mutating it is safe.
        """
        cfg = self.cfg
        delta = x_adv.sub_(x_clean)            # reuse x_adv's storage as delta
        if cfg.norm == "linf":
            delta.clamp_(-cfg.epsilon, cfg.epsilon)
        else:
            flat = delta.reshape(delta.shape[0], -1)
            scale = (cfg.epsilon / flat.norm(dim=1).clamp_min(_TINY)).clamp(max=1.0)
            flat.mul_(scale.unsqueeze(1))
        x_adv = delta.add_(x_clean)            # back to absolute coordinates, still in place

        if cfg.projection == "simplex":
            mass = self._grouped(x_clean).sum(dim=-1, keepdim=True)
            g_adv = project_simplex(self._grouped(x_adv), mass, dim=-1)
            x_adv = self._ungrouped(g_adv, x_clean)
        return x_adv.clamp_(0.0, cfg.domain_max)


def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    """Unit-L2 per example (flattened over all non-batch dims)."""
    flat = t.reshape(t.shape[0], -1)
    return flat.div(flat.norm(dim=1, keepdim=True).clamp_min(_TINY)).reshape_as(t)


@contextlib.contextmanager
def _attack_mode(model: nn.Module, eval_mode: bool = True):
    """Freeze parameters and (optionally) disable dropout for the duration of the attack.

    Freezing `requires_grad` keeps autograd from building/holding parameter-gradient state
    for the inner loop's `cfg.steps` backward passes; `eval()` makes the attack
    deterministic w.r.t. dropout / DropBlock so the perturbation targets the model rather
    than one dropout sample. Both are restored exactly on exit, including when the body
    raises — the outer training loop never sees a changed model.
    """
    was_training = model.training
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        if eval_mode:
            model.eval()
        yield
    finally:
        for p, was in flags:
            p.requires_grad_(was)
        model.train(was_training)


# --------------------------------------------------------------------------------------
# Distribution diagnostics: how far do the adversarial maps drift off-manifold?
# --------------------------------------------------------------------------------------
# The point of the constraint set is that adversarial maps stay *plausible*. These three
# per-example summaries are the axes an attack can drift along, and comparing their
# clean/adversarial distributions is how you tell "hardening against realistic attention"
# apart from "hardening against noise the DiT could never produce".
STAT_NAMES = ("mass", "peak", "entropy")


def attention_stats(x: torch.Tensor, group: int) -> dict[str, torch.Tensor]:
    """Per-example summaries of an attention map batch, pooled over (channel, frame).

    * `mass`    — Σ over the group axis: total attention the word receives. Rises under an
                  *inflation* attack; exactly constant under simplex/softmax.
    * `peak`    — max within the group: how spiky the hottest location is.
    * `entropy` — Shannon entropy of the group-normalized map / log(group) ∈ [0, 1]:
                  1 = uniform wash, 0 = all mass on one token. Falls when the attack
                  concentrates attention into a hot blob.
    """
    g = x.reshape(x.shape[0], -1, group).float()
    mass = g.sum(-1)
    p = g / mass.clamp_min(_TINY).unsqueeze(-1)
    ent = -(p * p.clamp_min(_TINY).log()).sum(-1) / float(np.log(group))
    return {"mass": mass.mean(1), "peak": g.amax(-1).mean(1), "entropy": ent.mean(1)}


def reduce_maps(x: torch.Tensor, mode: str, n_blocks: int = 0) -> torch.Tensor:
    """Reduce a `[B, C, T, h, w]` attention batch to the fidelity chosen for storage.

    Storing the raw maps is expensive and — because they are exactly reproducible from
    (checkpoint, clean features, config, seed) — largely redundant. See `regenerate_adv_maps`.

    * `pooled`    → `[B, T, h, w]`, uniform mean over every channel. Smallest, but the mean
      is the *wrong* reduction for a probe whose aggregator weights layers non-uniformly:
      an attack concentrated on a few heads is diluted by 1/C. Fine for the figure, weak
      for forensics.
    * `per_layer` → `[B, L, T, h, w]`, mean over heads *within* each layer. Keeps the axis
      the learned layer aggregator actually acts on, at L/C of the full size (1/32 at
      H=32). The default.
    * `full`      → unchanged `[B, C, T, h, w]`.
    """
    if mode == "full":
        return x
    if mode == "pooled":
        return x.mean(1)
    if mode == "per_layer":
        if n_blocks <= 0 or x.shape[1] % n_blocks != 0:
            raise ValueError(
                f"adv_save_maps='per_layer' needs n_blocks dividing C (got n_blocks="
                f"{n_blocks}, C={x.shape[1]}); use 'pooled' for an already-reduced input."
            )
        B, C = x.shape[:2]
        return x.reshape(B, n_blocks, C // n_blocks, *x.shape[2:]).mean(2)
    raise ValueError(f"Unknown save_maps mode {mode!r}.")


def regenerate_adv_maps(
    model: nn.Module,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    adv: dict | AdvConfig,
    feat_shape: tuple[int, int, int, int],
    forward: Optional[ForwardFn] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Recreate saved adversarial maps **exactly**, at full `[B, C, T, h, w]` fidelity.

    PGD is deterministic given the weights, the input and the config, so the random start
    is the only stochastic part — pinning it with the stored `example_seed` reproduces the
    attack bit-for-bit (verified in `_smoke_adversarial.py`, including across a
    `state_dict` round-trip). This is why the run stores a *recipe* (sample indices + seed
    + config, all in `adv_examples_*.pt`) rather than gigabytes of maps.

        ex = torch.load("adv_examples_step_040.pt", weights_only=False)
        # rebuild `model` from probe_step_040.pt and `x_clean` from ex["index"] over the val set
        full = regenerate_adv_maps(model, x_clean, y, ex["adv"], feat_shape, seed=ex["seed"])
    """
    cfg = AdvConfig.coerce(adv)
    attack = AttentionPGD(cfg, feat_shape=feat_shape)
    return attack.perturb(model, x_clean, y, forward=forward,
                          seed=cfg.example_seed if seed is None else seed)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic — 0 = identical distributions, 1 = disjoint.

    Used per summary stat as a single "how far off-manifold did the attack go" number that
    can be tracked across epochs without picking an arbitrary threshold.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.abs(ca - cb).max())


# --------------------------------------------------------------------------------------
# Robustness evaluation
# --------------------------------------------------------------------------------------
def evaluate_adversarial(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    attack: AttentionPGD,
    forward: Optional[ForwardFn] = None,
    max_batches: Optional[int] = None,
    dataset=None,
    perframe: Optional[tuple[int, int]] = None,   # (offset, T) into the label columns
    n_examples_kept: int = 0,
    save_maps: str = "per_layer",
    example_seed: Optional[int] = None,
) -> dict:
    """Run the attack over `loader` and report how exploitable the probe still is.

    Scalar metrics: robust AUROC / AP / balanced accuracy against their clean
    counterparts, plus the two numbers that map directly onto the inference-time failure —
    `flip_rate` (predictions the attack flips) and `neg_spike_rate` (negatives pushed above
    0.5, i.e. "score spikes, no object"). `ks_<stat>` tracks how far the adversarial maps
    drifted from the real ones along each summary in `STAT_NAMES`.

    Also returned for persisting to disk:
      * `per_example` — one row per attacked val example (scores + map statistics), the
        raw material for every adversarial plot.
      * `trace`       — mean `P(flipped label)` per PGD iteration (attack convergence).
      * `examples`    — the `n_examples_kept` **worst** clean/adversarial map pairs (largest
        score shift over the WHOLE loader, not the first ones seen), reduced to the
        `save_maps` fidelity and stored fp16, plus the `seed` that regenerates them exactly.

    Note the scalar metrics, `per_example` and `trace` all cover **every** attacked example;
    only the raw maps are subsampled, because only they are large.
    """
    import pandas as pd

    from will_it_render.training.evaluation import auroc as _auroc, average_precision as _ap

    was_training = model.training
    model.eval()
    fwd = forward or (lambda m, z: m(z))
    rows: list[dict] = []
    trace_sum: Optional[np.ndarray] = None
    trace_n = 0
    keeper = _TopKMaps(n_examples_kept)
    fr_rows: list[dict] = []
    offset = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        feats, labs = batch[0], batch[1]
        feats = feats.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        y = labs[:, :1]
        sel = attack.select(y)
        idx_global = offset + torch.nonzero(sel, as_tuple=False).reshape(-1).cpu().numpy()
        offset += labs.shape[0]
        if not bool(sel.any()):
            continue
        feats, y = feats[sel], y[sel]
        x_adv, trace = attack.perturb(model, feats, y, forward=forward, return_trace=True,
                                      seed=example_seed)
        if trace:
            t = np.asarray(trace, dtype=float)
            trace_sum = t * len(feats) if trace_sum is None else trace_sum + t * len(feats)
            trace_n += len(feats)
        with torch.no_grad():
            c = fwd(model, feats)
            a = fwd(model, x_adv)
            # Frame-level readouts, when the forward exposes them. The clip logit alone is
            # NOT enough: with adv_objective=perframe_min the attack targets min_t
            # frame_logit, and reporting only the clip head hid a total failure there
            # (clip adv AUROC 0.998 vs min-frame 0.000 on the 2026-08-13 runs).
            c_fr = c[1] if isinstance(c, tuple) and len(c) > 1 else None
            a_fr = a[1] if isinstance(a, tuple) and len(a) > 1 else None
            c = (c[0] if isinstance(c, tuple) else c).reshape(-1)
            a = (a[0] if isinstance(a, tuple) else a).reshape(-1)
            if perframe is not None and c_fr is not None:
                off, Tn = perframe
                ft = labs[sel][:, off:off + Tn]
                fm = ~torch.isnan(ft)
                fr_rows.append({
                    "fy": ft[fm].cpu().numpy(),
                    "fc": torch.sigmoid(c_fr)[fm].cpu().numpy(),
                    "fa": torch.sigmoid(a_fr)[fm].cpu().numpy(),
                    # min-frame: the readout the attack maximizes; its label is
                    # "present in EVERY decided frame"
                    "my": ((torch.nan_to_num(ft, nan=1.0) >= .5).all(1) & fm.any(1)).cpu().numpy(),
                    "mc": torch.sigmoid(c_fr.min(1).values).cpu().numpy(),
                    "ma": torch.sigmoid(a_fr.min(1).values).cpu().numpy(),
                })
            s_clean = attention_stats(feats, attack.group)
            s_adv = attention_stats(x_adv, attack.group)
            d = (x_adv - feats).reshape(len(feats), -1)
            row = {
                "index": idx_global,
                "gt": y.reshape(-1).cpu().numpy(),
                "clean_prob": torch.sigmoid(c).cpu().numpy(),
                "adv_prob": torch.sigmoid(a).cpu().numpy(),
                "delta_linf": d.abs().amax(1).cpu().numpy(),
                "delta_l2": d.norm(dim=1).cpu().numpy(),
            }
            for k in STAT_NAMES:
                row[f"clean_{k}"] = s_clean[k].cpu().numpy()
                row[f"adv_{k}"] = s_adv[k].cpu().numpy()
        rows.append(row)

        if n_examples_kept > 0:
            # Keep the WORST examples over the whole val set, not the first ones the loader
            # happens to yield: a running top-K by score shift costs one extra sort per batch
            # and bounds memory at K + batch_size maps.
            keeper.add(
                torch.from_numpy(np.abs(row["adv_prob"] - row["clean_prob"])),
                reduce_maps(feats, save_maps, attack.n_blocks).half().cpu(),
                reduce_maps(x_adv, save_maps, attack.n_blocks).half().cpu(),
                idx_global,
            )
    model.train(was_training)

    if not rows:
        return {}
    per = pd.DataFrame({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
    clean_p = per["clean_prob"].to_numpy()
    adv_p = per["adv_prob"].to_numpy()
    yb = (per["gt"].to_numpy() >= 0.5).astype(int)

    def _bal_acc(p):
        pred = (p >= 0.5).astype(int)
        tpr = float((pred[yb == 1] == 1).mean()) if (yb == 1).any() else float("nan")
        tnr = float((pred[yb == 0] == 0).mean()) if (yb == 0).any() else float("nan")
        return 0.5 * (tpr + tnr)

    if dataset is not None and hasattr(dataset, "records"):
        # records[i] == (absolute_extraction_dir, step, label, object_name) — the clip dir
        # IS record[0], so its .name is the clip, and the probe target is record[3].
        idx = per["index"].astype(int)
        per["sample_name"] = [Path(dataset.records[i][0]).name for i in idx]
        per["object_name"] = [dataset.records[i][3] for i in idx]

    def _bce(p):
        """Class-only BCE from the probabilities — the same quantity the train loop's
        `val_clean_loss` reports, so `clean_bce` should track it and `adv_bce` is the
        directly comparable 'what the loss would be under attack'."""
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return float(-(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean())

    neg = yb == 0
    out = {
        "clean_bce": _bce(clean_p),
        "adv_bce": _bce(adv_p),
        "clean_auroc": _auroc(yb, clean_p),
        "adv_auroc": _auroc(yb, adv_p),
        "clean_average_precision": _ap(yb, clean_p),
        "adv_average_precision": _ap(yb, adv_p),
        "clean_balanced_accuracy": _bal_acc(clean_p),
        "adv_balanced_accuracy": _bal_acc(adv_p),
        "flip_rate": float(((clean_p >= 0.5) != (adv_p >= 0.5)).mean()),
        "neg_spike_rate": float((adv_p[neg] >= 0.5).mean()) if neg.any() else float("nan"),
        "mean_score_shift": float(np.abs(adv_p - clean_p).mean()),
        "mean_delta_linf": float(per["delta_linf"].mean()),
        "mean_delta_l2": float(per["delta_l2"].mean()),
        "n_examples": int(yb.size),
    }
    for k in STAT_NAMES:
        out[f"ks_{k}"] = ks_distance(per[f"clean_{k}"].to_numpy(), per[f"adv_{k}"].to_numpy())
        out[f"mean_clean_{k}"] = float(per[f"clean_{k}"].mean())
        out[f"mean_adv_{k}"] = float(per[f"adv_{k}"].mean())
    out["per_example"] = per
    out["trace"] = (trace_sum / trace_n).tolist() if trace_n else []
    if fr_rows:
        F = {k: np.concatenate([r[k] for r in fr_rows]) for k in fr_rows[0]}
        fyb, myb = (F["fy"] >= .5).astype(int), F["my"].astype(int)
        def _b(p, y):
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out.update({
            "frame_clean_auroc": _auroc(fyb, F["fc"]), "frame_adv_auroc": _auroc(fyb, F["fa"]),
            "frame_clean_bce": _b(F["fc"], fyb), "frame_adv_bce": _b(F["fa"], fyb),
            "frame_neg_spike": float((F["fa"][fyb == 0] >= .5).mean()) if (fyb == 0).any() else float("nan"),
            "minframe_clean_auroc": _auroc(myb, F["mc"]), "minframe_adv_auroc": _auroc(myb, F["ma"]),
            "minframe_clean_bce": _b(F["mc"], myb), "minframe_adv_bce": _b(F["ma"], myb),
        })

    kept = keeper.result()
    if kept is not None:
        # `seed` + `save_maps` are the recipe: with them the FULL maps can be regenerated
        # exactly (regenerate_adv_maps), which is why we never store them by default.
        idx_kept, clean_maps, adv_maps = kept
        ex = {"index": idx_kept,
              "clean_maps": clean_maps,
              "adv_maps": adv_maps,
              "save_maps": save_maps,
              "seed": example_seed,
              "n_blocks": attack.n_blocks,
              "feat_shape": attack.feat_shape,
              "selection": "top_by_score_shift",
              "n_val_total": int(yb.size)}
        sub = per.set_index("index")
        for col in ("gt", "clean_prob", "adv_prob"):
            ex[col] = [float(sub.loc[i, col]) for i in idx_kept]
        for col in ("sample_name", "object_name"):
            if col in per.columns:
                ex[col] = [str(sub.loc[i, col]) for i in idx_kept]
        out["examples"] = ex
    return out


class _TopKMaps:
    """Running top-K map pairs by score shift, in one pass over the loader.

    Storing every example's raw maps is what makes adversarial dumps huge, and the first K
    the loader yields are not the interesting ones. This keeps only the K largest
    `|adv_prob − clean_prob|` seen so far — the biggest exploits the attack found — so the
    saved maps line up with what the examples figure wants to show.
    """

    def __init__(self, k: int) -> None:
        self.k = max(0, int(k))
        self._shift = self._clean = self._adv = self._idx = None

    def add(self, shift: torch.Tensor, clean: torch.Tensor, adv: torch.Tensor,
            idx: np.ndarray) -> None:
        if self.k == 0:
            return
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        if self._shift is None:
            self._shift, self._clean, self._adv, self._idx = shift, clean, adv, idx_t
        else:
            self._shift = torch.cat([self._shift, shift])
            self._clean = torch.cat([self._clean, clean])
            self._adv = torch.cat([self._adv, adv])
            self._idx = torch.cat([self._idx, idx_t])
        if self._shift.numel() > self.k:
            order = torch.topk(self._shift, self.k).indices
            self._shift, self._clean = self._shift[order], self._clean[order]
            self._adv, self._idx = self._adv[order], self._idx[order]

    def result(self):
        """`(indices, clean_maps, adv_maps)` sorted worst-first, or None if nothing kept."""
        if self._shift is None or self._shift.numel() == 0:
            return None
        order = torch.argsort(self._shift, descending=True)
        return (self._idx[order].tolist(), self._clean[order], self._adv[order])


def save_adv_examples(path: str | Path, examples: dict, adv_cfg: AdvConfig, step: int,
                      epoch: int) -> None:
    """Persist the saved clean/adversarial map pairs (+ their scores) for later inspection.

    A plain `torch.save` payload — the maps are already fp16/pooled, and unlike the feature
    safetensors this file carries mixed metadata (names, scores, the threat model that
    produced it), which is exactly what you need to re-open one of these months later.
    """
    torch.save({**examples, "adv": adv_cfg.to_dict(), "step": step, "epoch": epoch},
               str(path))
