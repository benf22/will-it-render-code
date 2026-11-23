"""Model-agnostic core of the inference-time attention-steering algorithm.

This module owns the *entire* steering optimization algorithm — the per-step
latent optimization, every regularizer, the optimizer/"simple"-GD update, and the
metrics + ``step_callback`` assembly. It contains **no model-specific code**: all
model coupling is isolated behind :class:`SteeringBackendAdapter`, which each
backend (LTX, WAN, ...) implements.

Keeping the algorithm single-sourced here is a correctness guarantee: LTX and WAN
(and future backends) run *the same* optimization; they differ only in the thin
adapter that plumbs the model's forward / scheduler / latent state. See
``steering/steering_loop.py`` (``LTXSteeringAdapter``) for the LTX implementation
and ``tests/test_steering_core.py`` for the mock-adapter test that pins this
algorithm.
"""
from __future__ import annotations

import logging
import math
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional

import torch
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def _make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer {name!r}. Choose 'adam', 'adamw' or 'sgd'.")


def _latent_softmax_kl(x_t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """KL between the two latents treated as 1D distributions via softmax.

    The whole latent is flattened and softmax-normalized into a probability distribution
    over its elements: ``P = softmax(flatten(x_t))`` (steered), ``Q = softmax(flatten(x0))``
    (original, detached target). Returns the forward KL
    ``KL(P‖Q) = Σ_i P_i·(log P_i − log Q_i)``. Uses ``log_softmax`` for numerical
    stability; differentiable w.r.t. ``x_t``.
    """
    logp = torch.log_softmax(x_t.reshape(-1).float(), dim=0)
    logq = torch.log_softmax(x0.reshape(-1).float(), dim=0).detach()
    return (logp.exp() * (logp - logq)).sum()


class SteeringBackendAdapter(ABC):
    """The narrow model-specific seam the steering core optimizes against.

    An adapter is a *stateful* object bound to a single generation: it holds the
    working latent state and advances it step by step. It is used as a context
    manager so backends can install/restore capture hooks, freeze params, and
    toggle gradient checkpointing around the whole run::

        with adapter:
            run_steering_optimization(adapter, steering_model, ...)

    Contract (all indices are 0-based denoising steps):

    - ``steps()`` — the step indices to iterate (``len``-able so tqdm shows a bar).
    - ``get_latent()`` — the current working video latent (read; the core detaches).
    - ``commit_latent(x_t)`` — write the optimized latent back into the state.
    - ``begin_optimization()`` / ``end_optimization()`` — bracket the per-step grad
      optimization; e.g. flip the model into the mode that enables its
      gradient-checkpointing gate, then back for the un-checkpointed stepping forward.
    - ``grad_forward(x_t, step_idx)`` — a **grad-enabled, cond-only** forward with
      cross-attention capture, returning ``(captures, pred_clean)`` where ``captures``
      is a list of ``BlockCapture`` (fed to ``steering_model.loss``) and ``pred_clean``
      is the denoiser's predicted-clean latent (or closest analog) used by ``pred_reg``.
      Called inside ``torch.enable_grad()`` by the core.
    - ``denoise_step(step_idx)`` — the normal **no-grad** denoise + scheduler advance
      on the committed latent (no capture).
    """

    def __enter__(self) -> "SteeringBackendAdapter":
        return self

    def __exit__(self, *exc) -> None:
        return None

    @abstractmethod
    def steps(self) -> Iterable[int]: ...

    @abstractmethod
    def get_latent(self) -> torch.Tensor: ...

    @abstractmethod
    def commit_latent(self, x_t: torch.Tensor) -> None: ...

    def begin_optimization(self) -> None:  # optional hook; default no-op
        return None

    def end_optimization(self) -> None:  # optional hook; default no-op
        return None

    #: Does ``grad_forward``'s second return actually hold the predicted-clean latent x̂0?
    #: False by default, matching the ``velocity()`` pass-through below: the default backend
    #: predicts ``v`` (WAN), so its ``pred`` is a VELOCITY and every consumer whose docstring
    #: says "x̂0" is silently reading the wrong quantity there. LTX (an X0Model) sets True.
    #: Consumers that genuinely need x̂0 — ``pred_reg`` and ``MOVE_BAND(space="pred")`` — assert
    #: on this in ``run_steering_optimization`` rather than degrading quietly.
    #:
    #: TODO(wan-pred-x0): give WanSteeringAdapter an x̂0 accessor (x̂0 = x_t − σ_t·v, the exact
    #: inverse of LTXSteeringAdapter.velocity) and route pred_reg / MOVE_BAND(space="pred")
    #: through it, then drop the assert. NOT done yet on purpose: it changes the quantity those
    #: terms penalize on WAN, so every historical WAN pred_reg number would move and would have
    #: to be re-run before it could be compared with anything new. See the memory note
    #: [[steering-flow-arg-is-x0hat-on-ltx]].
    pred_is_x0: bool = False

    #: On a v-prediction backend (pred_is_x0 False), can the core still recover x̂0 for
    #: pred_reg / MOVE_BAND(space="pred") via the flow-matching relation x̂0 = x_t − σ_t·v?
    #: True when the adapter exposes ``sigma(step_idx)`` (WAN). Lets the pred_reg assert pass
    #: and routes the term through genuine x̂0 drift instead of velocity drift. LTX leaves this
    #: False because pred_is_x0 already gives x̂0 directly. See TODO(wan-pred-x0) — now done.
    pred_to_x0_supported: bool = False

    @abstractmethod
    def grad_forward(self, x_t: torch.Tensor, step_idx: int) -> tuple[list, torch.Tensor]: ...

    def velocity(self, x_t: torch.Tensor, pred: torch.Tensor, step_idx: int) -> torch.Tensor:
        """The flow-matching velocity ``v`` at this step, from ``grad_forward``'s second return.

        ``grad_forward`` returns the backend's NATIVE prediction, and the two backends do not
        parameterize it the same way: WAN's model predicts ``v`` directly, LTX's is an X0Model
        and predicts ``x̂0``. ``pred_reg`` and ``MOVE_BAND(space="pred")`` want the prediction as
        the backend hands it over, but every ``flow_*`` rule wants ``v`` specifically — so the
        conversion lives here rather than inside ``grad_forward``, and each consumer asks for the
        quantity it actually means. Default: the backend's prediction already IS ``v`` (WAN).
        """
        return pred

    @abstractmethod
    def denoise_step(self, step_idx: int) -> None: ...


#: Optional diagnostic hook (off in normal steering). When set to a callable, the
#: optimization loop computes each target object's gradient on x_t separately — before
#: the joint update — and calls GRAD_DIAG_HOOK(step_idx, iter_idx, x_t_detached, {word:
#: grad}). Used only by the multi-object gradient-interaction diagnostic; adds K extra
#: backward passes per iteration, so leave it None for real runs.
GRAD_DIAG_HOOK = None

#: Optional attention-feature capture (off in normal steering). Set by steer.run_steering
#: from the steering config's ``attn_capture:`` block —
#: ``{"step": int, "after_updates": [int, ...], "abort_after": bool}`` — together with
#: ATTN_CAPTURE_SINK(n_updates, captures, target_words_indices, steering_model). At denoising
#: step ``step`` the sink receives the block captures of the working latent after EXACTLY
#: k gradient updates for each k in ``after_updates``: k < n_iters reuses the next
#: iteration's grad forward (free), k == n_iters costs one extra no-grad capturing forward.
#: The captures are the probe's own input (the sink aggregates them with the steering
#: model's agg_kwargs), so what is stored is bit-for-bit what the probe scored.
#: ``abort_after`` raises SteeringAborted right after that step's optimization — no flow
#: advance, no video — for capture-only runs (data augmentation).
ATTN_CAPTURE: Optional[dict] = None
ATTN_CAPTURE_SINK = None
#: LATENT_DUMP_SINK(step_idx, x_t, t) — optional per-step dump of the incoming latent on the
#: all-steps (LOG_ALL_STEPS) path, so an external objective (e.g. TempoControl's own loss code)
#: can be scored on exactly the latents this trajectory visited. Set per prompt by steer.py when
#: STEER_DUMP_LATENTS=1; ``None`` in every real run.
LATENT_DUMP_SINK = None
#: FLOW_DUMP_SINK(kind, step_idx, iter_idx, flow) — optional dump of the flow-matching velocity for
#: manifold-distance analysis. kind="opt" fires once per optimization iteration at each steered step
#: (flow = cond-only vel the update was built from; iter_idx = that iter); kind="denoise" fires once
#: per denoising step (flow = full-CFG velocity advancing the scheduler; iter_idx = -1), covering
#: steered AND un-steered steps. Set per prompt by steer.py when steering.dump_flow is true; None otherwise.
FLOW_DUMP_SINK = None


class SteeringAborted(Exception):
    """The trajectory was cut short on purpose (ATTN_CAPTURE.abort_after)."""

#: Optional gradient-aggregation override (off in normal steering). When set to a callable, the
#: applied update direction is NOT the plain gradient of the mean-BCE loss; instead the loop
#: computes each object's gradient on x_t separately and calls
#: GRAD_AGGREGATE(pgrads: dict[word, grad], flow, scores: dict[word, bce]) -> grad, whose return
#: replaces the applied gradient. `flow` is the flow-matching velocity v (via `adapter.velocity`,
#: NOT `grad_forward`'s raw prediction — those differ on LTX); `scores` are the per-object
#: BCEs (higher = lower probe, for anchor selection). Used to prototype token-aggregation rules
#: (flow-agreement / mutual-agreement / weakest-anchor). Regularizers ignored under this path (reg=0).
GRAD_AGGREGATE = None


def _token_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token cosine between two latent-shaped gradients (channel dim 0 reduced)."""
    a = a.float()
    b = b.float()
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0) + 1e-12)


#: Order in which the mutual-agree fold visits the objects (config ``steering.mutual_order``).
#: The fold is ORDER-DEPENDENT: the first object is kept as is (``MUTUAL_FOLD="anchor"``; the legacy fold
#: masked the first two symmetrically), and every later object only enters on patches where it agrees
#: with the running sum of the earlier ones, so a later object loses every patch-level conflict. Measured on CogVideoX (2026-10-21, k2-7): with
#: the prompt-text order, objects named 4th+ were lost 2.2x as often as with a plain unit-norm
#: mean (15.5% vs 7.0% of unsteered-present objects) and their alignment with the update fell
#: 0.61 -> 0.46. "worst_first" (default since 2026-10-21) folds the least-present object first
#: (ascending probe probability, ties keep text order), so the object that most needs the update
#: is never the one overruled. "text" = the legacy prompt order (every run before 2026-10-21).
#: Needs per-object scores; callers without scores keep the order they pass in.
MUTUAL_ORDER = "worst_first"


def _mutual_order(ws: list, p: Optional[dict]) -> list:
    """Fold order for the mutual-agree rules under :data:`MUTUAL_ORDER` (stable sort)."""
    if MUTUAL_ORDER == "text" or not p:
        return list(ws)
    if MUTUAL_ORDER != "worst_first":
        raise ValueError(f"MUTUAL_ORDER must be 'worst_first' or 'text' (got {MUTUAL_ORDER!r})")
    return sorted(ws, key=lambda w: p[w])


#: Latent dim holding the CHANNELS of one patch (token), per backbone — the dim a per-patch cosine
#: reduces: LTX tokens (1, N, C) -> -1, WAN [C, F, H, W] -> 0, CogVideoX [B, F, C, H, W] -> 2.
#: Set from ``model_type`` by steer.build_backend; None = infer from the latent's ndim (3/4/5).
CHANNEL_DIM: Optional[int] = None
CHANNEL_DIM_BY_MODEL = {"ltx": -1, "wan": 0, "cogvideox": 2}
_CHANNEL_DIM_BY_NDIM = {3: -1, 4: 0, 5: 2}


def _patch_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-PATCH cosine: one value per spatio-temporal patch, over its C-channel vector, keepdim (broadcasts
    back onto the latent). Unlike :func:`_token_cos` (always reduces dim 0 — the channels only on WAN; on
    LTX/CogVideoX dim 0 is the size-1 batch, which makes it a per-ELEMENT sign test)."""
    d = CHANNEL_DIM if CHANNEL_DIM is not None else _CHANNEL_DIM_BY_NDIM[a.dim()]
    a, b = a.float(), b.float()
    return (a * b).sum(d, keepdim=True) / (a.norm(dim=d, keepdim=True) * b.norm(dim=d, keepdim=True) + 1e-12)


#: How the mutual-agree fold combines the objects (config ``steering.mutual_fold``; 2026-10-25):
#:   "anchor" (default) — the first object in :data:`MUTUAL_ORDER` (the weakest, by default) is taken
#:       AS IS; each next object is added only on the PATCHES where it agrees with the running sum
#:       (``_patch_cos(g, agg) >= 0`` over the patch's channel vector); each patch is then divided by the
#:       number of objects summed THERE (a per-patch mean of the contributors, 1..k).
#:   "legacy" — every run before 2026-10-25: the first TWO objects share one mask (the weakest loses its
#:       disagreeing slots too; those slots become 0, so later objects then enter there unopposed), and
#:       the test is :func:`_token_cos`, i.e. per ELEMENT on LTX / CogVideoX, per patch only on WAN.
MUTUAL_FOLD = "anchor"

#: Denoising step indices at which the mutual-agree fold is SKIPPED (config ``steering.mutual_skip_steps``; 2026-10-26):
#: there :func:`agg_mutual_agree` returns the plain mean of the gradients it is given — for the gated rules that is the
#: "-mutual" update (mean of the survivors' unit-normed grads). Empty (default) = never skipped, bit-identical.
#: The step is read from :data:`AGG_STEP`, which the per-object GRAD_AGGREGATE loop sets next to AGG_ITER.
MUTUAL_SKIP_STEPS: frozenset = frozenset()
AGG_STEP = -1

#: Diagnostics only (config ``steering.log_mask_stats``; 2026-10-26): when True, the "anchor" fold records how much it
#: masks (+ a 40-bin histogram of the per-patch fold-test cosines and the sum / sum-sq / count of the negative ones)
#: and the loop adds a ``mask_stats`` dict to that iteration's record in steering_metrics.json. Never changes
#: the update. False (default) = nothing computed.
MASK_STATS = False
_MASK_LAST: Optional[dict] = None


def agg_mutual_agree(pgrads: dict, flow=None, scores=None) -> torch.Tensor:
    """Mutual-agreement aggregation: fold the objects in only where they agree, then average.

    The fold order is :data:`MUTUAL_ORDER` (least-present object first by default) when ``scores`` are
    given, else the dict order; the fold itself is :data:`MUTUAL_FOLD` ("anchor": first object as is,
    each next one added on the patches where it agrees with the running sum, each patch divided by the
    number of objects summed there; "legacy": masked first pair, per-element on LTX/CogX, / k everywhere).
    Degenerate (returns that object's plain gradient) for a single-object prompt, so it is a no-op on
    single-object prompt lists.
    """
    ws = _mutual_order(list(pgrads), _probs_from_scores(scores, list(pgrads)) if scores else None)
    if len(ws) < 2:
        return next(iter(pgrads.values()))
    if AGG_STEP in MUTUAL_SKIP_STEPS:
        return torch.stack([pgrads[w] for w in ws]).mean(0)
    if MUTUAL_FOLD == "legacy":
        g0, g1 = pgrads[ws[0]], pgrads[ws[1]]
        agg = (g0 + g1) * (_token_cos(g0, g1) >= 0).to(g0.dtype).unsqueeze(0)
        for w in ws[2:]:
            g = pgrads[w]
            agg = agg + g * (_token_cos(g, agg) >= 0).to(g.dtype).unsqueeze(0)
        return agg / len(ws)
    if MUTUAL_FOLD != "anchor":
        raise ValueError(f"unknown mutual_fold {MUTUAL_FOLD!r} (anchor | legacy)")
    agg = pgrads[ws[0]]
    cnt = 1.0                                         # per patch: how many objects were summed there
    if MASK_STATS:
        _d = CHANNEL_DIM if CHANNEL_DIM is not None else _CHANNEL_DIM_BY_NDIM[agg.dim()]
        _pe = lambda t: t.float().pow(2).sum(_d, keepdim=True)          # per-patch energy
        _ms = {"n": len(ws), "patches": int(agg.numel() // agg.shape[_d]), "masked_slots": 0.0,
               "e_total": sum(float(_pe(pgrads[w]).sum()) for w in ws), "e_masked": 0.0}
    for w in ws[1:]:
        g = pgrads[w]
        _c = _patch_cos(g, agg)
        m = (_c >= 0).to(g.dtype)
        if MASK_STATS:
            _ms["masked_slots"] += float((1 - m).sum()); _ms["e_masked"] += float((_pe(g) * (1 - m.float())).sum())
            _cf = _c.float().flatten(); _neg = _cf[_cf < 0]
            _h = torch.histc(_cf, bins=40, min=-1.0, max=1.0).tolist()          # per-patch cosines of this fold test
            _ms["cos_hist"] = [a + b for a, b in zip(_ms["cos_hist"], _h)] if "cos_hist" in _ms else _h
            _ms["neg_n"] = _ms.get("neg_n", 0) + int(_neg.numel())
            _ms["neg_sum"] = _ms.get("neg_sum", 0.0) + float(_neg.sum())
            _ms["neg_sumsq"] = _ms.get("neg_sumsq", 0.0) + float(_neg.pow(2).sum())
        agg = agg + g * m
        cnt = cnt + m
    if MASK_STATS:
        _ms["frac_masked"] = _ms["masked_slots"] / (_ms["n"] * _ms["patches"])
        _ms["e_ratio"] = _ms["e_masked"] / max(_ms["e_total"], 1e-30)
        globals()["_MASK_LAST"] = _ms
    return agg / cnt                                  # per-patch mean of the objects that contributed


def agg_mutual_agree_unitnorm(pgrads: dict, flow=None, scores=None) -> torch.Tensor:
    """:func:`agg_mutual_agree` after unit-normalising each object's gradient first.

    The fold is identical; only the inputs are rescaled to unit norm, so an object whose
    gradient happens to be large cannot dominate the sum (the multi-object balancing that
    ``grad_balance: unitnorm`` performs on the timeline branch). This is the rule the
    paper's LTX "+Ours" arm uses (#3 = mutual_agree + unitnorm, lr15, "a <noun>" spans);
    it was previously only reachable from the scratch driver
    ``_scratch/optrule/ltx_mutual_cfg_gen.py``, which defines the same aggregation inline.
    Single-object prompts reduce to that object's (unit-normed) gradient.
    """
    g = {w: v / (v.norm() + 1e-12) for w, v in pgrads.items()}
    return agg_mutual_agree(g, flow, scores)




#: Name → aggregation-rule lookup, so a steering config can request a rule by name
#: (``steering.grad_aggregate: mutual_agree``) instead of a driver assigning
#: :data:`GRAD_AGGREGATE` by hand. Keeps queue/sweep runs expressible in yaml — and
#: therefore visible in the fleet monitor and recorded in the sweep manifest.














def _flow_basis(flow: torch.Tensor) -> torch.Tensor:
    """Per-voxel channel basis of the flow: uncentered SVD of the [N, C] matrix of velocity
    vectors (N = F·H·W voxels) → U [C, C], columns ordered by flow energy (PC 1 ≈ the mean flow
    direction). Recomputed per call (the rules are called once per update; cheap at C=16)."""
    C = flow.shape[0]
    M = flow.float().reshape(C, -1).T                    # [N, C]
    _, _, Vh = torch.linalg.svd(M, full_matrices=False)  # Vh [C, C], rows = PCs
    return Vh.T                                         # [C, C], columns = PCs










def _probs_from_scores(scores: Optional[dict], ws: list) -> dict:
    """Per-object probability from the per-object BCE the loop hands the rule.

    ``scores[w]`` is that object's BCE toward the objective target; for the standard
    ``target = 1.0`` this is ``−log p`` so ``p = e^{−bce}``. Without scores (diagnostic
    callers) every object is treated as half-satisfied, which reduces the adaptive rules
    to plain unitnorm.
    """
    import math
    if not scores:
        return {w: 0.5 for w in ws}
    return {w: math.exp(-max(float(scores.get(w, 0.0)), 0.0)) for w in ws}


















def _make_gated_mutual(tau: float = 0.95, fallback: str = "worst", freeze: bool = False,
                       normalize: bool = True, budget_s: int = 0) -> Callable:
    """Gate FIRST, then mutual-agree among the survivors (2026-09-14).

    ``A+B`` chaining cannot express this: the chain hands rule B a single aggregated vector,
    while mutual agreement needs the per-object gradients. Fusing them in one rule keeps the
    full ``pgrads`` dict available to both halves:

      1. keep the objects still below ``tau`` — the ones that actually need work;
      2. among exactly those, fold in only the TOKENS where their gradients agree
         (:func:`agg_mutual_agree`), so objects that need help do not cancel each other.

    ``fallback`` decides what happens when every object is already above the gate — the case
    that idles 62-91% of the plain gate's iterations (see the 2026_09_14 rule-mechanics
    report). ``"worst"`` spends the iteration on the least-present object instead of emitting
    a zero gradient; ``"zero"`` reproduces the plain gate's behaviour.

    ``budget_s``: minimum number of optimization iterations per denoising step (default 0 =
    pure cutoff). When all objects reach ``tau`` after ``i`` iterations and ``i < budget_s``,
    the remaining ``budget_s - i`` iterations use the ``"worst"`` fallback; once ``AGG_ITER >=
    budget_s`` the rule falls back to zero. Requires ``stop_when_min_prob`` to be set so that
    the early-exit respects the budget (see the ``min_budget`` attribute read by the loop).
    Example: ``budget_s=3`` — if satisfied after 1 iter, push worst for 2 more; if already
    requiring 5 iters, stop immediately.

    ``freeze``: choose the surviving set ONCE, at gradient-iteration 0 of each steered step, and
    keep it for the rest of that step. The probe saturates within a step — both present and
    absent objects exceed 0.99 after a few updates and AUROC falls to chance — so re-reading the
    score mid-step re-selects on noise. Iteration 0 is the only reading taken on the latent as
    the solver delivered it.

    ``normalize``: unit-normalise each object's gradient before the gate/fold (default). ``False``
    runs the same gate + mutual fold (and fallback) on the RAW gradients — the "minus unitnorm"
    ablation (2026-10-19); the fixed lr is then not matched to the normalised rule.
    """
    state = {"iter": None, "keep": None}

    def _n(g):
        return g / (g.norm() + 1e-12) if normalize else g

    def rule(pgrads: dict, flow=None, scores=None) -> torch.Tensor:
        ws = list(pgrads)
        if len(ws) == 1:
            return _n(pgrads[ws[0]])
        u = {w: _n(pgrads[w]) for w in ws}
        p = _probs_from_scores(scores, ws)
        if freeze:
            it = AGG_ITER
            if it <= 0 or state["keep"] is None or state["iter"] is None or it <= state["iter"]:
                state["keep"] = [w for w in ws if p[w] < tau]      # (re)select at the step start
            state["iter"] = it
            keep = [w for w in state["keep"] if w in pgrads]
        else:
            keep = [w for w in ws if p[w] < tau]
        if not keep:
            # budget_s: use worst until AGG_ITER reaches the minimum budget, then zero.
            if fallback == "worst" or AGG_ITER < budget_s:
                return u[min(ws, key=lambda w: p[w])]
            return torch.zeros_like(pgrads[ws[0]])
        if len(keep) == 1:
            return u[keep[0]]
        keep = _mutual_order(keep, p)                         # MUTUAL_ORDER: least-present first
        return agg_mutual_agree({w: u[w] for w in keep}, flow, None)

    def grad_select(ws: list, scores=None) -> list:
        """Objects whose gradient the rule will use, so the loop skips the backward for the rest.

        Only the live (non-frozen) gate is a pure function of the current scores. The rule is
        idempotent under this pre-selection (same survivors, same order, same fallback), so the
        update is unchanged and only the skipped ``autograd.grad`` calls are saved.
        """
        ws = list(ws)
        if freeze or len(ws) == 1:
            return ws
        p = _probs_from_scores(scores, ws)
        keep = [w for w in ws if p[w] < tau]
        if keep:
            return keep
        if fallback == "worst" or AGG_ITER < budget_s:
            return [min(ws, key=lambda w: p[w])]
        return []
    rule.grad_select = grad_select
    rule.min_budget = budget_s
    rule.fallback = fallback      # "worst" | "zero"
    rule.gate_tau = tau           # gate/satisfaction threshold; a zero-fallback rule MUST early-stop
                                  # at this prob (see steer.py) — once every object is satisfied it only
                                  # applies zero updates, so the remaining iterations are a no-op.
    return rule


GRAD_AGGREGATE_REGISTRY = {
    # The shipped per-clip "Ours" rule (FORMAL_VERSIONS): gate p>=0.95 + mutual-agree fold
    # over survivors, worst-object fallback (LTX/WAN) or zero fallback (CogVideoX).
    "gated0.95_mutual": _make_gated_mutual(0.95, "worst"),
    "gated0.95_mutual_zero": _make_gated_mutual(0.95, "zero"),
}

#: Capture-space mutual-agree (see :func:`_capture_space_mutual_grad`). OFF by default; set from config
#: `steering.capture_space_mutual: true` (steer.py). Merged 2026-10-19 from the uncommitted
#: feat+wan-backend worktree (original: origin/wan-mutual-batchgrad). When True *and*
#: GRAD_AGGREGATE is set, per-object grads are aggregated at the attention captures and a SINGLE
#: transformer backward produces the update — O(1) in K transformer backwards, fits a memory-bound
#: GPU. A semantic variant of the injected latent-space rule (which is used only for the
#: compare-log direction cosine). Ported from the ``wan-mutual-batchgrad`` branch.
CAPTURE_SPACE_MUTUAL = False
#: Config: `steering.compare_pobj_grad: true`. Diagnostic compare for the capture-space path: also compute the EXACT per-object loop gradient
#: + the injected latent-space GRAD_AGGREGATE rule and log wall-time and the direction cosine
#: between the two updates. Slow (K extra full backwards); leave False for real runs.
#: Ported from the ``wan-mutual-batchgrad`` branch.
COMPARE_POBJ_GRAD = False


def _capture_space_mutual_grad(steering_model, captures, x_t, step_idx, target_words_indices):
    """Capture-space mutual-agree: O(1)-in-K transformer backwards (fits a memory-bound GPU).

    Ported from the ``wan-mutual-batchgrad`` branch (``vp_batchgrad``), with the unit-norm
    balancing of this branch's :func:`agg_mutual_agree_unitnorm` folded in (see below).

    The K per-object gradients all factor through the SAME transformer graph x_t → attention
    captures, differing only through the (tiny) probe head. So instead of K full backwards to
    x_t (the per-object loop in the GRAD_AGGREGATE branch, ~K× under checkpointing), we:
      1. K CHEAP backwards to the raw attention maps only  (u_i = ∂bce_i/∂attn — probe head, no
         transformer traversal),
      2. reduce each u_i to a common per-(block,head,Lq) direction r_i (sum over the object's
         text-token columns) and run the SAME mutual-agree fold there — agreement is measured
         per video token Lq via cos over (block,head), the capture-space analogue of
         :func:`agg_mutual_agree`'s per-token cosine in latent space,
      3. ONE transformer backward: vJP of the summed masked attention cotangent to x_t.
    Peak memory == one backward (fits H100); wall == 1 fwd + K cheap head-backwards + 1 bwd.
    Semantically a VARIANT of latent-space mutual-agree (agreement moved upstream to attention
    space), not bit-identical — validate it steers comparably. COMPARE_POBJ_GRAD also computes
    the exact loop grad + the injected GRAD_AGGREGATE rule and logs time + direction cosine.

    **unitnorm / grad_balance.** When the selected :data:`GRAD_AGGREGATE` rule is the unit-norm
    variant (:func:`agg_mutual_agree_unitnorm`, i.e. ``grad_balance: unitnorm``), each object's
    capture cotangent is scaled to unit norm before the fold, so no single object dominates the
    sum — the capture-space analogue of unit-norming the per-object *latent* gradients. It is an
    analogue, not an identity: the transformer vJP is linear, so ‖u_w‖=1 does not imply
    ‖J^T u_w‖=1; getting the exact latent norms back would require the K full backwards this
    path exists to avoid. The per-token cosines (and therefore the agree masks) are unaffected,
    since cosine is invariant to a per-object rescale.
    """
    C = [c.attn_map for c in captures]                       # graph tensors [1,H,Lq,Lk] per block
    pobj = steering_model.per_object_bce(
        captures, step_idx, target_words_indices=target_words_indices)
    words = list(pobj)
    # grad_balance: unitnorm — detected from the injected latent-space rule so one config knob
    # ("mutual_agree_unitnorm") drives both paths.
    unitnorm = (GRAD_AGGREGATE is agg_mutual_agree_unitnorm
                or getattr(GRAD_AGGREGATE, "__name__", "").endswith("unitnorm"))

    def tcos(a, b):                                          # [nblocks,H,Lq] -> [Lq]
        a = a.reshape(-1, a.shape[-1]).float(); b = b.reshape(-1, b.shape[-1]).float()
        return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0) + 1e-12)

    def _obj_cotangent(w):
        u = torch.autograd.grad(pobj[w], C, retain_graph=True, allow_unused=True)
        u = [ui if ui is not None else torch.zeros_like(ci) for ui, ci in zip(u, C)]
        if unitnorm:
            # Norm accumulated one block at a time (only one fp32 temporary alive), then scaled
            # IN PLACE. The obvious `[(ub.float() / n).to(ub.dtype) for ub in u]` materialises an
            # fp32 copy of every block simultaneously — on LTX (48 blocks) that transient OOM'd a
            # 95 GB GPU on top of the ~84 GB steering peak. `u` are fresh autograd.grad outputs,
            # not graph nodes, so mutating them is safe.
            n = torch.sqrt(sum(ub.pow(2).sum(dtype=torch.float32) for ub in u)) + 1e-12
            inv = (1.0 / n).to(u[0].dtype)
            for ub in u:
                ub.mul_(inv)
        return u

    def _capture_grad():
        K = len(words)
        # Pass 1: per-object REDUCED direction r_w [nblocks,H,Lq] (small); free the full u.
        R = {}
        for w in words:
            u = _obj_cotangent(w)
            R[w] = torch.stack([ub.sum(dim=-1).squeeze(0) for ub in u])   # sum over text-token cols
            del u
        # mutual-agree fold in capture space → per-object per-Lq keep-mask (small).
        M = {}
        if K < 2:
            M[words[0]] = None
        else:
            w0, w1 = words[0], words[1]
            m01 = (tcos(R[w0], R[w1]) >= 0)
            M[w0] = M[w1] = m01
            aggrR = (R[w0] + R[w1]) * m01.to(R[w0].dtype)
            for w in words[2:]:
                mw = (tcos(R[w], aggrR) >= 0)
                M[w] = mw
                aggrR = aggrR + R[w] * mw.to(R[w].dtype)
        del R
        # Pass 2: accumulate the masked full-attn cotangent, holding ≤1 extra u at a time.
        # Every step here is IN PLACE. The natural comprehensions
        # (`[ub * mv ...]`, `[a + b ...]`, `[ub / K ...]`) each duplicate all ~48 attention
        # blocks at once; stacked on LTX's ~84GB steering peak that OOM'd 95GB cards even
        # though a B200 was fine. `u` are fresh autograd.grad outputs and Uagg is ours, so
        # mutating both is safe.
        Uagg = None
        for w in words:
            u = _obj_cotangent(w)
            if M[w] is not None:
                mv = M[w].to(C[0].dtype).view(1, 1, -1, 1)
                for ub in u:
                    ub.mul_(mv)
            if Uagg is None:
                Uagg = u                       # takes ownership; u is dropped below
            else:
                for a, b in zip(Uagg, u):
                    a.add_(b)
            del u
        if K > 1:
            for ub in Uagg:
                ub.div_(K)
        # single transformer backward (the only expensive one, O(1) in K)
        return torch.autograd.grad(C, x_t, grad_outputs=Uagg,
                                   retain_graph=COMPARE_POBJ_GRAD)[0].detach()

    if not COMPARE_POBJ_GRAD:
        return _capture_grad()

    torch.cuda.synchronize(); t0 = time.perf_counter()
    g_cap = _capture_grad()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    # exact loop grad + injected latent-space aggregator, for a direction comparison
    _pg = {w: torch.autograd.grad(pobj[w], x_t, retain_graph=True)[0].detach() for w in words}
    g_loop = GRAD_AGGREGATE(_pg, None, {w: float(pobj[w].detach()) for w in words})
    torch.cuda.synchronize(); t2 = time.perf_counter()
    cos = float((g_cap.flatten() @ g_loop.flatten()) /
                (g_cap.norm() * g_loop.norm() + 1e-12))
    _logger.info("[capspace k=%d unitnorm=%s] capture=%.3fs loop=%.3fs speedup=%.2fx "
                 "cos(capture,loop)=%.3f |cap|=%.3e |loop|=%.3e", len(words), unitnorm,
                 t1 - t0, t2 - t1, (t2 - t1) / max(t1 - t0, 1e-9), cos,
                 float(g_cap.norm()), float(g_loop.norm()))
    return g_cap


#: Optional aggressive flow variation (off by default). When True *and* GRAD_AGGREGATE is set, a steered
#: step does NOT advance via the model's velocity: after the n_iters optimization it takes one more
#: aggregated-gradient step as the transition (the scheduler state is still advanced, its latent
#: discarded). Non-steered steps flow normally. Prototype for "grad step as the flow advance".
AGGRESSIVE_FLOW = False

#: Set by the optimisation loop before each GRAD_AGGREGATE call: the 0-based gradient-iteration
#: index within the CURRENT steered denoising step. Rules that must freeze a decision at the start
#: of a step (e.g. the *_iter0 gates) read it; everything else ignores it. -1 = unknown/diagnostic.
AGG_ITER = -1

#: A&E's required max-attention for the CURRENT denoising step, resolved by
#: :func:`_ae_sched_threshold` and set next to AGG_ITER by the loop. Read only by
#: ``gate="ae_sched"``. None = no A&E objective in play (the gate then keeps every object).
AGG_AE_THR: Optional[float] = None


def _ae_sched_threshold(steering_model, step_idx: int) -> Optional[float]:
    """A&E's ``thresholds`` dict -> the bar in force at ``step_idx``, held piecewise-constant.

    The paper stores a SPARSE schedule (``{0: 0.05, 10: 0.5, 20: 0.8}``) and consults it only at
    those exact steps, where it decides whether to spend extra refinement iterations. A gate runs
    at every step, so the schedule has to be extended; holding the last value forward reproduces
    the paper's escalation exactly (the bar jumps on step 10 and step 20 and nowhere else) rather
    than inventing intermediate values it never uses.
    """
    thr = getattr(steering_model, "thresholds", None)
    if not thr:
        return None
    past = [s for s in thr if int(s) <= int(step_idx)]
    return float(thr[max(past)]) if past else float(thr[min(thr)])


#: Held-out judge probes (``steering.judge_probes``): None/[] = off, "all" = every loaded probe
#: step other than the one being optimised, or an explicit list of probe steps. At every
#: gradient iteration each judge scores the SAME captures the optimised probe just saw, and the
#: result lands in that iteration's ``judge_prob`` — an independent read of whether the gain is
#: real, since the judge was never optimised against. Diagnostic only: never enters the loss.
JUDGE_PROBES = None

#: HINGE_JUDGE_MODEL (``steering.hinge_judge``): an Attend-and-Excite model instance scored
#: read-only, alongside the optimized objective, at every gradient iteration (and at every
#: non-steered step under LOG_ALL_STEPS). Never enters the loss. Lets a hinge-vs-BCE detector
#: comparison run on a SINGLE probe-steered trajectory instead of two separately-optimized runs.
HINGE_JUDGE_MODEL = None

#: PROBE_MIX (``steering.probe_mix``): use the held-out NEIGHBOUR probes DURING optimisation,
#: not just as a diagnostic. The natural-data judge experiment (2026-09-16) found steering leaks
#: into the optimised probe (≈99% of truly-absent objects read present) but NOT its neighbours
#: (37-55%), so a loss built from the neighbours should buy a gain the neighbours also recognise.
#: Both modes operate on the per-object BCE that GRAD_AGGREGATE consumes, so the objective
#: aggregation (worst / gated0.95) is unchanged — only WHICH probe(s) produce each object's BCE:
#:   "alternate" — iter 0 uses the own (nearest) probe, then each iteration switches to the next
#:                 probe in [own, *neighbours], cycling (2 neighbours → own,n1,n2,own,n1,...).
#:   "common"    — every iteration uses L = 0.5·(L_own + mean_neighbours(L)): a fixed 50/50 blend
#:                 of the own probe and the neighbour consensus.
#:   "neighbors_only" — every iteration uses L = mean_neighbours(L), dropping the own probe
#:                 entirely: steer PURELY by the honest (never-optimised-here) probes.
#: Requires a grad_aggregate rule (it routes through per_object_bce). None => own probe only.
PROBE_MIX = None


def _probe_mix_bce(steering_model, captures, step_idx, iter_idx, target_words_indices):
    """Per-object BCE dict under the active PROBE_MIX policy (own probe when PROBE_MIX is None)."""
    if PROBE_MIX is None or not hasattr(steering_model, "per_object_bce_with"):
        return steering_model.per_object_bce(
            captures, step_idx, target_words_indices=target_words_indices)
    own = steering_model._probe_step(step_idx)
    nbrs = steering_model.probe_neighbors(step_idx)
    mode = PROBE_MIX.get("mode", "common")
    if not nbrs:                               # no neighbour loaded → own only
        return steering_model.per_object_bce_with(
            captures, own, target_words_indices=target_words_indices)
    if mode == "alternate":
        cycle = [own] + list(nbrs)
        ps = cycle[iter_idx % len(cycle)]
        return steering_model.per_object_bce_with(
            captures, ps, target_words_indices=target_words_indices)
    if mode in ("common", "neighbors_only"):
        nbr_bces = [steering_model.per_object_bce_with(
            captures, s, target_words_indices=target_words_indices) for s in nbrs]
        if mode == "neighbors_only":
            return {w: torch.stack([d[w] for d in nbr_bces]).mean() for w in nbr_bces[0]}
        own_bce = steering_model.per_object_bce_with(
            captures, own, target_words_indices=target_words_indices)
        out = {}
        for w, b in own_bce.items():
            nb = torch.stack([d[w] for d in nbr_bces]).mean()
            out[w] = 0.5 * (b + nb)
        return out
    raise ValueError(
        f"PROBE_MIX.mode must be 'alternate', 'common' or 'neighbors_only' (got {mode!r})")


#: SELECT_PROBE (``steering.select_probe``): use a NEIGHBOUR probe to CHOOSE which object the rule
#: acts on (e.g. ``unitnorm_worst`` picks the least-present object), while the GRADIENT still comes
#: from the own/nearest probe. "Neighbour for choosing, not for optimising." The 4-regime judge
#: study showed a neighbour is only honest before it has been steered through, so this tests whether
#: an independent *selection* helps even though its gradient would not. int = a fixed probe step;
#: "neighbor" = the higher adjacent loaded probe (the stronger judge). None = select with the own
#: probe (default). Only affects GRAD_AGGREGATE rules that use ``scores`` (worst / gated).
SELECT_PROBE = None


def _resolve_select_probe(steering_model, step_idx):
    if SELECT_PROBE is None or not hasattr(steering_model, "per_object_bce_with"):
        return None
    if isinstance(SELECT_PROBE, int):
        return SELECT_PROBE if SELECT_PROBE in getattr(steering_model, "probes", {}) else None
    if SELECT_PROBE == "neighbor":
        nbrs = steering_model.probe_neighbors(step_idx)
        return nbrs[-1] if nbrs else None            # higher adjacent probe (stronger judge)
    raise ValueError(f"SELECT_PROBE must be an int, 'neighbor', or None (got {SELECT_PROBE!r})")


#: Which velocity a GRAD_AGGREGATE rule sees as ``flow`` at a steered step: "iter" (default) = the
#: cond-only velocity of the CURRENT iterate (re-evaluated after every update); "k0" = the velocity
#: of the incoming, un-steered latent (computed at iterate 0 and held fixed for all updates), i.e.
#: the natural flow before any optimization. Config knob ``steering.flow_ref``.
FLOW_REF = "iter"

# --- WHERE the flow task is allowed to act. All three default to the pre-2026-09-20 behaviour
# ("inside every optimisation iteration, nowhere else"), so an untouched config is bit-identical.
# They are independent knobs, but 2 and 3 are alternatives to 1 rather than additions to it: both
# assume the optimisation itself ran WITHOUT a flow task, so pair them with `flow_iters: []`.

#: VARIATION 1 — restrict the flow task to specific optimisation iterations within a steered step.
#: ``None`` (default) = every iteration. A list of iteration indices restricts it; ``-1`` means
#: "this step's last iteration", so ``[5, -1]`` is "iteration 5 and the final one". ``[]`` disables
#: the flow task inside the optimisation entirely (what variations 2 and 3 want).
#: On the iterations left out, the update is the argmax subject's RAW gradient — exactly what
#: plain A&E would have applied — not the unit-normed single-task shortcut, which would be a ~250x
#: step on the A&E path. Config knob ``steering.flow_iters``.
FLOW_ITERS: Optional[list] = None

#: VARIATION 2 — run the rule at the SOLVER step instead of inside the optimisation. Today the
#: denoise advance is untouched vanilla: it moves along the model's own velocity while the flow
#: task only ever biased the optimisation. With this on, after the optimisation commits, one
#: cond-only forward at the committed latent gives ``g_AE`` and ``v``; the rule arbitrates them
#: into ``d``; and the advance is taken along ``v' = ||v|| * d/||d||`` instead of ``v`` (fed back
#: to the sampler as ``x̂0' = x_t − σ·v'``, so the stepper itself is untouched). Magnitude is
#: preserved deliberately — only the DIRECTION of the advance is arbitrated, so this is not a
#: change of step size. Config knob ``steering.flow_at_denoise``.
FLOW_AT_DENOISE = False

#: VARIATION 3 — arbitrate the flow against the optimisation AS A WHOLE, once per steered step.
#: Let the optimisation run to completion, take ``Δ = x_end − x_start`` as ONE descent direction
#: (``g_opt = −Δ``, since the applied displacement is ``−lr·g``), CAGrad it against ``v``, and
#: apply the result to the PRE-optimisation latent: ``x = x_start − d``. The per-iteration path is
#: discarded; only its endpoint matters. ``rescale`` puts ``||d|| = ||Δ||``, so the step length is
#: the optimisation's own and the variation changes direction only. Config knob
#: ``steering.flow_on_total``.
FLOW_ON_TOTAL = False

#: v3b — sample the flow at the PRE-optimisation latent ``x0`` instead of at the post-optimisation
#: ``x^N``. The v3 site otherwise reads ``v`` where the latent is furthest off-manifold and uses it
#: to correct a displacement that started somewhere else (report 2026-09-22 §5.1). This is the only
#: one of the three v3b changes that alters the step's DIRECTION. Config: ``steering.flow_on_total_k0``.
FLOW_ON_TOTAL_K0 = False

#: v3b — hand the objective leg to the rule divided by the learning rate, i.e. as the accumulated
#: gradient ``Σg`` rather than the displacement ``−lr·Σg``, and multiply the result back by ``lr``
#: instead of force-rescaling it to ``||Δ||``. Makes the v3 site unit-consistent with the
#: per-iteration site. Direction-neutral for ``flow_scale="match"`` families; it changes the step
#: LENGTH only. Config: ``steering.flow_on_total_per_lr``.
FLOW_ON_TOTAL_PER_LR = False

#: Print one ``FLOWTOTALDIAG`` line per steered step from the v3 site: ``||Δ||``, the sampler's
#: own advance ``|dσ|·||v||`` for that step, their ratio, and ``cos(−Δ, v)``. Env-gated
#: (``STEER_FLOW_TOTAL_DIAG=1``) rather than config-plumbed because it answers ONE question and
#: must not become a knob: is the v3 flow leg's size — pinned to ``||Δ||`` by
#: ``flow_scale="match"`` plus the final rescale — anywhere near the advance the solver actually
#: schedules? If not, every v3 c-ladder reading is confounded with over-denoising.
FLOW_ON_TOTAL_DIAG = bool(os.environ.get("STEER_FLOW_TOTAL_DIAG"))

#: Number of optimisation iterations in the CURRENT steered step, so a rule can resolve
#: ``FLOW_ITERS``'s ``-1`` ("last"). Set next to AGG_ITER by the loop.
AGG_N_ITERS = 1

#: Learning rate and solver ``dσ`` for the CURRENT update, set next to AGG_ITER by the loop.
#: Only ``applied_space`` reads them; every other rule is scale-free and ignores them. ``AGG_LR``
#: is the DECAYED per-step lr the optimiser is about to apply, not the configured base lr.
#: ``AGG_DSIGMA`` is signed and NEGATIVE (σ decreases), so a rule that wants a positive multiple
#: of the advance must use ``abs``. Either may be None if the adapter exposes no schedule, in
#: which case ``applied_space`` falls back to its ordinary scale-free behaviour.
AGG_LR: Optional[float] = None
AGG_DSIGMA: Optional[float] = None


def _adapter_sigma(adapter, step_idx: int) -> tuple[Optional[float], Optional[float]]:
    """``(dσ, σ)`` at ``step_idx`` for whichever backend this is, or ``(None, None)``.

    ``dσ = σ_{k+1} − σ_k`` and is NEGATIVE (σ decreases along the sampler). The two backends carry
    the schedule in different shapes: WAN/diffusers hangs it off ``adapter.scheduler.sigmas`` and
    exposes a ``.sigma``; LTX's loop carries a bare ``adapter.sigmas`` tensor and has neither. The
    ``hasattr(adapter, "sigma")`` probe is what distinguishes them — note it is a SHAPE test, not a
    value test, which is also the reason ``MOVE_BAND(space="x0")`` silently falls back to latent
    space on LTX (see its docstring).

    Returns None on any failure rather than raising: callers (FIELD_HOOK's diagnostics,
    ``applied_space``) all have a defined no-schedule fallback, and a steering run should not die
    because a diagnostic could not read a tensor.
    """
    src = None
    if hasattr(adapter, "sigma"):
        src = getattr(getattr(adapter, "scheduler", None), "sigmas", None)
    elif hasattr(adapter, "sigmas"):
        src = adapter.sigmas
    if src is None:
        return None, None
    try:
        # LTX video edit runs a truncated schedule with absolute step numbers (step_offset).
        i = int(step_idx) - int(getattr(adapter, "step_offset", 0) or 0)
        nx = min(i + 1, len(src) - 1)
        return float((src[nx] - src[i]).item()), float(src[i].item())
    except Exception:
        return None, None



#: Velocity projection (regular generation, no probe needed). ``{"mode": "keep"|"remove",
#: "dims": [1-based flow-PC indices], "steps": None | [step indices]}``: at every denoising step
#: (or the listed ones) the CFG velocity used for the advance is replaced by its component inside
#: (keep) / outside (remove) the span of the listed PCs of the per-voxel channel basis computed
#: from THAT step's own velocity (uncentered SVD, PC 1 = the common-mode direction). Set from
#: ``steering.velocity_proj``; None = untouched.
VELOCITY_PROJ: Optional[dict] = None

#: Movement band regularizer ``{"weight": w, "lo": d_lo, "hi": d_hi, "space": "x0"|"latent"|"pred",
#: "steps": None | [step indices]}`` (steps: apply only at these steered steps — step 0's natural
#: relative displacement is ~0.5 because x̂0 there is nearly noise, so bands tuned for steps ≥1 blow up at 0):
#: penalizes the per-step relative displacement of the latent (or of the predicted clean latent /
#: the velocity) for being below ``lo`` or above ``hi``. The floor is what the existing trust-region
#: terms (anchor_reg / pred_reg, "not too much") lack: a "move at least this much" side. Set from
#: ``steering.optimization.move_band``; None = off. Note a floor in LATENT space rewards moving
#: anywhere (it amplifies whatever direction the gradient already has); the x0 space measures the
#: denoiser-visible change, which is the on-manifold notion of "moved".
MOVE_BAND: Optional[dict] = None

#: Persistence regularizers (2026-09-10 step-7 study) ``{"kind": "latent"|"flow"|"sawtooth",
#: "weight": w, "steps": None | [step indices], "detach_flow": bool}`` — ONE extra term added to
#: the BCE, loss = BCE + w·term, where (z_0 = the incoming latent of this step, z_i = the iterate,
#: v = cond-only velocity of the iterate, as in the aggregation rules):
#:   latent   : ‖z_i − z_0‖₂                     (L2 norm, not the mean-square of anchor_reg)
#:   flow     : ‖v(z_i) − v(z_0)‖₂ / ‖v(z_0)‖₂    (relative change of the flow)
#:   sawtooth : s(A(z_i)) − s(A(z_i + Δσ·v(z_i)))  s = mean target-word probe prob, the second
#:              term scored on the attention of the flow-advanced latent (one Euler step to the
#:              next timestep, forward at step_idx+1, differentiable through v unless detach_flow)
#: Raw term always logged as ``persist_term``. Set from ``steering.optimization.persist_reg``.
PERSIST_REG: Optional[dict] = None

#: Adaptive per-step iteration count: stop a step's optimization once ``min_i p_i >= value``
#: (the weakest object is satisfied). Set from ``steering.optimization.stop_when_min_prob``;
#: None = always run the full ``n_iters``. Pair it with a larger ``n_iters`` to get "up to N
#: updates, but only as many as this prompt needs".
STOP_WHEN_MIN_PROB: Optional[float] = None

#: Multistep-solver reset after a steered step (WAN UniPC). The UniPC *corrector* rebuilds the
#: current sample from ``last_sample`` (the previous, un-steered latent) and the velocities, so a
#: steered latent displacement Δx_t enters the advance only through Δv(x_t) — the direct latent
#: edit is discarded (= the "sawtooth"; an Euler look-ahead x_t + Δσ·v keeps the probe gain, the
#: UniPC advance does not). ``"corrector"``: skip the corrector at a steered step (last_sample=None);
#: ``"full"``: also drop the multistep history (order-1 predictor at that step). Config
#: ``steering.solver_reset``; None = stock solver.
SOLVER_RESET: Optional[str] = None


def project_velocity(v: torch.Tensor, step_idx: int) -> torch.Tensor:
    """Apply VELOCITY_PROJ to a [C, F, H, W] velocity (no-op when unset / step not listed)."""
    cfg = VELOCITY_PROJ
    if not cfg or (cfg.get("steps") is not None and step_idx not in cfg["steps"]):
        return v
    dims = [int(d) for d in cfg.get("dims", [1])]
    U = _flow_basis(v)[:, [d - 1 for d in dims]].to(v.dtype)     # [C, |dims|], basis of THIS step's flow
    C = v.shape[0]; V = v.reshape(C, -1)
    P = U @ (U.T @ V)
    return (P if cfg.get("mode", "remove") == "keep" else V - P).reshape_as(v)

#: Score the probe at EVERY denoising step, not just the steered ones (diagnostic). When on,
#: each non-steered step also runs a no-grad capturing forward + probe eval on the incoming
#: latent and emits a step_callback entry (``steered: False``, ``n_updates: 0``,
#: ``prob_before == prob_after`` since no optimization happened there). This makes the full
#: probe trajectory across all 50 steps visible for debugging (e.g. the reversion / decay on
#: the unsteered steps), at the cost of one extra capturing forward per non-steered step.
#: Requires the steering model to expose ``_probe_step`` (the per-step keyer) so a nearest
#: probe can score steps outside ``steps_to_steer``. Off by default (env ``STEER_LOG_ALL_STEPS=1``).
LOG_ALL_STEPS = os.environ.get("STEER_LOG_ALL_STEPS", "0") not in ("0", "", "false", "False")


def _nearest_probe_step(steering_model, step_idx: int):
    """The probe-step used to score ``step_idx`` (nearest/exact), or None if the model has no
    per-step probe keyer or no probe is available. Used only by the all-steps diagnostic."""
    ps = getattr(steering_model, "_probe_step", None)
    if not callable(ps):
        return None
    try:
        return ps(step_idx)
    except Exception:
        return None


#: Optional per-iteration FIELD capture (diagnostic). If set, the loop calls
#: FIELD_HOOK(step_idx, iter_idx, dict(x_t=…, grad=…, lr=…, flow=…, dsigma=…)) immediately BEFORE
#: each update is applied, with detached tensors. Lets a caller record the update field, the
#: flow field and their geometry per voxel without changing the optimisation.
#: Under GRAD_AGGREGATE the payload also carries ``pgrads`` (the per-object gradients the rule
#: aggregated) and ``pbce`` (each object's BCE, so p = e^{-bce}), which lets a capture decompose
#: the applied update into its per-object components — which object wrote which patch, and where
#: two objects cancel. ``None`` on the plain path, where no per-object gradient exists.
FIELD_HOOK = None

#: CAGRAD_PROBE(sel, coef) — optional diagnostic capture of the CAGrad solve's OWN output, called
#: once per rule invocation with the surviving object names and the per-object effective weights
#: (``[M, k]`` in the per_patch branch, ``[k]`` globally). FIELD_HOOK can recompute the Gram from
#: ``pgrads``, but not the coefficients: they depend on c, the gw_floor guard and the rescale
#: branch, so a recomputation would be a second implementation free to drift from the one that
#: actually stepped. Read-only; ``None`` in every real run.
CAGRAD_PROBE = None

#: With FIELD_HOOK set, also take one reading AFTER the last update of each steered step (recorded
#: as iteration ``n_updates``, zero gradient). Costs one conditional forward per steered step and
#: is what gives the final update a reaction / look-ahead of its own, on this side of the solver
#: advance. Diagnostic captures only.
FIELD_HOOK_POST = False

#: Optional x̂0 capture (diagnostic). If X0_HOOK is a callable and X0_POINTS a set of point
#: keys, the loop calls X0_HOOK(point_key, x0_latent) at those points: ("in", step) = the
#: incoming latent before that step's optimization; (step, n_updates) = after that many
#: gradient updates within the step. Used by the x̂0-decode diagnostic; None for real runs.
X0_HOOK = None
X0_POINTS = None


def _route_weights(c: torch.Tensor, lo: float = 0.0, hi: float = 1.0,
                   ramp: int = 1) -> torch.Tensor:
    """Control ``c`` ``[T]`` → the PRESENT half's per-frame routing weight ``m ∈ [lo, hi]^T``.

    The hard mask (``lo=0, hi=1, ramp=1``) is ``m = (c >= 0.5)``: each half writes only to its
    own frames, and the control's 0→1 seam becomes a routing discontinuity even though that is
    exactly where the object has to physically appear. Two independent softenings:

    * ``ramp`` (odd) box-smooths the binary control over that many frames, with edge replication,
      so the seam is crossed in ``ramp`` steps instead of one. ``c = [0,0,0,1,1,1]`` at ``ramp=3``
      → ``[0, 0, ⅓, ⅔, 1, 1]``.
    * ``lo``/``hi`` then rescale into that band, so each half keeps a minority share of every
      frame rather than being switched fully off.

    The absent half is always given ``1 - m``, so the two weights sum to 1 at every frame and the
    update is step-size-matched to the hard mask (nothing is globally scaled up or down).
    """
    m = (c >= 0.5).to(dtype=torch.float32)
    if ramp > 1:
        pad = ramp // 2
        # replicate-pad so a transition near the clip edge is not smoothed toward zero — the
        # control is a statement about the clip, not a signal that decays outside it.
        m = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(m[None, None], (pad, pad), mode="replicate"),
            kernel_size=ramp, stride=1)[0, 0]
    return lo + (hi - lo) * m


def _latent_frame_mask(x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """``present`` ``[T]`` (bool, or float routing weights) → a mask broadcastable over ``x``'s
    latent-frame axis.

    The axis is found by size rather than by index because the backends disagree on layout
    (WAN's working latent is ``[C, F, H, W]``, LTX's is not); ``T`` is the probe's frame count,
    so a unique match IS the frame axis. Ambiguity is an error rather than a guess — silently
    masking the channel axis would look like a working run and produce nonsense.
    """
    T = int(present.numel())
    axes = [i for i, s in enumerate(x.shape) if s == T]
    if len(axes) != 1:
        raise ValueError(
            f"frame_grad_mask: latent shape {tuple(x.shape)} has {len(axes)} axes of size "
            f"{T}; cannot identify the latent-frame axis unambiguously.")
    shape = [1] * x.ndim
    shape[axes[0]] = T
    return present.to(dtype=x.dtype, device=x.device).reshape(shape)


def _other_text_key_indices(
    Lk: int,
    target_indices: set[int],
    tokens_mode: str,
    n_text_tokens: Optional[int],
) -> list[int]:
    """Key positions for the attention-anchor term: every text token EXCEPT the target's.

    Ported from the pre-core LTX loop (master, attention-anchor regularizer). The
    cross-attention key axis is the text encoder's padded sequence (LTX: 256), right-padded,
    so the real prompt tokens occupy ``[0, n_text_tokens)`` in tokenizer order -- the same
    indexing ``target_words_indices`` use. ``"prompt"`` = real tokens only; ``"all"`` = every
    key position, padding included (the capture hook re-materialises ``softmax(QK^T)``
    WITHOUT the padding mask, so padding holds real mass). The target word's own tokens are
    dropped in both modes.
    """
    tokens_mode = str(tokens_mode).lower()
    if tokens_mode == "prompt":
        if n_text_tokens is None:
            raise ValueError(
                "attn_anchor_tokens='prompt' needs the prompt's real token count; the "
                "backend did not pass text_token_count (use 'all' instead)."
            )
        upper = min(int(n_text_tokens), Lk)
    elif tokens_mode == "all":
        upper = Lk
    else:
        raise ValueError(
            f"Unknown attn_anchor_tokens {tokens_mode!r}; choose 'prompt' (exclude "
            f"padding) or 'all' (include padding)."
        )
    idx = [i for i in range(upper) if i not in target_indices]
    if not idx:
        raise ValueError(
            f"attention-anchor token set is empty (Lk={Lk}, mode={tokens_mode!r}, "
            f"n_text_tokens={n_text_tokens}, targets={sorted(target_indices)})."
        )
    return idx


def run_steering_optimization(
    adapter: SteeringBackendAdapter,
    steering_model,
    target_words_indices: dict[str, list[int]],
    n_iters: int,
    optimizer_name: str = "adam",
    lr: float = 0.1,
    magnitude_reg: float = 0.0,
    anchor_reg: float = 0.0,
    pred_reg: float = 0.0,
    kl_reg: float = 0.0,
    entropy_reg: float = 0.0,
    attn_anchor_reg: float = 0.0,
    attn_anchor_tokens: str = "prompt",
    attn_anchor_ref: str = "step",
    text_token_count: Optional[int] = None,
    source_reg: float = 0.0,
    step_callback: Optional[Callable[[int, dict], None]] = None,
    desc: str = "Steering",
) -> None:
    """Run the per-step steering optimization over the whole denoising trajectory.

    ``n_iters`` — gradient steps per steered denoising step (0 disables steering;
    a step is also skipped when ``steering_model.should_steer`` is False).
    ``optimizer_name`` — ``"simple"`` (plain GD) or ``"adam"``/``"adamw"``/``"sgd"``.

    Regularizers keeping the steered latent close to the un-steered trajectory
    (added on top of the probe loss; all default to 0 = off):

    - ``magnitude_reg`` · ``(‖x_t‖ − ‖x_t0‖)²`` — preserve the latent *norm*.
    - ``anchor_reg``    · ``mean((x_t − x_t0)²)`` — trust region on the latent.
    - ``pred_reg``      · ``mean((x̂0(x_t) − x̂0(x_t0))²)`` — prediction-space consistency
      (``pred_clean`` from ``grad_forward``, cached at the first iteration).
    - ``kl_reg``        · ``KL(softmax(x_t) ‖ softmax(x_t0))`` — 1D-distribution KL.
    - ``entropy_reg``   · per-frame spatial entropy of the target-word attention
      (TempoControl eq. 8); only applied when ``steering_model`` exposes
      ``spatial_entropy`` and skipped entirely (not just unweighted) when off.
    - ``attn_anchor_reg`` · ``mean_blocks(mean((A_other(x_t) − A_other0)²))`` — attention-space
      trust region on every text key EXCEPT the steered words' tokens (``attn_anchor_tokens``:
      ``prompt`` | ``all``; see :func:`_other_text_key_indices`). ``attn_anchor_ref="step"``
      takes ``A_other0`` from this step's iteration 0 (term is 0 there); ``"unsteered"`` takes
      it from a never-steered shadow trajectory (``adapter.shadow_reference``; LTX only, one
      extra no-grad forward per step). Skipped entirely when 0 — no extra log keys.
    - ``source_reg`` · ``mean((x_t − x_orig(step))²)`` — video edit only: anchor to the SOURCE
      video's noising line (``adapter.source_reference``; LTX video-edit adapters). Raw
      ``source_term`` logged whenever a reference exists; nothing at all otherwise.
    """
    # Attention spatial-focus (entropy) regularizer is available only on steering
    # models that expose spatial_entropy (the probe steerer). Unlike the cheap
    # latent-space terms, this one re-runs aggregate_captures_to_vector per target
    # word, so it is computed only when entropy_reg > 0 rather than always-for-logging
    # (set a tiny entropy_reg to calibrate the raw term without moving the loss).
    use_entropy = bool(entropy_reg) and hasattr(steering_model, "spatial_entropy")
    # Attention-anchor trust region (off by default; every branch below is gated on it).
    use_attn_anchor = bool(attn_anchor_reg)
    if use_attn_anchor and str(attn_anchor_ref) not in ("step", "unsteered"):
        raise ValueError(
            f"Unknown attn_anchor_ref {attn_anchor_ref!r}; choose 'step' (this step's "
            f"pre-optimization latent) or 'unsteered' (a shadow un-steered trajectory).")
    use_shadow = use_attn_anchor and str(attn_anchor_ref) == "unsteered"
    if use_shadow and not hasattr(adapter, "shadow_reference"):
        raise NotImplementedError(
            f"attn_anchor_ref='unsteered' needs {type(adapter).__name__}.shadow_reference "
            f"(implemented on LTX only).")
    target_key_indices = {int(i) for idxs in target_words_indices.values() for i in idxs}
    # Source-video anchor (video edit): adapters without a source reference never pay for it.
    _source_ref_fn = getattr(adapter, "source_reference", None)
    attn_key_idx: Optional[torch.Tensor] = None

    # Attend-and-Excite is the only steering model with a per-step schedule (a variable
    # iteration budget + √-decayed step size + threshold-driven early exit). Gate the WHOLE
    # schedule on one explicit mode check so every other model (per_step "ours", tempo_control)
    # takes the exact fixed-n_iters / constant-lr path unchanged — the schedule hooks
    # (iters_for_step / lr_multiplier / stop_iteration) live on baselines/attend_excite/loss.py.
    is_attend_excite = getattr(steering_model, "mode", None) == "attend_excite"
    # Correlation-based early stopping (TempoControl, arXiv:2510.02226): a steering model
    # may ask to stop a step's optimization once a tracked metric reaches a target. Off
    # unless the model exposes both attrs — our probe steerers don't, so this is a no-op
    # for them.
    es_metric = getattr(steering_model, "early_stop_metric", None)
    es_threshold = getattr(steering_model, "early_stop_threshold", None)
    steps = list(adapter.steps())
    n_steps_total = len(steps)
    # Attention capture spec (None in every real steering run).
    _cap = ATTN_CAPTURE if (ATTN_CAPTURE and ATTN_CAPTURE_SINK is not None) else None
    _cap_step = int(_cap["step"]) if _cap else None
    _cap_k = {int(k) for k in _cap.get("after_updates", [])} if _cap else set()

    # pred_reg and MOVE_BAND(space="pred") are documented as x̂0-space terms, but they read
    # grad_forward's NATIVE prediction — which is x̂0 only on an X0Model backend. On WAN that
    # tensor is the velocity v, so both terms would silently penalize VELOCITY drift instead.
    # Fail loudly at configuration time rather than produce a number whose name is a lie.
    # See the TODO(wan-pred-x0) on SteeringBackendAdapter.pred_is_x0 for the real fix.
    _wants_x0 = []
    if pred_reg:
        _wants_x0.append(f"pred_reg={pred_reg}")
    if MOVE_BAND and MOVE_BAND.get("space", "x0") == "pred":
        _wants_x0.append('MOVE_BAND(space="pred")')
    assert adapter.pred_is_x0 or adapter.pred_to_x0_supported or not _wants_x0, (
        f"{', '.join(_wants_x0)} needs the predicted-clean latent x̂0, but "
        f"{type(adapter).__name__}.grad_forward returns the backend's native prediction, "
        f"pred_is_x0 is False and pred_to_x0_supported is False — on a v-prediction backend the "
        f"native prediction is the VELOCITY, so the term would penalize velocity drift under an "
        f"x̂0 name. Fix: set pred_to_x0_supported and expose sigma(step_idx) (x̂0 = x_t - sigma_t*v), "
        f'or use anchor_reg (latent space) / MOVE_BAND(space="latent") instead.'
    )

    with adapter:
        # Optional {step_idx: n_iters} map on the probe steerer — spend a different gradient
        # budget per denoising step (None => constant n_iters everywhere). Ignored for A&E,
        # which owns its own schedule.
        nps = None if is_attend_excite else getattr(steering_model, "n_iters_per_step", None)
        for step_idx in tqdm(steps, desc=desc, unit="step"):
            eff_iters = nps.get(step_idx, n_iters) if nps else n_iters
            do_steer = eff_iters > 0 and steering_model.should_steer(step_idx)

            # attn_anchor_ref="unsteered": advance the shadow trajectory every step and take
            # this step's reference attention from it when about to steer.
            shadow_ref: Optional[list] = None
            if use_shadow:
                _u_caps = adapter.shadow_reference(step_idx, capture=do_steer)
                if do_steer:
                    if attn_key_idx is None:
                        attn_key_idx = torch.as_tensor(
                            _other_text_key_indices(_u_caps[0].attn_map.shape[-1], target_key_indices,
                                                    attn_anchor_tokens, text_token_count),
                            dtype=torch.long, device=_u_caps[0].attn_map.device)
                    shadow_ref = [c.attn_map[0].index_select(-1, attn_key_idx).detach() for c in _u_caps]
                del _u_caps

            # x̂0 capture: incoming latent (before this step's optimization / after the
            # previous step's flow advance). Runs for steered AND non-steered steps.
            if X0_HOOK is not None and X0_POINTS is not None and ("in", step_idx) in X0_POINTS:
                X0_HOOK(("in", step_idx), adapter.estimate_x0(adapter.get_latent(), step_idx))

            if do_steer:
                # Per-step iteration budget and step size. Constant for every model except A&E,
                # whose schedule hooks override them; with the gate closed step_iters == n_iters
                # and step_lr == lr (no early exit) — byte-identical to the pre-A&E behaviour.
                if is_attend_excite:
                    step_iters = steering_model.iters_for_step(step_idx)
                    step_lr = lr * steering_model.lr_multiplier(step_idx, n_steps_total)
                else:
                    step_iters = eff_iters
                    step_lr = lr
                adapter.begin_optimization()  # enable the gradient-checkpointing gate
                # Pre-optimization latent (values of x_t before any step) — reference
                # for the norm / anchor regularizers.
                x0 = adapter.get_latent().detach()
                x_t = x0.clone().requires_grad_(True)
                x0_norm = x0.norm()
                # Un-steered predicted-clean latent x̂0(x_t0), cached on the first
                # iteration for the prediction-space regularizer (None until then).
                pred0: Optional[torch.Tensor] = None
                # Reference non-target attention: from the shadow (ref="unsteered") or cached
                # at iteration 0 below (ref="step"). Unused unless attn_anchor_reg is on.
                attn_ref: Optional[list] = shadow_ref
                # optimizer_name == "simple" → manual plain-GD update (opt stays None);
                # otherwise an Adam/AdamW/SGD optimizer over x_t.
                opt = None if optimizer_name.lower() == "simple" else _make_optimizer(optimizer_name, [x_t], step_lr)
                last_metrics: dict = {}
                first_metrics: dict = {}
                # Per-optimization-iteration snapshots (iter 0 == the pre-steer / "unsteer"
                # reading at this denoising step; iters 1..n-1 are the in-optimization
                # readings). Lets a trace show the full within-step optimization path,
                # not just before/after. Small (per-word probs + scalars); always logged.
                iter_snaps: list[dict] = []
                n_updates = 0   # gradient updates actually applied (varies under A&E's schedule)
                for iter_idx in range(step_iters):
                    with torch.enable_grad():
                        captures, pred = adapter.grad_forward(x_t, step_idx)
                        # `pred` is the backend's NATIVE prediction (x̂0 on LTX, v on WAN) and stays
                        # that way for pred_reg / MOVE_BAND. `vel` is the flow-matching velocity,
                        # which is what every flow_* rule and the field hook mean by "flow".
                        vel = adapter.velocity(x_t.detach(), pred.detach(), step_idx)
                        if FLOW_DUMP_SINK is not None:
                            FLOW_DUMP_SINK("opt", step_idx, iter_idx, vel)
                        if iter_idx == 0:
                            flow0 = vel   # natural (pre-optimization) flow, for FLOW_REF == "k0"
                        # iter_idx updates have been applied to x_t at this point, so this
                        # forward's captures ARE the "after iter_idx updates" attention.
                        if _cap_step == step_idx and iter_idx in _cap_k:
                            ATTN_CAPTURE_SINK(iter_idx, captures, target_words_indices, steering_model)
                        # On the GRAD_AGGREGATE path the update is built from per_object_bce, so this
                        # loss graph is never backpropagated; build it without grad (same values,
                        # no duplicate probe graph kept alive).
                        _agg_only = (GRAD_AGGREGATE is not None
                                     and hasattr(steering_model, "per_object_bce")
                                     and not (getattr(steering_model, "frame_grad_mask", False)
                                              and hasattr(steering_model, "per_frame_class_losses")))
                        with torch.set_grad_enabled(not _agg_only):
                            loss, last_metrics = steering_model.loss(
                                captures, step_idx, target_words_indices=target_words_indices
                            )
                        bce_loss = loss  # pre-regularizer BCE loss (kept so GRAD_AGGREGATE can add reg on top)
                        # Keep the noise magnitude similar to its pre-optimization
                        # value: + magnitude_reg * (||x_t|| - ||x_t0||)^2. Raw term always
                        # computed+logged (weight-free) for calibration, even at magnitude_reg == 0.
                        mag_term = (x_t.norm() - x0_norm) ** 2
                        last_metrics["mag_term"] = float(mag_term.detach())
                        if magnitude_reg:
                            loss = loss + magnitude_reg * mag_term
                        # Trust region: keep x_t near its pre-optimization value so the
                        # steered output stays close to the un-steered generation.
                        # + anchor_reg * mean((x_t - x_t0)^2). Raw term always logged
                        # (weight-free) for calibration, even when anchor_reg == 0.
                        anchor_term = ((x_t - x0) ** 2).mean()
                        last_metrics["anchor_term"] = float(anchor_term.detach())
                        if anchor_reg:
                            loss = loss + anchor_reg * anchor_term
                        # Prediction-space consistency: keep the model's predicted clean
                        # latent near the un-steered prediction (on-manifold/realism).
                        # + pred_reg * mean((x̂0(x_t) - x̂0(x_t0))^2). pred0 is the
                        # detached prediction from the first iter (x_t == x_t0 there, so
                        # pred_term == 0 on iter 0). Always logged for calibration.
                        if pred0 is None:
                            pred0 = pred.detach()
                        # x̂0 drift = ((x̂0(x_t) - x̂0(x_t0))^2).mean(). pred IS x̂0 on an X0Model
                        # (LTX) -> pred - pred0. On a v-prediction backend that exposes sigma (WAN,
                        # pred_to_x0_supported) convert IN-GRAPH via x̂0 = x_t - σ_t·v, so Δx̂0 =
                        # Δx_t - σ_t·Δv (same math as MOVE_BAND space="x0"). Backends that can do
                        # neither keep the velocity-drift value (logged only; pred_reg blocked above).
                        if adapter.pred_is_x0 or not adapter.pred_to_x0_supported:
                            _x0_drift = pred - pred0
                        else:
                            _x0_drift = (x_t - x0) - adapter.sigma(step_idx) * (pred - pred0)
                        pred_term = (_x0_drift ** 2).mean()
                        last_metrics["pred_term"] = float(pred_term.detach())
                        if pred_reg:
                            loss = loss + pred_reg * pred_term
                        # Movement band (MOVE_BAND): force the optimization to move the latent by AT
                        # LEAST `lo` but AT MOST `hi` (relative L2 displacement d = ‖Δ‖/‖ref‖),
                        # measured in latent space (Δx_t), velocity space (Δpred) or predicted-clean
                        # space (Δx̂0 = Δx_t − σ·Δpred, needs adapter.sigma). Penalty
                        # w·(relu(lo − d)² + relu(d − hi)²): zero inside the band, quadratic outside.
                        # Raw d always logged as move_term (weight-free) for calibration.
                        if MOVE_BAND and (MOVE_BAND.get("steps") is None or step_idx in MOVE_BAND["steps"]):
                            _sp = MOVE_BAND.get("space", "x0")
                            _eps = 1e-6                       # smooth norm: sqrt(‖Δ‖²+eps) has a finite gradient at Δ=0
                            _rn = lambda a, b: torch.sqrt((a ** 2).sum() + _eps) / (b.norm() + 1e-12)
                            if _sp == "latent":
                                _d = _rn(x_t - x0, x0)
                            elif _sp == "pred":
                                _d = _rn(pred - pred0, pred0)
                            else:  # "x0": Δx̂0 = Δx_t − σ·Δv, relative to x̂0(x_t0)
                                _sig = adapter.sigma(step_idx) if hasattr(adapter, "sigma") else None
                                if _sig is None:
                                    _d = _rn(x_t - x0, x0)
                                else:
                                    _d = _rn((x_t - x0) - _sig * (pred - pred0), x0 - _sig * pred0)
                            last_metrics["move_term"] = float(_d.detach())
                            _lo, _hi = float(MOVE_BAND.get("lo", 0.0)), float(MOVE_BAND.get("hi", float("inf")))
                            _pen = torch.relu(_lo - _d) ** 2 + torch.relu(_d - _hi) ** 2
                            loss = loss + float(MOVE_BAND.get("weight", 1.0)) * _pen
                        # Persistence regularizers (PERSIST_REG): loss = BCE + w·term, see the
                        # module docstring of PERSIST_REG for the three terms.
                        if PERSIST_REG and (PERSIST_REG.get("steps") is None or step_idx in PERSIST_REG["steps"]):
                            _kind = PERSIST_REG.get("kind", "latent")
                            if _kind == "latent":
                                _pt = torch.sqrt(((x_t - x0) ** 2).sum() + 1e-6)
                            elif _kind == "flow":
                                _pt = torch.sqrt(((pred - pred0) ** 2).sum() + 1e-6) / (pred0.norm() + 1e-12)
                            elif _kind == "sawtooth":
                                # s(A(z_i)): mean target-word prob from the per-word BCE (target 1 ⇒ p = e^{−bce})
                                _b_i = steering_model.per_object_bce(captures, step_idx, target_words_indices=target_words_indices)
                                _s_i = torch.stack([torch.exp(-b) for b in _b_i.values()]).mean()
                                _v = pred.detach() if PERSIST_REG.get("detach_flow", False) else pred
                                if PERSIST_REG.get("flow", "cond") == "cfg":
                                    # CFG velocity as the real advance uses: nu + g·(nc − nu), the
                                    # uncond part detached (no-grad forward), gradient via nc only.
                                    _nu = adapter.uncond_velocity(x_t.detach(), step_idx)
                                    _v = _nu + adapter.guide_scale * (_v - _nu)
                                _z_next = adapter.flow_advance(x_t, _v, step_idx)
                                _caps_next, _ = adapter.grad_forward(_z_next, step_idx + 1)
                                # scored with THIS step's probe (the next step may have none when
                                # nearest_probe is off; same-probe scoring is what the sawtooth means)
                                _b_n = steering_model.per_object_bce(_caps_next, step_idx, target_words_indices=target_words_indices)
                                _s_n = torch.stack([torch.exp(-b) for b in _b_n.values()]).mean()
                                last_metrics["persist_s_next"] = float(_s_n.detach())
                                _pt = _s_i - _s_n
                            else:
                                raise ValueError(f"persist_reg.kind={_kind!r}")
                            last_metrics["persist_term"] = float(_pt.detach())
                            loss = loss + float(PERSIST_REG.get("weight", 1.0)) * _pt
                        # KL between the steered and pre-steer latents as softmax 1D
                        # distributions. Always logged for calibration; only added
                        # when kl_reg > 0.
                        kl_term = _latent_softmax_kl(x_t, x0)
                        last_metrics["kl_term"] = float(kl_term.detach())
                        if kl_reg:
                            loss = loss + kl_reg * kl_term
                        # Attention-space entropy regularizer (TempoControl eq. 8) —
                        # concentrate the target-word attention spatially.
                        if use_entropy:
                            entropy_term = steering_model.spatial_entropy(
                                captures, step_idx,
                                target_words_indices=target_words_indices,
                            )
                            last_metrics["entropy_term"] = float(entropy_term.detach())
                            loss = loss + entropy_reg * entropy_term
                        # Attention-space trust region on the NON-target text tokens.
                        if use_attn_anchor:
                            if attn_key_idx is None:
                                attn_key_idx = torch.as_tensor(
                                    _other_text_key_indices(captures[0].attn_map.shape[-1], target_key_indices,
                                                            attn_anchor_tokens, text_token_count),
                                    dtype=torch.long, device=captures[0].attn_map.device)
                            _cur = [c.attn_map[0].index_select(-1, attn_key_idx) for c in captures]
                            if attn_ref is None:
                                attn_ref = [a.detach() for a in _cur]
                            attn_anchor_term = torch.stack([
                                ((a - r) ** 2).mean() for a, r in zip(_cur, attn_ref, strict=True)
                            ]).mean()
                            last_metrics["attn_anchor_term"] = float(attn_anchor_term.detach())
                            loss = loss + attn_anchor_reg * attn_anchor_term
                        # Anchor to the TRUE source video (video edit); None → skipped entirely.
                        _x_orig = _source_ref_fn(step_idx) if _source_ref_fn is not None else None
                        if _x_orig is not None:
                            source_term = ((x_t - _x_orig) ** 2).mean()
                            last_metrics["source_term"] = float(source_term.detach())
                            if source_reg:
                                loss = loss + source_reg * source_term
                        last_metrics["loss"] = float(loss.detach())
                    # Only x_t receives a gradient (model params are frozen); this
                    # also avoids populating .grad on the transformer.
                    if not first_metrics:
                        first_metrics = last_metrics
                    # HELD-OUT JUDGE PROBES (steering.judge_probes). The probe being optimised
                    # saturates within a step -- by the last update both present and absent
                    # objects read >0.99, final gap 0.000-0.005 -- so its own score cannot
                    # distinguish a real gain from a shortcut into its blind spot. A probe
                    # trained at a DIFFERENT denoising step was never optimised against here,
                    # so its reading of the SAME latent is an independent check. Cost is one
                    # probe forward per judge (negligible beside the DiT backward).
                    _judge = {}
                    if JUDGE_PROBES and hasattr(steering_model, "per_object_prob_with"):
                        _used = getattr(steering_model, "_probe_step", lambda s: None)(step_idx)
                        _cands = (steering_model.probe_steps if JUDGE_PROBES == "all"
                                  else list(JUDGE_PROBES))
                        for _js in _cands:
                            if _js == _used:
                                continue
                            _sc = steering_model.per_object_prob_with(
                                captures, _js, target_words_indices=target_words_indices)
                            if _sc:
                                _judge[str(_js)] = _sc
                    # HINGE JUDGE (steering.hinge_judge). Read-only Attend-and-Excite hinge on
                    # the SAME captures the optimized (probe-BCE) objective just saw -- lets a
                    # hinge-vs-BCE detector-quality comparison run on one trajectory instead of
                    # two separately-optimized ones. Never enters the loss.
                    _hinge_val = None
                    _hinge_pw = None
                    if HINGE_JUDGE_MODEL is not None:
                        with torch.no_grad():
                            _hloss, _hmetrics = HINGE_JUDGE_MODEL.loss(
                                captures, step_idx, target_words_indices=target_words_indices)
                            _hinge_val = float(_hloss.detach())
                            _hinge_pw = dict(_hmetrics.get("per_word_prob", {}))
                    iter_snaps.append({
                        "judge_prob": _judge,
                        "hinge": _hinge_val,
                        "hinge_per_word_prob": _hinge_pw,
                        "per_word_prob": dict(last_metrics.get("per_word_prob", {})),
                        # per_frame mode only: the mean prob split by the control bit, so a
                        # trace can show absent-frames↓ and present-frames↑ separately rather
                        # than their (misleading) mean. Absent from other steering models.
                        "per_word_prob_absent": dict(last_metrics.get("per_word_prob_absent", {})),
                        "per_word_prob_present": dict(last_metrics.get("per_word_prob_present", {})),
                        # per_frame mode only: per-frame probs at THIS iteration (loss-quality analysis).
                        "per_word_frame_prob": last_metrics.get("per_word_frame_prob"),
                        # ...and the same split of the objective itself: under
                        # frame_reduce=balanced the loss IS the (weighted) mean of these two.
                        "per_word_bce_absent": dict(last_metrics.get("per_word_bce_absent", {})),
                        "per_word_bce_present": dict(last_metrics.get("per_word_bce_present", {})),
                        # objective.class_select != "both": which half was optimized here.
                        "class_choice": list(last_metrics.get("class_choice", [])),
                        "prob": last_metrics.get("prob"),
                        "bce": last_metrics.get("bce"),
                        "loss": last_metrics.get("loss"),
                        # Regularizer raw terms (weight-free), persisted for calibration: pick
                        # the coefficient so coef*term is the desired fraction of the bce loss.
                        # pred_term = x̂0 drift (true x̂0 on WAN post wan-pred-x0 fix), kl_term =
                        # latent-softmax KL, anchor_term = latent L2 drift.
                        "pred_term": last_metrics.get("pred_term"),
                        "kl_term": last_metrics.get("kl_term"),
                        "anchor_term": last_metrics.get("anchor_term"),
                        # mag/entropy are only COMPUTED when their weight > 0 (mag is cheap;
                        # entropy re-aggregates captures per word, so it stays gated). Persist
                        # them so a tiny-weight calibration pass reads the raw term; .get is
                        # None when not computed, which is harmless.
                        "mag_term": last_metrics.get("mag_term"),
                        "entropy_term": last_metrics.get("entropy_term"),
                    })
                    # A&E early exit (their ``while loss > target_loss``): tested on the loss
                    # just computed, before it is applied, so breaking here leaves exactly
                    # iter_idx updates on the latent. Gate closed → never taken by other models.
                    if is_attend_excite and steering_model.stop_iteration(step_idx, last_metrics, iter_idx):
                        # Breaking skips the autograd.grad below, so the just-built forward
                        # graph (a FULL transformer grad-forward, ~49 GB on CogVideoX) would
                        # stay alive through these locals into the next step's forwards —
                        # enough to OOM a 94 GB H100. pred_term is the sneaky one: computed
                        # unconditionally ("always logged") and its grad_fn chains through
                        # pred into the ENTIRE graph. Drop every reference so it frees now.
                        loss = bce_loss = captures = pred = None
                        pred_term = _x0_drift = anchor_term = kl_term = mag_term = entropy_term = None
                        break
                    # Correlation-based early stopping: the update on the crossing
                    # iteration is SKIPPED (break before the grad step), matching
                    # TempoControl — iter 0 crossing == its per-step gate (skip the step
                    # entirely); a later crossing == its within-step early stop.
                    # Adaptive iteration count (steering.optimization.stop_when_min_prob): stop this
                    # step's optimization as soon as the WEAKEST target object passes the threshold —
                    # A&E's "refine until the most neglected subject is satisfied", so a prompt whose
                    # objects are already present spends (almost) no updates while a hard one uses the
                    # full n_iters budget. None = fixed n_iters.
                    if STOP_WHEN_MIN_PROB is not None:
                        _pw = (last_metrics.get("per_word_prob") or {}).values()
                        if _pw and min(_pw) >= STOP_WHEN_MIN_PROB:
                            _min_budget = getattr(GRAD_AGGREGATE, "min_budget", 0)
                            if iter_idx >= _min_budget:
                                # Release the un-backwarded forward graph before breaking, exactly
                                # as the A&E and correlation early-exits do — on CogVideoX this
                                # graph is ~49 GB and would otherwise stay alive into the next
                                # step's forwards and OOM a 94 GB H100. (WAN's graph is small, so
                                # this break historically "worked" there without the release.)
                                loss = bce_loss = captures = pred = None
                                pred_term = _x0_drift = anchor_term = kl_term = mag_term = entropy_term = None
                                break
                    if es_threshold is not None and es_metric is not None:
                        es_val = last_metrics.get(es_metric)
                        if es_val is not None and es_val >= es_threshold:
                            # Same un-backwarded-graph release as the A&E early exit above.
                            loss = bce_loss = captures = pred = None
                            pred_term = _x0_drift = anchor_term = kl_term = mag_term = entropy_term = None
                            break
                    # Multi-object gradient-interaction diagnostic (off by default): compute
                    # each object's gradient on x_t separately while the graph is still alive.
                    if GRAD_DIAG_HOOK is not None and hasattr(steering_model, "per_object_bce"):
                        with torch.enable_grad():
                            _pobj = steering_model.per_object_bce(
                                captures, step_idx, target_words_indices=target_words_indices)
                            _pg = {w: torch.autograd.grad(b, x_t, retain_graph=True)[0].detach()
                                   for w, b in _pobj.items()}
                        # x_t is passed live (not detached) so a hook can autograd.grad its own
                        # per-class losses against it while the graph is still alive; the branch
                        # is off in every real run, so this never touches steering behavior.
                        GRAD_DIAG_HOOK(step_idx, iter_idx, x_t, _pg,
                                       captures, steering_model, target_words_indices, pred.detach())
                    # Per-object grads of the applied update, for FIELD_HOOK: set only on the
                    # GRAD_AGGREGATE path (the only one that HAS per-object components), and
                    # cleared here so a hook never reads the previous iteration's tensors.
                    _pg = _pbce = None
                    if (getattr(steering_model, "frame_grad_mask", False)
                            and hasattr(steering_model, "per_frame_class_losses")):
                        # Control-masked update: derive the absent-frame loss and the
                        # present-frame loss SEPARATELY, and let each one write only to the
                        # latent frames of its own class. The absent half never touches a
                        # requested frame and the present half never touches a forbidden one,
                        # so the two can no longer cancel on shared latent elements — the
                        # failure mode behind the equalized fixed point (DEBUG.md §2, §6).
                        # Unmasked, the two grads would sum to exactly the frame_reduce=
                        # "balanced" gradient, so this is a change of *routing*, not of loss.
                        with torch.enable_grad():
                            parts = steering_model.per_frame_class_losses(
                                captures, step_idx, target_words_indices=target_words_indices)
                        grad = torch.zeros_like(x_t)
                        # lo/hi/ramp default to the hard 0/1 mask, so an existing config is
                        # bit-identical; softening only widens who may write to a frame.
                        _rw = dict(lo=float(getattr(steering_model, "frame_mask_lo", 0.0)),
                                   hi=float(getattr(steering_model, "frame_mask_hi", 1.0)),
                                   ramp=int(getattr(steering_model, "frame_mask_ramp", 1)))
                        for _lp, _la, _c in parts.values():
                            pm = _latent_frame_mask(x_t, _route_weights(_c, **_rw))
                            if _lp is not None:
                                grad += torch.autograd.grad(
                                    _lp, x_t, retain_graph=True)[0] * pm
                            if _la is not None:
                                grad += torch.autograd.grad(
                                    _la, x_t, retain_graph=True)[0] * (1.0 - pm)
                        grad = grad * (steering_model.weight / max(len(parts), 1))
                        # Regularizers are latent-global and stay unmasked: they are a trust
                        # region on the whole latent, not part of the timeline objective.
                        if loss is not bce_loss:
                            grad = grad + torch.autograd.grad(loss - bce_loss, x_t)[0]
                    elif (GRAD_AGGREGATE is not None and CAPTURE_SPACE_MUTUAL
                            and hasattr(steering_model, "per_object_bce")):
                        # O(1)-in-K: aggregate per-object grads at the attention captures, one
                        # transformer backward to x_t. Fits a memory-bound GPU (see helper).
                        # Ported from the wan-mutual-batchgrad branch; the helper folds in this
                        # branch's grad_balance/unitnorm when that aggregation rule is selected.
                        with torch.enable_grad():
                            # Regularizer gradient FIRST (retain_graph=True): the capture-space
                            # backward frees the graph, and the trust region must still apply
                            # here exactly as it does on the plain GRAD_AGGREGATE path below.
                            reg_grad = (torch.autograd.grad(loss - bce_loss, x_t,
                                                            retain_graph=True)[0]
                                        if loss is not bce_loss else None)
                            grad = _capture_space_mutual_grad(
                                steering_model, captures, x_t, step_idx, target_words_indices)
                        if reg_grad is not None:
                            grad = grad + reg_grad
                    elif GRAD_AGGREGATE is not None and hasattr(steering_model, "per_object_bce"):
                        # Custom token-aggregation rule: build the update from per-object grads
                        # (each the grad of that object's BCE) instead of the mean-BCE gradient.
                        globals()["AGG_ITER"] = int(iter_idx)
                        globals()["AGG_STEP"] = int(step_idx)
                        # A&E's own bar for this step, for gate="ae_sched". Resolved here rather
                        # than in the rule because only the loop holds the steering model.
                        globals()["AGG_AE_THR"] = _ae_sched_threshold(steering_model, step_idx)
                        # Everything a rule needs to reconstruct the ACTUAL move this iteration
                        # would make: how many iterations the step runs (FLOW_ITERS' "-1"), the
                        # decayed lr about to be applied, and the solver's dσ (applied_space).
                        globals()["AGG_N_ITERS"] = int(step_iters)
                        globals()["AGG_LR"] = float(step_lr)
                        globals()["AGG_DSIGMA"] = _adapter_sigma(adapter, step_idx)[0]
                        with torch.enable_grad():
                            _pobj = _probe_mix_bce(
                                steering_model, captures, step_idx, iter_idx, target_words_indices)
                            _pbce = {w: float(b.detach()) for w, b in _pobj.items()}
                            # Object SELECTION scores: by default the own-probe BCE, but SELECT_PROBE
                            # lets a NEIGHBOUR probe rank the objects (read-only) while the gradients
                            # (_pg) stay own-probe — "neighbour for choosing, not for optimising".
                            _agg_scores = _pbce
                            _sel_step = _resolve_select_probe(steering_model, step_idx)
                            if _sel_step is not None:
                                with torch.no_grad():
                                    _selbce = steering_model.per_object_bce_with(
                                        captures, _sel_step, target_words_indices=target_words_indices)
                                _agg_scores = {w: float(b.detach()) for w, b in _selbce.items()}
                            # Gate BEFORE autograd: only backprop the objects the rule will actually
                            # use (worst / live-gated). Idempotent under pre-selection → bit-identical,
                            # just fewer backward passes. FIELD_HOOK diagnostics want every object's
                            # grad, so keep all when it is on.
                            _selfn = getattr(GRAD_AGGREGATE, "grad_select", None)
                            _words = (_selfn(list(_pobj), _agg_scores)
                                      if _selfn is not None and FIELD_HOOK is None else list(_pobj))
                            # Free the graph on the LAST backward (unless the reg branch below
                            # still needs it): retain_graph=True on every call keeps the
                            # checkpointed capture-softmax recomputes alive into the next
                            # iteration's forward — +~12 GB, OOMs the 94 GB H100 on CogVideoX
                            # (from feat+cogvideox-backend). retain_graph does not change values.
                            _need_reg = loss is not bce_loss
                            _pg = {}
                            if _words:
                                for _i, _w in enumerate(_words):
                                    _retain = _need_reg or _i < len(_words) - 1
                                    _pg[_w] = torch.autograd.grad(
                                        _pobj[_w], x_t, retain_graph=_retain)[0].detach()
                            else:
                                # All objects satisfied → zero steering update. Still free the
                                # forward graph: one discarded backward does it.
                                torch.autograd.grad(next(iter(_pobj.values())), x_t,
                                                    retain_graph=_need_reg)
                        globals()["_MASK_LAST"] = None
                        grad = (GRAD_AGGREGATE(_pg, flow0 if FLOW_REF == "k0" else vel, _agg_scores)
                                if _pg else torch.zeros_like(x_t))
                        if MASK_STATS and iter_snaps:
                            # steering.log_mask_stats: what the fold masked this iteration (None = no fold ran:
                            # <2 gate survivors / skip step) + how far the update is from the -both update
                            # (plain mean of the RAW survivor grads) — diagnostics only, the update is unchanged.
                            _ms = dict(_MASK_LAST) if _MASK_LAST is not None else {}
                            if _pg:
                                _rm = torch.stack([g.float() for g in _pg.values()]).mean(0)
                                _gf = grad.float()
                                _ms["cos_vs_rawmean"] = float((_gf * _rm).sum() / (_gf.norm() * _rm.norm() + 1e-30))
                                _ms["n_grads"] = len(_pg)
                            iter_snaps[-1]["mask_stats"] = _ms
                        # Add the regularizer gradient on top of the aggregated BCE update, so a
                        # configured trust region (anchor/pred/magnitude/kl/entropy) still applies
                        # under GRAD_AGGREGATE. loss - bce_loss = the pure reg terms (BCE cancels);
                        # its grad is the reg gradient. No-op when all regularizers are 0 (loss is bce_loss).
                        if loss is not bce_loss:
                            grad = grad + torch.autograd.grad(loss - bce_loss, x_t)[0]
                    else:
                        grad = torch.autograd.grad(loss, x_t)[0]
                    if FIELD_HOOK is not None:
                        _ds, _sg = _adapter_sigma(adapter, step_idx)
                        FIELD_HOOK(int(step_idx), int(iter_idx),
                                   dict(x_t=x_t.detach(), grad=grad.detach(), lr=float(step_lr),
                                        flow=vel, pred=pred.detach(), dsigma=_ds, sigma=_sg,
                                        pgrads=_pg, pbce=_pbce))
                    if opt is None:
                        # "simple" update rule: plain gradient descent, x_t ← x_t − lr·grad
                        # (no Adam moments / per-element adaptive scaling). In-place under
                        # no_grad so x_t stays the same leaf for the next iteration.
                        with torch.no_grad():
                            x_t.add_(grad, alpha=-step_lr)
                    else:
                        x_t.grad = grad
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    n_updates += 1
                    # x̂0 capture: after this many gradient updates within the step (uses the
                    # UPDATED x_t — the mid-optimization latent).
                    if X0_HOOK is not None and X0_POINTS is not None and (step_idx, n_updates) in X0_POINTS:
                        X0_HOOK((step_idx, n_updates), adapter.estimate_x0(x_t.detach(), step_idx))
                if FIELD_HOOK is not None and FIELD_HOOK_POST and n_updates:
                    # One extra reading AFTER the step's last update, recorded as iteration
                    # n_updates with a zero gradient. Without it the final update has no
                    # "what did that do" reading at all: the next FIELD_HOOK call happens on the
                    # far side of the solver advance, which is a different question. The velocity
                    # comes from the same conditional forward the steered iterations use, so the
                    # x̂0 = x_t - sigma*v this yields is comparable with theirs (estimate_x0 would
                    # not be: it is the CFG velocity).
                    with torch.no_grad():
                        _, _post_v = adapter.grad_forward(x_t.detach(), step_idx)
                    _ds = _sg = None
                    if hasattr(adapter, "sigma"):
                        try:
                            _s = adapter.scheduler.sigmas
                            _nx = min(step_idx + 1, len(_s) - 1)
                            _ds = float((_s[_nx] - _s[step_idx]).item())
                            _sg = float(_s[step_idx].item())
                        except Exception:
                            _ds = _sg = None
                    FIELD_HOOK(int(step_idx), int(n_updates),
                               dict(x_t=x_t.detach(), grad=torch.zeros_like(x_t), lr=float(step_lr),
                                    flow=_post_v.detach(), dsigma=_ds, sigma=_sg,
                                    pgrads=None, pbce=None))
                if step_callback is not None:
                    # Model-specific extras (baselines report their own raw terms,
                    # e.g. energy / entropy / pearson) are forwarded generically as
                    # <key>_before / <key>_after so nothing a loss reports is lost.
                    known = {
                        "step", "loss", "prob", "per_word_prob", "balance", "bce",
                        "mag_term", "anchor_term", "pred_term", "kl_term", "entropy_term",
                    }
                    extras = {}
                    for key in last_metrics:
                        if key in known:
                            continue
                        extras[f"{key}_before"] = first_metrics.get(key)
                        extras[f"{key}_after"] = last_metrics.get(key)
                    step_callback(step_idx, {
                        **extras,
                        "step": step_idx,
                        "steered": True,
                        # Full per-optimization-iteration probe trajectory for this step
                        # (iter 0 = pre-steer/unsteer reading; rest = in-optimization).
                        "iters": iter_snaps,
                        # Realized schedule at this step: updates actually applied (variable
                        # under A&E's iters_for_step / stop_iteration) and the step size used.
                        # Constant (n_iters / lr) for models without the schedule.
                        "n_updates": n_updates,
                        "step_lr": step_lr,
                        "prob_before": first_metrics.get("prob"),
                        "prob_after": last_metrics.get("prob"),
                        "loss_before": first_metrics.get("loss"),
                        "loss_after": last_metrics.get("loss"),
                        # Per-object scores + spread, pre/post this step's optimization
                        # (for the balance-term before/after analysis).
                        "per_word_prob_before": first_metrics.get("per_word_prob"),
                        "per_word_prob_after": last_metrics.get("per_word_prob"),
                        "balance_before": first_metrics.get("balance"),
                        "balance_after": last_metrics.get("balance"),
                        # Raw (pre-weight) loss components for term-magnitude analysis:
                        # bce (×weight), balance (×balance_weight), mag_term (×magnitude_reg).
                        "bce_before": first_metrics.get("bce"),
                        "bce_after": last_metrics.get("bce"),
                        "mag_term_before": first_metrics.get("mag_term"),
                        "mag_term_after": last_metrics.get("mag_term"),
                        # Trust-region / prediction-space regularizer raw terms
                        # (weight-free), pre/post this step's optimization.
                        "anchor_term_before": first_metrics.get("anchor_term"),
                        "anchor_term_after": last_metrics.get("anchor_term"),
                        "pred_term_before": first_metrics.get("pred_term"),
                        "pred_term_after": last_metrics.get("pred_term"),
                        "kl_term_before": first_metrics.get("kl_term"),
                        "kl_term_after": last_metrics.get("kl_term"),
                        # Attention-space entropy regularizer raw term (weight-free).
                        "entropy_term_before": first_metrics.get("entropy_term"),
                        "entropy_term_after": last_metrics.get("entropy_term"),
                    })
                # "after the LAST update" is never seen by the loop above (the next forward is
                # the capture-free denoise step) — one extra no-grad capturing forward.
                if _cap_step == step_idx and n_updates in _cap_k:
                    with torch.no_grad():
                        _caps, _ = adapter.grad_forward(x_t.detach(), step_idx)
                    ATTN_CAPTURE_SINK(n_updates, _caps, target_words_indices, steering_model)
                    del _caps
                if FLOW_ON_TOTAL and GRAD_AGGREGATE is not None and n_updates:
                    # VARIATION 3 — arbitrate the flow against the optimisation AS A WHOLE.
                    # The per-iteration path is discarded: only its endpoint matters. Δ is the
                    # net displacement the optimisation produced, so the equivalent DESCENT
                    # direction is −Δ (the applied move is always −g), and the result is applied
                    # to the PRE-optimisation latent x0 rather than composed onto x_t.
                    # ANCHOR (v3b, FLOW_ON_TOTAL_K0). Sample the flow at the PRE-optimisation
                    # latent x0 instead of at the post-optimisation x_t. §5.1 of the 2026-09-22
                    # report showed the site otherwise straddles three anchor points — Δ measured
                    # from x0, v read at x^N, result applied at x0 — and at lr=500 x^N is the
                    # furthest off-manifold the latent ever gets, i.e. precisely where the
                    # denoiser is least trustworthy. With K0 all three collapse onto x0.
                    _ref3 = x0 if FLOW_ON_TOTAL_K0 else x_t.detach()
                    with torch.no_grad():
                        _caps3, _pred3 = adapter.grad_forward(_ref3, step_idx)
                        _v3 = adapter.velocity(_ref3, _pred3, step_idx)
                        del _caps3, _pred3
                    _delta = x_t.detach() - x0
                    globals()["AGG_ITER"] = 0
                    globals()["AGG_N_ITERS"] = 1
                    # UNITS (v3b, FLOW_ON_TOTAL_PER_LR). Default: d IS the displacement, so the
                    # effective lr is 1. Under PER_LR the objective leg reaches CAGrad divided by
                    # the learning rate — as the accumulated GRADIENT Σg rather than as the
                    # displacement −lr·Σg — which is what the per-iteration site receives, making
                    # the two placements unit-consistent. AGG_LR must follow, because rules that
                    # reconstruct the applied move internally (the applied_space family) read it
                    # to turn a gradient back into a displacement.
                    globals()["AGG_LR"] = float(step_lr) if FLOW_ON_TOTAL_PER_LR else 1.0
                    globals()["AGG_DSIGMA"] = _adapter_sigma(adapter, step_idx)[0]
                    # One pseudo-object whose "gradient" is the whole optimisation. scores must be
                    # non-empty or `objmax` falls back to sel[:1] — same thing at k=1, but explicit.
                    #
                    # FLOW_ITERS gates the OPTIMISATION's iterations; this is a DIFFERENT site and
                    # must not be gated by it — exactly the reason variation 2 clears it too. Every
                    # v3 config pairs `flow_on_total: true` with `flow_iters: []` (the flow is meant
                    # to act ONLY here), and `[]` made `_flow_iter_on()` false right where the flow
                    # is the entire point: an `aemax` rule then took its gated-OFF branch and
                    # returned the argmax gradient unchanged, i.e. `_d3 = -Δ` and `x_t = x0 + Δ` —
                    # a bit-exact NO-OP. Every flow_on_total run before 2026-09-22 reproduced its
                    # own un-arbitrated optimisation and must be re-read as such.
                    _keep3, globals()["FLOW_ITERS"] = FLOW_ITERS, None
                    _obj3 = (-_delta / step_lr) if FLOW_ON_TOTAL_PER_LR else -_delta
                    try:
                        _d3 = GRAD_AGGREGATE({"__opt__": _obj3}, _v3, {"__opt__": 1.0})
                    finally:
                        globals()["FLOW_ITERS"] = _keep3
                    # DIRECTION ONLY, enforced here rather than assumed. The variation's contract
                    # is that the step length stays the optimisation's own, and for the `aescale`
                    # family scale="argmax_score" already delivers ||d3|| = ||Δ|| (the lone
                    # pseudo-object IS the argmax). Every other family does not: a unit-normed
                    # rule returns ||d3|| = 1 against a ||Δ|| of several units, silently shrinking
                    # the whole step by that factor and confounding the placement with a learning
                    # rate. Only applied_space rules opt out — there the magnitude is the point.
                    #
                    # PER_LR DELIBERATELY SKIPS THIS RESCALE, and that is the whole reason the
                    # flag changes anything. `flow_scale="match"` puts the flow leg on the
                    # objective's scale, so dividing the objective by lr divides BOTH legs by lr;
                    # CAGrad is positively homogeneous, so _d3 comes back scaled by 1/lr with an
                    # IDENTICAL direction. Re-imposing ||_d3|| = ||Δ|| would then reproduce the
                    # un-divided arm bit-for-bit — the same class of silent no-op as bugs #1/#2.
                    # Under PER_LR the magnitude is instead whatever the rule returns, restored to
                    # displacement units by the lr multiply below.
                    if FLOW_ON_TOTAL_PER_LR:
                        _d3 = _d3 * step_lr
                    elif not getattr(GRAD_AGGREGATE, "_applied_space", False):
                        _d3 = _d3 * (_delta.norm() / (_d3.norm() + 1e-12))
                    # UNITS DIAGNOSTIC (env-gated, off by default, one line per steered step).
                    # The two tasks joined above are NOT commensurate: `flow_scale="match"` puts
                    # the velocity on the OBJECTIVE's scale, so the flow leg's physical size is
                    # ||Δ||, set by the steering LR — while the sampler's own advance for this
                    # step is |dσ|·||v||. Nothing in the rule ties those together, so at c=0 the
                    # move carries an unscheduled denoise kick of ~||Δ||/2 whose size is a free
                    # parameter. `adv_ratio` >> 1 means the v3 verdict measures over-denoising,
                    # not the placement, and the units-consistent re-test is the applied_space
                    # twin (`cagrad_*_aemax_flowapplied_dtn_c*`).
                    if FLOW_ON_TOTAL_DIAG:
                        _ds = abs(float(_adapter_sigma(adapter, step_idx)[0]))
                        _dn, _vn = float(_delta.norm()), float(_v3.norm())
                        _cos = float(torch.nn.functional.cosine_similarity(
                            (-_delta).flatten(), _v3.flatten(), dim=0))
                        print(f"FLOWTOTALDIAG step={step_idx} n_upd={n_updates} "
                              f"delta={_dn:.4g} adv={_ds * _vn:.4g} "
                              f"adv_ratio={_dn / (_ds * _vn + 1e-12):.4g} "
                              f"cos_negdelta_v={_cos:+.4f} "
                              # which v3b variant produced this line — without it a log cannot be
                              # told apart from a plain-v3 log, and the two are being compared.
                              f"k0={int(FLOW_ON_TOTAL_K0)} per_lr={int(FLOW_ON_TOTAL_PER_LR)} "
                              f"applied={float(_d3.norm()):.4g}", flush=True)
                    x_t = (x0 - _d3).detach().requires_grad_(False)
                    del _delta, _v3, _d3
                adapter.commit_latent(x_t.detach())
                adapter.end_optimization()  # un-checkpointed path for the stepping forward
            elif LOG_ALL_STEPS and step_callback is not None \
                    and _nearest_probe_step(steering_model, step_idx) is not None:
                # All-steps diagnostic: score the probe on this NON-steered step's incoming
                # latent (no optimization → prob_before == prob_after). One extra no-grad
                # capturing forward; makes the probe trajectory across every step visible.
                with torch.no_grad():
                    if LATENT_DUMP_SINK is not None:
                        LATENT_DUMP_SINK(step_idx, adapter.get_latent(),
                                         getattr(adapter, "timesteps", [None] * (step_idx + 1))[step_idx])
                    _caps, _ = adapter.grad_forward(adapter.get_latent(), step_idx)
                    _, m = steering_model.loss(
                        _caps, step_idx, target_words_indices=target_words_indices)
                    _hinge_val = None
                    _hinge_pw = None
                    if HINGE_JUDGE_MODEL is not None:
                        _hloss, _hmetrics = HINGE_JUDGE_MODEL.loss(
                            _caps, step_idx, target_words_indices=target_words_indices)
                        _hinge_val = float(_hloss.detach())
                        _hinge_pw = dict(_hmetrics.get("per_word_prob", {}))
                _pw = m.get("per_word_prob")
                step_callback(step_idx, {
                    "step": step_idx, "steered": False, "n_updates": 0, "step_lr": 0.0,
                    "iters": [{"per_word_prob": dict(_pw or {}),
                               "prob": m.get("prob"), "bce": m.get("bce"),
                               "loss": m.get("loss"), "hinge": _hinge_val,
                               "hinge_per_word_prob": _hinge_pw}],
                    "prob_before": m.get("prob"), "prob_after": m.get("prob"),
                    "hinge_before": _hinge_val, "hinge_after": _hinge_val,
                    "hinge_per_word_prob_before": _hinge_pw, "hinge_per_word_prob_after": _hinge_pw,
                    "per_word_prob_before": _pw, "per_word_prob_after": _pw,
                    "bce_before": m.get("bce"), "bce_after": m.get("bce"),
                    "loss_before": m.get("loss"), "loss_after": m.get("loss"),
                    "per_word_frame_prob_before": m.get("per_word_frame_prob"),
                    "frame_target_before": m.get("frame_target"),
                })
                del _caps

            # Capture-only run: everything wanted from this trajectory is on disk — skip the
            # flow advance and the remaining steps (the caller skips decoding too).
            if _cap and _cap_step == step_idx and _cap.get("abort_after"):
                raise SteeringAborted(step_idx)

            # Aggressive flow (variation 2): at steered steps replace the model flow advance with one
            # more aggregated-gradient step. The scheduler state is still advanced (history + step
            # index stay consistent for later steps) but its denoised latent is discarded — the actual
            # transition is the grad step on the post-optimization latent.
            if (AGGRESSIVE_FLOW and do_steer and GRAD_AGGREGATE is not None
                    and hasattr(steering_model, "per_object_bce")):
                adapter.begin_optimization()
                x_g = adapter.get_latent().detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    captures_g, pred_g = adapter.grad_forward(x_g, step_idx)
                    _pobj_g = steering_model.per_object_bce(
                        captures_g, step_idx, target_words_indices=target_words_indices)
                    _pg_g = {w: torch.autograd.grad(b, x_g, retain_graph=True)[0].detach()
                             for w, b in _pobj_g.items()}
                Jg = GRAD_AGGREGATE(_pg_g, adapter.velocity(x_g.detach(), pred_g.detach(), step_idx),
                                    {w: float(b.detach()) for w, b in _pobj_g.items()})
                with torch.no_grad():
                    x_g = (x_g - step_lr * Jg).detach()
                # Free the extra grad-forward graph/activations before the (heavy) denoise forwards,
                # else the aggressive path peaks ~1.5× and OOMs a 141 GB H200.
                del captures_g, pred_g, _pobj_g, _pg_g, Jg
                torch.cuda.empty_cache()
                adapter.end_optimization()
                adapter.denoise_step(step_idx)     # advance scheduler state (history + step index)
                adapter.commit_latent(x_g)          # ...but keep the grad-stepped latent as the transition
            else:
                # VARIATION 2 — arbitrate the flow at the SOLVER step, which until now was
                # untouched vanilla (the flow task only ever biased the optimisation, so "use it
                # in the denoising step" was genuinely not being done anywhere).
                #
                # The seam is the adapter's `denoise_adjust`: it receives the x̂0 the stepper is
                # about to consume, and returns a replacement. So we convert to a velocity, run
                # the SAME rule that the optimisation uses (A&E objective vs flow), and convert
                # back. ||v|| is preserved and only the DIRECTION is arbitrated — this is a
                # rotation of the advance, not a change of step size.
                #
                # The arbitrated direction is computed HERE, before `denoise_step`, and the hook
                # itself is pure tensor math. Doing the grad forward inside the hook instead OOMs
                # a 94 GB card: by then `denoise_step` is already holding the full CFG denoiser
                # forward (video_result + audio_result + x̂0) alive, and a 22B backward on top of
                # that has nowhere to go. Computing it first lets the graph be freed before the
                # heavy forward starts.
                _v2_on = (FLOW_AT_DENOISE and do_steer and GRAD_AGGREGATE is not None
                          and hasattr(steering_model, "per_object_bce")
                          and hasattr(adapter, "velocity"))
                _d2 = _sig2 = None
                if _v2_on:
                    _sig2 = _adapter_sigma(adapter, step_idx)[1]
                if _v2_on and _sig2:          # no schedule ⇒ cannot form a velocity; stay vanilla
                    adapter.begin_optimization()
                    try:
                        xg = adapter.get_latent().detach().clone().requires_grad_(True)
                        with torch.enable_grad():
                            caps, pr = adapter.grad_forward(xg, step_idx)
                            pobj = steering_model.per_object_bce(
                                caps, step_idx, target_words_indices=target_words_indices)
                            pg = {w: torch.autograd.grad(b, xg, retain_graph=True)[0].detach()
                                  for w, b in pobj.items()}
                        vel2 = adapter.velocity(xg.detach(), pr.detach(), step_idx)
                        globals()["AGG_ITER"] = 0
                        globals()["AGG_N_ITERS"] = 1
                        globals()["AGG_LR"] = 1.0
                        globals()["AGG_DSIGMA"] = _adapter_sigma(adapter, step_idx)[0]
                        # FLOW_ITERS gates the OPTIMISATION's iterations; the solver step is a
                        # different site and must not be gated by it, or `flow_iters: []` —
                        # which is exactly how this variation is meant to be configured —
                        # would silently disable the thing it is turning on.
                        _keep, globals()["FLOW_ITERS"] = FLOW_ITERS, None
                        try:
                            _d2 = GRAD_AGGREGATE(pg, vel2,
                                                 {w: float(b.detach()) for w, b in pobj.items()}).detach()
                        finally:
                            globals()["FLOW_ITERS"] = _keep
                        del caps, pr, pobj, pg, vel2, xg
                    finally:
                        adapter.end_optimization()
                    torch.cuda.empty_cache()

                    def _denoise_adjust(x_cur, x0_pred, k, _d=_d2, _s=float(_sig2)):
                        # Re-point the advance: same length as the model's own velocity, new
                        # direction. x̂0 = x_t − σ·v is the exact inverse of adapter.velocity, so
                        # rotating v and converting back leaves the stepper untouched. ||v|| is
                        # read off the CFG x̂0 the sampler actually got, not off the cond-only
                        # grad forward, so the step size is the sampler's own.
                        with torch.no_grad():
                            v_act = (x_cur - x0_pred.to(x_cur.dtype)) / _s
                            v_new = _d.to(x_cur.dtype) * (v_act.norm() / (_d.norm() + 1e-12))
                            return (x_cur - _s * v_new).to(x0_pred.dtype)

                    adapter.denoise_adjust = _denoise_adjust
                else:
                    _v2_on = False
                try:
                    # Actual denoise + scheduler step on the (possibly optimized) latent.
                    # No capture, no grad — mirrors the extraction loop's stepping.
                    adapter.denoise_step(step_idx)
                finally:
                    # Always clear: the hook must never leak into a NON-steered step, whose
                    # advance is supposed to be untouched.
                    if _v2_on:
                        adapter.denoise_adjust = None
