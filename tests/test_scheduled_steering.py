"""Gate OPEN: run_steering_optimization honors the A&E per-step schedule
(iters_for_step / lr_multiplier / stop_iteration) for a model whose ``mode ==
'attend_excite'``, and reports the realized ``n_updates`` / ``step_lr``. Pairs with
test_steering_core.py::test_gate_closed_ignores_schedule_hooks — together they pin the
explicit mode gate from both sides, so moving Attend-and-Excite from the old
make_ltx_steering_loop onto the shared core reproduces its schedule exactly.

Pure CPU (no model/GPU). ``pytest tests/test_scheduled_steering.py`` or
``python tests/test_scheduled_steering.py``.
"""
from __future__ import annotations

import os
import sys

import torch


from will_it_render.inference.steering_core import SteeringBackendAdapter, run_steering_optimization


class MockAdapter(SteeringBackendAdapter):
    """Deterministic in-memory adapter (same contract as test_steering_core's mock):
    grad_forward returns captures=[x_t] so the mock loss is a closed form of the latent."""

    def __init__(self, latent: torch.Tensor, n_steps: int):
        self.latent = latent.clone()
        self.n_steps = n_steps
        self.denoise_calls: list[int] = []

    def steps(self):
        return range(self.n_steps)

    def get_latent(self):
        return self.latent

    def commit_latent(self, x_t):
        self.latent = x_t

    def grad_forward(self, x_t, step_idx):
        return [x_t], x_t

    def denoise_step(self, step_idx):
        self.denoise_calls.append(step_idx)


class ScheduledModel:
    """mode == 'attend_excite' → the core applies the schedule. Loss = 0.5*sum(x**2), so
    grad == x and a simple-GD update gives x <- x*(1 - step_lr) per APPLIED iteration."""

    mode = "attend_excite"

    def __init__(self, iters_by_step: dict, lr_mult: float, stop_at_iter):
        self._iters_by_step = iters_by_step
        self._lr_mult = lr_mult
        self._stop_at_iter = stop_at_iter

    def should_steer(self, step_idx):
        return True

    def iters_for_step(self, step_idx):
        return self._iters_by_step[step_idx]

    def lr_multiplier(self, step_idx, n_steps=None):
        return self._lr_mult

    def stop_iteration(self, step_idx, metrics, iter_idx):
        return self._stop_at_iter is not None and iter_idx >= self._stop_at_iter

    def loss(self, captures, step_idx, *, target_words_indices):
        x = captures[0]
        loss = 0.5 * (x ** 2).sum()
        return loss, {
            "loss": float(loss.detach()),
            "prob": float(torch.sigmoid(-loss).detach()),
            "per_word_prob": {"a": 0.0},
            "balance": 0.0,
        }


def _x0():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


def test_schedule_iter_budget_and_lr_multiplier():
    """iters_for_step overrides the outer n_iters, and each update uses lr*lr_multiplier."""
    x0 = _x0()
    lr, mult = 0.1, 0.5
    snaps: dict = {}
    model = ScheduledModel(iters_by_step={0: 4}, lr_mult=mult, stop_at_iter=None)
    adapter = MockAdapter(x0, n_steps=1)
    run_steering_optimization(
        adapter, model, target_words_indices={"a": [0]},
        n_iters=1, optimizer_name="simple", lr=lr,        # n_iters=1 only enables steering
        step_callback=lambda s, m: snaps.update({s: dict(m)}),
    )
    step_lr = lr * mult
    expected = x0 * (1 - step_lr) ** 4                     # 4 updates (not 1), at lr*0.5
    assert torch.allclose(adapter.latent, expected, atol=1e-6), (adapter.latent, expected)
    assert snaps[0]["n_updates"] == 4, snaps[0]["n_updates"]
    assert abs(snaps[0]["step_lr"] - step_lr) < 1e-9, snaps[0]["step_lr"]


def test_schedule_early_stop_leaves_iter_idx_updates():
    """stop_iteration True at iter_idx>=2 → tested before the update is applied, so exactly
    2 updates land (iters 0 and 1) out of a 10-iteration budget."""
    x0 = _x0()
    lr = 0.1
    snaps: dict = {}
    model = ScheduledModel(iters_by_step={0: 10}, lr_mult=1.0, stop_at_iter=2)
    adapter = MockAdapter(x0, n_steps=1)
    run_steering_optimization(
        adapter, model, target_words_indices={"a": [0]},
        n_iters=1, optimizer_name="simple", lr=lr,
        step_callback=lambda s, m: snaps.update({s: dict(m)}),
    )
    expected = x0 * (1 - lr) ** 2
    assert torch.allclose(adapter.latent, expected, atol=1e-6), (adapter.latent, expected)
    assert snaps[0]["n_updates"] == 2, snaps[0]["n_updates"]


def test_schedule_multi_step():
    """Two steps, budgets 2 then 3, constant lr_mult=0.5 → latent = x0*(1-lr*0.5)^5, and
    every step still denoises."""
    x0 = _x0()
    lr, mult = 0.1, 0.5
    model = ScheduledModel(iters_by_step={0: 2, 1: 3}, lr_mult=mult, stop_at_iter=None)
    adapter = MockAdapter(x0, n_steps=2)
    run_steering_optimization(
        adapter, model, target_words_indices={"a": [0]},
        n_iters=1, optimizer_name="simple", lr=lr,
    )
    step_lr = lr * mult
    expected = x0 * (1 - step_lr) ** 5
    assert torch.allclose(adapter.latent, expected, atol=1e-6), (adapter.latent, expected)
    assert adapter.denoise_calls == [0, 1]


class _AECaptureAdapter(SteeringBackendAdapter):
    """Feeds the REAL AttendExciteSteeringModel A&E-shaped captures whose attention is a
    differentiable function of x_t (softmax over the text axis), so the whole loss →
    autograd path exercises. Grid: T×h×w = 2×4×4 (Lq=32), Lk=6 text positions."""

    T, hh, ww, Lk = 2, 4, 4, 6
    Lq = T * hh * ww

    def __init__(self, x0, n_steps):
        self.latent = x0.clone()
        self.n_steps = n_steps
        self.denoise_calls: list[int] = []

    def steps(self):
        return range(self.n_steps)

    def get_latent(self):
        return self.latent

    def commit_latent(self, x_t):
        self.latent = x_t

    def grad_forward(self, x_t, step_idx):
        from will_it_render.common.probe_types import BlockCapture, CaptureType
        attn = torch.softmax(x_t, dim=-1)[None, None]      # [1,1,Lq,Lk], depends on x_t
        cap = BlockCapture(attn_map=attn, text_key=x_t[None, :, None, :1],
                           capture_type=CaptureType.VIDEO_CA)
        return [cap], x_t

    def denoise_step(self, step_idx):
        self.denoise_calls.append(step_idx)


def test_real_attend_excite_schedule_on_core():
    """The REAL AttendExciteSteeringModel, driven through run_steering_optimization,
    reproduces its published schedule: 1+max_refinement_steps updates at a threshold step,
    exactly 1 at each non-threshold step, and a √-decayed step size — i.e. moving A&E off
    make_ltx_steering_loop onto the shared core changes nothing."""
    from will_it_render.evaluation.baselines.attend_excite.loss import AttendExciteSteeringModel

    ad = _AECaptureAdapter(0.1 * torch.ones(_AECaptureAdapter.Lq, _AECaptureAdapter.Lk), n_steps=4)
    agg = {"spatial_downsample": 1, "latent_hw": (_AECaptureAdapter.hh, _AECaptureAdapter.ww)}
    # threshold only at step 0 (target_loss = 1-0.99 = 0.01, unreachable here → full refinement
    # budget runs); guide steps 0..2 (max_step=3); max_refinement_steps=3 → step-0 budget = 4.
    model = AttendExciteSteeringModel(
        torch.device("cpu"), agg, sample_steps=50, max_step=3,
        thresholds={0: 0.99}, max_refinement_steps=3, smooth_attentions=True,
    )
    model.set_text_token_indices([1, 2, 3, 4])
    snaps: dict = {}
    run_steering_optimization(
        ad, model, target_words_indices={"a": [2, 3]},
        n_iters=1, optimizer_name="simple", lr=0.01,
        step_callback=lambda s, m: snaps.update({s: {"n_updates": m["n_updates"], "step_lr": m["step_lr"]}}),
    )
    assert snaps[0]["n_updates"] == 4, snaps[0]          # threshold step: 1 + max_refinement_steps
    assert snaps[1]["n_updates"] == 1 and snaps[2]["n_updates"] == 1, snaps  # non-threshold: 1 each
    # √-decay over the trajectory: step size strictly decreasing, step 0 == lr·√1.0.
    assert abs(snaps[0]["step_lr"] - 0.01) < 1e-9, snaps[0]["step_lr"]
    assert snaps[0]["step_lr"] > snaps[1]["step_lr"] > snaps[2]["step_lr"], snaps
    assert ad.denoise_calls == [0, 1, 2, 3]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} scheduled-steering tests passed.")


if __name__ == "__main__":
    _run_all()
