"""Pins the model-agnostic steering algorithm (steering/steering_core.py).

The whole point of extracting ``run_steering_optimization`` into a shared core is
that LTX and WAN (and future backends) run *the same* optimization — they differ
only in the thin ``SteeringBackendAdapter``. This test drives the core with a
deterministic CPU mock adapter + mock steering model whose loss is a known
closed-form function of the latent, and asserts the exact ``x_t`` trajectory and
each regularizer term against hand-computed values. If this passes, any backend
whose adapter satisfies the contract inherits provably-identical steering.

Runs anywhere (pure CPU torch, no model / GPU). ``pytest tests/test_steering_core.py``
or ``python tests/test_steering_core.py``.
"""
from __future__ import annotations

import os
import sys

import torch


from will_it_render.inference.steering_core import SteeringBackendAdapter, run_steering_optimization


class MockAdapter(SteeringBackendAdapter):
    """Deterministic in-memory adapter.

    - Holds a single latent tensor as the "video state".
    - ``grad_forward`` returns ``captures=[x_t]`` (so the mock loss can be a closed
      form of the latent) and ``pred_clean=x_t`` (identity — so pred_reg is a known
      function of the displacement).
    - ``denoise_step`` just records that it ran (no scheduler math needed here).
    """

    pred_is_x0 = True  # grad_forward returns pred_clean=x_t, i.e. x̂0 directly (identity)

    def __init__(self, latent: torch.Tensor, n_steps: int):
        self.latent = latent.clone()
        self.n_steps = n_steps
        self.denoise_calls: list[int] = []
        self.begin_calls = 0
        self.end_calls = 0

    def steps(self):
        return range(self.n_steps)

    def get_latent(self):
        return self.latent

    def commit_latent(self, x_t):
        self.latent = x_t

    def begin_optimization(self):
        self.begin_calls += 1

    def end_optimization(self):
        self.end_calls += 1

    def grad_forward(self, x_t, step_idx):
        return [x_t], x_t  # captures=[x_t], pred_clean=x_t

    def denoise_step(self, step_idx):
        self.denoise_calls.append(step_idx)


class MockSteeringModel:
    """Loss = 0.5 * sum(captures[0]**2)  →  grad wrt x_t is exactly x_t.

    ``steer_steps=None`` steers every step; otherwise only the listed steps.
    """

    def __init__(self, steer_steps=None):
        self.steer_steps = steer_steps

    def should_steer(self, step_idx):
        return True if self.steer_steps is None else (step_idx in self.steer_steps)

    def loss(self, captures, step_idx, *, target_words_indices):
        x = captures[0]
        loss = 0.5 * (x ** 2).sum()
        # A realistic-shaped metrics dict so the callback assembly is exercised.
        prob = float(torch.sigmoid(-loss).detach())
        metrics = {
            "loss": float(loss.detach()),
            "bce": float(loss.detach()),
            "prob": prob,
            "per_word_prob": {"dog": prob},
            "balance": 0.0,
        }
        return loss, metrics


def _x0():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


def test_simple_gd_trajectory():
    """simple-GD: loss grad == x_t, so x ← x*(1-lr) per iter → x0*(1-lr)^n_iters."""
    x0 = _x0()
    lr, n_iters = 0.1, 3
    adapter = MockAdapter(x0, n_steps=1)
    run_steering_optimization(
        adapter, MockSteeringModel(), target_words_indices={"dog": [0]},
        n_iters=n_iters, optimizer_name="simple", lr=lr,
    )
    expected = x0 * (1 - lr) ** n_iters
    assert torch.allclose(adapter.latent, expected, atol=1e-6), (adapter.latent, expected)
    assert adapter.denoise_calls == [0]
    assert adapter.begin_calls == 1 and adapter.end_calls == 1


def test_no_steer_leaves_latent_untouched_but_still_denoises():
    """n_iters=0 disables optimization; every step still denoises."""
    x0 = _x0()
    adapter = MockAdapter(x0, n_steps=4)
    run_steering_optimization(
        adapter, MockSteeringModel(), target_words_indices={"dog": [0]},
        n_iters=0, optimizer_name="simple", lr=0.1,
    )
    assert torch.allclose(adapter.latent, x0)
    assert adapter.denoise_calls == [0, 1, 2, 3]
    assert adapter.begin_calls == 0  # never entered optimization


def test_should_steer_gates_steps():
    """Only steps the model opts into are optimized; the rest pass through."""
    x0 = _x0()
    lr, n_iters = 0.1, 1
    adapter = MockAdapter(x0, n_steps=5)
    run_steering_optimization(
        adapter, MockSteeringModel(steer_steps={1, 3}), target_words_indices={"dog": [0]},
        n_iters=n_iters, optimizer_name="simple", lr=lr,
    )
    # Two steered steps, each x ← x*(1-lr): total factor (1-lr)^2.
    expected = x0 * (1 - lr) ** 2
    assert torch.allclose(adapter.latent, expected, atol=1e-6)
    assert adapter.denoise_calls == [0, 1, 2, 3, 4]
    assert adapter.begin_calls == 2  # only steps 1 and 3


def test_adam_single_step():
    """Adam iter-1 update ≈ lr * g / (sqrt(g^2)+eps) = lr*sign(g); g = x0 here."""
    x0 = _x0()
    lr = 0.1
    adapter = MockAdapter(x0, n_steps=1)
    run_steering_optimization(
        adapter, MockSteeringModel(), target_words_indices={"dog": [0]},
        n_iters=1, optimizer_name="adam", lr=lr,
    )
    eps = 1e-8
    g = x0  # grad of 0.5*sum(x^2)
    expected = x0 - lr * g / (torch.sqrt(g * g) + eps)
    assert torch.allclose(adapter.latent, expected, atol=1e-5), (adapter.latent, expected)


def test_regularizer_terms_are_zero_at_first_iter():
    """At iter 0, x_t == x0, so magnitude/anchor/pred/kl raw terms are all 0.

    Verified by capturing the metrics via step_callback (which reports *_before as the
    first-iter reading).
    """
    x0 = _x0()
    captured = {}

    def cb(step_idx, m):
        captured.update(m)

    adapter = MockAdapter(x0, n_steps=1)
    run_steering_optimization(
        adapter, MockSteeringModel(), target_words_indices={"dog": [0]},
        n_iters=1, optimizer_name="simple", lr=0.1,
        magnitude_reg=1.0, anchor_reg=1.0, pred_reg=1.0, kl_reg=1.0,
        step_callback=cb,
    )
    for key in ("mag_term_before", "anchor_term_before", "pred_term_before", "kl_term_before"):
        assert abs(captured[key]) < 1e-6, (key, captured[key])
    # iters trajectory recorded (iter 0 snapshot present).
    assert isinstance(captured["iters"], list) and len(captured["iters"]) == 1


def test_anchor_term_matches_formula_second_iter():
    """anchor_term at iter k == mean((x_k - x0)^2). With plain probe-loss simple-GD
    and no active reg, x_1 = x0*(1-lr); check the raw anchor term logged at iter 2."""
    x0 = _x0()
    lr = 0.1
    snaps = {}

    def cb(step_idx, m):
        snaps.update(m)

    adapter = MockAdapter(x0, n_steps=1)
    # anchor_reg=0 keeps it out of the loss but the raw term is still logged each iter.
    run_steering_optimization(
        adapter, MockSteeringModel(), target_words_indices={"dog": [0]},
        n_iters=2, optimizer_name="simple", lr=lr, anchor_reg=0.0,
        step_callback=cb,
    )
    # After iter 1, x = x0*(1-lr); the *_after term is read at the last (2nd) iter,
    # computed on x after iter-1 update: mean((x0*(1-lr) - x0)^2) = mean((x0*lr)^2).
    expected_after = ((x0 * (1 - lr) - x0) ** 2).mean().item()
    assert abs(snaps["anchor_term_after"] - expected_after) < 1e-6, (snaps["anchor_term_after"], expected_after)


class _ScheduleSpyModel(MockSteeringModel):
    """A NON-A&E model that nonetheless *implements* the schedule hooks. The explicit
    ``mode == 'attend_excite'`` gate in run_steering_optimization must ignore them — the
    hooks record any call and return absurd values (99 iters, ×100 lr, stop-immediately)
    that would wreck the trajectory if ever honored. Proves the gate keys on mode, not on
    hasattr, so ``per_step`` / ``tempo_control`` are unaffected by the A&E schedule."""

    def __init__(self, steer_steps=None):
        super().__init__(steer_steps)
        self.mode = "per_step"   # NOT attend_excite → gate stays closed
        self.calls = {"iters_for_step": 0, "lr_multiplier": 0, "stop_iteration": 0}

    def iters_for_step(self, step_idx):
        self.calls["iters_for_step"] += 1
        return 99

    def lr_multiplier(self, step_idx, n_steps=None):
        self.calls["lr_multiplier"] += 1
        return 100.0

    def stop_iteration(self, step_idx, metrics, iter_idx):
        self.calls["stop_iteration"] += 1
        return True


def test_gate_closed_ignores_schedule_hooks():
    """Non-A&E model: the schedule hooks are never called and the trajectory is the fixed
    n_iters / constant-lr path — byte-identical to a model without the hooks."""
    x0 = _x0()
    lr, n_iters = 0.1, 3
    model = _ScheduleSpyModel()
    adapter = MockAdapter(x0, n_steps=1)
    run_steering_optimization(
        adapter, model, target_words_indices={"dog": [0]},
        n_iters=n_iters, optimizer_name="simple", lr=lr,
    )
    # Gate closed → hooks untouched (had any fired, iters=99 / lr×100 / stop would show).
    assert model.calls == {"iters_for_step": 0, "lr_multiplier": 0, "stop_iteration": 0}, model.calls
    expected = x0 * (1 - lr) ** n_iters
    assert torch.allclose(adapter.latent, expected, atol=1e-6), (adapter.latent, expected)


class _FrameMaskModel:
    """Per-frame model whose two class losses have CONSTANT, distinguishable gradients.

    ``L_present = (x·P).sum()`` and ``L_absent = (x·A).sum()`` with P ≡ 1 and A ≡ −1, so the
    unmasked gradient of their sum is exactly 0 everywhere. Any nonzero result therefore comes
    from the routing and nothing else — which is precisely what this pins.
    """

    weight = 1.0
    frame_grad_mask = True

    def __init__(self, control):
        self.control = control

    def should_steer(self, step_idx):
        return True

    def loss(self, captures, step_idx, *, target_words_indices):
        x = captures[0]
        loss = (x * 1.0).sum() + (x * -1.0).sum()      # == 0, grad == 0
        return loss, {"loss": 0.0, "bce": 0.0, "prob": 0.5,
                      "per_word_prob": {"dog": 0.5}, "balance": 0.0}

    def per_frame_class_losses(self, captures, step_idx, *, target_words_indices):
        x = captures[0]
        return {"dog": ((x * 1.0).sum(), (x * -1.0).sum(), self.control)}


def test_frame_grad_mask_routes_each_class_to_its_own_frames():
    """control=1 frames move by −lr·(+1); control=0 frames move by −lr·(−1).

    Unmasked the two gradients cancel exactly (the model is built so their sum is 0) and the
    latent would not move at all — so this asserts the routing, not just that something changed.
    """
    T, lr = 4, 0.1
    control = torch.tensor([1.0, 1.0, 0.0, 0.0])
    x0 = torch.zeros(2, T, 3, 3)                       # [C, F, H, W]; F is the only axis of size T
    adapter = MockAdapter(x0, n_steps=1)
    run_steering_optimization(
        adapter, _FrameMaskModel(control), target_words_indices={"dog": [0]},
        n_iters=1, optimizer_name="simple", lr=lr,
    )
    got = adapter.latent[0, :, 0, 0]
    expected = torch.tensor([-lr, -lr, lr, lr])        # present: −lr·(+1); absent: −lr·(−1)
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    # And every element of a frame moves together — the mask is over frames, not elements.
    for t in range(T):
        assert torch.allclose(adapter.latent[:, t], torch.full((2, 3, 3), float(expected[t])),
                              atol=1e-6)


def test_frame_grad_mask_off_is_the_plain_gradient():
    """Same model with the flag off: the two halves cancel and the latent does not move."""
    control = torch.tensor([1.0, 1.0, 0.0, 0.0])
    model = _FrameMaskModel(control)
    model.frame_grad_mask = False
    x0 = torch.zeros(2, 4, 3, 3)
    adapter = MockAdapter(x0, n_steps=1)
    run_steering_optimization(
        adapter, model, target_words_indices={"dog": [0]},
        n_iters=1, optimizer_name="simple", lr=0.1,
    )
    assert torch.allclose(adapter.latent, x0, atol=1e-7), adapter.latent


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} steering-core tests passed.")


if __name__ == "__main__":
    _run_all()
