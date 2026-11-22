"""Nearest-probe step resolution for StepKeyedSteeringModel.

WAN captures probes at a sparse step subset ({0,1,4,7,10,15}); nearest_probe lets a
contiguous steps_to_steer range steer every step by borrowing the closest probe. These
tests exercise the pure resolver / gate logic on a bare instance (via __new__), so they
need no GPU or checkpoints.
"""
from will_it_render.inference.steering_model import StepKeyedSteeringModel

WAN_PROBE_STEPS = {0, 1, 4, 7, 10, 15}


def _model(nearest_probe, steps_filter):
    o = StepKeyedSteeringModel.__new__(StepKeyedSteeringModel)  # bypass __init__ (no GPU load)
    o.probes = {s: object() for s in WAN_PROBE_STEPS}
    o._probe_steps = sorted(o.probes)
    o.nearest_probe = nearest_probe
    o.steps_filter = set(steps_filter) if steps_filter is not None else None
    return o


def test_off_only_steers_exact_probe_steps():
    o = _model(nearest_probe=False, steps_filter=range(0, 11))
    assert [s for s in range(16) if o.should_steer(s)] == [0, 1, 4, 7, 10]
    assert o._probe_step(2) is None  # no borrowing when off


def test_on_steers_every_step_in_range():
    o = _model(nearest_probe=True, steps_filter=range(0, 11))
    assert [s for s in range(16) if o.should_steer(s)] == list(range(0, 11))
    o15 = _model(nearest_probe=True, steps_filter=range(0, 16))
    assert [s for s in range(16) if o15.should_steer(s)] == list(range(0, 16))


def test_nearest_mapping_and_tie_breaks_earlier():
    o = _model(nearest_probe=True, steps_filter=range(0, 16))
    assert {s: o._probe_step(s) for s in range(16)} == {
        0: 0, 1: 1, 2: 1, 3: 4, 4: 4, 5: 4, 6: 7, 7: 7,
        8: 7, 9: 10, 10: 10, 11: 10, 12: 10, 13: 15, 14: 15, 15: 15,
    }
    # equidistant between two probes → the earlier one
    o.probes = {0: object(), 4: object()}
    o._probe_steps = [0, 4]
    assert o._probe_step(2) == 0


def test_exact_probe_step_never_borrows():
    o = _model(nearest_probe=True, steps_filter=None)
    for s in WAN_PROBE_STEPS:
        assert o._probe_step(s) == s


def test_n_iters_per_step_map_resolution():
    # Per-step gradient budget: run_steering_optimization resolves eff_iters as
    # n_iters_per_step.get(step, global_n_iters) — steps in the map override, others fall back.
    o = _model(nearest_probe=False, steps_filter=None)
    o.n_iters_per_step = {0: 16, 1: 8, 4: 4, 7: 4, 10: 4}
    GLOBAL = 8
    assert [o.n_iters_per_step.get(s, GLOBAL) for s in (0, 1, 4, 7, 10)] == [16, 8, 4, 4, 4]
    assert o.n_iters_per_step.get(15, GLOBAL) == GLOBAL  # unmapped step -> global fallback
