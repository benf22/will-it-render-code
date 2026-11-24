"""External baselines to compare against our probe-guided attention steering.

Each baseline is a drop-in :class:`will_it_render.inference.steering_model.SteeringModel`: it turns one
denoising step's live cross-attention captures into a scalar loss that
``steering/steering_loop.py`` backpropagates to ``x_t``. Reusing that loop (rather than
porting each paper's own denoising code) means every baseline runs through the *identical*
generation path as our method — same backend, seed latents, output layout, sweep driver
and labeling — so the comparison is controlled.

Adding a baseline:

1. implement ``SteeringModel`` under ``baselines/<name>/``,
2. register its ``mode`` string in :data:`BASELINE_BUILDERS`,
3. write a steering-style config with ``steering.model.mode: <mode>``.

``will_it_render.inference.steering_model.build_steering_model`` dispatches any mode it does not
implement here.
"""
from __future__ import annotations

from typing import Callable, Optional


def _build_tempo_control(cfg, device, agg_kwargs, probe_config, steps_filter):
    from will_it_render.evaluation.baselines.tempocontrol.loss import build_tempo_control_model

    return build_tempo_control_model(cfg, device, agg_kwargs, probe_config, steps_filter)


def _build_attend_excite(cfg, device, agg_kwargs, probe_config, steps_filter):
    from will_it_render.evaluation.baselines.attend_excite.loss import build_attend_excite_model

    return build_attend_excite_model(cfg, device, agg_kwargs, probe_config, steps_filter)


# mode string → builder with the signature of build_steering_model.
BASELINE_BUILDERS: dict[str, Callable] = {
    "tempo_control": _build_tempo_control,
    "attend_excite": _build_attend_excite,
}


def build_baseline_steering_model(
    mode: str,
    cfg: dict,
    device,
    agg_kwargs: dict,
    probe_config: dict,
    steps_filter: Optional[list[int]] = None,
):
    """Factory for baseline steering models. Raises on an unknown ``mode``."""
    try:
        builder = BASELINE_BUILDERS[mode]
    except KeyError:
        raise ValueError(
            f"Unknown steering model mode {mode!r}. Implemented: 'per_step' (ours) + "
            f"baselines {sorted(BASELINE_BUILDERS)}."
        ) from None
    return builder(cfg, device, agg_kwargs, probe_config, steps_filter)
