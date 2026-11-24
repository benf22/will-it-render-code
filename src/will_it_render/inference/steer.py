"""Entry point for inference-time attention steering.

    python -m steering.steer --config steering/steering.yaml
    python -m steering.steer --config steering/steering.yaml \
        --prompts_csv example_prompts/steer_lion.csv --output_dir /tmp/steer

Loads ``steering/steering.yaml`` plus the shared per-model config
``configs/models/<model_type>.yaml`` (the only config reused from the rest of the
pipeline), builds the backend and the steering model, then runs steered generation
for each prompt — saving ``output.mp4`` / ``output.wav`` and a per-step
``steering_metrics.json`` per prompt.

The body is factored into reusable functions so other drivers (e.g.
a caller) can load the configs / build the expensive backend ONCE and
call :func:`run_steering` many times without reloading the model:

    load_configs  → parse the steering yaml + resolve the shared model config
    build_backend → create the (expensive) model backend; call ONCE and reuse
    run_steering  → build the (cheap) steering model + run every prompt into an
                    explicit, fully-resolved output dir (caller owns timestamping)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make the repo root importable when run as a script.

import numpy as np
import scipy.io.wavfile as wavfile_io
import torch
import yaml
from omegaconf import OmegaConf

from will_it_render.evaluation.baselines import build_baseline_steering_model
from will_it_render.common.agg import build_agg_kwargs
from will_it_render.common.prompts import load_prompts, normalize_row, parse_target_words
from will_it_render.backends.model_backend import create_backend
from will_it_render.common.storage import make_output_dir
from will_it_render.inference import steering_core
from will_it_render.inference.steering_model import build_steering_model


def parse_timelines(raw: str | None, obj_sep: str = ";", kv_sep: str = ":") -> dict[str, list[float]]:
    """Parse the prompts-CSV ``timelines`` column → ``{word: [presence per frame]}``.

    Format: one ``word<kv_sep>timeline`` entry per object, entries joined by ``obj_sep``::

        "dog:1111100000000;cat:0000000011111"

    Each timeline is a **binary string** (one ``0``/``1`` char per frame) or a
    comma/space-separated list of numbers; either length is fine — the steering model
    resamples it to the probe's ``T`` latent frames. Empty / missing → ``{}`` (every
    target defaults to all-ones = the per-clip objective). Keyed by the same surface
    target word as ``target_words``.
    """
    if not raw or not str(raw).strip():
        return {}
    out: dict[str, list[float]] = {}
    for entry in str(raw).split(obj_sep):
        entry = entry.strip()
        if not entry:
            continue
        if kv_sep not in entry:
            raise ValueError(
                f"timelines entry {entry!r} missing '{kv_sep}' (expected "
                f"'word{kv_sep}timeline', entries joined by '{obj_sep}').")
        word, sig = entry.split(kv_sep, 1)
        word, sig = word.strip(), sig.strip()
        if ("," in sig) or (" " in sig):
            values = [float(x) for x in sig.replace(",", " ").split()]
        else:  # bare bit-string, one char per frame
            values = [float(ch) for ch in sig]
        if not values:
            raise ValueError(f"timelines entry for {word!r} has an empty timeline.")
        out[word] = values
    return out


def load_configs(config_path: str):
    """Load the steering config + its resolved shared per-model config.

    Returns ``(raw_cfg, steering, model_cfg, model_type, model_cfg_path,
    model_extraction_cfg)``. ``engine_cfg`` is rebuilt by :func:`build_backend` from
    ``model_cfg["engine"]`` (kept out of the tuple to avoid carrying an OmegaConf
    object across the boundary).
    """
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    steering = raw_cfg["steering"]
    model_type: str = raw_cfg.get("model_type", "ltx")

    # The per-model config is the only shared config; it lives in the repo's configs/models/
    # dir, located via common.paths.config_dir() (VP_CONFIG_DIR or an upward search).
    from will_it_render.common import paths

    model_cfg_path = paths.config_dir() / "models" / f"{model_type}.yaml"
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    model_extraction_cfg: dict = model_cfg.get("extraction", {})
    # Per-run extraction overrides: the probe's TRAIN-time feature recipe (e.g. a spatial
    # avg-pool via train_spatial_downsample, or a block subset via layer_subset) must be
    # reproduced at steer time by the aggregator/hooks, but those knobs live in the shared
    # configs/models/<model>.yaml. Rather than mutate the shared extraction default, a
    # steering config may carry `model_extraction_overrides:` (e.g. attn_spatial_downsample: 2)
    # that we shallow-merge here. Keeps the shared config's extraction defaults intact.
    # Also used to match LTX's official guider recipe (video_rescale_scale / video_stg_scale /
    # video_stg_blocks) for a single run. Absent key → no change.
    overrides = raw_cfg.get("model_extraction_overrides") or {}
    if overrides:
        model_extraction_cfg = {**model_extraction_cfg, **overrides}
    return raw_cfg, steering, model_cfg, model_type, model_cfg_path, model_extraction_cfg


#: cuBLAS needs a fixed workspace to make its GEMM reductions reproducible. It reads this
#: when the handle is created (first CUDA GEMM), so it must be in the environment before
#: then — setting it later silently does nothing.
_CUBLAS_DETERMINISTIC = ":4096:8"


def enable_determinism(logger: logging.Logger | None = None) -> bool:
    """Make steered generation bit-reproducible. On by default; ``STEER_DETERMINISTIC=0`` opts out.

    Steered generation was **not** reproducible run-to-run even on one GPU type, while
    un-steered generation is bit-exact — so the divergence entered through the backward
    pass. Measured (2 prompts, identical config/seed/latent):

        flags off ............ CLIP drift 0.19 / 0.38 vs its own re-run
        warn_only=True ....... 0.14 / 0.39   (fixes every op EXCEPT the one that mattered)
        full determinism ..... 0.0000, identical md5

    ``warn_only=True`` isolated the single culprit: **Flash Attention backward**, which
    accumulates dQ/dK/dV with atomics. Nothing else in the steering path lacks a
    deterministic kernel, so this does not raise. Cost ≈ +24% wall-clock per clip
    (46 s → 57 s); most of that is `cudnn.benchmark=False` plus the other
    deterministic kernels, not the attention fix itself.

    Without this, a config's own re-run differs about as much as a *different prompt*
    would — which is why small-n steering sweeps measured noise (empirically).

    Call BEFORE the model is built. Returns True when determinism is on.
    """
    if os.environ.get("STEER_DETERMINISTIC", "").strip().lower() in ("0", "off", "false"):
        if logger:
            logger.warning("determinism DISABLED (STEER_DETERMINISTIC=0) — steered output "
                           "will not be reproducible run-to-run")
        return False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_DETERMINISTIC)
    cfgv = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    if cfgv not in (":4096:8", ":16:8"):
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG={cfgv!r} is set to a non-deterministic value; use "
            f"':4096:8' (or export STEER_DETERMINISTIC=0 to opt out of determinism).")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if logger:
        logger.info("determinism ON: use_deterministic_algorithms(True), cudnn.deterministic=True, "
                    "CUBLAS_WORKSPACE_CONFIG=%s (~+24%% wall-clock; STEER_DETERMINISTIC=0 to opt out)",
                    cfgv)
    return True


def build_backend(model_type: str, model_cfg: dict, model_extraction_cfg: dict,
                  steering: dict, device: int, logger: logging.Logger):
    """Create the model backend. EXPENSIVE — call once and reuse across configs.

    The backend depends only on engine paths + extraction params (and the
    ``steering`` dict as the experiment cfg); none of the swept steering params feed
    it, so a single backend is reusable for an entire parameter sweep.
    """
    # Before the pipeline is constructed: cuBLAS must see its workspace config prior to
    # creating a handle, and the deterministic-algorithm switch must precede any autograd.
    enable_determinism(logger)
    # Which latent dim holds a patch's channels (per-patch cosine of the mutual-agree fold).
    steering_core.CHANNEL_DIM = steering_core.CHANNEL_DIM_BY_MODEL.get(model_type)
    engine_cfg = OmegaConf.create(model_cfg.get("engine", {}))
    return create_backend(model_type, engine_cfg, model_extraction_cfg, steering, device, logger)


def _make_attn_capture_sink(root: Path, stem: str, step: int, logger: logging.Logger):
    """Build the steering_core.ATTN_CAPTURE_SINK for one prompt.

    Writes ``<root>/<stem>__upd<k>/step_<step>.safetensors`` with one ``features__<word>``
    fp16 tensor per steered word — the probe's exact input (aggregated with the steering
    model's own agg_kwargs, i.e. the same block subset / spatial pool the probe was trained
    with), flattened like ``data_extraction.storage.save_features`` stores extractions. The
    first capture also writes ``<root>/probe_config.yaml`` with the STORED geometry (pooled
    ``latent_hw`` / ``Lq``), so ``<root>`` is a valid ProbeDataset extraction root.
    """
    from safetensors.torch import save_file
    from will_it_render.common.aggregator import aggregate_captures_to_vector, probe_shape_from_captures

    def sink(n_updates: int, captures, target_words_indices: dict, steering_model) -> None:
        feats = {}
        for word, idx in target_words_indices.items():
            vec = aggregate_captures_to_vector(captures, idx, **steering_model.agg_kwargs)
            feats[f"features__{word}"] = vec.detach().flatten().to(torch.float16).cpu().contiguous()
        d = root / f"{stem}__upd{n_updates}"
        d.mkdir(parents=True, exist_ok=True)
        final = d / f"step_{step:03d}.safetensors"
        tmp = d / f"step_{step:03d}.safetensors.tmp"
        save_file(feats, str(tmp))
        os.replace(tmp, final)  # atomic: a reader never sees a half-written file
        pc = root / "extraction_config.yaml"
        if not pc.exists():
            shape = probe_shape_from_captures(captures, **steering_model.agg_kwargs)
            h, w = shape["latent_hw"]
            shape.update({
                "latent_num_frames": int(shape["Lq"] // (h * w)),
                "feature_dim": int(shape["n_blocks_heads"] * shape["Lq"]),
                "capture_blocks": getattr(steering_model, "capture_blocks", None),
                "n_blocks_total": int(getattr(steering_model, "n_blocks_total", shape["n_blocks"])),
                "spatial_downsample": int(steering_model.agg_kwargs.get("spatial_downsample", 1)),
                "source": "steering attn_capture (probe-input features, already pooled)",
            })
            pc.write_text(yaml.dump(shape, default_flow_style=False))
        n = next(iter(feats.values())).numel()
        logger.info("attn_capture: %s after %d update(s): %d word(s) x %d → %s",
                    stem, n_updates, len(feats), n, final)

    return sink


def run_steering(steering: dict, backend, model_extraction_cfg: dict, raw_cfg: dict,
                 model_cfg: dict, model_cfg_path: Path, output_dir: Path,
                 device: int, logger: logging.Logger, prompts_csv: str | None = None,
                 extra_step_callback=None) -> dict:
    """Run steered generation for every prompt into ``output_dir``.

    ``backend`` is prebuilt and reused — never recreated here. ``output_dir`` is
    explicit and fully resolved (the caller owns any timestamping).

    Returns ``{"n_prompts": int, "prompt_dirs": list[Path]}`` so callers can aggregate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot the configs used (reproducibility).
    (output_dir / "steering.yaml").write_text(yaml.dump(raw_cfg, default_flow_style=False))
    (output_dir / model_cfg_path.name).write_text(yaml.dump(model_cfg, default_flow_style=False))

    # Feature-aggregation params must match how the probe's training features were
    # produced — read them from the shared model extraction config.
    agg_kwargs = build_agg_kwargs(model_extraction_cfg)
    steps_to_steer = steering.get("steps_to_steer", None)
    # Extraction probe_config.yaml — provides capture_blocks + pre-reduction
    # n_blocks/n_heads needed to reshape captures for the probe. Optional: baselines
    # (baselines/) have no trained probe and infer the shapes from the captures.
    probe_config_path = steering["model"].get("extraction_config", steering["model"].get("probe_config"))
    if probe_config_path:
        with open(probe_config_path) as f:
            probe_config = yaml.safe_load(f)
    else:
        probe_config = {}
    model_block = {**steering["model"], "objective": steering.get("objective", {}),
                   "n_iters_per_step": steering.get("optimization", {}).get("n_iters_per_step")}
    steering_model = build_steering_model(
        model_block, torch.device(f"cuda:{device}"), agg_kwargs, probe_config,
        steps_filter=steps_to_steer,
    )
    logger.debug("steering model built: mode=%s checkpoints_dir=%s",
                 steering["model"].get("mode"), steering["model"].get("checkpoints_dir"))

    prompts_csv = prompts_csv or steering["prompts_csv"]

    opt = steering.get("optimization", {})
    n_iters = int(opt.get("n_iters", 5))
    optimizer_name = str(opt.get("optimizer", "adam"))
    lr = float(opt.get("lr", 0.1))
    magnitude_reg = float(opt.get("magnitude_reg", 0.0))
    anchor_reg = float(opt.get("anchor_reg", 0.0))
    pred_reg = float(opt.get("pred_reg", 0.0))
    kl_reg = float(opt.get("kl_reg", 0.0))
    entropy_reg = float(opt.get("entropy_reg", 0.0))
    # Attention-space trust region on the non-target text tokens (see steering_loop).
    attn_anchor_reg = float(opt.get("attn_anchor_reg", 0.0))
    attn_anchor_tokens = str(opt.get("attn_anchor_tokens", "prompt"))
    attn_anchor_ref = str(opt.get("attn_anchor_ref", "step"))

    # Gradient-aggregation rule by name (e.g. `grad_aggregate: mutual_agree`). Only touched
    # when the key is present, so the scratch drivers that assign core.GRAD_AGGREGATE
    # themselves before calling in here keep working. An explicit null/"" clears it, which is
    # what a sweep whose combos toggle the rule needs (one worker process serves many combos).
    if "grad_aggregate" in steering:
        name = str(steering.get("grad_aggregate") or "").strip()
        # "A+B[+C]": chain rules — A aggregates the per-object gradients, each later rule is
        # applied to that single aggregate (e.g. "unitnorm_gated0.95+flow_pca_drop_3" = gate out
        # satisfied objects, then project the update off flow PC 3). "unitnorm+X" is the common
        # case (unit-norm balancing, then the flow-geometry rule X).
        parts = [q for q in name.split("+") if q]
        for q in parts:
            if q not in steering_core.GRAD_AGGREGATE_REGISTRY:
                raise ValueError(f"unknown grad_aggregate {q!r} (in {name!r}); "
                                 f"known: {sorted(steering_core.GRAD_AGGREGATE_REGISTRY)}")
        if not parts:
            steering_core.GRAD_AGGREGATE = None
        elif len(parts) == 1:
            steering_core.GRAD_AGGREGATE = steering_core.GRAD_AGGREGATE_REGISTRY[parts[0]]
        else:
            _rules = [steering_core.GRAD_AGGREGATE_REGISTRY[q] for q in parts]

            def _chained(pgrads, flow=None, scores=None, _rules=_rules):
                g = _rules[0](pgrads, flow, scores)
                for r in _rules[1:]:
                    g = r({"_agg": g}, flow, None)
                return g

            steering_core.GRAD_AGGREGATE = _chained
        logger.info("grad_aggregate: %s", name or "none (plain mean-BCE gradient)")

    # Aggressive flow: at steered steps, DON'T advance via the model velocity — take one more
    # aggregated-gradient step and commit THAT as the transition (the gradient becomes the
    # velocity). Requires a grad_aggregate rule (uses per_object_bce grads). Reset per combo.
    # Movement-band regularizer (optimization.move_band: {weight, lo, hi, space}); None/absent = off.
    mb = steering.get("optimization", {}).get("move_band")
    steering_core.MOVE_BAND = dict(mb) if mb else None
    if mb:
        logger.info("move_band: %s", mb)
    # Adaptive iteration count (optimization.stop_when_min_prob): stop the step when the weakest object passes it.
    _sw = steering.get("optimization", {}).get("stop_when_min_prob")
    steering_core.STOP_WHEN_MIN_PROB = float(_sw) if _sw else None
    if _sw:
        logger.info("stop_when_min_prob: %s", _sw)
    # Invariant: a zero-fallback gated rule MUST early-stop. Once every object is satisfied the
    # rule applies only zero updates (bit-exact no-op when regularizers are off), so running the
    # remaining iterations is pure wasted compute. Auto-enable the early-exit at the rule's gate
    # threshold when the config did not set one. (budget_s variants still push the worst object for
    # their first min_budget iters; the loop's `iter_idx >= min_budget` guard preserves that, so the
    # committed latent — and the output — is unchanged.) Worst-fallback rules keep pushing when all
    # objects are satisfied, so they are intentionally left alone.
    _agg = steering_core.GRAD_AGGREGATE
    if steering_core.STOP_WHEN_MIN_PROB is None and getattr(_agg, "fallback", None) == "zero":
        steering_core.STOP_WHEN_MIN_PROB = float(getattr(_agg, "gate_tau", 0.95))
        logger.info("stop_when_min_prob auto-enabled at %.2f (zero-fallback rule requires early-stop)",
                    steering_core.STOP_WHEN_MIN_PROB)
    # Mutual-agree fold order (steering.mutual_order): "worst_first" (default) folds the
    # least-present object first; "text" = the legacy prompt order (reproduce old runs).
    steering_core.MUTUAL_ORDER = str(steering.get("mutual_order", "worst_first"))
    # Mutual-agree fold (steering.mutual_fold): "anchor" (default) keeps the first object
    # as is and adds each next one on the PATCHES where it agrees; "legacy" = the earlier behaviour.
    steering_core.MUTUAL_FOLD = str(steering.get("mutual_fold", "anchor"))
    # steering.mutual_skip_steps: denoising steps where the fold is replaced by the plain mean ("-mutual" there).
    steering_core.MUTUAL_SKIP_STEPS = frozenset(int(x) for x in (steering.get("mutual_skip_steps") or []))
    # steering.log_mask_stats: diagnostics only — per-iteration mask stats of the anchor fold in steering_metrics.json.
    steering_core.MASK_STATS = bool(steering.get("log_mask_stats", False))
    if "mutual" in str(steering.get("grad_aggregate", "")):
        logger.info("mutual_order: %s  mutual_fold: %s  mutual_skip_steps: %s", steering_core.MUTUAL_ORDER,
                    steering_core.MUTUAL_FOLD, sorted(steering_core.MUTUAL_SKIP_STEPS))
    # Capture-space mutual-agree (one transformer backward per iteration instead of one per object;
    # a semantic VARIANT of latent-space mutual-agree, not bit-identical). Off unless configured.
    steering_core.CAPTURE_SPACE_MUTUAL = bool(steering.get("capture_space_mutual", False))
    steering_core.COMPARE_POBJ_GRAD = bool(steering.get("compare_pobj_grad", False))
    if steering_core.CAPTURE_SPACE_MUTUAL:
        logger.info("capture_space_mutual: ON (compare_pobj_grad=%s)", steering_core.COMPARE_POBJ_GRAD)

    # Persistence regularizer (optimization.persist_reg: {kind, weight, steps, detach_flow}); None/absent = off.
    pr = steering.get("optimization", {}).get("persist_reg")
    steering_core.PERSIST_REG = dict(pr) if pr else None
    if pr:
        logger.info("persist_reg: %s", pr)
    # Multistep-solver reset after a steered step (steering.solver_reset: corrector | full); None = stock.
    sr = steering.get("solver_reset") or None
    steering_core.SOLVER_RESET = str(sr) if sr else None
    if sr:
        logger.info("solver_reset: %s", sr)

    # Velocity projection of the denoising advance itself (keep / remove flow PCs), probe-free.
    if "velocity_proj" in steering:
        vp = steering.get("velocity_proj") or None
        steering_core.VELOCITY_PROJ = dict(vp) if vp else None
        logger.info("velocity_proj: %s", vp)

    # Flow reference for the aggregation rules: "iter" (velocity of the current iterate, default)
    # or "k0" (velocity of the incoming un-steered latent, held fixed across the updates).
    if "flow_ref" in steering:
        fr = str(steering.get("flow_ref") or "iter")
        if fr not in ("iter", "k0"):
            raise ValueError(f"flow_ref must be 'iter' or 'k0' (got {fr!r})")
        steering_core.FLOW_REF = fr
        logger.info("flow_ref: %s", fr)

    # Held-out judge probes: "all" or a list of probe steps. Diagnostic only (never in the loss)
    # — each judge scores the same captures the optimised probe saw, so a saturated optimised
    # score can be checked against a probe that was never optimised against here.
    if steering.get("judge_probes"):
        jp = steering.get("judge_probes")
        steering_core.JUDGE_PROBES = "all" if str(jp) == "all" else [int(x) for x in jp]
        logger.info("judge_probes: %s", steering_core.JUDGE_PROBES)

    # Held-out hinge judge (steering.hinge_judge: true or {objective: {...}}): a read-only
    # Attend-and-Excite model scored on the SAME captures the optimized (probe-BCE) objective
    # just saw, at every gradient iteration. Never enters the loss — lets a hinge-vs-BCE
    # detector-quality comparison run on one probe-steered trajectory. Reset every call so a
    # sweep worker reusing this process doesn't leak the judge into a combo that didn't ask for it.
    _hj = steering.get("hinge_judge")
    if _hj:
        _hj_cfg = {"objective": (_hj.get("objective", {}) if isinstance(_hj, dict) else {})}
        steering_core.HINGE_JUDGE_MODEL = build_baseline_steering_model(
            "attend_excite", _hj_cfg, torch.device(f"cuda:{device}"), agg_kwargs, {},
        )
        logger.info("hinge_judge: attend_excite read-only judge enabled")
    else:
        steering_core.HINGE_JUDGE_MODEL = None

    # Probe-mix (steering.probe_mix): fold the held-out NEIGHBOUR probes into the optimisation.
    # {mode: alternate|common}. Off by default; requires a grad_aggregate rule (routes through
    # per_object_bce). Reset per combo so a sweep arm without it reverts to the own probe.
    _pm = steering.get("probe_mix") or None
    steering_core.PROBE_MIX = dict(_pm) if _pm else None
    if _pm:
        logger.info("probe_mix: %s", steering_core.PROBE_MIX)

    # Neighbour-as-selector: pick the object the rule acts on with a neighbour probe (read-only);
    # gradient stays own-probe. int step or "neighbor"; reset per combo.
    _sp = steering.get("select_probe")
    steering_core.SELECT_PROBE = (int(_sp) if isinstance(_sp, (int, str)) and str(_sp).lstrip("-").isdigit()
                                  else (_sp or None))
    if steering_core.SELECT_PROBE is not None:
        logger.info("select_probe: %s", steering_core.SELECT_PROBE)

    if "aggressive_flow" in steering:
        steering_core.AGGRESSIVE_FLOW = bool(steering.get("aggressive_flow"))
        logger.info("aggressive_flow: %s", steering_core.AGGRESSIVE_FLOW)

    # WHERE the flow task acts. Reset on EVERY combo rather than
    # gated behind `in steering`: these are module globals, so a sweep arm that omits the key
    # would otherwise inherit the previous arm's setting and be silently mislabelled.
    _fi = steering.get("flow_iters", None)
    steering_core.FLOW_ITERS = None if _fi is None else [int(x) for x in _fi]
    steering_core.FLOW_AT_DENOISE = bool(steering.get("flow_at_denoise"))
    steering_core.FLOW_ON_TOTAL = bool(steering.get("flow_on_total"))
    steering_core.FLOW_ON_TOTAL_K0 = bool(steering.get("flow_on_total_k0"))
    steering_core.FLOW_ON_TOTAL_PER_LR = bool(steering.get("flow_on_total_per_lr"))
    if _fi is not None or steering_core.FLOW_AT_DENOISE or steering_core.FLOW_ON_TOTAL:
        logger.info("flow placement: iters=%s at_denoise=%s on_total=%s",
                    steering_core.FLOW_ITERS, steering_core.FLOW_AT_DENOISE,
                    steering_core.FLOW_ON_TOTAL)

    # All-steps probe-score logging (diagnostic). Config knob mirrors the STEER_LOG_ALL_STEPS
    # env flag so it travels through the sweep queue; when set it wins over the env default.
    # When on, every non-steered denoising step is also probe-scored (no-grad) so
    # steering_metrics.json carries the probe trajectory across ALL steps, not just steered ones.
    if "log_all_steps" in steering:
        steering_core.LOG_ALL_STEPS = bool(steering.get("log_all_steps"))
        logger.info("log_all_steps: %s", steering_core.LOG_ALL_STEPS)

    # Attention-feature capture (augmentation data / diagnostics). Config block
    #     attn_capture: {step: 7, after_updates: [4, 8], abort_after: true}
    # At that denoising step the probe's aggregated attention feature is written after
    # exactly k updates, one training-format extraction dir per k:
    #     <output_dir>/prompt_<idx>__seed_<s>__upd<k>/step_<step>.safetensors   (features__<word>, fp16)
    # plus <output_dir>/probe_config.yaml with the stored (pooled) geometry, so the dirs plug
    # straight into ProbeDataset. abort_after ends the trajectory there (no video). Travels
    # through the sweep queue like log_all_steps; off unless the block is present.
    attn_capture = steering.get("attn_capture") or None
    steering_core.ATTN_CAPTURE = dict(attn_capture) if attn_capture else None
    steering_core.ATTN_CAPTURE_SINK = None
    if attn_capture:
        logger.info("attn_capture: %s", attn_capture)

    steps = int(steering.get("sample_steps", 50))
    seed_default = int(steering.get("seed", 42))
    vh, vw = model_extraction_cfg.get("video_frame_height_width", [512, 512])
    vgs = float(steering.get("guidance_scale", model_extraction_cfg.get("video_guidance_scale", 5.0)))
    ags = float(model_extraction_cfg.get("audio_guidance_scale", 4.0))
    vneg = model_extraction_cfg.get("video_negative_prompt", "")
    frame_rate = float(model_extraction_cfg.get("frame_rate", 25.0))
    sample_rate = int(model_extraction_cfg.get("sample_rate", 16000))

    rows = load_prompts(prompts_csv)
    logger.info("loaded %d prompt(s) from %s; output → %s", len(rows), prompts_csv, output_dir)
    prompt_dirs: list[Path] = []
    for idx, raw_row in enumerate(rows):
        row = normalize_row(raw_row)
        if "prompt_idx" not in row:
            row["prompt_idx"] = idx
        prompt = row["prompt"]
        target_words = parse_target_words(row["target_words"], sep=row.get("_target_words_sep", ","))
        # Optional first-frame image conditioning (I2V). Absent/empty → pure T2V.
        conditioning_image = str(row.get("conditioning_image", "") or "").strip() or None
        seed_str = str(row.get("seed", "")).strip()
        seed = int(seed_str) if seed_str else seed_default

        out_dir = make_output_dir(output_dir, int(row["prompt_idx"]), seed=int(seed_str) if seed_str else None)

        metrics_log: list[dict] = []

        def _cb(step_idx: int, m: dict) -> None:
            metrics_log.append(m)
            if extra_step_callback is not None:
                extra_step_callback(step_idx, m)
            pb, pa = m.get("prob_before"), m.get("prob_after")
            logger.debug("    step %3d: prob %.4f → %.4f  (target=%s)",
                         step_idx, pb, pa, steering["objective"].get("target", 1.0))

        # Per-prompt hook for steering models that need prompt identity (e.g. the
        # TempoControl baseline's random-window control signal picks a fresh window
        # seeded by the prompt text). No-op for the probe steerer.
        if hasattr(steering_model, "set_prompt_context"):
            steering_model.set_prompt_context(prompt_idx=int(row["prompt_idx"]), prompt=prompt)

        # Per-object presence timeline (per_frame / timeline steering). Parsed from the
        # optional `timelines` CSV column; a target word without an entry defaults to
        # all-ones (present in every frame = the per-clip objective). No-op for steerers
        # without the hook (per_clip / baselines).
        if hasattr(steering_model, "set_prompt_timelines"):
            timelines = parse_timelines(row.get("timelines"))
            steering_model.set_prompt_timelines(timelines)
            row = {**row, "timelines_parsed": timelines}

        print(f"[steering] prompt_idx={row['prompt_idx']} seed={seed} "
              f"targets={target_words}: {prompt[:60]}")
        if attn_capture:
            steering_core.ATTN_CAPTURE_SINK = _make_attn_capture_sink(
                output_dir, out_dir.name, int(attn_capture["step"]), logger)
        if os.environ.get("STEER_DUMP_LATENTS", "0") not in ("0", "", "false"):
            _ldir = out_dir / "latents"
            _ldir.mkdir(parents=True, exist_ok=True)

            def _dump(step_idx, x_t, t, _ldir=_ldir):
                torch.save({"x_t": x_t.detach().to(torch.bfloat16).cpu(),
                            "t": None if t is None else float(t)}, _ldir / f"step_{step_idx:03d}.pt")
            steering_core.LATENT_DUMP_SINK = _dump

            # Steered steps: the latent each iteration was scored on (FIELD_HOOK fires before
            # every update; iter == n_updates is the post-optimization latent).
            def _field(step_idx, iter_idx, payload, _ldir=_ldir):
                torch.save({"x_t": payload["x_t"].to(torch.bfloat16).cpu(), "sigma": payload.get("sigma")},
                           _ldir / f"step_{step_idx:03d}_iter_{iter_idx:02d}.pt")
            steering_core.FIELD_HOOK = _field
        else:
            steering_core.LATENT_DUMP_SINK = None
        # Flow dump (steering.dump_flow): save the flow-matching velocity at every optimization
        # iteration (cond-only vel) and every denoising step (full-CFG vel) for later
        # manifold-distance analysis. fp16 tensors under <clip>/flows/.
        if steering.get("dump_flow"):
            _fdir = out_dir / "flows"
            _fdir.mkdir(parents=True, exist_ok=True)

            def _flow_dump(kind, step_idx, iter_idx, flow, _fdir=_fdir):
                nm = (f"denoise_step{step_idx:03d}.pt" if kind == "denoise"
                      else f"opt_step{step_idx:03d}_iter{iter_idx:02d}.pt")
                torch.save(flow.detach().to(torch.float16).cpu(), _fdir / nm)
            steering_core.FLOW_DUMP_SINK = _flow_dump
        else:
            steering_core.FLOW_DUMP_SINK = None
        aborted_after: int | None = None
        gen_started_unix = time.time()
        _gen_t0 = time.perf_counter()
        try:
            gen_video, gen_audio = backend.run_with_steering(
                prompt=prompt,
                video_negative_prompt=vneg,
                target_words=target_words,
                steering_model=steering_model,
                video_h=vh,
                video_w=vw,
                steps=steps,
                n_iters=n_iters,
                optimizer_name=optimizer_name,
                lr=lr,
                video_guidance_scale=vgs,
                audio_guidance_scale=ags,
                seed=seed,
                magnitude_reg=magnitude_reg,
                anchor_reg=anchor_reg,
                pred_reg=pred_reg,
                kl_reg=kl_reg,
                entropy_reg=entropy_reg,
                # I2V kwarg only when set: WAN/CogX run_with_steering don't accept it.
                **({"conditioning_image": conditioning_image} if conditioning_image else {}),
                # Attention-anchor trust region (LTX only): passed only when on, so WAN/CogX
                # run_with_steering signatures stay untouched.
                **({"attn_anchor_reg": attn_anchor_reg, "attn_anchor_tokens": attn_anchor_tokens,
                    "attn_anchor_ref": attn_anchor_ref} if attn_anchor_reg else {}),
                step_callback=_cb,
            )
        except steering_core.SteeringAborted as e:
            # Capture-only run: the trajectory was cut after the capture step — no video.
            aborted_after = int(e.args[0])
            gen_video, gen_audio = None, None
        finally:
            steering_core.ATTN_CAPTURE_SINK = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_duration_s = time.perf_counter() - _gen_t0

        media_stem = out_dir.name
        if gen_video is not None:
            mp4_path = out_dir / f"{media_stem}.mp4"
            backend.save_video(
                str(mp4_path),
                gen_video, gen_audio,
                fps=frame_rate, sample_rate=sample_rate, rank=device,
            )
        else:
            row = {**row, "aborted_after_step": aborted_after}
        # Video-only backends (e.g. WAN) return gen_audio=None — save_video already
        # handles that; only write a standalone WAV when there is audio.
        if gen_audio is not None:
            audio_int16 = (np.clip(gen_audio, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile_io.write(str(out_dir / f"{media_stem}.wav"), sample_rate, audio_int16)
        # Record the actual control window (random_window mode) for in-window fidelity.
        if hasattr(steering_model, "current_control"):
            cc = steering_model.current_control()
            if cc is not None:
                row = {**row, "control_window": cc}
        row = {**row, "gen_duration_s": round(gen_duration_s, 3),
               "gen_started_unix": round(gen_started_unix, 3)}
        (out_dir / "prompt_info.json").write_text(json.dumps(row, indent=2))
        (out_dir / "steering_metrics.json").write_text(json.dumps(metrics_log, indent=2))
        print(f"[steering] saved → {out_dir} (gen {gen_duration_s:.1f}s)")
        prompt_dirs.append(out_dir)

        torch.cuda.empty_cache()

    return {"n_prompts": len(rows), "prompt_dirs": prompt_dirs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="steering/steering.yaml")
    parser.add_argument("--prompts_csv", default=None, help="overrides steering.prompts_csv")
    parser.add_argument("--output_dir", default=None, help="overrides steering.output_dir")
    args = parser.parse_args()

    raw_cfg, steering, model_cfg, model_type, model_cfg_path, model_extraction_cfg = load_configs(args.config)

    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    verbose = bool(steering.get("verbose", False))
    logger = logging.getLogger("steering")
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    logger.debug("config=%s model_type=%s device=cuda:%d", args.config, model_type, device)
    logger.debug("steering knobs: steps_to_steer=%s optimization=%s objective=%s",
                 steering.get("steps_to_steer"), steering.get("optimization"), steering.get("objective"))

    backend = build_backend(model_type, model_cfg, model_extraction_cfg, steering, device, logger)

    output_dir = Path(args.output_dir or steering["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_steering(steering, backend, model_extraction_cfg, raw_cfg, model_cfg,
                 model_cfg_path, output_dir, device, logger, prompts_csv=args.prompts_csv)


if __name__ == "__main__":
    main()
