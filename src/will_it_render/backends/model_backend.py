"""Model backend abstraction for probe feature extraction.

Each backend wraps a specific generative model and exposes a single
``run_with_probes`` method that runs a full generation pass while collecting
per-step probe features.

To add a new model, subclass ``ModelBackend`` and register it in
``create_backend``'s dispatch table.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np
import torch

from will_it_render.common.probe_types import CaptureType

# Type alias for what a step callback receives — either a single merged tensor
# (per_word_features=False) or a dict of per-word tensors (per_word_features=True).
StepFeature = Union[torch.Tensor, Dict[str, torch.Tensor]]


class _PrecomputedNoiser:
    """Replay precomputed per-seed initial latents instead of sampling from the
    (GPU-architecture-dependent) CUDA RNG, so a given seed yields the identical
    video on any GPU.

    The LTX stage builds the video latent first, then the audio latent, calling
    this noiser once per modality with the same instance (see
    ``ltx_pipelines/utils/blocks.py``). We therefore return the stored
    ``[video_noise, audio_noise]`` in call order. The blend matches the original
    ``GaussianNoiser`` (``noise*mask + latent*(1-mask)``); for these stages the
    clean latent is zeros and the mask is all-ones, so the result is the raw noise.

    Generate the files with ``data_extraction/gen_seed_latents.py``.
    """

    def __init__(self, path: Path, seed: int, gen_hint: str) -> None:
        assert path.exists(), (
            f"Precomputed seed-latent file is missing for seed={seed}:\n  {path}\n"
            f"All seeds must be precomputed on the canonical GPU before extraction.\n"
            f"Generate it (on ctxa100-01) with:\n  {gen_hint}"
        )
        # weights_only=False: our own precomputed seed-latent file (trusted); it carries
        # a torch TorchVersion tag that the PyTorch>=2.6 weights_only=True default rejects.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self._noises = [blob["video_noise"], blob["audio_noise"]]
        self._i = 0
        self._seed = seed
        self._path = path

    def __call__(self, latent_state, noise_scale: float = 1.0):
        from dataclasses import replace as _replace

        assert self._i < len(self._noises), (
            f"_PrecomputedNoiser for seed={self._seed} called {self._i + 1} times but only "
            f"{len(self._noises)} latents are stored (expected 2: video then audio). "
            f"File: {self._path}"
        )
        noise = self._noises[self._i].to(latent_state.latent.device, latent_state.latent.dtype)
        assert noise.shape == latent_state.latent.shape, (
            f"Seed-latent shape {tuple(noise.shape)} != expected "
            f"{tuple(latent_state.latent.shape)} (modality #{self._i}) for seed={self._seed}. "
            f"Resolution/frames likely changed — regenerate. File: {self._path}"
        )
        self._i += 1
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
        return _replace(latent_state, latent=latent.to(latent_state.latent.dtype))


@contextmanager
def _cpu_pinned_randn():
    """Force ``torch.randn`` calls that carry a ``generator`` to draw on the CPU
    (host RNG is identical across GPU architectures) and move the result to the
    requested device. This makes the VAE decode's ``inject_noise`` draws
    reproducible across GPUs. The patch is scoped to the ``with`` block and
    restored via ``try/finally``; inference is single-threaded so the global swap
    is safe here.

    Callers must pass a CPU ``torch.Generator`` (a CUDA generator cannot draw a
    CPU tensor).
    """
    _orig = torch.randn

    def _patched(*args, **kwargs):
        gen = kwargs.get("generator")
        dev = kwargs.get("device")
        if gen is not None and dev is not None and torch.device(dev).type == "cuda":
            dtype = kwargs.pop("dtype", None)
            kwargs.pop("device")
            out = _orig(*args, **kwargs)  # drawn on CPU with the CPU generator
            return out.to(device=dev, dtype=dtype) if dtype is not None else out.to(device=dev)
        return _orig(*args, **kwargs)

    torch.randn = _patched
    try:
        yield
    finally:
        torch.randn = _orig


class ModelBackend(ABC):
    @abstractmethod
    def run_with_probes(
        self,
        prompt: str,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        target_words: list[str],
        hook_type: CaptureType,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
        step_callback: Callable[[int, StepFeature], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list, list[int], Dict[str, List[int]], dict]:
        """Run generation with probe hooks.

        When ``step_callback`` is provided it is called with (step_idx, feature)
        after each denoising step and features are NOT accumulated in memory.
        ``feature`` is a flat Tensor when per_word_features=False, or a
        Dict[word, Tensor] when per_word_features=True.

        Returns:
            generated_video:  ndarray [C, F, H, W]
            generated_audio:  ndarray [T]
            step_features:    list of features (empty when step_callback used)
            step_indices:     corresponding denoising step indices
            token_index_map:  {word: [token_indices]} for the tokenized prompt
            probe_shape:      dict with n_blocks, n_heads, n_blocks_heads, Lq, latent_hw
        """

    @abstractmethod
    def run_generation_only(
        self,
        prompt: str,
        video_negative_prompt: str,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run generation without any probe hooks."""

    @abstractmethod
    def run_with_steering(
        self,
        prompt: str,
        video_negative_prompt: str,
        target_words: list[str],
        steering_model: "object",
        video_h: int,
        video_w: int,
        steps: int,
        n_iters: int,
        optimizer_name: str,
        lr: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        seed: int,
        magnitude_reg: float = 0.0,
        step_callback: Callable[[int, dict], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inference-time attention steering.

        Runs generation while, at each denoising step the ``steering_model`` opts
        into, optimizing the current-step latent ``x_t`` for ``n_iters`` iterations
        to minimize a loss the model computes from the live cross-attention maps
        (gradients are backpropagated to ``x_t``). ``step_callback(step_idx, metrics)``
        receives per-step steering metrics (loss, probe probability).

        Returns ``(generated_video, generated_audio)``.
        """

    @abstractmethod
    def save_video(
        self,
        path: str,
        video: np.ndarray,
        audio: np.ndarray,
        fps: float,
        sample_rate: int,
        rank: int,
    ) -> None:
        """Save generated video and audio to *path* using the backend's environment."""


# ``build_agg_kwargs`` hoisted to ``common.agg`` (WILL_IT_RENDER_PLAN.md §10b R1) so the
# steering path can derive agg flags without importing this backend module. Re-exported
# under both the public name and the historical private ``_build_agg_kwargs`` alias so
# existing callers (here and in steering/) keep working.
from will_it_render.common.agg import build_agg_kwargs  # noqa: E402

_build_agg_kwargs = build_agg_kwargs


class OviBackend(ModelBackend):
    """Backend for the Ovi audio-video fusion model."""

    def __init__(
        self,
        engine_cfg,
        model_cfg: dict,
        experiment_cfg: dict,
        device: int,
        logger: logging.Logger,
    ):
        from ovi.ovi_fusion_engine import OviFusionEngine

        self._engine = OviFusionEngine(config=engine_cfg, device=device, logger=logger)
        self._model_cfg = model_cfg
        self._experiment_cfg = experiment_cfg
        raw_seq_lens: dict = model_cfg.get("capture_type_seq_lens", {})
        self._capture_type_seq_lens: dict[CaptureType, int] = {
            CaptureType(k): int(v) for k, v in raw_seq_lens.items()
        }

    def run_with_probes(
        self,
        prompt: str,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        target_words: list[str],
        hook_type: CaptureType,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
        step_callback: Callable[[int, StepFeature], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list, list[int], Dict[str, List[int]]]:
        from will_it_render.common.aggregator import (
            aggregate_captures_per_word,
            aggregate_captures_to_vector,
            probe_shape_from_captures,
        )
        engine = self._engine
        cfg = self._model_cfg

        # The exact word→token mapping (build_token_index_map) requires a fast
        # tokenizer's character offset_mapping. Ovi's text path
        # (engine.text_model) does not expose offsets here, so per-word extraction
        # is not wired for Ovi yet. Implement an offset source before enabling it —
        # the previous substring-based matcher was removed because it mis-assigned
        # article tokens to concepts and dropped punctuation-glued tokens.
        raise NotImplementedError(
            "Ovi backend: exact offset-based token mapping is not yet wired. "
            "Provide a return_offsets_mapping-capable tokenizer for the Ovi text "
            "encoder and call build_token_index_map(prompt, target_words, offsets)."
        )

        per_word = bool(cfg.get("per_word_features", False))  # noqa: F841  (kept for when Ovi is wired)
        agg_kwargs = _build_agg_kwargs(cfg)
        capture_every: int = cfg.get("capture_every_n_steps", 1)
        first_step: int = cfg.get("first_step_to_capture", 0)
        capture_steps = cfg.get("capture_steps", None)

        step_features: list = []
        step_indices: list[int] = []
        _probe_shape: dict = {}

        capture_step_set = set(capture_steps) if capture_steps is not None else None

        def step_probe_fn(step_idx: int, all_captures) -> None:
            if capture_step_set is not None:
                if step_idx not in capture_step_set:
                    return
            else:
                is_last = step_idx == steps - 1
                after_first = step_idx >= first_step
                on_schedule = (step_idx - first_step) % capture_every == 0
                if not (is_last or (after_first and on_schedule)):
                    return
            typed = [c for c in all_captures if c.capture_type == hook_type]
            if not _probe_shape and typed:
                _probe_shape.update(probe_shape_from_captures(typed, **agg_kwargs))
            if per_word:
                vec = {w: t.cpu() for w, t in
                       aggregate_captures_per_word(typed, token_index_map, **agg_kwargs).items()}
            else:
                vec = aggregate_captures_to_vector(typed, target_indices, **agg_kwargs).cpu()
            if step_callback is not None:
                step_callback(step_idx, vec)
            else:
                step_features.append(vec)
                step_indices.append(step_idx)

        gen_video, gen_audio, *_ = engine.generate(
            text_prompt=prompt,
            video_frame_height_width=[video_h, video_w],
            seed=seed,
            solver_name=solver,
            sample_steps=steps,
            shift=shift,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            slg_layer=slg_layer,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
            step_probe_fn=step_probe_fn,
            capture_type_seq_lens=self._capture_type_seq_lens,
        )

        return gen_video, gen_audio, step_features, step_indices, token_index_map, _probe_shape

    def run_generation_only(
        self,
        prompt: str,
        video_negative_prompt: str,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        gen_video, gen_audio, *_ = self._engine.generate(
            text_prompt=prompt,
            video_frame_height_width=[video_h, video_w],
            seed=seed,
            solver_name=solver,
            sample_steps=steps,
            shift=shift,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            slg_layer=slg_layer,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt="",
        )
        return gen_video, gen_audio

    def run_with_steering(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        # Steering reuses the cross-attention capture hooks, which are not yet wired
        # for Ovi (its run_with_probes is also NotImplemented — no offset-based token
        # mapping). Implement the Ovi probe path first, then mirror LTXBackend here.
        raise NotImplementedError(
            "Ovi backend: run_with_steering is not implemented yet. Wire the Ovi "
            "cross-attention capture + token mapping (see run_with_probes) first."
        )

    def save_video(
        self,
        path: str,
        video: np.ndarray,
        audio: np.ndarray,
        fps: float,
        sample_rate: int,
        rank: int,
    ) -> None:
        from ovi.utils.io_utils import save_video as ovi_save_video

        ovi_save_video(path, video, audio, sample_rate=sample_rate, fps=fps, rank=rank)


class LTXBackend(ModelBackend):
    """Backend for the LTX-2 video model."""

    def __init__(
        self,
        engine_cfg,
        model_cfg: dict,
        experiment_cfg: dict,
        device: int,
        logger: logging.Logger,
    ) -> None:
        from will_it_render.common import paths

        # Resolve the LTX-2 checkout from VP_LTX_ROOT (or engine_cfg.ltx_root). The old
        # parents[3] directory-walk broke once this module moved under src/ (plan §10b R5).
        _LTX_ROOT = paths.ltx_root(getattr(engine_cfg, "ltx_root", None))
        for pkg in ("ltx-core", "ltx-pipelines"):
            paths.add_to_syspath(_LTX_ROOT / "packages" / pkg / "src")

        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
        from ltx_pipelines.utils.blocks import AudioConditioner

        self._device = torch.device(f"cuda:{device}")
        self._logger = logger
        self._model_cfg = model_cfg
        self._experiment_cfg = experiment_cfg
        # Model identity for the precomputed seed-latent path (checkpoint filename
        # stem, e.g. "ltx-2.3-22b-distilled-1.1"). Other backends have other shapes.
        self._model_name = Path(str(engine_cfg.checkpoint_path)).stem

        self._pipeline = TI2VidOneStagePipeline(
            checkpoint_path=engine_cfg.checkpoint_path,
            gemma_root=engine_cfg.gemma_root,
            loras=[],
            device=self._device,
        )
        self._tokenizer = LTXVGemmaTokenizer(engine_cfg.gemma_root)
        # Only needed for run_with_steering_from_video (VAE-encodes a source video's
        # audio track); TI2VidOneStagePipeline has no audio_conditioner of its own
        # (unlike RetakePipeline), so it's built the same way here.
        self._audio_conditioner = AudioConditioner(
            checkpoint_path=engine_cfg.checkpoint_path,
            dtype=self._pipeline.dtype,
            device=self._device,
        )

    def _seed_latent_path(self, seed: int, video_h: int, video_w: int) -> Path:
        """Path to the precomputed initial-latent file for ``seed``. Mirrors the
        path construction in ``data_extraction/gen_seed_latents.py`` (model name +
        resolution/fps + gpu tag); both must stay in sync."""
        cfg = self._model_cfg
        root = cfg.get("seed_latents_dir") or self._experiment_cfg.get("seed_latents_dir")
        assert root, (
            "extraction.seed_latents_dir is not set — add it to configs/models/ltx.yaml. "
            "Precomputed per-seed latents are required (seed noise is GPU-dependent)."
        )
        frames = int(cfg.get("num_frames", 97))
        fps = float(cfg.get("frame_rate", 25.0))
        tag = cfg.get("seed_latents_gpu_tag", "ctxa100")
        res = f"{video_h}x{video_w}x{frames}_{fps:g}fps"
        return Path(root) / self._model_name / res / f"seed_{int(seed):05d}__{tag}.pt"

    def _seed_latent_gen_hint(self, seed: int, video_h: int, video_w: int) -> str:
        cfg = self._model_cfg
        frames = int(cfg.get("num_frames", 97))
        fps = float(cfg.get("frame_rate", 25.0))
        tag = cfg.get("seed_latents_gpu_tag", "ctxa100")
        return (
            f"python -m data_extraction.gen_seed_latents --seeds {int(seed)} "
            f"--model-name {self._model_name} --height {video_h} --width {video_w} "
            f"--frames {frames} --fps {fps:g} --gpu-tag {tag}"
        )

    def _build_image_conditionings(
        self,
        conditioning_image: str | None,
        video_h: int,
        video_w: int,
    ):
        """First-frame image conditioning for I2V. Returns ``[]`` when no image
        is given (pure T2V — the historical behaviour).

        Replaces the latent at frame 0 with the VAE-encoded input image
        (``VideoConditionByLatentIndex``, ``latent_idx=0``) — the exact path
        LTX's ti2vid CLI uses. The image is resized/cropped to ``(video_h,
        video_w)`` by ``load_image_and_preprocess`` inside the combiner.

        Built under ``torch.no_grad()`` (not ``inference_mode``) so the produced
        conditioning latents are ordinary tensors: the steering path runs a
        grad-tracked ``stage`` forward that consumes them, and inference-mode
        tensors cannot enter autograd (same reason the text contexts are cloned
        in ``_build_pipeline_inputs``)."""
        if not conditioning_image:
            return []
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.helpers import combined_image_conditionings

        images = [ImageConditioningInput(path=str(conditioning_image), frame_idx=0, strength=1.0)]
        with torch.no_grad():
            return self._pipeline.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=images,
                    height=video_h,
                    width=video_w,
                    video_encoder=enc,
                    dtype=self._pipeline.dtype,
                    device=self._device,
                )
            )

    def _build_pipeline_inputs(
        self,
        prompt: str,
        video_negative_prompt: str,
        video_h: int,
        video_w: int,
        steps: int,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        seed: int,
        grad: bool = False,
        skip_noiser: bool = False,
    ):
        """Build text contexts, scheduler sigmas, denoiser and noiser.

        ``grad=False`` (default, extraction/generation): runs under
        ``torch.inference_mode()`` exactly as before. ``grad=True`` (steering):
        runs under ``torch.no_grad()`` instead, and *clones* the text-context
        tensors so they are not inference tensors — the Gemma encoder produces
        them under its own ``inference_mode``, and inference tensors cannot be
        saved for backward when they are later consumed as cross-attention keys
        in a grad-tracked forward.

        ``skip_noiser=True`` (video-edit steering only): the caller discards the
        returned noiser and builds its own seeded ``GaussianNoiser`` for the
        real video's latent, so skip constructing ``_PrecomputedNoiser`` — its
        __init__ asserts a canonical seed-latent file exists on disk, which is
        both unnecessary work and a false dependency here (video-edit seeds are
        real dataset seeds, not necessarily ones with a precomputed pure-noise
        latent).
        """
        from ltx_core.components.guiders import (
            MultiModalGuiderParams,
            create_multimodal_guider_factory,
        )
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser

        def _materialize(t):
            # Clone out of the encoder's inference-mode so the tensor can enter autograd.
            return t.clone() if (grad and torch.is_tensor(t)) else t

        pipeline = self._pipeline
        negative_prompt = video_negative_prompt or ""
        build_ctx = torch.no_grad() if grad else torch.inference_mode()
        with build_ctx:
            ctx_p, ctx_n = pipeline.prompt_encoder([prompt, negative_prompt])
            v_context_p = _materialize(ctx_p.video_encoding)
            a_context_p = _materialize(ctx_p.audio_encoding)
            v_context_n = _materialize(ctx_n.video_encoding)
            a_context_n = _materialize(ctx_n.audio_encoding)

            sigmas = LTX2Scheduler().execute(steps=steps).to(dtype=torch.float32, device=self._device)

            # LTX's oversaturation/quality knobs. LTX's own presets set these
            # (video: cfg=3, stg=1, rescale=0.7 in ltx_pipelines.utils.constants), but that
            # preset layer only runs in the official ti2vid_* CLIs — our custom denoising loop
            # builds the guider directly, so anything not passed here falls to the neutral
            # dataclass defaults (rescale=0, stg=0). We expose them via the model config so a
            # run can match LTX's look; defaults preserve historical behaviour (rescale/stg off).
            #   rescale_scale — CFG-rescale (Lin et al.); pure std-match on the guided pred,
            #                   NO extra forward pass. This is the real anti-oversaturation lever.
            #   stg_scale     — spatiotemporal guidance; needs a PERTURBED forward (uncond_perturbed)
            #                   our probe/steering loop does not run, so keep 0 unless that path
            #                   is added. Exposed for completeness / non-probe generation.
            mc = self._model_cfg
            def _guider_params(cfg_scale, prefix):
                return MultiModalGuiderParams(
                    cfg_scale=cfg_scale,
                    stg_scale=float(mc.get(f"{prefix}_stg_scale", 0.0) or 0.0),
                    stg_blocks=list(mc.get(f"{prefix}_stg_blocks") or []),
                    rescale_scale=float(mc.get(f"{prefix}_rescale_scale", 0.0) or 0.0),
                )
            video_guider_factory = create_multimodal_guider_factory(
                params=_guider_params(video_guidance_scale, "video"),
                negative_context=v_context_n,
            )
            audio_guider_factory = create_multimodal_guider_factory(
                params=_guider_params(audio_guidance_scale, "audio"),
                negative_context=a_context_n,
            )
            denoiser = FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
            )
            base_gen = torch.Generator(device=self._device).manual_seed(seed)
            # --- GPU-noise experiment harness (off unless LTX_SAVE_LATENT/LTX_LOAD_LATENT set) ---
            # Tests whether cross-GPU video divergence originates at seed→noise (the CUDA RNG's
            # parallel layout is arch-dependent, so torch.randn(generator) can differ across GPU
            # types) or in the denoising matmuls. SAVE captures the sampled noise on one GPU;
            # LOAD replays it on another so both start from the *identical* initial latent.
            import os as _os
            # When LTX_NOISE_KEY is set (run_extraction sets it per prompt), treat the env
            # paths as DIRECTORIES and key the noise file per prompt → one run, many prompts.
            _key = _os.environ.get("LTX_NOISE_KEY")
            def _resolve(_p):
                if _p and _key:
                    _os.makedirs(_p, exist_ok=True)
                    return _os.path.join(_p, _key + ".pt")
                return _p
            _save_p = _resolve(_os.environ.get("LTX_SAVE_LATENT"))
            _load_p = _resolve(_os.environ.get("LTX_LOAD_LATENT"))
            if skip_noiser:
                noiser = None
            elif not (_save_p or _load_p):
                # Default (mandatory): replay the canonical, GPU-independent per-seed
                # initial latents precomputed on ctxa100. Seed noise is GPU-architecture
                # dependent (see seed_gpu_analysis/), so sampling here would make the same
                # seed produce a different video per GPU. Asserts if the seed is missing.
                noiser = _PrecomputedNoiser(
                    self._seed_latent_path(seed, video_h, video_w),
                    seed,
                    self._seed_latent_gen_hint(seed, video_h, video_w),
                )
            else:
                from dataclasses import replace as _replace

                class _SaveLoadNoiser:
                    def __init__(self, gen):
                        self.generator = gen
                        self._i = 0
                        self._saved: list = []
                        self._loaded = torch.load(_load_p, map_location="cpu") if _load_p else None

                    def __call__(self, latent_state, noise_scale: float = 1.0):
                        if self._loaded is not None:
                            noise = self._loaded[self._i].to(
                                latent_state.latent.device, latent_state.latent.dtype)
                        else:
                            noise = torch.randn(
                                *latent_state.latent.shape,
                                device=latent_state.latent.device,
                                dtype=latent_state.latent.dtype,
                                generator=self.generator,
                            )
                            if _save_p:
                                self._saved.append(noise.detach().to("cpu"))
                                torch.save(self._saved, _save_p)
                        self._i += 1
                        scaled_mask = latent_state.denoise_mask * noise_scale
                        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
                        return _replace(latent_state, latent=latent.to(latent_state.latent.dtype))

                noiser = _SaveLoadNoiser(base_gen)
                print(f"[noise-exp] {'LOAD ' + _load_p if _load_p else 'SAVE ' + _save_p}", flush=True)

        return v_context_p, a_context_p, sigmas, denoiser, noiser

    @torch.inference_mode()
    def run_with_probes(
        self,
        prompt: str,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        target_words: list[str],
        hook_type: CaptureType,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
        conditioning_image: str | None = None,
        step_callback: Callable[[int, StepFeature], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list, list[int], Dict[str, List[int]], dict]:
        from ltx_pipelines.utils.types import ModalitySpec

        from will_it_render.common.aggregator import (
            aggregate_captures_per_word,
            aggregate_captures_to_vector,
            probe_shape_from_captures,
        )
        from will_it_render.backends.hooks.ltx_hooks import make_ltx_extraction_loop
        from will_it_render.common.token_mapper import build_token_index_map

        cfg = self._model_cfg
        num_frames: int = cfg.get("num_frames", 97)
        frame_rate: float = cfg.get("frame_rate", 25.0)

        encoded = self._tokenizer.tokenizer(
            prompt.strip(),
            padding=False,
            truncation=True,
            max_length=self._tokenizer.max_length,
            return_offsets_mapping=True,
        )
        token_index_map = build_token_index_map(
            prompt.strip(), target_words, encoded["offset_mapping"]
        )
        target_indices = sorted({idx for idxs in token_index_map.values() for idx in idxs})

        per_word = bool(cfg.get("per_word_features", False))
        agg_kwargs = _build_agg_kwargs(cfg)
        capture_every: int = cfg.get("capture_every_n_steps", 1)
        first_step: int = cfg.get("first_step_to_capture", 0)
        capture_steps = cfg.get("capture_steps", None)
        capture_blocks = cfg.get("capture_blocks", None)
        if capture_blocks is not None:
            capture_blocks = sorted(int(b) for b in capture_blocks)
        block_meta: dict = {}

        v_context_p, a_context_p, sigmas, denoiser, noiser = self._build_pipeline_inputs(
            prompt, video_negative_prompt, video_h, video_w, steps,
            video_guidance_scale, audio_guidance_scale, seed,
        )

        step_features: list = []
        step_indices: list[int] = []
        _probe_shape: dict = {}

        def step_probe_fn(step_idx: int, captures) -> None:
            typed = [c for c in captures if c.capture_type == CaptureType.VIDEO_CA]
            if not _probe_shape and typed:
                _probe_shape.update(probe_shape_from_captures(typed, **agg_kwargs))
            if per_word:
                vec = {w: t.cpu() for w, t in
                       aggregate_captures_per_word(typed, token_index_map, **agg_kwargs).items()}
            else:
                vec = aggregate_captures_to_vector(typed, target_indices, **agg_kwargs).cpu()
            if step_callback is not None:
                step_callback(step_idx, vec)
            else:
                step_features.append(vec)
                step_indices.append(step_idx)

        sf = cfg.get("latent_scale_factors", {})
        if sf:
            lat_t = int(sf.get("time", 8))
            lat_h = int(sf.get("height", 32))
            lat_w = int(sf.get("width", 32))
            lat_frames = (num_frames - 1) // lat_t + 1
            video_seq_len: int | None = lat_frames * (video_h // lat_h) * (video_w // lat_w)
        else:
            video_seq_len = None

        extraction_loop = make_ltx_extraction_loop(
            step_probe_fn,
            desc=f"Extracting [{prompt[:35]}]",
            capture_every=capture_every,
            first_step_to_capture=first_step,
            capture_steps=capture_steps,
            capture_blocks=capture_blocks,
            block_meta=block_meta,
            video_seq_len=video_seq_len,
        )

        video_state, audio_state = self._pipeline.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=video_w,
            height=video_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=self._build_image_conditionings(conditioning_image, video_h, video_w),
            ),
            audio=ModalitySpec(context=a_context_p),
            max_batch_size=1,
            loop=extraction_loop,
        )
        del v_context_p, a_context_p, sigmas, denoiser, noiser
        torch.cuda.empty_cache()

        gen_video = self._decode_video(video_state, seed)
        gen_audio = self._decode_audio(audio_state)
        del video_state, audio_state

        # Record which transformer blocks were actually captured so downstream can
        # restore their original indices (the stored tensor holds only these blocks).
        if capture_blocks is not None and _probe_shape:
            _probe_shape["capture_blocks"] = capture_blocks
            _probe_shape["n_blocks_total"] = block_meta.get("n_blocks_total")

        return gen_video, gen_audio, step_features, step_indices, token_index_map, _probe_shape

    @torch.inference_mode()
    def run_generation_only(
        self,
        prompt: str,
        video_negative_prompt: str,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
        conditioning_image: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from ltx_pipelines.utils.types import ModalitySpec

        from will_it_render.backends.hooks.ltx_hooks import make_ltx_extraction_loop

        cfg = self._model_cfg
        num_frames: int = cfg.get("num_frames", 97)
        frame_rate: float = cfg.get("frame_rate", 25.0)

        v_context_p, a_context_p, sigmas, denoiser, noiser = self._build_pipeline_inputs(
            prompt, video_negative_prompt, video_h, video_w, steps,
            video_guidance_scale, audio_guidance_scale, seed,
        )

        generation_loop = make_ltx_extraction_loop(
            None, desc=f"Generating [{prompt[:35]}]"
        )

        video_state, audio_state = self._pipeline.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=video_w,
            height=video_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=self._build_image_conditionings(conditioning_image, video_h, video_w),
            ),
            audio=ModalitySpec(context=a_context_p),
            max_batch_size=1,
            loop=generation_loop,
        )
        del v_context_p, a_context_p, sigmas, denoiser, noiser
        torch.cuda.empty_cache()

        gen_video = self._decode_video(video_state, seed)
        gen_audio = self._decode_audio(audio_state)
        del video_state, audio_state

        return gen_video, gen_audio

    def run_with_steering(
        self,
        prompt: str,
        video_negative_prompt: str,
        target_words: list[str],
        steering_model: "object",
        video_h: int,
        video_w: int,
        steps: int,
        n_iters: int,
        optimizer_name: str,
        lr: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        seed: int,
        magnitude_reg: float = 0.0,
        anchor_reg: float = 0.0,
        pred_reg: float = 0.0,
        kl_reg: float = 0.0,
        entropy_reg: float = 0.0,
        attn_anchor_reg: float = 0.0,
        attn_anchor_tokens: str = "prompt",
        attn_anchor_ref: str = "step",
        conditioning_image: str | None = None,
        step_callback: Callable[[int, dict], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from ltx_core.components.guiders import (
            MultiModalGuiderParams,
            create_multimodal_guider_factory,
        )
        from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
        from ltx_pipelines.utils.types import ModalitySpec

        from will_it_render.common.token_mapper import build_token_index_map
        from will_it_render.inference.ltx_adapter import make_ltx_steering_loop

        cfg = self._model_cfg
        num_frames: int = cfg.get("num_frames", 97)
        frame_rate: float = cfg.get("frame_rate", 25.0)

        # target word → token indices (same path as run_with_probes).
        encoded = self._tokenizer.tokenizer(
            prompt.strip(),
            padding=False,
            truncation=True,
            max_length=self._tokenizer.max_length,
            return_offsets_mapping=True,
        )
        token_index_map = build_token_index_map(
            prompt.strip(), target_words, encoded["offset_mapping"]
        )
        # Per-word steering: keep each word's FULL subword-token index list (e.g.
        # "teddy-bear" → teddy+bear tokens) so the loss can pool + score each word
        # separately, mirroring how training built one feature per word. Drop words
        # that matched no tokens so they don't contribute a zeros-pass to the average.
        target_words_indices = {w: idxs for w, idxs in token_index_map.items() if idxs}
        if not target_words_indices:
            # Nothing matched (e.g. typo'd target). Fall back to a single all-token
            # pass so generation still runs, rather than steering on nothing.
            all_idx = sorted({idx for idxs in token_index_map.values() for idx in idxs})
            self._logger.warning(
                "no target word matched any prompt token (target_words=%s); "
                "falling back to a single combined-target pass", target_words,
            )
            target_words_indices = {",".join(target_words): all_idx}

        # Baselines that re-normalize the attention over the *text* axis (Attend-and-Excite
        # softmaxes over the prompt tokens) need to know which key positions hold real
        # tokens: the attention key axis is padded to the encoder's max length, so a
        # softmax over all of Lk would be diluted by padding. Special tokens (BOS/EOS/pad)
        # carry an empty offset span, which is exactly the slice A&E takes by index
        # arithmetic — derived here instead, so it holds for any tokenizer.
        if hasattr(steering_model, "set_text_token_indices"):
            steering_model.set_text_token_indices(
                [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > a]
            )

        # grad=True: contexts cloned out of inference-mode; no inference_mode decorator
        # here so the per-step optimization forward can build an autograd graph to x_t.
        v_context_p, a_context_p, sigmas, denoiser, noiser = self._build_pipeline_inputs(
            prompt, video_negative_prompt, video_h, video_w, steps,
            video_guidance_scale, audio_guidance_scale, seed, grad=True,
        )

        # Positive-only (cfg=1.0) denoiser for the optimization forward: it skips the
        # uncond pass, ~halving grad-forward memory. This is exact for steering — the
        # probe was trained on the COND-pass attention, which is identical regardless
        # of CFG scale (CFG only blends cond/uncond *after* the transformer). The real
        # CFG `denoiser` is still used for the actual denoise + scheduler step.
        opt_v_factory = create_multimodal_guider_factory(params=MultiModalGuiderParams(cfg_scale=1.0))
        opt_a_factory = create_multimodal_guider_factory(params=MultiModalGuiderParams(cfg_scale=1.0))
        opt_denoiser = FactoryGuidedDenoiser(
            v_context=v_context_p,
            a_context=a_context_p,
            video_guider_factory=opt_v_factory,
            audio_guider_factory=opt_a_factory,
        )

        sf = cfg.get("latent_scale_factors", {})
        if sf:
            lat_t = int(sf.get("time", 8))
            lat_h = int(sf.get("height", 32))
            lat_w = int(sf.get("width", 32))
            lat_frames = (num_frames - 1) // lat_t + 1
            video_seq_len: int | None = lat_frames * (video_h // lat_h) * (video_w // lat_w)
        else:
            video_seq_len = None

        steering_loop = make_ltx_steering_loop(
            steering_model=steering_model,
            target_words_indices=target_words_indices,
            n_iters=n_iters,
            optimizer_name=optimizer_name,
            lr=lr,
            video_seq_len=video_seq_len,
            capture_blocks=getattr(steering_model, "capture_blocks", None),
            opt_denoiser=opt_denoiser,
            magnitude_reg=magnitude_reg,
            anchor_reg=anchor_reg,
            pred_reg=pred_reg,
            kl_reg=kl_reg,
            entropy_reg=entropy_reg,
            attn_anchor_reg=attn_anchor_reg,
            attn_anchor_tokens=attn_anchor_tokens,
            attn_anchor_ref=attn_anchor_ref,
            # Real (un-padded) prompt-token count — the attention key axis is padded to
            # the encoder's max length, and the attention-anchor term's "prompt" mode
            # needs to know where the real tokens end.
            text_token_count=len(encoded["offset_mapping"]),
            step_callback=step_callback,
            desc=f"Steering [{prompt[:30]}]",
        )

        video_state, audio_state = self._pipeline.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=video_w,
            height=video_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=self._build_image_conditionings(conditioning_image, video_h, video_w),
            ),
            audio=ModalitySpec(context=a_context_p),
            max_batch_size=1,
            loop=steering_loop,
        )
        del v_context_p, a_context_p, sigmas, denoiser, noiser
        torch.cuda.empty_cache()

        gen_video = self._decode_video(video_state, seed)
        gen_audio = self._decode_audio(audio_state)
        del video_state, audio_state

        return gen_video, gen_audio

    def run_with_steering_from_video(
        self,
        video_path: str,
        prompt: str,
        video_negative_prompt: str,
        target_words: list[str],
        steering_model: "object",
        noise_strength: float,
        video_h: int,
        video_w: int,
        steps: int,
        n_iters: int,
        optimizer_name: str,
        lr: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        seed: int,
        magnitude_reg: float = 0.0,
        anchor_reg: float = 0.0,
        pred_reg: float = 0.0,
        kl_reg: float = 0.0,
        entropy_reg: float = 0.0,
        source_reg: float = 0.0,
        source_reg_every_step: float = 0.0,
        step_callback: Callable[[int, dict], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Object-removal / video-editing steering (SDEdit-style).

        VAE-encodes an existing ``video_path`` (+ its audio track) to a latent, blends
        in Gaussian noise up to ``noise_strength`` (the flow-matching sigma to start
        from — 1.0 reproduces from-scratch generation; smaller values keep more of the
        source and only re-denoise the tail of the schedule), then runs the same
        probe-gradient steering loop as ``run_with_steering`` starting mid-schedule on
        that noised real latent instead of from scratch on pure noise. Pass
        ``objective.target: 0.0`` on ``steering_model`` to suppress ``target_words``
        (object removal) instead of inducing them.

        Under LTX-2's linear flow-matching parameterization, ``GaussianNoiser``'s
        ``noise*mask*noise_scale + latent*(1-mask*noise_scale)`` blend IS the forward
        noising process at sigma=``noise_scale`` — so setting ``noise_scale`` to the
        sigma value the truncated schedule starts at reproduces exactly the noise
        level the (un-truncated) schedule would have had at that step, and the
        remaining steps denoise it back out consistently.

        Requires the source video to have an audio track (this pipeline's own
        extraction/generation outputs always mux one): the diffusion stage shares a
        single ``sigmas`` schedule across both modalities, so audio must be noised the
        same way as video rather than generated from scratch under a truncated
        video-only schedule.

        Uses a plain seeded ``GaussianNoiser`` (not the deterministic
        ``_PrecomputedNoiser`` used for from-scratch generation, which only replays
        precomputed *pure-noise* seed latents) — steered edits from this method are
        therefore not guaranteed bit-exact across GPU architectures, unlike
        ``run_with_steering``.

        ``source_reg`` / ``source_reg_every_step`` anchor the steered latent to the
        TRUE source video (as opposed to ``anchor_reg``, which only bounds drift
        from one steered step to the next) — see ``steering.steering_loop`` for the
        exact mechanism. Both need the noise sample the initial re-noising drew, so
        the noiser passed to ``pipeline.stage`` here is wrapped to record it into
        ``source_noise_sink`` before the loop's first step runs. Returns a third
        element, ``source_trace``: a list of per-denoising-step
        ``{step, did_steer, source_gap_before, source_gap_after}`` dicts logging the
        (weight-free) MSE to the source trajectory at every step, whether or not
        either knob is actually nonzero — the caller can dump it for diagnostics
        even when only measuring drift, not correcting it.
        """
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.helpers import audio_latent_from_file, video_latent_from_file
        from ltx_pipelines.utils.types import ModalitySpec

        from will_it_render.common.token_mapper import build_token_index_map
        from will_it_render.inference.ltx_adapter import make_ltx_steering_loop

        assert 0.0 < noise_strength <= 1.0, f"noise_strength must be in (0, 1], got {noise_strength}"

        cfg = self._model_cfg
        num_frames: int = cfg.get("num_frames", 97)
        frame_rate: float = cfg.get("frame_rate", 25.0)

        # target word -> token indices (identical path to run_with_steering).
        encoded = self._tokenizer.tokenizer(
            prompt.strip(),
            padding=False,
            truncation=True,
            max_length=self._tokenizer.max_length,
            return_offsets_mapping=True,
        )
        token_index_map = build_token_index_map(
            prompt.strip(), target_words, encoded["offset_mapping"]
        )
        target_words_indices = {w: idxs for w, idxs in token_index_map.items() if idxs}
        if not target_words_indices:
            all_idx = sorted({idx for idxs in token_index_map.values() for idx in idxs})
            self._logger.warning(
                "no target word matched any prompt token (target_words=%s); "
                "falling back to a single combined-target pass", target_words,
            )
            target_words_indices = {",".join(target_words): all_idx}
        if hasattr(steering_model, "set_text_token_indices"):
            steering_model.set_text_token_indices(
                [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > a]
            )

        # VAE-encode the source video + audio to latents at the target pixel shape.
        # Assumes video_path already matches (video_h, video_w, num_frames, frame_rate)
        # by construction (e.g. one of this pipeline's own extraction/generation
        # outputs) — no resolution/frame-count renegotiation is attempted here.
        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=video_h, width=video_w, fps=frame_rate
        )
        initial_video_latent = self._pipeline.image_conditioner(
            lambda enc: video_latent_from_file(
                video_encoder=enc,
                file_path=video_path,
                output_shape=pixel_shape,
                device=self._device,
                dtype=self._pipeline.dtype,
            )
        )
        initial_audio_latent = self._audio_conditioner(
            lambda enc: audio_latent_from_file(
                audio_encoder=enc,
                file_path=video_path,
                output_shape=pixel_shape,
                device=self._device,
                dtype=self._pipeline.dtype,
            )
        )
        assert initial_audio_latent is not None, (
            f"Source video has no audio track: {video_path}. Object-removal editing "
            f"requires a muxed video+audio file (this pipeline's own extraction/"
            f"generation outputs always have one), because the diffusion stage shares "
            f"one sigma schedule across both modalities."
        )

        # grad=True: contexts cloned out of inference-mode so the per-step optimization
        # forward can build an autograd graph to x_t. sigmas here is the FULL schedule
        # (length steps+1, 1.0 -> 0.0); we truncate it below and discard the returned
        # noiser (built for from-scratch pure-noise generation, not applicable here).
        v_context_p, a_context_p, sigmas_full, denoiser, _unused_noiser = self._build_pipeline_inputs(
            prompt, video_negative_prompt, video_h, video_w, steps,
            video_guidance_scale, audio_guidance_scale, seed, grad=True, skip_noiser=True,
        )
        del _unused_noiser

        # Truncate the schedule to start at the sigma nearest noise_strength: higher
        # noise_strength -> earlier start_idx -> more of the schedule is re-denoised
        # (1.0 == start_idx 0 == from-scratch). Clamp so at least one denoising step
        # remains (sigmas_full has length steps+1; sigmas[:-1] must be non-empty).
        start_idx = int(round((1.0 - noise_strength) * steps))
        start_idx = max(0, min(start_idx, steps - 1))
        sigmas = sigmas_full[start_idx:].clone()
        noise_scale = float(sigmas[0].item())

        generator = torch.Generator(device=self._device).manual_seed(seed)
        # Recording wrapper so the steering loop can anchor to the source video's
        # actual noising line (source_reg / source_reg_every_step) — captures the
        # raw noise tensor GaussianNoiser draws, keyed by shape (video vs. audio
        # latents always differ in shape, so no ambiguity regardless of which
        # modality stage() noises first). Reimplements GaussianNoiser.__call__
        # exactly (see ltx_core.components.noisers) rather than calling it and
        # re-drawing, which would advance the generator twice.
        source_noise_sink: dict = {}

        class _RecordingGaussianNoiser(GaussianNoiser):
            def __call__(self, latent_state, noise_scale: float = 1.0):
                noise = torch.randn(
                    *latent_state.latent.shape,
                    device=latent_state.latent.device,
                    dtype=latent_state.latent.dtype,
                    generator=self.generator,
                )
                source_noise_sink[tuple(latent_state.latent.shape)] = noise
                scaled_mask = latent_state.denoise_mask * noise_scale
                latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
                from dataclasses import replace as _replace
                return _replace(latent_state, latent=latent.to(latent_state.latent.dtype))

        noiser = _RecordingGaussianNoiser(generator=generator)

        # Positive-only (cfg=1.0) denoiser for the optimization forward — see
        # run_with_steering for why this is exact for probe-based steering.
        from ltx_core.components.guiders import (
            MultiModalGuiderParams,
            create_multimodal_guider_factory,
        )
        from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser

        opt_v_factory = create_multimodal_guider_factory(params=MultiModalGuiderParams(cfg_scale=1.0))
        opt_a_factory = create_multimodal_guider_factory(params=MultiModalGuiderParams(cfg_scale=1.0))
        opt_denoiser = FactoryGuidedDenoiser(
            v_context=v_context_p,
            a_context=a_context_p,
            video_guider_factory=opt_v_factory,
            audio_guider_factory=opt_a_factory,
        )

        sf = cfg.get("latent_scale_factors", {})
        if sf:
            lat_t = int(sf.get("time", 8))
            lat_h = int(sf.get("height", 32))
            lat_w = int(sf.get("width", 32))
            lat_frames = (num_frames - 1) // lat_t + 1
            video_seq_len: int | None = lat_frames * (video_h // lat_h) * (video_w // lat_w)
        else:
            video_seq_len = None

        source_trace: list[dict] = []
        steering_loop = make_ltx_steering_loop(
            steering_model=steering_model,
            target_words_indices=target_words_indices,
            n_iters=n_iters,
            optimizer_name=optimizer_name,
            lr=lr,
            video_seq_len=video_seq_len,
            capture_blocks=getattr(steering_model, "capture_blocks", None),
            opt_denoiser=opt_denoiser,
            magnitude_reg=magnitude_reg,
            anchor_reg=anchor_reg,
            pred_reg=pred_reg,
            kl_reg=kl_reg,
            entropy_reg=entropy_reg,
            source_noise_sink=source_noise_sink,
            sigmas_full=sigmas_full,
            source_reg=source_reg,
            source_reg_every_step=source_reg_every_step,
            source_trace=source_trace,
            step_callback=step_callback,
            desc=f"Video-edit steering [{prompt[:30]}]",
            step_offset=start_idx,
        )

        video_state, audio_state = self._pipeline.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=video_w,
            height=video_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p, conditionings=[],
                noise_scale=noise_scale, initial_latent=initial_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p, conditionings=[],
                noise_scale=noise_scale, initial_latent=initial_audio_latent,
            ),
            max_batch_size=1,
            loop=steering_loop,
        )
        del v_context_p, a_context_p, sigmas, sigmas_full, denoiser, noiser
        torch.cuda.empty_cache()

        gen_video = self._decode_video(video_state, seed)
        gen_audio = self._decode_audio(audio_state)
        del video_state, audio_state

        return gen_video, gen_audio, source_trace

    def save_video(
        self,
        path: str,
        video: np.ndarray,
        audio: np.ndarray,
        fps: float,
        sample_rate: int,
        rank: int,
    ) -> None:
        from ltx_core.types import Audio
        from ltx_pipelines.utils.media_io import encode_video

        video_tensor = torch.from_numpy(video).permute(1, 2, 3, 0)  # [C,F,H,W] → [F,H,W,C]
        waveform = torch.from_numpy(audio).unsqueeze(0)
        waveform = waveform.repeat(2, 1)
        audio_obj = Audio(waveform=waveform, sampling_rate=sample_rate)
        encode_video(
            video=video_tensor,
            fps=int(fps),
            audio=audio_obj,
            output_path=path,
            video_chunks_number=1,
        )
        _ = rank

    @torch.inference_mode()
    def _decode_video(self, video_state, seed: int) -> np.ndarray:
        # Pin the VAE-decode noise (inject_noise draws) to the CPU RNG so the decode
        # is reproducible across GPU architectures — the decoder's torch.randn(generator)
        # draws are otherwise GPU-arch dependent, like the initial-latent noise.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with _cpu_pinned_randn():
            chunks = list(self._pipeline.video_decoder(video_state.latent, generator=generator))
        video = torch.cat(chunks, dim=0)
        del chunks
        video = video.permute(3, 0, 1, 2)
        result = video.float().cpu().numpy()
        del video
        return result

    @torch.inference_mode()
    def _decode_audio(self, audio_state) -> np.ndarray:
        audio = self._pipeline.audio_decoder(audio_state.latent)
        # waveform is [C, T]; take first channel to get [T]
        return audio.waveform[0].float().cpu().numpy()


class WanBackend(ModelBackend):
    """Backend for the WAN (Wan2.1) text→video model (task ``t2v-1.3B``).

    Video-only (no audio track); the audio-shaped interface params are accepted and
    ignored, and ``run_*`` return ``audio=None`` so ``run_extraction`` skips the WAV write.
    The DiT runs on an SDPA implementation of WAN's fused ``flash_attention`` (see
    ``data_extraction/wan_probe.install_sdpa_flash_attention``): no ``flash_attn``
    dependency, and a deterministic backward for steering.

    Builds against whatever is checked out in ``~/repos/Wan2.1`` (the user's fork), which
    is path-injected at runtime — mirroring the LTX/Ovi convention.
    """

    def __init__(
        self,
        engine_cfg,
        model_cfg: dict,
        experiment_cfg: dict,
        device: int,
        logger: logging.Logger,
    ) -> None:
        from will_it_render.common import paths

        # Locate the Wan2.1 checkout: an explicit engine.wan_root (configs/models/wan.yaml)
        # wins, else VP_WAN_ROOT (plan §10b R5). `wan` is not pip-installed, so we path-inject.
        _WAN_ROOT = paths.wan_root(getattr(engine_cfg, "wan_root", None))
        if not (_WAN_ROOT / "wan").is_dir():
            raise FileNotFoundError(
                f"Wan2.1 repo at {_WAN_ROOT} has no 'wan/' package. Set VP_WAN_ROOT (or "
                f"engine.wan_root in configs/models/wan.yaml) to a valid Wan2.1 checkout."
            )
        paths.add_to_syspath(_WAN_ROOT)

        from wan.configs import WAN_CONFIGS
        from wan.text2video import WanT2V

        from will_it_render.backends.hooks.wan_hooks import install_sdpa_flash_attention

        self._device = torch.device(f"cuda:{device}")
        self._logger = logger
        self._model_cfg = model_cfg
        self._experiment_cfg = experiment_cfg
        self._task = str(getattr(engine_cfg, "task", "t2v-1.3B"))
        self._ckpt_dir = str(engine_cfg.checkpoint_dir)
        self._cfg = WAN_CONFIGS[self._task]

        # Swap WAN's fused flash-attention for the SDPA path before any forward runs
        # (both self- and cross-attention call flash_attention directly; without this the
        # model can't run without flash_attn installed). Held for the process lifetime.
        self._restore_attn = install_sdpa_flash_attention()

        # Determinism (mirrors steering.steer.enable_determinism; inlined to avoid a
        # circular import). Must be set before CUDA/cuBLAS init below. DEFAULT = strict
        # (use_deterministic_algorithms warn_only=False): SDPA's flash backward then uses its
        # deterministic algorithm, so a steered run is bit-reproducible on the same GPU type.
        # Under warn_only the backward was atomicAdd-based and steered runs differed run-to-run
        # even on the same GPU (2026-10-09 hash test). Measured cost of strict: +5% wall-clock,
        # no extra VRAM; the full WAN steer + VAE-decode path has no op without a deterministic
        # kernel. STEER_DETERMINISTIC=warn restores warn_only; =0 opts out entirely.
        import os as _os
        _det = _os.environ.get("STEER_DETERMINISTIC", "").strip().lower()
        if _det not in ("0", "off", "false"):
            _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            _warn = _det == "warn"
            torch.use_deterministic_algorithms(True, warn_only=_warn)
            logger.info("WAN determinism ON (cudnn.deterministic, use_deterministic_algorithms "
                        "%s; STEER_DETERMINISTIC=warn for warn_only, =0 to opt out)",
                        "warn_only" if _warn else "STRICT")

        # Precomputed per-seed initial latents (data_extraction/gen_seed_latents_wan.py) —
        # like LTX, seeds are canonical/GPU-independent because the pipeline LOADS the
        # initial noise instead of sampling (CUDA RNG is GPU-arch-dependent).
        self._seed_latents_dir = model_cfg.get("seed_latents_dir") or experiment_cfg.get("seed_latents_dir")
        self._seed_latents_tag = model_cfg.get("seed_latents_gpu_tag", "h100")

        self._pipe = WanT2V(
            config=self._cfg,
            checkpoint_dir=self._ckpt_dir,
            device_id=device,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
        )
        self._model = self._pipe.model
        # HF fast tokenizer (umt5-xxl, whitespace-cleaned) the text encoder uses — reused
        # for offset-based target-word → token-index mapping.
        self._hf_tokenizer = self._pipe.text_encoder.tokenizer

    # ------------------------------------------------------------------ helpers

    def _model_tag(self) -> str:
        return f"wan-{self._task}"

    def _target_shape(self, video_h: int, video_w: int) -> tuple[int, int, int, int]:
        """Initial-latent shape [z_dim, F', H/8, W/8] — the exact shape WAN's generate()
        samples. Single source of truth shared with gen_seed_latents_wan.py."""
        cfg = self._cfg
        F = int(self._model_cfg.get("num_frames", 81))
        z_dim = self._pipe.vae.model.z_dim
        return (
            z_dim,
            (F - 1) // cfg.vae_stride[0] + 1,
            video_h // cfg.vae_stride[1],
            video_w // cfg.vae_stride[2],
        )

    def _res_tag(self, video_h: int, video_w: int) -> str:
        F = int(self._model_cfg.get("num_frames", 81))
        fps = float(self._model_cfg.get("frame_rate", 16.0))
        return f"{video_h}x{video_w}x{F}_{fps:g}fps"

    def _seed_latent_path(self, seed: int, video_h: int, video_w: int) -> Path:
        """Path to the precomputed initial-latent file for ``seed`` — mirrors
        LTXBackend._seed_latent_path and gen_seed_latents_wan.py."""
        assert self._seed_latents_dir, (
            "extraction.seed_latents_dir is not set — add it to configs/models/wan.yaml. "
            "Precomputed per-seed latents are required (seed noise is GPU-dependent); "
            "generate them with data_extraction/gen_seed_latents_wan.py."
        )
        return (Path(self._seed_latents_dir) / self._model_tag() / self._res_tag(video_h, video_w)
                / f"seed_{int(seed):05d}__{self._seed_latents_tag}.pt")

    def _seed_latent_gen_hint(self, video_h: int, video_w: int) -> str:
        F = int(self._model_cfg.get("num_frames", 81))
        fps = float(self._model_cfg.get("frame_rate", 16.0))
        return (f"python -m data_extraction.gen_seed_latents_wan --seeds <SEEDS> "
                f"--gpu-tag {self._seed_latents_tag} --height {video_h} --width {video_w} "
                f"--frames {F} --fps {fps:g}")

    def _token_index_map(self, prompt: str, target_words: list[str]):
        """Map target words → token indices against the *cleaned* prompt the encoder sees.

        Returns ``(cleaned, {word: [token_idx]}, real_token_indices)`` — the last is the list
        of key-axis positions holding real prompt tokens (special tokens have an empty offset
        span), consumed by A&E's text-axis softmax via ``set_text_token_indices``.
        """
        from wan.modules.tokenizers import basic_clean, whitespace_clean

        from will_it_render.common.token_mapper import build_token_index_map

        cleaned = whitespace_clean(basic_clean(prompt))
        enc = self._hf_tokenizer.tokenizer(
            cleaned, return_offsets_mapping=True, add_special_tokens=True
        )
        real_token_indices = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a]
        return (cleaned,
                build_token_index_map(cleaned, target_words, enc["offset_mapping"]),
                real_token_indices)

    def _build_scheduler(self, solver: str, steps: int, shift: float):
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        cfg = self._cfg
        if solver == "unipc":
            s = FlowUniPCMultistepScheduler(
                num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
            s.set_timesteps(steps, device=self._device, shift=shift)
            return s, s.timesteps
        if solver == "dpm++":
            s = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
            sig = get_sampling_sigmas(steps, shift)
            ts, _ = retrieve_timesteps(s, device=self._device, sigmas=sig)
            return s, ts
        raise NotImplementedError(f"WAN solver {solver!r} not supported (use 'unipc' or 'dpm++').")

    def _prepare(self, prompt, video_negative_prompt, video_h, video_w, steps, solver, shift, seed):
        """Build (context, context_null, latents, scheduler, timesteps, seq_len).

        The initial noise is LOADED from a precomputed per-seed file (generated once on
        the canonical GPU by gen_seed_latents_wan.py) instead of sampled — so a given seed
        reproduces the identical video on any GPU (CUDA RNG is arch-dependent). Hard-fails
        if the file is missing, mirroring the LTX backend.
        """
        import math

        cfg = self._cfg
        pipe = self._pipe
        target_shape = self._target_shape(video_h, video_w)
        seq_len = int(math.ceil(
            (target_shape[2] * target_shape[3]) / (cfg.patch_size[1] * cfg.patch_size[2])
            * target_shape[1]
        ))
        neg = video_negative_prompt or cfg.sample_neg_prompt
        pipe.text_encoder.model.to(self._device)
        context = pipe.text_encoder([prompt], self._device)
        context_null = pipe.text_encoder([neg], self._device)
        # Free the umt5-xxl encoder (~11 GB bf16) right after encoding, mirroring upstream
        # WanT2V.generate's offload_model path (wan/text2video.py:178). This hand-rolled loop
        # otherwise keeps it GPU-resident for the whole denoise, which OOMs 24 GB cards (L4)
        # and needlessly narrows headroom everywhere else. Costs one H2D reload per clip.
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()

        path = self._seed_latent_path(seed, video_h, video_w)
        assert path.exists(), (
            f"Precomputed seed latent missing for seed={seed}:\n  {path}\n"
            f"Generate it on the canonical GPU first:\n  {self._seed_latent_gen_hint(video_h, video_w)}"
        )
        # weights_only=False: our own trusted file, and it carries a meta dict.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        noise = blob["video_noise"]
        assert tuple(noise.shape) == tuple(target_shape), (
            f"seed-latent shape {tuple(noise.shape)} != expected {tuple(target_shape)} for "
            f"seed={seed} (resolution/frames changed?) — regenerate. File: {path}"
        )
        latents = [noise.to(self._device, dtype=torch.float32)]
        scheduler, timesteps = self._build_scheduler(solver, steps, shift)
        return context, context_null, latents, scheduler, timesteps, seq_len

    def _sample(self, latents, context, context_null, scheduler, timesteps, seq_len,
                guide_scale, *, capture_ctx=None, is_active=None, on_capture=None, desc="Generating"):
        """Run the WAN denoising loop (replicates ``WanT2V.generate``'s inner loop).

        When ``capture_ctx`` is given, the cond pass is captured (gated by ``is_active``)
        and ``on_capture(step_idx, captures)`` is invoked; the uncond pass never captures.
        """
        import torch.cuda.amp as amp
        from tqdm import tqdm

        model = self._model
        n_steps = len(timesteps)
        for step_idx, t in enumerate(tqdm(timesteps, desc=desc, unit="step")):
            timestep = torch.stack([t])
            active = bool(capture_ctx is not None and (is_active is None or is_active(step_idx, n_steps)))
            with amp.autocast(dtype=self._pipe.param_dtype), torch.no_grad():
                if capture_ctx is not None:
                    capture_ctx.active = active
                    capture_ctx.captures = []
                nc = model(latents, t=timestep, context=context, seq_len=seq_len)[0]
                if capture_ctx is not None:
                    capture_ctx.active = False
                nu = model(latents, t=timestep, context=context_null, seq_len=seq_len)[0]
                noise_pred = nu + guide_scale * (nc - nu)
                if active and on_capture is not None:
                    on_capture(step_idx, list(capture_ctx.captures))
                x0 = scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=None,
                )[0]
                latents = [x0.squeeze(0)]
        return latents

    def _decode(self, latents) -> np.ndarray:
        # WanVAE.decode returns a list of [C, F, H, W] float tensors clamped to [-1, 1].
        videos = self._pipe.vae.decode(latents)
        return videos[0].float().cpu().numpy()

    @staticmethod
    def _make_is_active(capture_steps, capture_every, first_step):
        """Mirror make_ltx_extraction_loop's step gating."""
        capture_step_set = set(capture_steps) if capture_steps is not None else None

        def is_active(step_idx, n_steps):
            if capture_step_set is not None:
                return step_idx in capture_step_set
            is_last = step_idx == n_steps - 1
            after_first = step_idx >= first_step
            on_schedule = (step_idx - first_step) % capture_every == 0
            return is_last or (after_first and on_schedule)

        return is_active

    # ------------------------------------------------------------------ interface

    def run_with_probes(
        self,
        prompt: str,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        target_words: list[str],
        hook_type: CaptureType,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
        step_callback: Callable[[int, StepFeature], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list, list[int], Dict[str, List[int]], dict]:
        from will_it_render.common.aggregator import (
            aggregate_captures_per_word,
            aggregate_captures_to_vector,
            probe_shape_from_captures,
        )
        from will_it_render.backends.hooks.wan_hooks import WanCrossAttnProbeContext

        cfg = self._model_cfg
        cleaned, token_index_map, _ = self._token_index_map(prompt, target_words)
        target_indices = sorted({idx for idxs in token_index_map.values() for idx in idxs})

        per_word = bool(cfg.get("per_word_features", False))
        agg_kwargs = _build_agg_kwargs(cfg)
        capture_every = cfg.get("capture_every_n_steps", 1)
        first_step = cfg.get("first_step_to_capture", 0)
        capture_steps = cfg.get("capture_steps", None)
        capture_blocks = cfg.get("capture_blocks", None)
        if capture_blocks is not None:
            capture_blocks = sorted(int(b) for b in capture_blocks)

        context, context_null, latents, scheduler, timesteps, seq_len = self._prepare(
            prompt, video_negative_prompt, video_h, video_w, steps, solver, shift, seed,
        )

        step_features: list = []
        step_indices: list[int] = []
        _probe_shape: dict = {}

        def on_capture(step_idx: int, captures) -> None:
            typed = [c for c in captures if c.capture_type == CaptureType.VIDEO_CA]
            if not _probe_shape and typed:
                _probe_shape.update(probe_shape_from_captures(typed, **agg_kwargs))
            if per_word:
                vec = {w: t.cpu() for w, t in
                       aggregate_captures_per_word(typed, token_index_map, **agg_kwargs).items()}
            else:
                vec = aggregate_captures_to_vector(typed, target_indices, **agg_kwargs).cpu()
            if step_callback is not None:
                step_callback(step_idx, vec)
            else:
                step_features.append(vec)
                step_indices.append(step_idx)

        is_active = self._make_is_active(capture_steps, capture_every, first_step)
        with WanCrossAttnProbeContext(self._model, capture_blocks=capture_blocks, detach=True) as cctx:
            latents = self._sample(
                latents, context, context_null, scheduler, timesteps, seq_len,
                video_guidance_scale, capture_ctx=cctx, is_active=is_active,
                on_capture=on_capture, desc=f"Extracting [{prompt[:35]}]",
            )
            n_blocks_total = cctx.n_blocks_total

        gen_video = self._decode(latents)

        if capture_blocks is not None and _probe_shape:
            _probe_shape["capture_blocks"] = capture_blocks
            _probe_shape["n_blocks_total"] = n_blocks_total

        return gen_video, None, step_features, step_indices, token_index_map, _probe_shape

    def run_generation_only(
        self,
        prompt: str,
        video_negative_prompt: str,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        context, context_null, latents, scheduler, timesteps, seq_len = self._prepare(
            prompt, video_negative_prompt, video_h, video_w, steps, solver, shift, seed,
        )
        latents = self._sample(
            latents, context, context_null, scheduler, timesteps, seq_len,
            video_guidance_scale, desc=f"Generating [{prompt[:35]}]",
        )
        return self._decode(latents), None

    def run_with_steering(
        self,
        prompt: str,
        video_negative_prompt: str,
        target_words: list[str],
        steering_model: "object",
        video_h: int,
        video_w: int,
        steps: int,
        n_iters: int,
        optimizer_name: str,
        lr: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        seed: int,
        magnitude_reg: float = 0.0,
        anchor_reg: float = 0.0,
        pred_reg: float = 0.0,
        kl_reg: float = 0.0,
        entropy_reg: float = 0.0,
        step_callback: Callable[[int, dict], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from will_it_render.inference.steering_core import run_steering_optimization
        from will_it_render.inference.wan_adapter import WanSteeringAdapter

        cleaned, token_index_map, text_token_indices = self._token_index_map(prompt, target_words)
        # Per-word steering: keep each word's full subword-token index list; drop words
        # that matched no tokens, falling back to a single combined pass if none match
        # (mirrors the LTX backend).
        target_words_indices = {w: idxs for w, idxs in token_index_map.items() if idxs}
        if not target_words_indices:
            all_idx = sorted({idx for idxs in token_index_map.values() for idx in idxs})
            self._logger.warning(
                "no target word matched any prompt token (target_words=%s); "
                "falling back to a single combined-target pass", target_words,
            )
            target_words_indices = {",".join(target_words): all_idx}
        # A&E re-normalizes attention over the real-token key positions only (drops special
        # tokens); mirrors the LTX backend. No-op for steering models without the hook.
        if hasattr(steering_model, "set_text_token_indices"):
            steering_model.set_text_token_indices(text_token_indices)

        context, context_null, latents, scheduler, timesteps, seq_len = self._prepare(
            prompt, video_negative_prompt, video_h, video_w, steps, solver="unipc",
            shift=self._model_cfg.get("shift", 5.0), seed=seed,
        )

        adapter = WanSteeringAdapter(
            model=self._model,
            param_dtype=self._pipe.param_dtype,
            context=context,
            context_null=context_null,
            latents=latents,
            scheduler=scheduler,
            timesteps=timesteps,
            seq_len=seq_len,
            guide_scale=video_guidance_scale,
            seed_g=None,
            capture_blocks=getattr(steering_model, "capture_blocks", None),
            video_seq_len=None,
        )
        run_steering_optimization(
            adapter,
            steering_model,
            target_words_indices=target_words_indices,
            n_iters=n_iters,
            optimizer_name=optimizer_name,
            lr=lr,
            magnitude_reg=magnitude_reg,
            anchor_reg=anchor_reg,
            pred_reg=pred_reg,
            kl_reg=kl_reg,
            entropy_reg=entropy_reg,
            step_callback=step_callback,
            desc=f"Steering [{prompt[:30]}]",
        )
        return self._decode(adapter.latents), None

    def save_video(
        self,
        path: str,
        video: np.ndarray,
        audio: np.ndarray,
        fps: float,
        sample_rate: int,
        rank: int,
    ) -> None:
        from wan.utils.utils import cache_video

        # video: [C, F, H, W] float in [-1, 1]. cache_video wants [B, C, F, H, W].
        tensor = torch.from_numpy(video)[None]
        cache_video(
            tensor=tensor, save_file=path, fps=int(fps), nrow=1,
            normalize=True, value_range=(-1, 1),
        )
        _ = (audio, sample_rate, rank)


class CogVideoXBackend(ModelBackend):
    """Backend for CogVideoX-5B (THUDM) text→video — video-only (no audio track).

    CogVideoX runs one JOINT self-attention over ``cat([text(226), video])`` (no dedicated
    cross-attention module), so the probe hook (``cogvideox_probe``) materializes the
    video-query × text-key sub-block of that joint attention — yielding the same
    ``[n_blocks, n_heads, Lq]`` feature layout as LTX/Wan. Uses ``diffusers`` (pip-installed
    in the cogvideox refmodel venv, NOT path-injected): the fiddly pieces (T5 prompt
    encoding, RoPE, VAE scaling, DDIM scheduler) are reused from ``CogVideoXPipeline``;
    only the denoising loop is owned here so we can load precomputed seed noise, gate the
    capture, and run cond/uncond as two separate forwards (capture the cond pass only).
    """

    def __init__(
        self,
        engine_cfg,
        model_cfg: dict,
        experiment_cfg: dict,
        device: int,
        logger: logging.Logger,
    ) -> None:
        from diffusers import CogVideoXPipeline

        self._device = torch.device(f"cuda:{device}")
        self._logger = logger
        self._model_cfg = model_cfg
        self._experiment_cfg = experiment_cfg
        self._ckpt_dir = str(engine_cfg.checkpoint_dir)
        self._model_tag_str = str(getattr(engine_cfg, "model_tag", "cogvideox-5b"))
        dtype_str = str(getattr(engine_cfg, "dtype", "bfloat16")).lower()
        self._dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(dtype_str, torch.bfloat16)

        # Determinism (mirrors WanBackend): a seed + its precomputed initial latent then
        # reproduce the same video run-to-run on a given GPU tier. STEER_DETERMINISTIC=0 opts out.
        import os as _os
        if _os.environ.get("STEER_DETERMINISTIC", "").strip().lower() not in ("0", "off", "false"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Intentionally NOT torch.use_deterministic_algorithms(True): it forces
            # memory-heavy deterministic conv3d/group_norm kernels that blow the CogVideoX
            # VAE decode past 80 GB. Forward-only generation is already reproducible on a
            # fixed GPU tier with cudnn.deterministic + precomputed seed latents (pass-1 and
            # pass-2 are both forward-only, so bit-exact backward is not required here).
            logger.info("CogVideoX determinism: cudnn.deterministic ON (use_deterministic_algorithms "
                        "left OFF to avoid VAE-decode OOM; STEER_DETERMINISTIC=0 to opt out)")

        self._pipe = CogVideoXPipeline.from_pretrained(self._ckpt_dir, torch_dtype=self._dtype)
        # Model CPU offload keeps only ONE component on-GPU at a time (T5 for encode, then
        # the DiT for the loop, then the VAE for decode). Essential on 80 GB: the tiled VAE
        # decode of 49×480×720 needs a clean device — leaving the 20 GB DiT+T5 resident (or
        # manually offloading them, which leaves ~10 GB of unreclaimable allocator
        # fragmentation from the per-step capture matmuls) OOMs the decode. The accelerate
        # hooks move each module on its forward, which our custom loop drives correctly.
        self._pipe.enable_model_cpu_offload(gpu_id=device)
        # Tiled/sliced VAE decode keeps the 49×480×720 reconstruction within budget.
        self._pipe.vae.enable_tiling()
        self._pipe.vae.enable_slicing()
        self._transformer = self._pipe.transformer
        self._scheduler = self._pipe.scheduler
        self._hf_tokenizer = self._pipe.tokenizer  # T5 fast tokenizer (offset mapping)

        self._seed_latents_dir = model_cfg.get("seed_latents_dir") or experiment_cfg.get("seed_latents_dir")
        self._seed_latents_tag = model_cfg.get("seed_latents_gpu_tag", "a100")

    # ------------------------------------------------------------------ helpers

    def _model_tag(self) -> str:
        return self._model_tag_str

    def _num_frames(self) -> int:
        return int(self._model_cfg.get("num_frames", 49))

    def _target_shape(self, video_h: int, video_w: int) -> tuple[int, int, int, int]:
        """Batchless initial-latent shape [F', C, H/8, W/8] — shared with
        gen_seed_latents_cogx.py (CogVideoX VAE: temporal 4×, spatial 8×, 16 channels)."""
        F = self._num_frames()
        sf = self._model_cfg.get("latent_scale_factors", {"time": 4, "height": 8, "width": 8})
        return (
            (F - 1) // int(sf["time"]) + 1,
            16,
            video_h // int(sf["height"]),
            video_w // int(sf["width"]),
        )

    def _res_tag(self, video_h: int, video_w: int) -> str:
        F = self._num_frames()
        fps = float(self._model_cfg.get("frame_rate", 8.0))
        return f"{video_h}x{video_w}x{F}_{fps:g}fps"

    def _seed_latent_path(self, seed: int, video_h: int, video_w: int) -> Path:
        assert self._seed_latents_dir, (
            "extraction.seed_latents_dir is not set — add it to configs/models/cogvideox.yaml. "
            "Precomputed per-seed latents are required (seed noise is GPU-dependent); "
            "generate them with data_extraction/gen_seed_latents_cogx.py."
        )
        return (Path(self._seed_latents_dir) / self._model_tag() / self._res_tag(video_h, video_w)
                / f"seed_{int(seed):05d}__{self._seed_latents_tag}.pt")

    def _seed_latent_gen_hint(self, video_h: int, video_w: int) -> str:
        F = self._num_frames()
        fps = float(self._model_cfg.get("frame_rate", 8.0))
        return (f"python -m data_extraction.gen_seed_latents_cogx --seeds <SEEDS> "
                f"--gpu-tag {self._seed_latents_tag} --height {video_h} --width {video_w} "
                f"--frames {F} --fps {fps:g}")

    def _token_index_map(self, prompt: str, target_words: list[str]):
        """Map target words → T5 token indices via the fast tokenizer's offset mapping.

        Returns ``(prompt, {word: [token_idx]}, real_token_indices)``. CogVideoX applies no
        prompt cleaning (T5 sentencepiece handles it), so the offsets index the raw prompt.
        """
        from will_it_render.common.token_mapper import build_token_index_map

        enc = self._hf_tokenizer(
            prompt, return_offsets_mapping=True, add_special_tokens=True,
            truncation=True, max_length=int(self._model_cfg.get("max_sequence_length", 226)),
        )
        offsets = enc["offset_mapping"]
        real_token_indices = [i for i, (a, b) in enumerate(offsets) if b > a]
        return (prompt,
                build_token_index_map(prompt, target_words, offsets),
                real_token_indices)

    def _prepare(self, prompt, video_negative_prompt, video_h, video_w, steps, seed):
        """Build (prompt_embeds, neg_embeds, latents, timesteps, image_rotary_emb).

        Initial noise is LOADED from a precomputed per-seed file (gen_seed_latents_cogx.py)
        instead of sampled — so a seed reproduces the identical initial latent on any GPU.
        Hard-fails if missing, mirroring the LTX/Wan backends.
        """
        # enable_model_cpu_offload normally manages placement via forward hooks, but the
        # steering/decode paths manually `.to("cpu")` the text encoder (to free ~9 GB for the
        # grad-forward peak), which desyncs accelerate's hook so a LATER encode_prompt in the
        # same worker process finds T5 weights stranded on CPU ("two devices cuda:0 and cpu").
        # Restore it explicitly before encoding — idempotent, and it's needed on-GPU here anyway.
        self._pipe.text_encoder.to(self._device)
        max_seq = int(self._model_cfg.get("max_sequence_length", 226))
        prompt_embeds, negative_prompt_embeds = self._pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=video_negative_prompt or "",
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self._device,
            dtype=self._dtype,
        )

        target_shape = self._target_shape(video_h, video_w)
        path = self._seed_latent_path(seed, video_h, video_w)
        assert path.exists(), (
            f"Precomputed seed latent missing for seed={seed}:\n  {path}\n"
            f"Generate it on the canonical GPU first:\n  {self._seed_latent_gen_hint(video_h, video_w)}"
        )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        noise = blob["video_noise"]
        assert tuple(noise.shape) == tuple(target_shape), (
            f"seed-latent shape {tuple(noise.shape)} != expected {tuple(target_shape)} for "
            f"seed={seed} (resolution/frames changed?) — regenerate. File: {path}"
        )
        # [F', C, H/8, W/8] → [1, F', C, H/8, W/8]; init_noise_sigma == 1.0 (no scaling).
        latents = noise.unsqueeze(0).to(self._device, dtype=self._dtype)
        latents = latents * self._scheduler.init_noise_sigma

        self._scheduler.set_timesteps(steps, device=self._device)
        timesteps = self._scheduler.timesteps

        image_rotary_emb = None
        if self._transformer.config.use_rotary_positional_embeddings:
            image_rotary_emb = self._pipe._prepare_rotary_positional_embeddings(
                video_h, video_w, latents.size(1), self._device
            )
        return prompt_embeds, negative_prompt_embeds, latents, timesteps, image_rotary_emb

    def _sample(self, latents, prompt_embeds, negative_prompt_embeds, timesteps, image_rotary_emb,
                guidance_scale, use_dynamic_cfg, *, capture_ctx=None, is_active=None,
                on_capture=None, desc="Generating"):
        """Denoising loop mirroring CogVideoXPipeline.__call__.

        Cond and uncond are run as TWO separate transformer forwards so the capture hook can
        record the cond pass only (gated by ``is_active``); the uncond pass never captures.
        """
        import math

        from tqdm import tqdm

        n_steps = len(timesteps)
        # The WHOLE loop runs under no_grad. Extraction and generation are inference-only,
        # and (crucially) leaving scale_model_input / step outside no_grad let an autograd
        # graph accumulate across all 50 steps — which the VAE decode then retained, growing
        # to fill the entire GPU (78 GB on A100, 92 GB on H100) and OOMing. Steering uses a
        # separate grad-enabled path (run_with_steering), not this method.
        with torch.no_grad():
            for step_idx, t in enumerate(tqdm(timesteps, desc=desc, unit="step")):
                active = bool(capture_ctx is not None and (is_active is None or is_active(step_idx, n_steps)))
                latent_model_input = self._scheduler.scale_model_input(latents, t)
                timestep = t.expand(latent_model_input.shape[0])
                if capture_ctx is not None:
                    capture_ctx.active = active
                    capture_ctx.captures = []
                noise_cond = self._transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0].float()
                if capture_ctx is not None:
                    capture_ctx.active = False
                noise_uncond = self._transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0].float()

                gs = guidance_scale
                if use_dynamic_cfg:
                    # Dynamic CFG is scheduled on the TIMESTEP VALUE t (999→0), not the loop
                    # index — matching CogVideoXPipeline.__call__ (`num_inference_steps -
                    # t.item()`). Using step_idx collapsed guidance to ~1 for most of the loop
                    # (mean gs 1.94 vs 3.82), over-brightening output and weakening prompt
                    # adherence (fixed 2026-10-09).
                    gs = 1.0 + guidance_scale * (
                        (1 - math.cos(math.pi * ((n_steps - t.item()) / n_steps) ** 5.0)) / 2.0
                    )
                noise_pred = noise_uncond + gs * (noise_cond - noise_uncond)

                if active and on_capture is not None:
                    on_capture(step_idx, list(capture_ctx.captures))

                latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents = latents.to(self._dtype)
        return latents

    def _decode(self, latents) -> np.ndarray:
        # decode_latents: [B,F,C,H,W] → permute+unscale → vae.decode → [B, C, F, H, W] in [-1,1].
        # The CogVideoX VAE decode of 49×480×720 is ~50 GB of fp32 conv activations even
        # with tiling; offload the DiT + text encoder to CPU first so it fits on an 80 GB
        # card (they are restored on-GPU at the next _prepare).
        # Force the DiT + text encoder off-GPU before the VAE decode. enable_model_cpu_offload's
        # hook does not reliably offload the DiT here (our custom loop calls the module directly,
        # so the pipe's prev-module offload chain isn't advanced), leaving ~11 GB resident plus
        # allocator fragmentation from the per-step capture matmuls — enough to OOM the ~50 GB
        # tiled VAE decode of 49×480×720 on an 80 GB card. gc + empty_cache reclaim it.
        import gc as _gc
        self._pipe.transformer.to("cpu")
        self._pipe.text_encoder.to("cpu")
        _gc.collect()
        torch.cuda.empty_cache()
        self._pipe.vae.to(self._device)  # steering path offloads VAE to CPU; restore for decode
        self._pipe.vae.enable_tiling()  # re-assert (defensive: ensure the tiled decode path)
        self._pipe.vae.enable_slicing()
        # MUST be under no_grad: the VAE params require grad, so decoding 49×480×720 outside
        # no_grad builds an autograd graph that retains every conv3d activation for backward —
        # tens of GB that fill the whole GPU and OOM. (This, not the DiT residency, was the
        # real cause of the decode OOM.)
        with torch.no_grad():
            frames = self._pipe.decode_latents(latents)
        return frames[0].float().cpu().numpy()

    @staticmethod
    def _make_is_active(capture_steps, capture_every, first_step):
        capture_step_set = set(capture_steps) if capture_steps is not None else None

        def is_active(step_idx, n_steps):
            if capture_step_set is not None:
                return step_idx in capture_step_set
            is_last = step_idx == n_steps - 1
            after_first = step_idx >= first_step
            on_schedule = (step_idx - first_step) % capture_every == 0
            return is_last or (after_first and on_schedule)

        return is_active

    # ------------------------------------------------------------------ interface

    def run_with_probes(
        self,
        prompt: str,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        target_words: list[str],
        hook_type: CaptureType,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
        step_callback: Callable[[int, StepFeature], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list, list[int], Dict[str, List[int]], dict]:
        from will_it_render.common.aggregator import (
            aggregate_captures_per_word,
            aggregate_captures_to_vector,
            probe_shape_from_captures,
        )
        from will_it_render.backends.hooks.cogvideox_hooks import CogVideoXCrossAttnProbeContext

        cfg = self._model_cfg
        _, token_index_map, _ = self._token_index_map(prompt, target_words)
        target_indices = sorted({idx for idxs in token_index_map.values() for idx in idxs})

        per_word = bool(cfg.get("per_word_features", False))
        agg_kwargs = _build_agg_kwargs(cfg)
        capture_every = cfg.get("capture_every_n_steps", 1)
        first_step = cfg.get("first_step_to_capture", 0)
        capture_steps = cfg.get("capture_steps", None)
        capture_blocks = cfg.get("capture_blocks", None)
        if capture_blocks is not None:
            capture_blocks = sorted(int(b) for b in capture_blocks)
        use_dynamic_cfg = bool(cfg.get("use_dynamic_cfg", True))

        prompt_embeds, neg_embeds, latents, timesteps, image_rotary_emb = self._prepare(
            prompt, video_negative_prompt, video_h, video_w, steps, seed,
        )

        step_features: list = []
        step_indices: list[int] = []
        _probe_shape: dict = {}

        def on_capture(step_idx: int, captures) -> None:
            typed = [c for c in captures if c.capture_type == CaptureType.VIDEO_CA]
            if not _probe_shape and typed:
                _probe_shape.update(probe_shape_from_captures(typed, **agg_kwargs))
            if per_word:
                vec = {w: t.cpu() for w, t in
                       aggregate_captures_per_word(typed, token_index_map, **agg_kwargs).items()}
            else:
                vec = aggregate_captures_to_vector(typed, target_indices, **agg_kwargs).cpu()
            if step_callback is not None:
                step_callback(step_idx, vec)
            else:
                step_features.append(vec)
                step_indices.append(step_idx)

        is_active = self._make_is_active(capture_steps, capture_every, first_step)
        with CogVideoXCrossAttnProbeContext(self._transformer, capture_blocks=capture_blocks, detach=True) as cctx:
            latents = self._sample(
                latents, prompt_embeds, neg_embeds, timesteps, image_rotary_emb,
                video_guidance_scale, use_dynamic_cfg, capture_ctx=cctx, is_active=is_active,
                on_capture=on_capture, desc=f"Extracting [{prompt[:35]}]",
            )
            n_blocks_total = cctx.n_blocks_total

        gen_video = self._decode(latents)

        if capture_blocks is not None and _probe_shape:
            _probe_shape["capture_blocks"] = capture_blocks
            _probe_shape["n_blocks_total"] = n_blocks_total

        return gen_video, None, step_features, step_indices, token_index_map, _probe_shape

    def run_generation_only(
        self,
        prompt: str,
        video_negative_prompt: str,
        video_h: int,
        video_w: int,
        steps: int,
        solver: str,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        use_dynamic_cfg = bool(self._model_cfg.get("use_dynamic_cfg", True))
        prompt_embeds, neg_embeds, latents, timesteps, image_rotary_emb = self._prepare(
            prompt, video_negative_prompt, video_h, video_w, steps, seed,
        )
        latents = self._sample(
            latents, prompt_embeds, neg_embeds, timesteps, image_rotary_emb,
            video_guidance_scale, use_dynamic_cfg, desc=f"Generating [{prompt[:35]}]",
        )
        return self._decode(latents), None

    def run_with_steering(
        self,
        prompt: str,
        video_negative_prompt: str,
        target_words: list[str],
        steering_model: "object",
        video_h: int,
        video_w: int,
        steps: int,
        n_iters: int,
        optimizer_name: str,
        lr: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        seed: int,
        magnitude_reg: float = 0.0,
        anchor_reg: float = 0.0,
        pred_reg: float = 0.0,
        kl_reg: float = 0.0,
        entropy_reg: float = 0.0,
        step_callback: Callable[[int, dict], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inference-time steering for CogVideoX (joint-attention DiT).

        Mirrors the WAN path: build the per-word token-index map, prepare (T5 context,
        seed latents, DDIM timesteps, RoPE), then hand a :class:`CogVideoXSteeringAdapter`
        to the shared steering core. The adapter runs the cond pass with grad (capturing the
        video→text softmax) and the full-CFG advance under ``no_grad``.
        """
        from will_it_render.inference.steering_core import run_steering_optimization
        from will_it_render.inference.cogx_adapter import CogVideoXSteeringAdapter

        cleaned, token_index_map, text_token_indices = self._token_index_map(prompt, target_words)
        # Per-word steering: keep each word's full subword-token index list; drop words that
        # matched no tokens, falling back to a single combined pass if none match (mirrors WAN/LTX).
        target_words_indices = {w: idxs for w, idxs in token_index_map.items() if idxs}
        if not target_words_indices:
            all_idx = sorted({idx for idxs in token_index_map.values() for idx in idxs})
            self._logger.warning(
                "no target word matched any prompt token (target_words=%s); "
                "falling back to a single combined-target pass", target_words,
            )
            target_words_indices = {",".join(target_words): all_idx}
        if hasattr(steering_model, "set_text_token_indices"):
            steering_model.set_text_token_indices(text_token_indices)

        prompt_embeds, negative_prompt_embeds, latents, timesteps, image_rotary_emb = self._prepare(
            prompt, video_negative_prompt, video_h, video_w, steps, seed,
        )
        # Free the T5 text encoder (+VAE) for the steering loop. encode_prompt already ran in
        # _prepare (embeds are cached) and neither is needed until decode, but
        # enable_model_cpu_offload's hook does not reliably offload them from our custom loop
        # (same issue _decode documents for the DiT), leaving ~9 GB of T5 resident — enough to
        # tip a 94 GB card over at the grad-forward capture peak. The accelerate hook restores
        # them on the next encode_prompt / decode forward.
        import gc as _gc
        self._pipe.text_encoder.to("cpu")
        self._pipe.vae.to("cpu")
        _gc.collect()
        torch.cuda.empty_cache()
        use_dynamic_cfg = bool(self._model_cfg.get("use_dynamic_cfg", True))

        # Capture-time Lq pooling (steering VRAM): pool the video-query grid to the probe's
        # resolution INSIDE the capture so only the reduced [B,H,Lq/f²,Lk] map is kept on-GPU
        # with grad (~4× less than the full map — what lets the ≤96 GB tiers fit). The pool
        # reuses the aggregator's own _spatial_downsample; we then switch the downstream
        # aggregation to spatial_downsample=1 with the post-pool grid so it does NOT re-pool.
        # Patched once per steering_model (idempotent across prompts).
        agg = getattr(steering_model, "agg_kwargs", None)
        if isinstance(agg, dict) and not getattr(steering_model, "_cogx_pool_patched", False):
            f = int(agg.get("spatial_downsample", 1) or 1)
            hlw = agg.get("latent_hw")
            if f > 1 and hlw is not None:
                _, _, hlat, wlat = self._target_shape(video_h, video_w)
                patch = int(getattr(self._transformer.config, "patch_size", 2) or 2)
                h_p, w_p = hlat // patch, wlat // patch
                steering_model._cogx_steer_pool = (f, int(hlw[0]), int(hlw[1]))
                agg["spatial_downsample"] = 1
                agg["latent_hw"] = (h_p // f, w_p // f)
            steering_model._cogx_pool_patched = True
        steer_pool = getattr(steering_model, "_cogx_steer_pool", None)

        adapter = CogVideoXSteeringAdapter(
            transformer=self._transformer,
            scheduler=self._scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            latents=latents,
            timesteps=timesteps,
            image_rotary_emb=image_rotary_emb,
            guidance_scale=video_guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            capture_blocks=getattr(steering_model, "capture_blocks", None),
            dtype=self._dtype,
            steer_pool=steer_pool,
        )
        run_steering_optimization(
            adapter,
            steering_model,
            target_words_indices=target_words_indices,
            n_iters=n_iters,
            optimizer_name=optimizer_name,
            lr=lr,
            magnitude_reg=magnitude_reg,
            anchor_reg=anchor_reg,
            pred_reg=pred_reg,
            kl_reg=kl_reg,
            entropy_reg=entropy_reg,
            step_callback=step_callback,
            desc=f"Steering [{prompt[:30]}]",
        )
        _ = audio_guidance_scale
        return self._decode(adapter.latents), None

    def save_video(
        self,
        path: str,
        video: np.ndarray,
        audio: np.ndarray,
        fps: float,
        sample_rate: int,
        rank: int,
    ) -> None:
        from diffusers.utils import export_to_video

        # video: [C, F, H, W] float in [-1, 1] → [F, H, W, C] float [0,1] frame list.
        # export_to_video does `(frame * 255).astype(uint8)` itself, so frames MUST be float
        # in [0,1]. Passing uint8 here multiplies 0-255 by 255 in uint8 arithmetic, which
        # wraps mod 256 (255·x ≡ 256−x) and stored every clip as its photographic NEGATIVE
        # (found 2026-10-09; all cogx eval clips before this fix are negated).
        vid = torch.from_numpy(video).permute(1, 2, 3, 0)  # [F, H, W, C]
        vid = ((vid.clamp(-1, 1) + 1.0) / 2.0).float().numpy()
        frames = [vid[i] for i in range(vid.shape[0])]
        export_to_video(frames, path, fps=int(fps))
        _ = (audio, sample_rate, rank)


def create_backend(
    model_type: str,
    engine_cfg,
    model_extraction_cfg: dict,
    experiment_cfg: dict,
    device: int,
    logger: logging.Logger,
) -> ModelBackend:
    """Factory that returns the appropriate ModelBackend for ``model_type``."""
    backends: dict[str, type[ModelBackend]] = {
        "ovi": OviBackend,
        "ltx": LTXBackend,
        "wan": WanBackend,
        "cogvideox": CogVideoXBackend,
    }
    if model_type not in backends:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {list(backends)}"
        )
    return backends[model_type](engine_cfg, model_extraction_cfg, experiment_cfg, device, logger)
