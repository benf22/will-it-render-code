"""Video-realism / visual-quality scorers (word-free analogue of visual_classifier.py).

Unlike the object classifiers in ``labeling/visual_classifier.py`` (which score a
*(video, word)* pair), a ``QualityScorer`` scores a whole video and returns one or more
named continuous sub-scores. Two scorers are provided:

  - ``DOVERScorer``  — DOVER, the standard standalone no-reference video quality model
    (technical + aesthetic, fused), via the ``pyiqa`` library. Captures per-frame realism
    and aesthetics.
  - ``VBenchQualityScorer`` — VBench's quality / temporal-realism dimensions
    (motion_smoothness, temporal_flickering, subject/background consistency,
    dynamic_degree, imaging_quality, aesthetic_quality). Captures temporal coherence that
    DOVER underweights. Reuses VBench exactly as ``evaluation/vbench_eval.py`` does, but
    loads the heavy submodules **once** (in ``__init__``) and scores one video at a time.

All scores are no-reference (no ground-truth video needed). Both models are trained on
natural / UGC video, so on AIGC output they are a strong but imperfect proxy.

``pyiqa`` is required for ``DOVERScorer`` (``pip install pyiqa``); DOVER / MUSIQ / LAION
weights auto-download to their caches on first use, mirroring the VBench grit weights in
``~/.cache/vbench/``.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import torch

# VBench on sys.path (mirrors labeling/visual_classifier.py and evaluation/vbench_eval.py)
_VBENCH_ROOT = Path(__file__).parents[2] / "VBench"
if str(_VBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_VBENCH_ROOT))


# VBench dimensions that measure realism / quality without needing the prompt text.
DEFAULT_VBENCH_DIMS: List[str] = [
    "motion_smoothness",
    "temporal_flickering",
    "subject_consistency",
    "background_consistency",
    "dynamic_degree",
    "imaging_quality",
    "aesthetic_quality",
]

# Defaults for the prompt-alignment (CLIP) scorer.
DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_CLIP_FRAMES = 8


class QualityScorer(ABC):
    name: str               # short identifier (e.g. "dover", "vbench")
    sub_scores: List[str]   # sub-score names this scorer emits (CSV columns sans "score_")

    @abstractmethod
    def score(self, mp4_path: str | Path) -> Dict[str, float]:
        """Return {sub_score_name: value} for one video. Higher = better quality."""


class DOVERScorer(QualityScorer):
    """DOVER no-reference video quality (technical + aesthetic, fused).

    NOTE: DOVER is **not** available through ``pyiqa`` (no ``dover`` metric/arch ships
    with it), and there is no official pip package. This scorer is an opt-in stub: to
    enable it, vendor the DOVER repo (e.g. ``~/repos/DOVER`` from github.com/teowu/DOVER)
    plus its checkpoints and wire it up here, mirroring how VBench is injected on
    ``sys.path``. Until then ``--scorers dover`` is disabled by default and constructing
    this raises a clear error.

    Returns a single fused ``dover`` score (higher = better); its scale is *not*
    normalized to [0, 1], so calibrate the labeler threshold on a sample.
    """

    name = "dover"

    def __init__(self, device: str = "cuda"):
        import pyiqa  # local import: optional dependency

        self._device = torch.device(device)
        if "dover" not in set(pyiqa.list_models()):
            raise RuntimeError(
                "DOVER is not available via pyiqa (no 'dover' metric in this install). "
                "Vendor the DOVER repo + checkpoints and wire it into DOVERScorer, or run "
                "with --scorers vbench. See the DOVERScorer docstring."
            )
        self._metric = pyiqa.create_metric("dover", device=self._device)
        self.sub_scores = ["dover"]
        print(f"DOVERScorer initialized (pyiqa dover) on {device}")

    def score(self, mp4_path: str | Path) -> Dict[str, float]:
        with torch.no_grad():
            out = self._metric(str(mp4_path))
        val = float(out.item()) if hasattr(out, "item") else float(out)
        return {"dover": val}


class CLIPPromptScorer(QualityScorer):
    """Text↔video CLIP alignment: mean per-frame CLIP cosine between a video and the
    prompt that generated it.

    Unlike the word-free realism scorers, this one is *prompt-aware*. To keep the
    ``score(mp4_path)`` interface unchanged (so the labeler and its sweep wrapper need no
    signature change), it reads the prompt itself from the sibling ``prompt_info.json``
    (top-level ``"prompt"`` key) that every extraction dir carries next to its mp4 — the
    same file ``csv_writer.n_objects_in_clip`` reads.

    Loads OpenAI CLIP (``ViT-B/32`` by default; importable in the LTX-2 venv, weights
    auto-download to ``~/.cache/clip/``) once in ``__init__``. ``score`` samples
    ``num_frames`` evenly-spaced frames with decord, encodes each with CLIP,
    L2-normalizes, and returns the **mean per-frame cosine** to the CLIP text embedding
    of the prompt. Higher = better prompt alignment (raw cosine, typically ~0.2–0.35 —
    *not* the 2.5×max(·,0) rescaling some papers apply). Returns ``nan`` when the prompt
    is missing/empty or the video cannot be read, so the labeler records it without
    poisoning the primary label.
    """

    name = "clip"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = DEFAULT_CLIP_MODEL,
        num_frames: int = DEFAULT_CLIP_FRAMES,
    ):
        import clip  # local import: optional dependency (OpenAI CLIP)

        self._device = torch.device(device)
        self._num_frames = int(num_frames)
        self._model, self._preprocess = clip.load(model_name, device=self._device)
        self._model.eval()
        self._tokenize = clip.tokenize
        self.sub_scores = ["clip"]
        print(f"CLIPPromptScorer initialized ({model_name}, {self._num_frames} frames) on {device}")

    @staticmethod
    def _prompt_for(mp4_path: Path) -> str | None:
        info = mp4_path.parent / "prompt_info.json"
        if not info.exists():
            return None
        prompt = json.loads(info.read_text()).get("prompt")
        prompt = str(prompt).strip() if prompt else ""
        return prompt or None

    def _sample_frames(self, mp4_path: str | Path):
        import numpy as np
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(str(mp4_path), ctx=cpu(0))
        n = len(vr)
        if n == 0:
            return []
        idx = np.linspace(0, n - 1, num=min(self._num_frames, n)).round().astype(int)
        frames = vr.get_batch(idx).asnumpy()  # (n, H, W, 3)
        return [Image.fromarray(f) for f in frames]

    def score(self, mp4_path: str | Path) -> Dict[str, float]:
        mp4_path = Path(mp4_path)
        prompt = self._prompt_for(mp4_path)
        if not prompt:
            return {"clip": float("nan")}
        try:
            frames = self._sample_frames(mp4_path)
        except Exception as e:  # unreadable / corrupt video
            print(f"CLIPPromptScorer: failed to read {mp4_path}: {e}")
            return {"clip": float("nan")}
        if not frames:
            return {"clip": float("nan")}

        batch = torch.stack([self._preprocess(im) for im in frames]).to(self._device)
        with torch.no_grad():
            img_f = self._model.encode_image(batch).float()
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt = self._tokenize([prompt], truncate=True).to(self._device)
            txt_f = self._model.encode_text(txt).float()
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            cos = (img_f @ txt_f.T).squeeze(1)  # (n_frames,)
        return {"clip": float(cos.mean().item())}


class ViClipPromptScorer(QualityScorer):
    """Text↔video **ViCLIP** alignment (VBench's ``overall_consistency`` dimension),
    scored per video.

    A video-native counterpart to :class:`CLIPPromptScorer`: instead of averaging
    per-frame OpenAI-CLIP cosines, it embeds the whole 8-frame clip with ViCLIP
    (InternVid-10M-FLT) and takes a single cosine to the ViCLIP text embedding of the
    prompt — the exact computation VBench uses for ``overall_consistency``. Reuses
    VBench's ``ViCLIP`` model, ``SimpleTokenizer``, ``clip_transform`` and
    ``read_frames_decord_by_fps`` unchanged; the model is built once in ``__init__``.

    Prompt-aware like :class:`CLIPPromptScorer`, reading the prompt from the sibling
    ``prompt_info.json``. Higher = better prompt alignment. Weights
    (``ViClip-InternVid-10M-FLT.pth``, ~1 GB) auto-download to ``~/.cache/vbench/ViCLIP/``
    on first use. Returns ``nan`` when the prompt is missing/empty or the video cannot
    be read.
    """

    name = "viclip"

    def __init__(self, device: str = "cuda", num_frames: int = DEFAULT_CLIP_FRAMES):
        import os

        from vbench.utils import CACHE_DIR, clip_transform, init_submodules, read_frames_decord_by_fps
        from vbench.overall_consistency import get_text_features, get_vid_features
        from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
        from vbench.third_party.ViCLIP.viclip import ViCLIP

        self._device = torch.device(device)
        self._num_frames = int(num_frames)
        submodules = init_submodules(["overall_consistency"], local=True)["overall_consistency"]
        self._tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
        self._model = ViCLIP(tokenizer=self._tokenizer, **submodules).to(self._device)
        self._transform = clip_transform(224)
        self._read_frames = read_frames_decord_by_fps
        self._get_vid = get_vid_features
        self._get_txt = get_text_features
        self.sub_scores = ["viclip"]
        print(f"ViClipPromptScorer initialized ({self._num_frames} frames) on {device}")

    def score(self, mp4_path: str | Path) -> Dict[str, float]:
        mp4_path = Path(mp4_path)
        prompt = CLIPPromptScorer._prompt_for(mp4_path)
        if not prompt:
            return {"viclip": float("nan")}
        try:
            images = self._read_frames(str(mp4_path), num_frames=self._num_frames, sample="middle")
            images = self._transform(images).to(self._device)
            with torch.no_grad():
                vid_feat = self._get_vid(self._model, images.unsqueeze(0))
                # Fresh dict per call → no unbounded GPU text-feature cache across 16k prompts.
                txt_feat = self._get_txt(self._model, prompt, self._tokenizer, {})
                cos = float((vid_feat @ txt_feat.T)[0][0].cpu())
        except Exception as e:  # unreadable / corrupt video
            print(f"ViClipPromptScorer: failed to read {mp4_path}: {e}")
            return {"viclip": float("nan")}
        return {"viclip": cos}


class VBenchQualityScorer(QualityScorer):
    """VBench quality / temporal-realism dimensions, scored one video at a time.

    Loads the (heavy) per-dimension submodules once in ``__init__`` via
    ``init_submodules`` and caches each ``compute_{dim}`` function — so model loads stay
    out of the per-video hot path. ``score`` writes a one-entry temp full_info json
    (the shape VBench's compute functions expect) and runs every dimension on it.
    """

    name = "vbench"

    def __init__(self, device: str = "cuda", dims: List[str] | None = None):
        from vbench.utils import init_submodules

        self._device = torch.device(device)
        self.dims = list(dims) if dims else list(DEFAULT_VBENCH_DIMS)
        self.sub_scores = list(self.dims)
        self._submodules = init_submodules(self.dims, local=True)
        self._compute_fns = {
            dim: getattr(importlib.import_module(f"vbench.{dim}"), f"compute_{dim}")
            for dim in self.dims
        }
        print(f"VBenchQualityScorer initialized with dims: {self.dims}")

    def score(self, mp4_path: str | Path) -> Dict[str, float]:
        mp4_path = str(mp4_path)
        prompt_en = Path(mp4_path).stem.replace("_", " ")
        # Single full_info entry listing all dims for this one video (same shape as
        # evaluation/vbench_eval.build_full_info produces for non-object dimensions).
        entries = [{"prompt_en": prompt_en, "dimension": self.dims, "video_list": [mp4_path]}]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(entries, f)
            full_info_path = f.name

        results: Dict[str, float] = {}
        try:
            for dim in self.dims:
                _, video_results = self._compute_fns[dim](
                    full_info_path, self._device, self._submodules[dim]
                )
                results[dim] = self._extract_score(video_results, mp4_path)
        finally:
            try:
                Path(full_info_path).unlink()
            except OSError:
                pass
        return results

    @staticmethod
    def _extract_score(video_results: list, mp4_path: str) -> float:
        """Pull this video's score out of VBench's per-video result list."""
        if not video_results:
            return float("nan")
        target = Path(mp4_path).resolve()
        for vr in video_results:
            if len(video_results) == 1 or Path(str(vr.get("video_path", ""))).resolve() == target:
                return float(vr["video_results"])
        return float(video_results[0]["video_results"])
