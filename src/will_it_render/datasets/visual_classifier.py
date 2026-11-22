from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set

import torch
from PIL import Image
from torchvision import transforms

_VBENCH_ROOT = Path(__file__).parents[2] / "VBench"
if str(_VBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_VBENCH_ROOT))

from vbench.utils import load_video, get_frame_indices
from vbench.object_class import get_dect_from_grit
from vbench.third_party.grit_model import DenseCaptioning

# Frame count the legacy (pre-per-frame) pipeline scored: VBench middle-samples
# 16 frames per clip. The `score_<clf>` / `label_<clf>` / `label` CSV columns are
# reproduced exactly by sub-selecting these same indices from the native frames.
LEGACY_NUM_FRAMES = 16


# Frame-colour handling. VBench ``load_video`` returns float32 in [0, 255]; torchvision
# ``ToPILImage`` treats float input as [0, 1] and does ``mul(255).byte()`` → 255·x mod 256,
# i.e. an exact photographic NEGATIVE. Every PIL-based classifier (GDINO, OWLv2, YOLO, VLM)
# has scored negatives since 2026-05-31; GRiT (tensor path) saw true colours. Both flags
# default OFF so existing runs/labels reproduce byte-for-byte. Set once per process via
# ``set_frame_colour`` (the labelers read them from run_config.json / CLI).
#   fix_pil_negation: cast to uint8 before ToPILImage → PIL classifiers see true colours.
#   source_negated:   the mp4 itself is stored as a negative (CogX clips before 2026-10-09)
#                     → invert on load, for every classifier (GRiT included).
_FIX_PIL_NEGATION = False
_SOURCE_NEGATED = False


def set_frame_colour(fix_pil_negation: bool = False, source_negated: bool = False) -> None:
    global _FIX_PIL_NEGATION, _SOURCE_NEGATED
    _FIX_PIL_NEGATION = bool(fix_pil_negation)
    _SOURCE_NEGATED = bool(source_negated)
    print(f"frame colour: fix_pil_negation={_FIX_PIL_NEGATION} source_negated={_SOURCE_NEGATED}")


def _load_video(mp4_path: str | Path, num_frames: int | None = None) -> torch.Tensor:
    """VBench ``load_video`` ([T, C, H, W] float32 in [0, 255]), un-negated if the source is."""
    video_tensor = load_video(str(mp4_path), num_frames=num_frames)
    if _SOURCE_NEGATED:
        video_tensor = 255.0 - video_tensor
    return video_tensor


def legacy_frame_indices(n_frames: int, num_frames: int = LEGACY_NUM_FRAMES) -> list[int]:
    """Indices into a native-length per-frame list that reproduce the frames the
    old pipeline decoded (``load_video(num_frames=16)`` → VBench middle-sampling)."""
    return [int(i) for i in get_frame_indices(num_frames, n_frames, sample="middle")]


class VisualClassifier(ABC):
    name: str  # short identifier used as CSV column suffix

    @abstractmethod
    def in_vocabulary(self, word: str) -> bool:
        """Return True if this model can potentially detect the given word."""

    @abstractmethod
    def score_frames(
        self, mp4_path: str | Path, target_word: str, quick_frames: int | None = None
    ) -> list[int]:
        """Return per-frame 0/1 detections for target_word.

        ``quick_frames=None`` scores every native frame (length = native frame
        count). ``quick_frames=N`` (quick mode) decodes and scores only N frames
        evenly middle-sampled from the clip (VBench ``get_frame_indices``), so the
        returned list has length ``min(N, native_frames)`` and inference runs on
        N frames instead of all of them.
        """

    def score(
        self, mp4_path: str | Path, target_word: str, quick_frames: int | None = None
    ) -> float:
        """Fraction of frames [0, 1] where target_word was detected."""
        frames = self.score_frames(mp4_path, target_word, quick_frames=quick_frames)
        return sum(frames) / len(frames) if frames else 0.0


class GRiTClassifier(VisualClassifier):
    """Closed-set object detector using VBench's GRiT (LVIS v1, 1203 categories)."""

    name = "grit"

    def __init__(self, device: str = "cpu"):
        model_weight = Path.home() / ".cache" / "vbench" / "grit_model" / "grit_b_densecap_objectdet.pth"
        self._model = DenseCaptioning(torch.device(device))
        self._model.initialize_model_det(model_weight=str(model_weight))
        self._vocab: Set[str] = self._build_vocab()
        print(f"GRiTClassifier initialized with {len(self._vocab)} unique object names in vocabulary")

    @staticmethod
    def _build_vocab() -> Set[str]:
        # Import after VBench path injection so detectron2 is reachable.
        from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
        names = {cat["name"].lower() for cat in LVIS_CATEGORIES}
        # Also index individual words within compound names (e.g. "golden_retriever" → "retriever").
        words = {w for name in names for w in name.split("_")}
        return names | words

    def in_vocabulary(self, word: str) -> bool:
        return word.lower() in self._vocab

    def score_frames(
        self, mp4_path: str | Path, target_word: str, quick_frames: int | None = None
    ) -> list[int]:
        # quick_frames=None → all native frames; quick_frames=N → N evenly middle-sampled.
        video_tensor = _load_video(mp4_path, num_frames=quick_frames)
        _, _, h, w = video_tensor.size()
        if min(h, w) > 768:
            scale = 720.0 / min(h, w)
            video_tensor = transforms.Resize(size=(int(scale * h), int(scale * w)))(video_tensor)

        per_frame_sets = get_dect_from_grit(self._model, video_tensor.permute(0, 2, 3, 1))
        target = target_word.lower()
        return [
            int(any(target == cls.lower() or target in cls.lower().split("_") for cls in pred))
            for pred in per_frame_sets
        ]

    def score_pil_frames(self, frames: "list[Image.Image]", target_word: str) -> list[int]:
        import numpy as np
        arrs = [np.array(f.convert("RGB"), dtype=np.uint8) for f in frames]
        video_tensor = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)  # [T, C, H, W]
        _, _, h, w = video_tensor.size()
        if min(h, w) > 768:
            scale = 720.0 / min(h, w)
            video_tensor = transforms.Resize(size=(int(scale * h), int(scale * w)))(video_tensor)

        per_frame_sets = get_dect_from_grit(self._model, video_tensor.permute(0, 2, 3, 1))
        target = target_word.lower()
        return [
            int(any(target == cls.lower() or target in cls.lower().split("_") for cls in pred))
            for pred in per_frame_sets
        ]


def _video_to_pil_frames(mp4_path: str | Path, num_frames: int | None = None) -> list[Image.Image]:
    """Load a video and return a list of PIL Images. num_frames=None loads all native frames."""
    video_tensor = _load_video(mp4_path, num_frames=num_frames)  # [T, C, H, W], float32 0-255
    if _FIX_PIL_NEGATION:
        video_tensor = video_tensor.round().clamp(0, 255).to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return [to_pil(frame) for frame in video_tensor]


class GroundingDINOClassifier(VisualClassifier):
    """Open-vocabulary object detector using Grounding DINO via HuggingFace transformers."""

    name = "gdino"

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cpu",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        print(f"GroundingDINOClassifier initialized ({model_id})")

    def in_vocabulary(self, word: str) -> bool:
        return True  # open-vocabulary

    def score_frames(
        self, mp4_path: str | Path, target_word: str, quick_frames: int | None = None
    ) -> list[int]:
        frames = _video_to_pil_frames(mp4_path, num_frames=quick_frames)
        return self.score_pil_frames(frames, target_word)

    def score_pil_frames(self, frames: "list[Image.Image]", target_word: str) -> list[int]:
        # Grounding DINO expects the text prompt to end with a period.
        text_prompt = f"{target_word.lower()}."
        per_frame: list[int] = []
        with torch.no_grad():
            for frame in frames:
                inputs = self._processor(
                    images=frame, text=text_prompt, return_tensors="pt"
                ).to(self._device)
                outputs = self._model(**inputs)
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=self._box_threshold,
                    text_threshold=self._text_threshold,
                    target_sizes=[frame.size[::-1]],
                )
                per_frame.append(int(len(results[0]["scores"]) > 0))
        return per_frame


class OWLv2Classifier(VisualClassifier):
    """Open-vocabulary object detector using OWL-ViT v2 via HuggingFace transformers."""

    name = "owl"

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: str = "cpu",
        score_threshold: float = 0.1,
    ):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self._processor = Owlv2Processor.from_pretrained(model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
        self._device = device
        self._score_threshold = score_threshold
        print(f"OWLv2Classifier initialized ({model_id})")

    def in_vocabulary(self, word: str) -> bool:
        return True  # open-vocabulary

    def score_frames(
        self, mp4_path: str | Path, target_word: str, quick_frames: int | None = None
    ) -> list[int]:
        frames = _video_to_pil_frames(mp4_path, num_frames=quick_frames)
        return self.score_pil_frames(frames, target_word)

    def score_pil_frames(self, frames: "list[Image.Image]", target_word: str) -> list[int]:
        text_queries = [[f"a {target_word.lower()}"]]
        per_frame: list[int] = []
        with torch.no_grad():
            for frame in frames:
                inputs = self._processor(
                    text=text_queries, images=frame, return_tensors="pt"
                ).to(self._device)
                outputs = self._model(**inputs)
                target_sizes = torch.tensor([frame.size[::-1]], device=self._device)
                results = self._processor.post_process_grounded_object_detection(
                    outputs, threshold=self._score_threshold, target_sizes=target_sizes
                )
                per_frame.append(int(len(results[0]["scores"]) > 0))
        return per_frame


class YoloV10Classifier(VisualClassifier):
    """Closed-set object detector using YOLOv10x (COCO, 80 classes).

    The detector TempoControl adopts for its temporal-accuracy metric (it replaced VBench's
    GRiT, unreliable on small / partial objects). Vocabulary is YOLO's 80 COCO classes —
    which are exactly the single-object benchmark's 80 prompts. Needs ``ultralytics``
    (lazy import).
    """

    name = "yolov10"

    def __init__(self, weights: str = "yolov10x.pt", device: str = "cpu",
                 conf: float = 0.25):
        # TempoControl loads `jameslahm/yolov10x` via the old `YOLOv10.from_pretrained`
        # API; ultralytics ≥8.4 dropped that top-level class, so prefer the generic
        # `YOLO(weights)` loader with the official `yolov10x.pt` (identical 80 COCO
        # classes — the single-object benchmark's vocabulary). Fall back to the HF
        # `YOLOv10.from_pretrained` path for older ultralytics.
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(weights if weights.endswith(".pt") else "yolov10x.pt")
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "YoloV10Classifier needs `ultralytics` (pip install ultralytics)."
            ) from exc
        except Exception:  # weights name not a local .pt → try the HF from_pretrained API
            from ultralytics import YOLOv10  # type: ignore
            self._model = YOLOv10.from_pretrained(weights)
        self._device = None if device == "cpu" else device
        self._conf = conf
        self._names = {n.lower() for n in self._model.names.values()}
        print(f"YoloV10Classifier initialized ({weights}, {len(self._names)} COCO classes)")

    def in_vocabulary(self, word: str) -> bool:
        return word.lower() in self._names

    def _detect(self, frames) -> "list[set[str]]":
        import numpy as np
        sets: list[set[str]] = []
        for f in frames:
            arr = f if isinstance(f, np.ndarray) else np.array(f.convert("RGB"))[:, :, ::-1]
            out = self._model.predict(source=arr, save=False, verbose=False,
                                      conf=self._conf, device=self._device)
            r = out[0]
            sets.append({r.names[int(k)].lower() for k in r.boxes.cls.cpu().numpy()})
        return sets

    def score_frames(self, mp4_path, target_word, quick_frames=None) -> list[int]:
        frames = _video_to_pil_frames(mp4_path, num_frames=quick_frames)
        return self.score_pil_frames(frames, target_word)

    def score_pil_frames(self, frames: "list[Image.Image]", target_word: str) -> list[int]:
        target = target_word.lower()
        return [int(target in s) for s in self._detect(frames)]


class VLMClassifier(VisualClassifier):
    """Vision-language-model judge: per-frame yes/no presence via first-token log-probs.

    Asks a *neutral* question per frame — the prompt that generated the clip is never shown, so
    the model cannot be primed by it — and reads the probability of "Yes" vs "No" from the first
    generated token's logits (no free-text parsing, no sampling). ``p(yes) >= threshold`` → the
    frame counts as a detection, so the output matches the detector contract (per-frame 0/1) and
    plugs into the existing majority / frames_final machinery.

    Default model: Qwen2.5-VL-32B-Instruct (bf16). Set ``model_id`` to the 72B for the most
    reliable judge (needs ~145 GB → one B200/H200, or 2×80 GB with device_map="auto"), or to the
    7B for a fast pass. Frames are batched in one forward.
    """

    name = "vlm"
    QUESTION = ("Look at this image. Is there a clearly visible {word} in it? "
                "Answer with a single word: Yes or No.")

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        device: str = "cuda",
        threshold: float = 0.5,
        max_pixels: int = 768 * 28 * 28,
        batch_size: int = 8,
    ):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="auto" if device.startswith("cuda") else None,
        ).eval()
        self._device = device
        self._threshold = float(threshold)
        self._batch = int(batch_size)
        tok = self._processor.tokenizer
        # first-token ids for the two answers (with and without the leading space variant)
        self._yes_ids = {tok.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes", " yes")}
        self._no_ids = {tok.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no", " no")}
        print(f"VLMClassifier initialized ({model_id}, threshold={threshold})")

    def in_vocabulary(self, word: str) -> bool:
        return True  # open vocabulary

    def _p_yes(self, frames, question: str) -> list[float]:
        out: list[float] = []
        for i in range(0, len(frames), self._batch):
            chunk = frames[i:i + self._batch]
            msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                    for _ in chunk]
            texts = [self._processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in msgs]
            inputs = self._processor(text=texts, images=[[f] for f in chunk], padding=True,
                                     return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            for row in probs:
                py = float(sum(row[i] for i in self._yes_ids))
                pn = float(sum(row[i] for i in self._no_ids))
                out.append(py / (py + pn + 1e-9))
        return out

    def score_frames(
        self, mp4_path: str | Path, target_word: str, quick_frames: int | None = None
    ) -> list[int]:
        frames = _video_to_pil_frames(mp4_path, num_frames=quick_frames)
        ps = self._p_yes(frames, self.QUESTION.format(word=target_word.lower()))
        return [int(p >= self._threshold) for p in ps]
