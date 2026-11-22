from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

_AV_BENCH_ROOT = Path(__file__).parents[3] / "av-benchmark"
if str(_AV_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_AV_BENCH_ROOT))

# AudioSet class label CSV — prefer panns_inference install, fall back to known path
_FALLBACK_CSV = Path(__file__).parents[3] / "LanguageBind" / "a_cls" / "class_labels_indices.csv"


def _load_audioset_labels() -> Dict[int, str]:
    """Return {index: display_name} for all AudioSet classes."""
    try:
        import panns_inference
        csv_path = Path(panns_inference.__file__).parent / "class_labels_indices.csv"
    except ImportError:
        csv_path = _FALLBACK_CSV

    labels: Dict[int, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[int(row["index"])] = row["display_name"]
    return labels


TARGET_TO_AUDIOSET: Dict[str, List[str]] = {
    "dog": ["Bark", "Bow-wow", "Howl", "Dog", "Whimper (dog)"],
    "cat": ["Meow", "Purr", "Hiss", "Cat"],
    "car": ["Car", "Vehicle", "Engine", "Motor vehicle (road)", "Car passing by"],
    "horn": ["Car alarm", "Beep, bleep", "Horn"],
    "bird": ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk"],
    "rain": ["Rain", "Raindrop", "Rain on surface"],
    "thunder": ["Thunder", "Thunderstorm"],
    "music": ["Music", "Musical instrument"],
    "speech": ["Speech", "Narration, monologue"],
    "baby": ["Baby cry, infant cry", "Crying, sobbing", "Baby laughter"],
    "fire": ["Fire", "Crackling"],
    "water": ["Water", "Stream", "Splash, splatter", "Pour"],
    "wind": ["Wind", "Whoosh, swoosh, swish"],
    "knock": ["Knock", "Tap"],
    "footstep": ["Footsteps"],
    "laugh": ["Laughter"],
    "cough": ["Cough"],
    "sneeze": ["Sneeze"],
    "glass": ["Glass", "Shatter"],
    "explosion": ["Explosion"],
}


class AudioSetClassifier:
    def __init__(self, device: str = "cpu", sample_rate: int = 16000):
        from av_bench.panns import Cnn14

        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.model = Cnn14(
            features_list=["logits"],
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        ).to(self.device).eval()

        self._labels = _load_audioset_labels()
        self._name_to_indices: Dict[str, List[int]] = {}
        for idx, name in self._labels.items():
            self._name_to_indices.setdefault(name, []).append(idx)

    def _resolve_indices(self, target_name: str) -> List[int]:
        cache_key = target_name.lower()
        if cache_key in self._name_to_indices:
            return self._name_to_indices[cache_key]

        audioset_names = TARGET_TO_AUDIOSET.get(cache_key, [target_name])
        indices: List[int] = []
        for aname in audioset_names:
            for idx, label in self._labels.items():
                if aname.lower() in label.lower() or label.lower() in aname.lower():
                    indices.append(idx)
        return sorted(set(indices))

    @torch.no_grad()
    def score(self, wav_path: str, target_name: str) -> float:
        """Return soft probability [0, 1] that target_name is present in the WAV."""
        indices = self._resolve_indices(target_name)
        if not indices:
            return 0.0

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        audio = waveform.squeeze(0).to(self.device)  # [T]

        out = self.model(audio.unsqueeze(0))
        logits = out["logits"]  # [1, 527]
        probs = torch.sigmoid(logits[0, indices])
        return probs.max().item()
