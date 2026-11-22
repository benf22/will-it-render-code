from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List
from typing import Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.data import BatchSampler, Dataset
from tqdm import tqdm

from will_it_render.training.augment import parse_clip_percentiles, percentile_clip


def bits_to_latent_frames(
    bits: str, n_latent: int, mode: str = "majority", model_type: str = "ltx"
) -> list[float]:
    """Object-labeler ``frames_final`` bit-string → one presence target per LATENT frame.

    The labeler emits one character per *native* decoded frame (``'1'``/``'0'``/``'u'`` for a
    detector tie), while the probe sees the VAE's latent frames. LTX-2's temporal VAE is
    causal with factor ``s``: the first frame is uncompressed and every later latent frame
    covers ``s`` native frames, so ``native == 1 + s*(n_latent-1)`` — for the 512×512×97
    config, 97 native → 13 latent with ``s = 8``:

        latent 0  ← native [0]
        latent t  ← native [s*(t-1)+1 … s*t]      for t = 1 … n_latent-1

    ``s`` is derived from the string length and the mapping is verified exactly, so a
    quick-mode file (e.g. ``--quick-frames 32``, whose frames are VBench middle-samples and
    do *not* tile the timeline this way) raises instead of being silently mis-mapped.

    ``'u'`` characters are excluded from a bin's denominator — the repo-wide convention for
    undecided labels (cf. ``evaluation/metrics_utils.py``). A bin with no decided frame
    yields ``nan``, which the training loop masks out of the loss.

    ``mode``: ``majority`` → ``float(ones/decided >= 0.5)`` (default) | ``any`` →
    ``float(ones > 0)`` | ``soft`` → the raw ``ones/decided`` fraction.
    """
    # The native→latent frame grouping below is a CAUSAL-VAE tiling: latent 0 covers native
    # frame 0 alone, every later latent frame covers ``s`` native frames, so
    # ``native == 1 + s*(n_latent-1)``. This holds for every backend we support, only the
    # temporal factor differs — LTX-2 s=8 (97 native → 13 latent), WAN 2.1 s=4 (81 native →
    # 21 latent, verified against the 2026_08_16 labeling), CogVideoX-5B s=4 (49 native → 13
    # latent; 1 + 4*(13-1) == 49, the same causal stride-4 VAE as WAN, confirmed against the
    # 2026_09_30 cogvideox extraction's frames_final). ``s`` is derived from the string length
    # below and the tiling is checked exactly, so a mismatch still raises. Other backends with
    # a non-causal or differently-patchified temporal VAE must be validated before being added
    # here.
    if model_type not in ("ltx", "wan", "cogvideox"):
        raise NotImplementedError(
            f"bits_to_latent_frames encodes a causal-VAE frame grouping validated for "
            f"model_type in ('ltx','wan','cogvideox'); got {model_type!r}. Verify this backend's VAE uses "
            f"the same native==1+s*(n_latent-1) tiling before enabling per-frame targets "
            f"(training/probe_dataset.py)."
        )
    if mode not in ("majority", "any", "soft"):
        raise ValueError(f"Unknown per-frame mode {mode!r}. Choose 'majority', 'any' or 'soft'.")
    if n_latent < 2:
        raise ValueError(f"n_latent must be >= 2 (got {n_latent}).")
    n_native = len(bits)
    stride = (n_native - 1) // (n_latent - 1)
    if stride < 1 or n_native != 1 + stride * (n_latent - 1):
        raise ValueError(
            f"per-frame bit-string of length {n_native} does not tile {n_latent} latent "
            f"frames (expected 1 + s*({n_latent}-1) for an integer VAE stride s). Per-frame "
            f"labels must come from a FULL-frame labeling run (no --quick-frames)."
        )
    out: list[float] = []
    for t in range(n_latent):
        chunk = bits[0:1] if t == 0 else bits[stride * (t - 1) + 1: stride * t + 1]
        decided = len(chunk) - chunk.count("u")
        if decided == 0:
            out.append(float("nan"))
            continue
        frac = chunk.count("1") / decided
        if mode == "majority":
            out.append(float(frac >= 0.5))
        elif mode == "any":
            out.append(float(chunk.count("1") > 0))
        else:
            out.append(frac)
    return out


class ProbeDataset(Dataset):
    def __init__(
        self,
        labels_csv: str | Path,
        denoising_step: int,
        n_blocks_heads: Optional[int] = None,
        latent_thw: Optional[tuple[int, int, int]] = None,
        avg_mode: str = "none",
        block_index: Optional[int] = None,
        n_blocks: Optional[int] = None,
        capture_blocks: Optional[list[int]] = None,
        n_blocks_total: Optional[int] = None,
        layer_subset: Optional[list[int]] = None,
        root_override: Optional[str | Path] = None,
        preload: bool = True,
        device: Optional[str | torch.device] = None,
        half: bool = False,
        pin_memory: bool = False,
        aux_csv: Optional[str | Path] = None,
        aux_col: str = "score_aesthetic_quality",
        perframe_csv: Optional[str | Path] = None,
        perframe_col: str = "frames_final",
        perframe_T: Optional[int] = None,
        perframe_mode: str = "majority",
        class_head: bool = False,
        class_vocab: Optional[dict[str, int]] = None,
        train_spatial_downsample: int = 1,
        feature_cache_dir: Optional[str | Path] = None,
        model_type: str = "ltx",
        aug_w_path: Optional[str | Path] = None,
        aug_w_prob: float = 0.0,
        aug_w_alpha: tuple[float, float] = (0.1, 0.4),
        aug_w_on: str = "neg",
        clip_percentiles: Optional[Sequence[float]] = None,
        clip_device: Optional[str | torch.device] = None,
    ):
        # ---- steering-direction augmentation (see _augment) ---------------------------
        # `aug_w_path` holds `w_pooled`: a unit direction over the n_blocks_heads channels,
        # measured as the per-(block,head) attention-mass shift that separates STEERED from
        # natural maps. Re-weighting a natural map along -w reproduces the probe gain real
        # steering produces, so applying it to NEGATIVES and keeping the label 0 is the
        # hard-negative the probe currently has no defence against.
        self.aug_w = None
        if aug_w_path is not None and float(aug_w_prob) > 0:
            import numpy as _np
            _z = _np.load(str(aug_w_path))
            _w = _z["w_pooled"] if "w_pooled" in _z.files else _z[_z.files[0]]
            self.aug_w = torch.from_numpy(_np.asarray(_w, dtype=_np.float32))
        self.aug_w_prob = float(aug_w_prob)
        self.aug_w_alpha = tuple(float(a) for a in aug_w_alpha)
        self.aug_w_on = str(aug_w_on)
        self.model_type = model_type
        self.labels_csv = str(labels_csv)
        # Local-disk cache of the fully-preloaded (post avg_mode / spatial-pool / fp16) feature
        # tensor + labels + records, keyed by everything that determines them. The first build
        # pays the (slow, NFS-bound) per-sample load; later runs with the SAME feature shape
        # (e.g. a hyperparameter/arch sweep over the same pooled stack) load the whole tensor in
        # one sequential read. Only the INPUT tensor is cached, so it is shared across all probe
        # arch/LR/epoch variants that consume the same features. None disables it.
        self.feature_cache_dir = str(feature_cache_dir) if feature_cache_dir is not None else None
        # Train-time 2D average-pool of the spatial grid (h, w) by this integer factor,
        # applied AFTER the feature is reshaped to (C, T, h, w) in _load_feat. Features are
        # stored at full spatial resolution (downsample=1 at extraction); this shrinks the
        # per-sample tensor at load time so a bigger backend (e.g. WAN's 30x52 grid) fits
        # GPU-resident. latent_thw here still describes the FULL grid (used for the reshape);
        # the caller must build the probe with the POOLED (h//f, w//f) dims. Mirrors
        # data_extraction/aggregator.py::_spatial_downsample.
        self.train_spatial_downsample = int(train_spatial_downsample)
        if self.train_spatial_downsample < 1:
            raise ValueError(
                f"train_spatial_downsample must be >= 1 (got {train_spatial_downsample})."
            )
        df = pd.read_csv(str(labels_csv), skip_blank_lines=True).dropna(
            subset=["root_extraction_dir", "relative_extraction_dir", "label"]
        )
        n_ignored = int((df["label"] < 0).sum())
        if n_ignored:
            print(f"[ProbeDataset] step {denoising_step}: ignored {n_ignored} records with label < 0")
        df = df[df["label"] >= 0]
        # Optional auxiliary (multi-task) target, e.g. the video's aesthetic score, joined
        # from a separate per-video CSV on (root_extraction_dir, relative_extraction_dir).
        # Aesthetic is a per-CLIP property, so it's keyed by the clip's absolute dir and shared
        # across that clip's per-object rows. Clips absent from aux_csv get NaN (masked in loss).
        self.multitask = aux_csv is not None
        self.aux_col = aux_col
        self._aux_by_dir: dict[str, float] = {}
        aux_lookup: dict[tuple[str, str], float] = {}
        if self.multitask:
            adf = pd.read_csv(str(aux_csv), skip_blank_lines=True)
            if aux_col not in adf.columns:
                raise ValueError(f"aux_col {aux_col!r} not in {aux_csv} (cols: {list(adf.columns)})")
            for _, ar in adf.iterrows():
                aux_lookup[(str(ar["root_extraction_dir"]), str(ar["relative_extraction_dir"]))] = float(ar[aux_col])
        # Optional per-latent-frame presence targets, joined from one or more object-labeler
        # CSVs (comma-separated) carrying the `frames_final` bit-string. Unlike the aesthetic
        # aux — a per-CLIP property — presence is per (clip, OBJECT), so the join key carries
        # object_name too. Rows absent from the CSVs get an all-NaN target (masked in the
        # loss), never a negative.
        self.perframe = perframe_csv is not None
        self.perframe_col = perframe_col
        self.perframe_mode = perframe_mode
        self.perframe_T = int(perframe_T) if perframe_T is not None else (
            int(latent_thw[0]) if latent_thw is not None else 0
        )
        self._perframe_by_key: dict[tuple[str, str], list[float]] = {}
        perframe_lookup: dict[tuple[str, str, str], str] = {}
        if self.perframe:
            if self.perframe_T < 2:
                raise ValueError(
                    "perframe_csv given but the latent frame count is unknown; pass "
                    "perframe_T (or latent_thw) so the bit-strings can be binned."
                )
            paths = [p.strip() for p in str(perframe_csv).split(",") if p.strip()]
            for path in paths:
                pdf = pd.read_csv(path, skip_blank_lines=True)
                if perframe_col not in pdf.columns:
                    raise ValueError(
                        f"perframe_col {perframe_col!r} not in {path} (cols: {list(pdf.columns)})"
                    )
                for _, pr in pdf.iterrows():
                    bits = pr[perframe_col]
                    if not isinstance(bits, str) or not bits:
                        continue  # no in-vocabulary detector for this word → no target
                    perframe_lookup[(
                        str(pr["root_extraction_dir"]),
                        str(pr["relative_extraction_dir"]),
                        str(pr["object_name"]),
                    )] = bits
        # Optional class-identity target: an integer class index per record, mapped from the
        # `object_name` column (which is the canonical class — dog / cat / … — set at labeling
        # time from prompt_info.json["class_name"]). class_vocab is the {name: index} map built
        # once from the train split; object_names absent from it (e.g. a val-only class) get -1,
        # which the training loop masks out of the cross-entropy. Packed as the LAST label
        # column (after class-presence, aesthetic aux, and per-frame columns).
        self.class_head = bool(class_head)
        self.class_vocab = dict(class_vocab) if class_vocab is not None else None
        if self.class_head and not self.class_vocab:
            raise ValueError("class_head=True requires a non-empty class_vocab {name: index}.")
        self.records = []
        missing = 0
        for _, row in df.iterrows():
            root = root_override if root_override is not None else row["root_extraction_dir"]
            extraction_dir = Path(root) / row["relative_extraction_dir"]
            feat_path = extraction_dir / f"step_{denoising_step:03d}.safetensors"
            if not feat_path.exists():
                missing += 1
                continue
            self.records.append((
                str(extraction_dir),
                denoising_step,
                float(row["label"]),
                str(row["object_name"]),
            ))
            if self.multitask:
                # Key by absolute extraction_dir (== record[0]) so the aux value survives the
                # missing/corrupt record filtering and is found regardless of root_override.
                self._aux_by_dir[str(extraction_dir)] = aux_lookup.get(
                    (str(row["root_extraction_dir"]), str(row["relative_extraction_dir"])), float("nan")
                )
            if self.perframe:
                # Keyed by (absolute extraction_dir, object_name) == (record[0], record[3])
                # so the target survives root_override and the missing/corrupt filtering.
                bits = perframe_lookup.get((
                    str(row["root_extraction_dir"]),
                    str(row["relative_extraction_dir"]),
                    str(row["object_name"]),
                ))
                self._perframe_by_key[(str(extraction_dir), str(row["object_name"]))] = (
                    bits_to_latent_frames(bits, self.perframe_T, perframe_mode, self.model_type)
                    if bits is not None else [float("nan")] * self.perframe_T
                )
        if self.perframe:
            n_cov = sum(1 for v in self._perframe_by_key.values() if not all(x != x for x in v))
            n_tot = len(self._perframe_by_key)
            print(f"[ProbeDataset] step {denoising_step}: per-frame targets for {n_cov}/{n_tot} "
                  f"records ({n_cov / n_tot:.4f})" if n_tot else "[ProbeDataset] no records")
        if missing:
            print(f"[ProbeDataset] step {denoising_step}: skipped {missing} records with missing safetensors")
        self.n_blocks_heads = n_blocks_heads
        self.latent_thw = latent_thw
        # Percentile clipping (outlier removal) on the RAW per-(block, head) maps — applied
        # before any block averaging/slicing, so the band is a property of the map the model
        # actually saw during extraction and is identical for every avg_mode. Deterministic
        # preprocessing, so it runs here (once, at load) for BOTH train and val rather than
        # as a per-batch train-only augmentation.
        self.clip_percentiles = parse_clip_percentiles(clip_percentiles)
        # The two kthvalue selections cost ~54 ms/sample on CPU — 27 min over a 30k-row
        # preload. Run them on the training GPU instead (~20 MB round-trip per sample); the
        # result is bit-identical, since order statistics and clamp are exact.
        self.clip_device = torch.device(clip_device) if clip_device is not None else None
        # avg_mode: "none" → [n_blocks_heads, T, h, w] | "all" → [1, T, h, w]
        #           "layers" → average over blocks, keep heads → [n_heads, T, h, w]
        #           "stack"  → blocks kept un-averaged → [n_blocks*n_heads, T, h, w]
        #                      (block-major; for learned layer aggregation in the model,
        #                       e.g. ResNetProbe layer_agg. Differs from "none" in the
        #                       capture_blocks case: no NaN-scatter to n_blocks_total.)
        self.avg_mode = avg_mode
        self.block_index = block_index
        self.n_blocks = n_blocks
        # When the extraction stored only a subset of transformer blocks, these map
        # the stored blocks back to their original indices (the rest are zero-filled
        # at load time so block_index keeps its original-48 meaning).
        self.capture_blocks = list(capture_blocks) if capture_blocks is not None else None
        self.n_blocks_total = n_blocks_total
        # layer_subset: train on only this subset of the STORED transformer blocks (by ORIGINAL
        # block index). Only meaningful for avg_mode in ("stack", "layers"); it slices the layer
        # axis at load time so the resident tensor holds just the selected blocks. _subset_positions
        # maps the requested original indices to their positions in the stored block array
        # (identity when capture_blocks is None, since stored position == original index).
        self.layer_subset = [int(b) for b in layer_subset] if layer_subset is not None else None
        self._subset_positions: Optional[list[int]] = None
        if self.layer_subset is not None:
            if self.capture_blocks is not None:
                pos = []
                for b in self.layer_subset:
                    if b not in self.capture_blocks:
                        raise ValueError(
                            f"layer_subset block {b} not in capture_blocks {self.capture_blocks}"
                        )
                    pos.append(self.capture_blocks.index(b))
                self._subset_positions = pos
            else:
                self._subset_positions = list(self.layer_subset)  # stored pos == original index

        # When set, the preloaded feature/label tensors are moved to this device
        # once (GPU-resident data) so training reads batches straight from VRAM
        # with no per-batch host→device copy. None keeps them in CPU RAM, which is
        # the streaming path (per-batch .to(device), benefits from pin_memory +
        # non_blocking). Only meaningful when preload=True.
        self.device = torch.device(device) if device is not None else None
        # half=True stores the preloaded feature tensor in float16 (halving the resident
        # footprint, e.g. so the un-averaged "stack" features fit GPU-resident on a B200).
        # Features are upcast back to float32 per-sample in __getitem__, so the probe still
        # trains in fp32 and nothing downstream changes — only the storage dtype differs.
        self.half = half
        # pin_memory=True allocates the CPU-resident tensor in pinned host memory so the
        # GpuBatchStreamer can copy streamed batches to the GPU at full bandwidth (only
        # meaningful when device is None / CPU-resident).
        self.pin_memory = pin_memory

        self._data: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        if preload:
            cache_path = self._cache_path(denoising_step) if self.feature_cache_dir else None
            if cache_path is not None and cache_path.exists():
                self._load_from_cache(cache_path, denoising_step)
            else:
                self._preload(denoising_step)
                if cache_path is not None:
                    self._save_to_cache(cache_path)

    def _cache_path(self, step: int) -> Path:
        """Local-disk path for this dataset's cached preloaded tensor. The key hashes every
        field that changes the INPUT tensor or the packed labels, so an incompatible request
        never reuses a stale cache."""
        import hashlib
        key = "|".join(str(x) for x in [
            self.labels_csv, step, self.avg_mode, self.train_spatial_downsample,
            int(self.half), self.n_blocks_heads, self.latent_thw, self.n_blocks,
            self.block_index, self.layer_subset, self.capture_blocks, self.n_blocks_total,
            self.multitask, getattr(self, "aux_col", None),
            self.perframe, self.perframe_col if self.perframe else None,
            self.perframe_T, self.perframe_mode,
            self.class_head,
            tuple(sorted(self.class_vocab.items())) if self.class_head else None,
        ])
        # Appended only when set, so every pre-existing cache keeps its hash.
        if getattr(self, "clip_percentiles", None) is not None:
            key += f"|clip{tuple(self.clip_percentiles)}"
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        stem = Path(self.labels_csv).stem
        fp = 16 if self.half else 32
        return Path(self.feature_cache_dir) / (
            f"{stem}_step{step:03d}_{self.avg_mode}_pool{self.train_spatial_downsample}_fp{fp}_{h}.pt"
        )

    def _load_from_cache(self, path: Path, step: int) -> None:
        """Load a previously-built cache and place the tensors exactly where _preload would."""
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        data, labels = blob["data"], blob["labels"]
        self.records = blob["records"]
        if self.half and data.dtype != torch.float16:
            data = data.half()
        target = self.device
        if target is not None:                     # GPU-resident: move both to the device
            data = data.to(target)
            labels = labels.to(target)
        elif self.pin_memory:                      # streamer path: page-lock the host tensor
            data = data.pin_memory()
        self._data, self._labels = data, labels
        gb = self._data.numel() * self._data.element_size() / 1e9
        loc = str(self.device) if self.device is not None else "cpu"
        print(f"[ProbeDataset] step {step}: loaded feature cache {path.name} "
              f"({self._data.shape}, {gb:.2f} GB) on {loc}")

    def _save_to_cache(self, path: Path) -> None:
        """Atomically write the preloaded tensor + labels + records to local disk (tmp+rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"data": self._data.detach().to("cpu"),
                    "labels": self._labels.detach().to("cpu"),
                    "records": self.records}, str(tmp))
        tmp.rename(path)
        gb = self._data.numel() * self._data.element_size() / 1e9
        print(f"[ProbeDataset] wrote feature cache {path.name} ({gb:.2f} GB)")

    def _load_feat(self, extraction_dir: str, step: int, word: str) -> torch.Tensor:
        f = self.train_spatial_downsample
        if f > 1 and self.n_blocks_heads is not None and self.latent_thw is not None:
            # A root may store features ALREADY pooled by f (steer-time captures built with
            # attn_spatial_downsample=f, e.g. the step-7 augmentation set): the flat size is
            # 1/f^2 of the full grid. Reshape those with the pooled grid and skip the pool —
            # pooling twice would silently shrink them to h/f^2.
            T, h, w = self.latent_thw
            pooled = (T, h // f, w // f)
            raw = self._read_feat(extraction_dir, step, word)
            if raw.numel() == self.n_blocks_heads * T * pooled[1] * pooled[2]:
                return self._reshape_feat(raw, pooled)
            feat = self._reshape_feat(raw, self.latent_thw)
        else:
            feat = self._load_feat_raw(extraction_dir, step, word)
        # 2x (or Nx) spatial average-pool of the (h, w) grid, applied to the reshaped
        # (C, T, h, w) feature so the resident tensor is 1/f^2 the spatial size. No-op when
        # factor==1 or the feature was left flat (n_blocks_heads/latent_thw not supplied).
        if f > 1 and feat.dim() == 4:
            C, T, h, w = feat.shape
            feat = F.avg_pool2d(
                feat.reshape(C * T, 1, h, w), kernel_size=f, stride=f
            ).reshape(C, T, h // f, w // f)
        return feat

    def _read_feat(self, extraction_dir: str, step: int, word: str) -> torch.Tensor:
        """The flat stored feature (fp32), no reshape."""
        feat_path = Path(extraction_dir) / f"step_{step:03d}.safetensors"
        with safe_open(str(feat_path), framework="pt") as f:
            word_key = f"features__{word}"
            key = word_key if word_key in f.keys() else "feature"
            feat = f.get_tensor(key)
        feat = feat.float()
        # Percentile clipping on the RAW stored map (before any reshape/pool), so the band is
        # identical for every avg_mode and for train + val. Off unless clip_percentiles is set.
        if self.clip_percentiles is not None:
            if self.n_blocks_heads is None:
                raise ValueError("clip_percentiles requires n_blocks_heads (the channel count)")
            lo, hi = self.clip_percentiles
            if self.clip_device is not None:
                feat = percentile_clip(feat.to(self.clip_device), lo, hi,
                                       self.n_blocks_heads).to("cpu")
            else:
                feat = percentile_clip(feat, lo, hi, self.n_blocks_heads)
        return feat

    def _load_feat_raw(self, extraction_dir: str, step: int, word: str) -> torch.Tensor:
        return self._reshape_feat(self._read_feat(extraction_dir, step, word), self.latent_thw)

    def _reshape_feat(self, feat: torch.Tensor, thw) -> torch.Tensor:
        """Flat stored feature → the model-facing layout for this dataset's avg_mode /
        block_index / layer_subset, with ``thw`` as the (T, h, w) grid."""
        if self.n_blocks_heads is not None and thw is not None:
            T, h, w = thw
            if self.capture_blocks is not None:
                # Stored tensor holds only the captured blocks; scatter them back to
                # the full [n_blocks_total, n_heads, T, h, w] layout so block_index
                # addresses the original block index. Non-captured blocks are filled
                # with NaN (not zero) so any accidental use of un-extracted data is
                # immediately visible rather than silently training on fake zeros.
                n_stored = len(self.capture_blocks)
                n_heads = self.n_blocks_heads // n_stored
                # Guard: never hand back a block that was not extracted.
                if self.block_index is not None:
                    orig_idx = self.block_index % self.n_blocks_total
                    if orig_idx not in self.capture_blocks:
                        raise ValueError(
                            f"block_index {self.block_index} (block {orig_idx}) was not "
                            f"extracted; captured blocks are {self.capture_blocks}. "
                            f"Choose a layer from capture_blocks."
                        )
                stored = feat.view(n_stored, n_heads, T, h, w)
                if self.avg_mode == "stack":
                    # Captured blocks only, stacked (block-major) and un-averaged — and
                    # WITHOUT the NaN-scatter to the full n_blocks_total layout below. The
                    # model (ResNetProbe layer_agg) reshapes [L*H, T, h, w] →
                    # [L, H, T, h, w] and learns the layer combine. per_layer/block_index
                    # is disallowed for this mode (guarded in trainer.train_one_step).
                    # layer_subset (when set) keeps only the selected blocks → L = len(subset).
                    if self._subset_positions is not None:
                        stored = stored[self._subset_positions]
                    return stored.reshape(stored.shape[0] * n_heads, T, h, w)
                full = stored.new_full((self.n_blocks_total, n_heads, T, h, w), float("nan"))
                for pos, orig in enumerate(self.capture_blocks):
                    full[orig] = stored[pos]
                if self.block_index is not None:
                    return full[self.block_index]
                if self.avg_mode == "layers":
                    # Average over the captured blocks (the only ones with real data;
                    # full[] has NaN for un-captured blocks), keep heads → [n_heads, T, h, w].
                    # layer_subset restricts the mean to the selected blocks.
                    if self._subset_positions is not None:
                        return stored[self._subset_positions].mean(dim=0)
                    return stored.mean(dim=0)
                feat = full.reshape(self.n_blocks_total * n_heads, T, h, w)
                if self.avg_mode == "all":
                    feat = feat.mean(dim=0, keepdim=True)
                return feat
            feat = feat.view(self.n_blocks_heads, T, h, w)
            if self.block_index is not None:
                n_heads = self.n_blocks_heads // self.n_blocks
                feat = feat.view(self.n_blocks, n_heads, T, h, w)[self.block_index]
            elif self.avg_mode == "layers":
                assert self.n_blocks, "avg_mode='layers' requires n_blocks"
                n_heads = self.n_blocks_heads // self.n_blocks
                blocks = feat.view(self.n_blocks, n_heads, T, h, w)
                if self._subset_positions is not None:
                    blocks = blocks[self._subset_positions]
                feat = blocks.mean(dim=0)  # [n_heads, T, h, w]
            elif self.avg_mode == "stack":
                # Already [n_blocks*n_heads, T, h, w] in block-major order (no subset
                # extraction → nothing to scatter); the model splits it into [L, H, ...]
                # using n_blocks. Just require n_blocks to be known.
                assert self.n_blocks, "avg_mode='stack' requires n_blocks"
                if self._subset_positions is not None:
                    # Keep only the selected blocks → L = len(subset) (block-major).
                    n_heads = self.n_blocks_heads // self.n_blocks
                    feat = feat.view(self.n_blocks, n_heads, T, h, w)[self._subset_positions]
                    feat = feat.reshape(len(self._subset_positions) * n_heads, T, h, w)
            elif self.avg_mode == "all":
                feat = feat.mean(dim=0, keepdim=True)
        return feat

    def _preload(self, step: int) -> None:
        # Preallocate the stack ONCE and fill it row-by-row, instead of building a Python
        # list and torch.stack()-ing it. stack doubles peak memory (list + new contiguous
        # tensor), which for the large "stack" avg_mode (~86 GB) blows past host RAM. Here:
        #   - GPU-resident (device set): allocate on the GPU and copy each sample straight
        #     in, so the big tensor never lives in host RAM (host peak ≈ one sample).
        #   - CPU (device None): allocate once on CPU and fill in place — no 2× spike.
        target = self.device  # torch.device or None (→ CPU)
        data: Optional[torch.Tensor] = None
        kept = []
        corrupt = 0
        n = 0  # number of successfully loaded rows written into `data`
        n_total = len(self.records)
        for rec in tqdm(self.records, desc=f"[ProbeDataset] preloading step {step}", unit="sample"):
            extraction_dir, s, label, word = rec
            try:
                feat = self._load_feat(extraction_dir, s, word)
            except Exception as e:
                # Some extractions leave a full-size but sparse/zeroed (corrupt-header)
                # safetensors from an interrupted write; existence checks pass but loading
                # fails. Skip the record rather than crashing the whole run.
                corrupt += 1
                if corrupt <= 10:
                    print(f"[ProbeDataset] WARNING: skipping corrupt feature "
                          f"{extraction_dir}/step_{s:03d}.safetensors: {type(e).__name__}: {e}")
                continue
            if data is None:
                # Optimistic size n_total; any corrupt skips leave an unused tail trimmed below.
                store_dtype = torch.float16 if self.half else feat.dtype
                pin = self.pin_memory and target is None  # pinned only valid for CPU tensors
                data = torch.empty((n_total, *feat.shape), dtype=store_dtype,
                                   device=target, pin_memory=pin)
            data[n] = feat  # copies into the slot (→ device, → fp16 when half — casts on copy)
            kept.append(rec)
            n += 1
        if corrupt:
            print(f"[ProbeDataset] step {step}: skipped {corrupt} corrupt/unreadable safetensors")
            self.records = kept
        if not self.records:
            raise RuntimeError(f"[ProbeDataset] step {step}: no loadable samples after skipping "
                               f"{corrupt} corrupt files")
        # Trim the unused tail (corrupt skips) via a view — no copy, so no transient 2×.
        # The trimmed rows are a negligible few (corrupt is rare); their storage stays put.
        self._data = data[:n]
        self._labels = torch.tensor(
            [r[2] for r in self.records], dtype=torch.float32, device=target
        ).unsqueeze(1)
        if self.multitask:
            # Pack the aux target as a 2nd label column → [N, 2] (class, aux). This rides the
            # existing labels slot through both the DataLoader and GpuBatchStreamer paths, so
            # no transport code changes; the train loop splits the columns.
            aux = torch.tensor(
                [self._aux_by_dir.get(r[0], float("nan")) for r in self.records],
                dtype=torch.float32, device=target,
            ).unsqueeze(1)
            self._labels = torch.cat([self._labels, aux], dim=1)
        if self.perframe:
            # Pack the T per-latent-frame targets as the LAST columns → [N, 1+T] (or
            # [N, 2+T] alongside the aesthetic aux, which keeps column 1). Same trick as
            # above: extra label columns ride the existing transports untouched.
            pf = torch.tensor(
                [self._perframe_by_key.get((r[0], r[3]), [float("nan")] * self.perframe_T)
                 for r in self.records],
                dtype=torch.float32, device=target,
            )
            self._labels = torch.cat([self._labels, pf], dim=1)
        if self.class_head:
            # Class index per record as the LAST column → [N, ..., 1]. Unknown object_name → -1
            # (masked in the CE loss). Same extra-column trick as the aux/per-frame targets, so
            # it rides the DataLoader / GpuBatchStreamer transports unchanged.
            cls = torch.tensor(
                [[float(self.class_vocab.get(r[3], -1))] for r in self.records],
                dtype=torch.float32, device=target,
            )
            self._labels = torch.cat([self._labels, cls], dim=1)
        gb = self._data.numel() * self._data.element_size() / 1e9
        loc = str(self.device) if self.device is not None else "cpu"
        print(f"[ProbeDataset] preloaded {self._data.shape} ({gb:.2f} GB) on {loc}")

    @property
    def perframe_offset(self) -> int:
        """Column index of the first per-frame target in the packed label row
        (0 = class, 1 = aesthetic aux when multitask, then the T per-frame columns)."""
        return 1 + int(self.multitask)

    @property
    def class_offset(self) -> int:
        """Column index of the class-identity target (the last packed column): after the
        class-presence label (0), the aesthetic aux (when multitask), and the T per-frame
        columns (when perframe)."""
        return 1 + int(self.multitask) + (self.perframe_T if self.perframe else 0)

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Re-weight per-(block,head) attention mass along the steering direction w.

        feat: [C, T, h, w] with C == n_blocks_heads. Each channel's mass is moved by
        `-alpha * ||base|| * w[c]` and the spatial map rescaled to match, so the map's SHAPE
        is untouched (shape was shown indistinguishable between natural and steered; the
        cross-head mass PATTERN is what separates them and what moves the probe).

        Applied to NEGATIVES only by default, label unchanged: the point is to teach the
        probe that a map carrying the steering signature is still an absent object.
        """
        if self.aug_w is None or self.aug_w_prob <= 0:
            return feat
        y = float(label.reshape(-1)[0])
        if self.aug_w_on == "neg" and not y < 0.5:
            return feat
        if self.aug_w_on == "pos" and not y >= 0.5:
            return feat
        if torch.rand(()) >= self.aug_w_prob:
            return feat
        w = self.aug_w.to(feat.device, feat.dtype)
        if w.numel() != feat.shape[0]:
            return feat                     # layout mismatch: never silently mis-augment
        lo, hi = self.aug_w_alpha
        alpha = float(torch.empty(()).uniform_(lo, hi))
        s = feat.sum(dim=(-2, -1))                       # [C, T] mass per channel/frame
        base = s.mean(dim=-1)                            # [C]
        tgt = s - alpha * torch.linalg.vector_norm(base) * w[:, None]
        fac = torch.where(s.abs() > 1e-9, tgt / torch.where(s.abs() > 1e-9, s,
                                                           torch.ones_like(s)),
                          torch.ones_like(s))
        return feat * fac[..., None, None]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._data is not None:
            feat = self._data[idx]
            # Upcast fp16-resident storage back to fp32 per-sample so the probe trains in
            # fp32 (the float() is a no-op when storage is already fp32).
            feat = feat.float() if self.half else feat
            if self.aug_w is not None:
                feat = self._augment(feat.clone(), self._labels[idx])   # never touch storage
            return feat, self._labels[idx]

        extraction_dir, step, label, word = self.records[idx]
        feat = self._load_feat(extraction_dir, step, word)
        cols = [label]
        if self.multitask:
            cols.append(self._aux_by_dir.get(extraction_dir, float("nan")))
        if self.perframe:
            cols.extend(self._perframe_by_key.get(
                (extraction_dir, word), [float("nan")] * self.perframe_T))
        if self.class_head:
            cols.append(float(self.class_vocab.get(word, -1)))
        lab = torch.tensor(cols, dtype=torch.float32)
        if self.aug_w is not None:
            feat = self._augment(feat, lab)
        return feat, lab


class MultiStepProbeDataset(Dataset):
    """Pool several denoising steps into ONE dataset for a step-conditioned probe.

    Builds one :class:`ProbeDataset` per step (reusing all its loading / preload / device /
    capture_blocks machinery) and concatenates them. Each item is
    ``(feat, step_value, label)`` so a step-conditioned model can FiLM on the denoising step.

    The label is per-clip (does the class appear in the final video), identical across steps,
    so pooling multiplies the (step, sample) rows per model — the direct lever against the
    small-dataset overfit. **No new leakage**: the train/val split is done upstream on the
    CSVs (grouped by prompt identity), so every step of a train clip stays in train and every
    step of a val clip stays in val — as long as the same train.csv / val.csv are pooled here.

    ``records`` mirrors ProbeDataset's ``(extraction_dir, step, label, object_name)`` tuples
    (flattened across steps) so the existing ``_dump_epoch_preds`` / predictions code works.
    """

    def __init__(self, labels_csv: str | Path, steps: list[int], *,
                 unified_fp16: bool = False, pin_memory: bool = False,
                 sample_frac: float = 1.0, sample_seed: int = 0, **probe_ds_kwargs) -> None:
        self.steps = [int(s) for s in steps]
        self.unified = unified_fp16
        # sample_frac < 1 randomly keeps that fraction of each step's records (independently
        # per step, fixed seed) — a cheap way to cut compute while still pooling more total
        # data than a single step (e.g. 0.5 × 3 steps = 1.5× the per-step amount).
        self._rng = random.Random(sample_seed)
        self.sample_frac = float(sample_frac)

        def _subsample(records: list) -> list:
            if self.sample_frac >= 1.0:
                return records
            k = max(1, int(round(len(records) * self.sample_frac)))
            return [records[i] for i in sorted(self._rng.sample(range(len(records)), k))]

        if not unified_fp16:
            # Resident/mean path: one preloaded ProbeDataset per step; index into them.
            self.subsets = [
                ProbeDataset(labels_csv, denoising_step=s, **probe_ds_kwargs) for s in self.steps
            ]
            self._index: List[tuple[int, int]] = []
            for si, ds in enumerate(self.subsets):
                keep = range(len(ds)) if self.sample_frac >= 1.0 else sorted(
                    self._rng.sample(range(len(ds)), max(1, int(round(len(ds) * self.sample_frac)))))
                self._index += [(si, li) for li in keep]
            self.records = [self.subsets[si].records[li] for si, li in self._index]
            self._data = self._labels = self._steps = None
            return

        # Streamer path (conv_k / stacked features): the per-step "stack" features are far
        # too big to hold one preloaded tensor PER step (would peak at 2× host RAM when
        # concatenated). Instead build ONE big fp16 host tensor directly from disk — each
        # row filled in place — plus a parallel per-sample step tensor for FiLM conditioning,
        # then feed it to a GpuBatchStreamer (hybrid GPU cache + host stream). Mirrors
        # ProbeDataset._preload's preallocate-and-fill + corrupt-skip, across all pooled steps.
        subs = [ProbeDataset(labels_csv, denoising_step=s, preload=False, **probe_ds_kwargs)
                for s in self.steps]
        for sub in subs:              # subsample each step's records (no-op when frac >= 1)
            sub.records = _subsample(sub.records)
        n_total = sum(len(s) for s in subs)
        data: Optional[torch.Tensor] = None
        steps_list: list[float] = []
        records: list = []
        n = 0
        corrupt = 0
        for si, ds in enumerate(subs):
            sv = float(self.steps[si])
            for rec in tqdm(ds.records, desc=f"[MultiStep] preloading step {self.steps[si]}", unit="sample"):
                extraction_dir, s, label, word = rec
                try:
                    feat = ds._load_feat(extraction_dir, s, word)
                except Exception as e:
                    corrupt += 1
                    if corrupt <= 10:
                        print(f"[MultiStep] WARNING: skipping corrupt {extraction_dir}/step_{s:03d}: "
                              f"{type(e).__name__}: {e}")
                    continue
                if data is None:
                    data = torch.empty((n_total, *feat.shape), dtype=torch.float16,
                                       pin_memory=pin_memory)
                data[n] = feat  # casts to fp16 on copy
                steps_list.append(sv)
                records.append(rec)
                n += 1
        if data is None:
            raise RuntimeError("[MultiStep] no loadable samples across pooled steps")
        self._data = data[:n]
        self._steps = torch.tensor(steps_list, dtype=torch.float32)
        self._labels = torch.tensor([r[2] for r in records], dtype=torch.float32).unsqueeze(1)
        self.records = records
        self._index = None
        gb = self._data.numel() * self._data.element_size() / 1e9
        print(f"[MultiStep] unified preload {tuple(self._data.shape)} ({gb:.1f} GB fp16, "
              f"{corrupt} corrupt skipped) over steps {self.steps}")

    def __len__(self) -> int:
        return len(self.records) if self.unified else len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.unified:
            return self._data[idx].float(), self._steps[idx], self._labels[idx]
        si, li = self._index[idx]
        feat, label = self.subsets[si][li]
        return feat, torch.tensor(self.steps[si], dtype=torch.float32), label


class BalancedBatchSampler(BatchSampler):
    """Yields batches with a controlled pos/neg ratio.

    Two modes, selected by ``balanced``:

    - ``balanced=False`` (default): each batch's pos/neg split is **proportional**
      to the overall class distribution. Both index lists are shuffled
      independently each epoch; the minority class is never oversampled — the
      epoch ends when the smaller pool is exhausted.

    - ``balanced=True``: each batch is **50/50** pos/neg (``batch_size // 2`` each).
      The majority pool is shuffled and walked once per epoch; the minority pool is
      oversampled (reshuffled and cycled) so every majority sample is seen once.
      The epoch length is set by the majority class.

    Raises ValueError if either class is absent (caller should skip training).
    """

    def __init__(self, dataset: "ProbeDataset", batch_size: int, balanced: bool = False) -> None:
        self.pos_indices: List[int] = [i for i, r in enumerate(dataset.records) if r[2] >= 0.5]
        self.neg_indices: List[int] = [i for i, r in enumerate(dataset.records) if r[2] < 0.5]

        n_pos = len(self.pos_indices)
        n_neg = len(self.neg_indices)
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"BalancedBatchSampler requires both classes to be present "
                f"(n_pos={n_pos}, n_neg={n_neg})"
            )

        self.balanced = balanced
        if balanced:
            # 50/50 per batch; minority oversampled (cycled) to match majority count.
            self.n_pos_per_batch: int = max(1, batch_size // 2)
            self.n_neg_per_batch: int = max(1, batch_size // 2)
            n_majority = max(n_pos, n_neg)
            self._n_batches = max(
                n_majority // self.n_pos_per_batch,
                n_majority // self.n_neg_per_batch,
            )
        else:
            total = n_pos + n_neg
            self.n_pos_per_batch = max(1, round(batch_size * n_pos / total))
            self.n_neg_per_batch = batch_size - self.n_pos_per_batch
            self._n_batches = min(
                n_pos // self.n_pos_per_batch,
                n_neg // self.n_neg_per_batch,
            )

    @staticmethod
    def _draw(pool: List[int], n_needed: int) -> List[int]:
        """Return a shuffled list of length >= n_needed, cycling the pool with reshuffles."""
        out = pool[:]
        random.shuffle(out)
        while len(out) < n_needed:
            extra = pool[:]
            random.shuffle(extra)
            out += extra
        return out

    def __iter__(self) -> Iterator[List[int]]:
        if self.balanced:
            # Oversample each class up to the number of slots the epoch needs so the
            # minority is cycled rather than exhausted.
            pos = self._draw(self.pos_indices, self._n_batches * self.n_pos_per_batch)
            neg = self._draw(self.neg_indices, self._n_batches * self.n_neg_per_batch)
        else:
            pos = self.pos_indices[:]
            neg = self.neg_indices[:]
            random.shuffle(pos)
            random.shuffle(neg)

        for i in range(self._n_batches):
            batch = (
                pos[i * self.n_pos_per_batch : (i + 1) * self.n_pos_per_batch]
                + neg[i * self.n_neg_per_batch : (i + 1) * self.n_neg_per_batch]
            )
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._n_batches


def sequential_batches(n: int, batch_size: int) -> List[List[int]]:
    """Contiguous index batches [[0..bs), [bs..2bs), ...] for sequential eval passes."""
    return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]


class GpuBatchStreamer:
    """Yields ``(feats_fp32_gpu, labels_gpu)`` batches for the large un-averaged ("stack")
    features that don't fit GPU-resident in fp32.

    Hybrid residence: the first ``n_resident`` samples live on the GPU in fp16 (no per-epoch
    gather or host→device copy — read straight from VRAM); the remaining samples stay in
    pinned host RAM and are gathered per batch into a reusable pinned buffer and copied to the
    GPU. Each batch is assembled on the GPU and upcast to fp32 so the probe trains exactly as
    before. ``n_resident`` auto-sizes to the GPU's free VRAM (minus ``reserve_gb`` for the
    batch + activations), so a B200 caches almost everything and an H200 caches less — both
    correct, only the streamed fraction (and thus speed) differs.

    One streamer owns the (single) resident GPU copy; call ``.loader(batch_index_lists)`` to
    get a cheap iterable for any sampler (the train BalancedBatchSampler, or
    ``sequential_batches`` for eval) without re-allocating the cache.
    """

    def __init__(self, data_cpu: torch.Tensor, labels: torch.Tensor, device: torch.device,
                 batch_size: int, resident_gb: Optional[float] = None, reserve_gb: float = 28.0,
                 steps: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None):
        self.device = torch.device(device)
        self.data_cpu = data_cpu                       # [N, *feat] pinned host (all rows)
        self.dtype = data_cpu.dtype                    # fp16 (compact) or fp32 (reference)
        self.labels = labels.to(self.device).float()
        # steps: optional per-sample denoising-step value for step-conditioned (pooled) probes.
        # When set, batches are yielded as (feats, steps, labels) instead of (feats, labels).
        self.steps = steps.to(self.device).float() if steps is not None else None
        # weights: optional per-sample loss weights (source/class balancing). Yielded only when a
        # loader is built with with_weights=True (train loop) — eval loaders never see them.
        if weights is not None and steps is not None:
            raise ValueError("GpuBatchStreamer: per-sample weights + step-conditioning together "
                             "are unsupported (both use the extra batch slot).")
        self.weights = torch.as_tensor(weights).to(self.device).float() if weights is not None else None
        n = data_cpu.shape[0]
        self.feat_shape = tuple(data_cpu.shape[1:])
        per_sample = data_cpu[0].numel() * data_cpu.element_size()
        if resident_gb is not None:
            budget = resident_gb * 1e9
        else:
            free = torch.cuda.mem_get_info(self.device)[0]
            budget = max(0.0, free - reserve_gb * 1e9)
        self.n_resident = int(min(n, budget // per_sample))
        # Resident GPU copy of the first n_resident rows (fp16). The full set also stays on the
        # host so streamed indices (>= n_resident) read from data_cpu directly.
        self.res = (self.data_cpu[:self.n_resident].to(self.device)
                    if self.n_resident > 0 else None)
        self.buf = torch.empty((batch_size, *self.feat_shape), dtype=self.dtype,
                               pin_memory=(self.device.type == "cuda"))
        free_now = (torch.cuda.mem_get_info(self.device)[0] / 1e9
                    if self.device.type == "cuda" else float("nan"))
        print(f"[GpuBatchStreamer] N={n} resident={self.n_resident} "
              f"({self.n_resident*per_sample/1e9:.0f} GB {self.dtype} on GPU) "
              f"streamed={n - self.n_resident}  free now={free_now:.0f} GB")

    def _yield(self, batch_lists, with_weights: bool = False):
        dev = self.device
        for batch_idx in batch_lists:
            idx = torch.as_tensor(batch_idx, device=dev)
            res_mask = idx < self.n_resident
            feats = torch.empty((idx.numel(), *self.feat_shape), dtype=self.dtype, device=dev)
            if self.res is not None and res_mask.any():
                feats[res_mask] = self.res[idx[res_mask]]
            stream_mask = ~res_mask
            if stream_mask.any():
                si = idx[stream_mask].cpu()                       # absolute rows on host
                k = si.numel()
                g = torch.index_select(self.data_cpu, 0, si, out=self.buf[:k])
                feats[stream_mask] = g.to(dev, non_blocking=True)
            if self.steps is not None:
                yield feats.float(), self.steps[idx], self.labels[idx]
            elif with_weights and self.weights is not None:
                yield feats.float(), self.labels[idx], self.weights[idx]
            else:
                yield feats.float(), self.labels[idx]

    def loader(self, batch_lists, with_weights: bool = False) -> "_StreamLoader":
        return _StreamLoader(self, batch_lists, with_weights)


class _StreamLoader:
    """Cheap iterable view over a GpuBatchStreamer for a given list of index-batches."""

    def __init__(self, streamer: GpuBatchStreamer, batch_lists, with_weights: bool = False):
        # batch_lists may be a re-iterable sampler (e.g. BalancedBatchSampler, which reshuffles
        # on each __iter__) or a fixed list (sequential eval) — keep the reference so the train
        # loader reshuffles every epoch.
        self.streamer = streamer
        self.batch_lists = batch_lists
        self.with_weights = with_weights

    def __iter__(self):
        return self.streamer._yield(self.batch_lists, self.with_weights)

    def __len__(self) -> int:
        return len(self.batch_lists)
