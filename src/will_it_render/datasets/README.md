# datasets

Builds the training/eval data: validate prompts, extract per-step cross-attention features, and
label the generated clips with object detectors.

**Standalone (N2):** imports only `common` and `backends`.

## Files

| File | What it does | Entry point |
|------|--------------|-------------|
| `generate_extraction_csv.py` | Validate/normalize a prompts CSV (`prompt,object_name,target_words`) for extraction. | `python -m will_it_render.datasets.generate_extraction_csv --manual_prompts_csv in.csv --output val.csv` |
| `run_extraction.py` | DDP entry point: drive the DiT denoising loop and store one feature vector per word per captured step (`step_NNN.safetensors`) + `extraction_config.yaml`. | `torchrun --nproc_per_node=N -m will_it_render.datasets.run_extraction --config configs/attnsteer.yaml --prompts_csv data/WIR_testset/wir_testset.csv --output_dir out/feats` |
| `video_object_labeler.py` | Label each clip with object detectors (YOLO / GDINO / OWLv2 / GRiT), majority-vote → `labels.csv` (clip + per-frame). | `torchrun --nproc_per_node=N -m will_it_render.datasets.video_object_labeler --root out/feats --out labels.csv` |
| `visual_classifier.py` | → `common/visual_classifier.py` (the detector wrappers live in `common`, shared with evaluation). | — |
| `quality_scorer.py` | VBench-based realism/quality scoring of clips (optional `[vbench]` extra). | `python -m will_it_render.datasets.quality_scorer --root out/feats` |
| `csv_reader.py` / `splitter.py` | object-CSV dataclass; train/val split helper. | library |
| `convert_vbench_prompts.py` | Convert VBench prompt lists into the extraction CSV format. | `python -m will_it_render.datasets.convert_vbench_prompts …` |
| `visualize_features.py` | Flask tool to inspect stored attention features per clip. | `python -m will_it_render.datasets.visualize_features <extraction_dir>` |

Detector/VBench dependencies are optional extras (`.[detectors]`, `.[vbench]`) — never
import-time-required by the core path.

> The research repo's Llama-3 prompt-enrichment and PANNs audio-labeling paths are **not** part
> of this release (legacy). Author prompt CSVs directly or reuse `data/WIR_testset/`.
