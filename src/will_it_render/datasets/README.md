# datasets

Builds the training data: generate prompts, extract per-step cross-attention features, and
label the generated clips.

**Standalone (N2):** imports only `common` and `backends`.

- prompt generation (`generate_extraction_csv`, optional Llama-3 enrichment, CSV reader, split).
- feature extraction (`run_extraction`) — drives the DiT denoising loop and stores one flat
  feature vector per target word per captured step (`step_NNN.safetensors`).
- labeling — produce binary / per-frame presence labels via PANNs audio tagging and object
  detectors (YOLO / GDINO / OWLv2 / GRiT) + quality scoring.

Heavy detector/audio dependencies are optional extras (`.[detectors]`, `.[audio]`) and must
never be import-time-required by the core path.

> Scaffold — ported in Phase 2b.
