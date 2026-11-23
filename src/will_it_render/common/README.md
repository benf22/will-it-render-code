# common

Dependency-sink shared by every other subpackage (`datasets` / `training` / `inference` /
`evaluation` all import `common`, never each other). It depends on nothing else in the package
(stdlib + torch + the leaf AttnSteer-architecture modules), so it never creates a cycle. **Not an
entry point — a library.**

## Files

| File | What it provides |
|------|------------------|
| `attnsteer_factory.py` | `build_attnsteer(model_arch, …)` — instantiate an AttnSteer architecture by name (the training stack is not pulled in) |
| `steering_base.py` | the `SteeringModel` ABC (subclassed by the per-step steerer and the baselines) |
| `aggregator.py` | captures → flat feature vector (`aggregate_captures_to_vector`) |
| `agg.py` | `build_agg_kwargs` — derive aggregation flags from a model config |
| `control_signal.py` | `resolve_control_signal` — per-frame control schedule (shared by the timeline steerer + TempoControl baseline) |
| `token_mapper.py` | keyword → text-token indices (subword-aware) |
| `storage.py` | safetensors / MP4 / WAV save helpers |
| `probe_types.py` | `BlockCapture` dataclass, `CaptureType` enum |
| `extraction_config.py` | locate the per-extraction geometry config (`extraction_config.yaml`, legacy `probe_config.yaml`) |
| `prompts.py` | prompt-CSV parsing helpers |
| `paths.py` | **env-var resolver** for the engine checkouts (`VP_LTX_ROOT`/`VP_WAN_ROOT`/`VP_COGX_ROOT`), configs (`VP_CONFIG_DIR`), artifact/seed-latent dirs |
| `visual_classifier.py` | object-detector wrappers (YOLO / GDINO / OWLv2 / GRiT) — shared by labeling + evaluation |
