# tests

Reproduce current behavior (plan §10b R7 / §7):

- ported unit tests (steering_core, nearest_attnsteer, mutual_fold, scheduled_steering,
  feature_dataset VAE guard) — CPU-runnable, in CI.
- `test_wan_hooks_capture` — needs the WAN checkout + GPU-adjacent env → `@pytest.mark.gpu`,
  excluded from CI.
- golden-value tests: det-rate metrics, temporal_accuracy terms, token mapping, aggregator
  vector layout — pinned numeric outputs.
- CPU end-to-end smoke: tiny committed feature sample → AttnSteer forward → steering-loss.
- a **module-independence test** (N2): walks intra-package imports, fails on any
  `datasets`/`training`/`inference`/`evaluation` → sibling edge.
- a **secret/path scanner** (with an allowlist for its own fixtures + the gated GPU test):
  fails on `/cortex`, `/private/schwartz-lab`, `/home/lab/fishmab1`, `fishmab1`, `benfish`,
  `video_probing`, or HF/wandb token patterns in tracked files.

CI: GitHub Actions running the CPU suite + both scanners.
