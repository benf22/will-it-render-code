# Version history

Replaces the research monorepo's development DEVLOG. Public, user-facing changes only.

## 0.1.0 — unreleased (migrated, pre-release)

- Initial public extraction from an internal research monorepo.
- Full `src/will_it_render/` package: common / backends (+hooks) / inference (+run) /
  datasets / training / evaluation (+baselines). AttnSteer naming throughout; env-var engine
  resolution (VP_LTX_ROOT / VP_WAN_ROOT / VP_COGX_ROOT); sanitized configs; MIT + THIRD_PARTY.
- Tests: CPU suite (training/inference/metrics) + module-independence, secret/path, and
  stale-module scanners + CI; the WAN hook test is gpu-marked (needs a WAN checkout).
- GPU-validated, all three backends end-to-end with real trained AttnSteer weights:
  - WAN (L40S): `steer.py` + `wir-run` steered videos (determinism on/off) + 7/7 hook tests.
  - LTX (H100): 48-layer conv_k champion, steered video+audio clip, probe 0.48→0.99 over steps.
  - CogVideoX (H100): steered clip across two steered steps, probe ~0.98.
  (LTX 43 GB / CogVideoX 5B steering grad OOMs on a 48 GB L40S — both need an H100-class card.)
- Not yet done: probe-weight / bundle upload to HF; the canonical per-backend steering configs
  (pinned to paper recipes).
