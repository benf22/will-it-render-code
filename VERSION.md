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
  - WAN: `steer.py` + `wir-run` steered videos (determinism on/off) + 7/7 hook tests.
  - LTX: 48-layer conv_k probe, steered video+audio clip, probe 0.48→0.99 over steps.
  - CogVideoX: steered clip across two steered steps, probe ~0.98.
  (LTX / CogVideoX 5B steering backward needs more than ~48 GB.)
- Not yet done: probe-weight / bundle upload to HF; the canonical per-backend steering configs
  (pinned to paper recipes).
