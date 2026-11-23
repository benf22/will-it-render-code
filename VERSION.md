# Version history

Replaces the research monorepo's development DEVLOG. Public, user-facing changes only.

## 0.1.0 — unreleased (migrated, pre-release)

- Initial public extraction from an internal research monorepo.
- Full `src/will_it_render/` package: common / backends (+hooks) / inference (+run) /
  datasets / training / evaluation (+baselines). AttnSteer naming throughout; env-var engine
  resolution (VP_LTX_ROOT / VP_WAN_ROOT / VP_COGX_ROOT); sanitized configs; MIT + THIRD_PARTY.
- Tests: CPU suite (training/inference/metrics) + module-independence, secret/path, and
  stale-module scanners + CI; the WAN hook test is gpu-marked (needs a WAN checkout).
- GPU-validated on an L40S: WAN end-to-end via `steer.py` and `wir-run` (steered videos,
  determinism on/off) + 7/7 WAN hook tests against the real engine; LTX env-var path
  resolution and the CogVideoX hook/backend validated (complete steered LTX/CogVideoX clips
  need a B200/H200-class card — the 48 GB L40S OOMs on their steering grad).
- Not yet done: probe-weight / bundle upload to HF; the canonical per-backend steering configs
  (pinned to paper recipes); complete steered LTX/CogVideoX clips on a larger GPU.
