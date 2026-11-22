# common

Dependency-sink shared by every other subpackage. **Depends on nothing else in the package**
(stdlib + torch + leaf AttnSteer-architecture modules only), so it can be imported from
anywhere without creating a cycle.

Will hold (ported in Phase 2.5 / 2b):
- `attnsteer_factory.py` — `build_attnsteer` (`build_probe` in the source), instantiating an
  architecture by name without dragging in the training stack.
- `aggregation.py` — captures → feature vector, and the agg-flag derivation (`build_agg_kwargs`).
- `prompts.py` — prompt-CSV parsing helpers.
- `config.py` — config loading / merging.
- `paths.py` — env-var resolver for the DiT engine checkouts (`VP_LTX_ROOT` / `VP_WAN_ROOT` /
  `VP_COGX_ROOT`) and the artifact cache, replacing `sys.path` injection (plan §10b R5).

> Scaffold — see `WILL_IT_RENDER_PLAN.md`. The R1 hoist already created the equivalent
> `common/` package in the source repo; this is its destination.
