"""agg_mutual_agree fold variants (steering.mutual_fold, 2026-10-25): "anchor" (default) vs "legacy"."""
import pytest
import torch

import will_it_render.inference.steering_core as core

LAYOUTS = {"ltx": ((1, 6, 4), -1), "wan": ((4, 2, 3, 3), 0), "cogvideox": ((1, 2, 4, 3, 3), 2)}


@pytest.fixture(autouse=True)
def _reset():
    old = (core.MUTUAL_FOLD, core.MUTUAL_ORDER, core.CHANNEL_DIM, core.MUTUAL_SKIP_STEPS, core.AGG_STEP)
    core.MUTUAL_ORDER = "text"                       # dict order, so the tests pick the anchor explicitly
    yield
    core.MUTUAL_FOLD, core.MUTUAL_ORDER, core.CHANNEL_DIM, core.MUTUAL_SKIP_STEPS, core.AGG_STEP = old


def _grads(shape, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {f"o{i}": torch.randn(shape, generator=g) for i in range(n)}


def _legacy(pg):
    ws = list(pg); g0, g1 = pg[ws[0]], pg[ws[1]]
    agg = (g0 + g1) * (core._token_cos(g0, g1) >= 0).to(g0.dtype).unsqueeze(0)
    for w in ws[2:]:
        agg = agg + pg[w] * (core._token_cos(pg[w], agg) >= 0).to(agg.dtype).unsqueeze(0)
    return agg / len(ws)


@pytest.mark.parametrize("model", list(LAYOUTS))
def test_legacy_is_bit_identical_to_the_old_fold(model):
    core.MUTUAL_FOLD = "legacy"
    pg = _grads(LAYOUTS[model][0], 4)
    assert torch.equal(core.agg_mutual_agree(pg), _legacy(pg))


@pytest.mark.parametrize("model", list(LAYOUTS))
@pytest.mark.parametrize("by", ["model_type", "ndim"])
def test_anchor_keeps_first_whole_and_masks_per_patch(model, by):
    shape, cdim = LAYOUTS[model]
    core.MUTUAL_FOLD = "anchor"
    core.CHANNEL_DIM = core.CHANNEL_DIM_BY_MODEL[model] if by == "model_type" else None
    pg = _grads(shape, 3)
    g0, g1, g2 = pg.values()
    out = core.agg_mutual_agree(pg)
    # reference, written out per patch: cosine over the channel vector of each patch
    cos = lambda a, b: (a * b).sum(cdim, keepdim=True) / (a.norm(dim=cdim, keepdim=True) * b.norm(dim=cdim, keepdim=True))
    m1 = (cos(g1, g0) >= 0).float(); agg = g0 + g1 * m1
    m2 = (cos(g2, agg) >= 0).float(); cnt = 1 + m1 + m2
    assert torch.allclose(out, (agg + g2 * m2) / cnt, atol=1e-6)
    # first object is never masked: out * cnt - g0 is exactly the masked others
    assert torch.allclose(out * cnt - g0, g1 * m1 + g2 * m2, atol=1e-6)
    # per-patch divisor: a patch where only the anchor survives is the anchor itself
    solo = (cnt == 1).expand_as(g0)
    assert solo.any() and torch.allclose(out[solo], g0[solo])
    # the mask is constant across the channels of a patch (one decision per patch, not per element)
    assert m1.shape[cdim] == 1
    assert 0 < m1.mean() < 1                         # the test data has both agreeing and disagreeing patches


def test_anchor_two_objects_toy():
    core.MUTUAL_FOLD = "anchor"; core.CHANNEL_DIM = -1               # LTX layout (1, N tokens, C)
    g0 = torch.tensor([[[1., 1., 1., 1.], [1., 0., 0., 0.]]])
    g1 = torch.tensor([[[1., 1., 1., -2.], [-1., 5., 0., 0.]]])        # patch 0 cos > 0 despite one opposite channel
    out = core.agg_mutual_agree({"a": g0, "b": g1})
    assert torch.equal(out[0, 0], (g0[0, 0] + g1[0, 0]) / 2)           # whole patch kept (incl. the -2 channel), mean of 2
    assert torch.equal(out[0, 1], g0[0, 1])                            # patch 1 cos < 0: b dropped, a alone (/1)


@pytest.mark.parametrize("fold", ["anchor", "legacy"])
def test_skip_steps_give_the_plain_mean_only_at_those_steps(fold):
    core.MUTUAL_FOLD = fold; core.CHANNEL_DIM = -1
    pg = _grads(LAYOUTS["ltx"][0], 3)
    folded = core.agg_mutual_agree(pg)
    core.MUTUAL_SKIP_STEPS = frozenset({0})
    core.AGG_STEP = 0
    assert torch.allclose(core.agg_mutual_agree(pg), torch.stack(list(pg.values())).mean(0))
    core.AGG_STEP = 1
    assert torch.equal(core.agg_mutual_agree(pg), folded)


def test_mask_stats_do_not_change_the_fold():
    import will_it_render.inference.steering_core as core
    torch.manual_seed(0)
    g = {w: torch.randn(1, 64, 16) for w in "abc"}
    old = (core.MASK_STATS, core.CHANNEL_DIM, core.MUTUAL_FOLD)
    try:
        core.CHANNEL_DIM, core.MUTUAL_FOLD = -1, "anchor"
        core.MASK_STATS = False; a = core.agg_mutual_agree(g)
        core.MASK_STATS = True; b = core.agg_mutual_agree(g)
        assert torch.equal(a, b)
        ms = core._MASK_LAST
        assert ms["n"] == 3 and ms["patches"] == 64 and 0 <= ms["frac_masked"] <= 2 / 3
        assert abs(ms["e_total"] - sum(float(v.pow(2).sum()) for v in g.values())) < 1e-3
        assert 0 <= ms["e_ratio"] <= 1
    finally:
        core.MASK_STATS, core.CHANNEL_DIM, core.MUTUAL_FOLD = old
