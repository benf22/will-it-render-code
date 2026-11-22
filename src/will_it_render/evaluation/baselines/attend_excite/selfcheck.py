"""CPU self-check of the Attend-and-Excite loss — no GPU, no model, seconds to run.

    python -m baselines.attend_excite.selfcheck

Builds synthetic captures with a known attention map and checks every piece of the port
against a hand-computed value: the text-axis softmax (temperature, BOS/pad exclusion),
the Gaussian smoothing kernel, the three ``frame_reduce`` variants, the max-over-subjects
loss, and the paper's schedule (guided-step range, √-decayed step size, threshold-driven
refinement). Also asserts the loss stays differentiable w.r.t. the captured attention,
which is what makes it usable as a steering objective at all.
"""
from __future__ import annotations

import math
import sys

import torch

from will_it_render.evaluation.baselines.attend_excite.loss import AttendExciteSteeringModel, _gaussian_kernel_2d
from will_it_render.common.probe_types import BlockCapture, CaptureType

N_BLOCKS, N_HEADS, T, H, W = 3, 2, 5, 4, 4
LK = 8
# Key layout: 0 = BOS, 1..5 = prompt tokens, 6 = EOS, 7 = padding.
TEXT_IDX = [1, 2, 3, 4, 5]
WORD_A = [2, 3]   # two subword tokens of the first target word
WORD_B = [5]      # a second target word

_failures: list[str] = []


def _scalar(x) -> float:
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def check(name: str, got, want, tol: float = 1e-6) -> None:
    got, want = _scalar(got), _scalar(want)
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got:.8f}, want {want:.8f}")
    if not ok:
        _failures.append(name)


def check_true(name: str, ok: bool) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        _failures.append(name)


def make_captures(logits: torch.Tensor) -> tuple[list[BlockCapture], torch.Tensor]:
    """Captures whose pre-softmax attention row is ``logits`` [Lq, LK].

    Every block and head carries the same values, so the head- and block-means the model
    takes are identity operations and the expected value stays hand-computable.
    """
    attn = logits.unsqueeze(0).unsqueeze(0).repeat(1, N_HEADS, 1, 1).clone()  # [1,H,Lq,LK]
    attn.requires_grad_(True)
    key = torch.zeros(1, LK, N_HEADS, 8, dtype=torch.float64)
    caps = [
        BlockCapture(attn_map=attn, text_key=key, capture_type=CaptureType.VIDEO_CA)
        for _ in range(N_BLOCKS)
    ]
    return caps, attn


def build_model(**kw) -> AttendExciteSteeringModel:
    return AttendExciteSteeringModel(
        device=torch.device("cpu"),
        agg_kwargs={"spatial_downsample": 1, "latent_hw": (H * 2, W * 2)},
        sample_steps=50,
        **kw,
    )


def main() -> int:
    torch.manual_seed(0)

    # ---------------------------------------------------------------- Gaussian kernel
    print("\nGaussian smoothing kernel (their k=3, sigma=0.5):")
    k = _gaussian_kernel_2d(3, 0.5, torch.device("cpu"), torch.float64)
    check("kernel sums to 1", k.sum(), 1.0)
    check_true("kernel is symmetric", bool(torch.allclose(k, k.flip(-1))) and bool(torch.allclose(k, k.flip(-2))))
    # 1D weights: exp(-((x-1)/0.5)^2/2) = [e^-2, 1, e^-2], normalized.
    w = torch.tensor([math.exp(-2.0), 1.0, math.exp(-2.0)], dtype=torch.float64)
    w = w / w.sum()
    check("kernel center weight", k[0, 0, 1, 1], w[1] * w[1])
    check("kernel corner weight", k[0, 0, 0, 0], w[0] * w[0])

    # ------------------------------------------------------- text-axis softmax + max
    print("\nText-axis softmax and per-subject max attention:")
    # One patch of one frame is spiked for WORD_A; BOS and padding are given huge logits
    # so that INCLUDING them would visibly change the normalization.
    logits = torch.zeros(T * H * W, LK, dtype=torch.float64)
    logits[:, 0] = 10.0   # BOS — must be excluded
    logits[:, 7] = 10.0   # padding — must be excluded
    spike_q = 1 * H * W + 5   # frame 1, some patch
    logits[spike_q, WORD_A] = 1.0
    caps, attn_leaf = make_captures(logits)

    m = build_model(frame_reduce="global", smooth_attentions=False)
    m.set_text_token_indices(TEXT_IDX)
    loss, metrics = m.loss(caps, 0, target_words_indices={"a": WORD_A, "b": WORD_B})

    # Expected: softmax over the 5 text logits, x100 temperature. At the spiked query the
    # two WORD_A columns hold logit 1.0 and the other three hold 0.0.
    e_hi, e_lo = math.exp(100.0 * 1.0), math.exp(0.0)
    p_spike = e_hi / (2 * e_hi + 3 * e_lo)          # per WORD_A column, mean over 2 == same
    p_flat = 1.0 / 5.0                               # uniform elsewhere
    check("max attention, spiked word (global)", metrics["per_word_prob"]["a"], p_spike)
    check("max attention, flat word", metrics["per_word_prob"]["b"], p_flat)
    check("loss = max over subjects of (1 - M_s)", loss, 1.0 - min(p_spike, p_flat))
    check_true("padding/BOS excluded (flat word stays uniform over 5 text tokens)",
               abs(metrics["per_word_prob"]["b"] - p_flat) < 1e-9)
    check_true("min_max_attention names the driving subject",
               abs(metrics["min_max_attention"] - min(p_spike, p_flat)) < 1e-9)

    # Including the padding columns must change the answer — proves the span is load-bearing.
    m_nospan = build_model(frame_reduce="global", smooth_attentions=False)
    _, metrics_nospan = m_nospan.loss(caps, 0, target_words_indices={"b": WORD_B})
    check_true("softmax span matters (all-keys normalization differs)",
               abs(metrics_nospan["per_word_prob"]["b"] - p_flat) > 1e-6)

    # ------------------------------------------------------- aggregation ORDER (heads)
    print("\nAggregation order — head/layer mean BEFORE the softmax (their order):")
    # Heads must disagree, otherwise the two orders coincide and the test is vacuous.
    logits_h = torch.zeros(T * H * W, LK, dtype=torch.float64)
    attn_h = logits_h.unsqueeze(0).unsqueeze(0).repeat(1, N_HEADS, 1, 1).clone()
    attn_h[0, 0, :, WORD_A] = 1.0     # head 0 attends to WORD_A
    attn_h[0, 1, :, WORD_A] = -1.0    # head 1 attends away from it
    attn_h.requires_grad_(True)
    key_h = torch.zeros(1, LK, N_HEADS, 8, dtype=torch.float64)
    caps_h = [BlockCapture(attn_map=attn_h, text_key=key_h, capture_type=CaptureType.VIDEO_CA)
              for _ in range(N_BLOCKS)]
    m_h = build_model(frame_reduce="global", smooth_attentions=False)
    m_h.set_text_token_indices(TEXT_IDX)
    _, met_h = m_h.loss(caps_h, 0, target_words_indices={"a": WORD_A})

    sl = attn_h[0][:, :, TEXT_IDX]                                  # [H, Lq, n_text]
    want_order = torch.softmax(sl.mean(dim=0) * 100.0, dim=-1)[:, [1, 2]].mean(-1).max()
    wrong_order = torch.softmax(sl * 100.0, dim=-1).mean(dim=0)[:, [1, 2]].mean(-1).max()
    check("M_s = max of softmax(mean over heads)", met_h["per_word_prob"]["a"], want_order)
    check_true("the two orders genuinely differ on this input",
               abs(float(want_order) - float(wrong_order)) > 1e-6)
    check_true("NOT mean(softmax per head)",
               abs(met_h["per_word_prob"]["a"] - float(wrong_order)) > 1e-6)

    # ------------------------------------------------------------------ frame_reduce
    print("\nframe_reduce variants:")
    # Frame f gets a uniform value (f+1)/10 for WORD_A, so per-frame maxima are known.
    logits2 = torch.full((T * H * W, LK), -50.0, dtype=torch.float64)
    for f in range(T):
        logits2[f * H * W:(f + 1) * H * W, WORD_A] = 0.0
    # Frame 0 is suppressed for one subword so its softmax mass is lower.
    caps2, _ = make_captures(logits2)
    for reduce_mode in ("global", "min", "mean"):
        mm = build_model(frame_reduce=reduce_mode, smooth_attentions=False)
        mm.set_text_token_indices(TEXT_IDX)
        _, met = mm.loss(caps2, 0, target_words_indices={"a": WORD_A})
        # Every frame is identical here, so all three reductions must agree.
        check(f"{reduce_mode}: uniform clip gives the same M_s", met["per_word_prob"]["a"],
              math.exp(0.0) / (2 * math.exp(0.0) + 3 * math.exp(-50.0 * 100.0)) * 1.0)

    # Now make exactly one frame strong: global == that frame's max, min == the weak
    # frames' value, mean sits between.
    logits3 = torch.zeros(T * H * W, LK, dtype=torch.float64)
    logits3[2 * H * W:(2 + 1) * H * W, WORD_A] = 1.0     # frame 2 strong
    caps3, _ = make_captures(logits3)
    vals = {}
    for reduce_mode in ("global", "min", "mean"):
        mm = build_model(frame_reduce=reduce_mode, smooth_attentions=False)
        mm.set_text_token_indices(TEXT_IDX)
        _, met = mm.loss(caps3, 0, target_words_indices={"a": WORD_A})
        vals[reduce_mode] = met["per_word_prob"]["a"]
    check("global = the strong frame's value", vals["global"], p_spike)
    check("min = the weak frames' value", vals["min"], p_flat)
    check("mean = (4 weak + 1 strong) / 5", vals["mean"], (4 * p_flat + p_spike) / 5)
    check_true("global >= mean >= min", vals["global"] >= vals["mean"] >= vals["min"])

    # ---------------------------------------------------------------- differentiable
    print("\nDifferentiability (the objective must reach x_t through the captures):")
    caps4, leaf = make_captures(logits.clone())
    m4 = build_model(frame_reduce="global", smooth_attentions=True)
    m4.set_text_token_indices(TEXT_IDX)
    loss4, _ = m4.loss(caps4, 0, target_words_indices={"a": WORD_A, "b": WORD_B})
    grad = torch.autograd.grad(loss4, leaf)[0]
    check_true("loss is differentiable w.r.t. the attention", grad is not None)
    check_true("gradient is non-zero", float(grad.abs().sum()) > 0)
    check_true("gradient is finite", bool(torch.isfinite(grad).all()))

    # ------------------------------------------------- per-subject components (CAGrad)
    # steering_core enters its GRAD_AGGREGATE branch on hasattr(model, "per_object_bce")
    # and differentiates each returned scalar against x_t separately, so these are the
    # components a rule aggregates INSTEAD of the outer max. See loss.per_object_bce.
    print("\nPer-subject components for the GRAD_AGGREGATE path:")
    caps5, leaf5 = make_captures(logits.clone())
    m5 = build_model(frame_reduce="global", smooth_attentions=True)
    m5.set_text_token_indices(TEXT_IDX)
    twi = {"a": WORD_A, "b": WORD_B}
    pobj = m5.per_object_bce(caps5, 0, target_words_indices=twi)
    loss5, met5 = m5.loss(caps5, 0, target_words_indices=twi)

    check_true("the loop's dispatch attribute exists", hasattr(m5, "per_object_bce"))
    check_true("per_object_bce_with does NOT exist (probe_mix/select_probe must no-op)",
               not hasattr(m5, "per_object_bce_with"))
    check_true("one component per (unsatisfied) subject", set(pobj) == {"a", "b"})
    # The outer max of loss() must be exactly the max of the components it now delegates to.
    check("loss() == max over the components", loss5, max(float(v) for v in pobj.values()))
    for w in pobj:
        check(f"component {w} == 1 - M_{w}", pobj[w], 1.0 - met5["per_word_prob"][w])
    # Each component must reach the captures on its own — this is what lets the loop take
    # k separate autograd.grad calls and hand the rule k distinct directions.
    gs = {w: torch.autograd.grad(b, leaf5, retain_graph=True)[0] for w, b in pobj.items()}
    check_true("every component is differentiable w.r.t. the attention",
               all(g is not None for g in gs.values()))
    check_true("every component's gradient is non-zero",
               all(float(g.abs().sum()) > 0 for g in gs.values()))
    check_true("the components give DIFFERENT directions (else aggregation is vacuous)",
               float((gs["a"] - gs["b"]).abs().sum()) > 1e-12)

    # A satisfied subject (M_s >= 1) has an exactly-zero gradient, which would become
    # unit-norm garbage and an all-zero Gram row. It must be dropped, not returned.
    # Only a SINGLE-subword subject can reach M_s = 1: word_reduce="mean" averages a word's
    # subword columns, and the text-axis softmax splits the mass between them, so a
    # 2-subword word saturates at 1/len(subwords). WORD_B is the single-token one.
    logits_sat = torch.full((T * H * W, LK), -50.0, dtype=torch.float64)
    logits_sat[:, WORD_B] = 50.0        # WORD_B takes essentially all the softmax mass
    caps6, _ = make_captures(logits_sat)
    m6 = build_model(frame_reduce="global", smooth_attentions=False)
    m6.set_text_token_indices(TEXT_IDX)
    p6 = m6.per_object_bce(caps6, 0, target_words_indices=twi)
    _, met6 = m6.loss(caps6, 0, target_words_indices=twi)
    check_true("a satisfied subject is reported as such", met6["per_word_prob"]["b"] >= 1.0)
    check_true("the satisfied subject is dropped from the components", "b" not in p6)
    check_true("the still-neglected subject survives", "a" in p6)
    check_true("no component is ever exactly zero",
               all(float(v) > 0 for v in p6.values()))

    # The multi-subword ceiling, asserted so the above reasoning is not folklore: a
    # k-subword subject cannot exceed 1/k, so its hinge never reaches 0 and the drop path
    # is insurance rather than a routine occurrence. WORD_A has 2 subwords -> M_a <= 0.5.
    logits_sat_a = torch.full((T * H * W, LK), -50.0, dtype=torch.float64)
    logits_sat_a[:, WORD_A] = 50.0
    caps7, _ = make_captures(logits_sat_a)
    m7 = build_model(frame_reduce="global", smooth_attentions=False)
    m7.set_text_token_indices(TEXT_IDX)
    _, met7 = m7.loss(caps7, 0, target_words_indices={"a": WORD_A})
    check("a fully-excited 2-subword subject saturates at 1/2",
          met7["per_word_prob"]["a"], 1.0 / len(WORD_A), tol=1e-9)

    # ------------------------------------------------- the rule consumes them (LTX shape)
    # End-to-end on the aggregation side: LTX-shaped per-object latent gradients through
    # the arm we actually intend to run. Shape convention is channel-dim-0, no batch.
    print("\nCAGrad over the per-subject gradients (LTX latent shape [C, F, H, W]):")
    from will_it_render.inference.steering_core import GRAD_AGGREGATE_REGISTRY
    rule = GRAD_AGGREGATE_REGISTRY["cagrad_patch_nounit_c8.0"]
    C, LF, LH, LW = 128, 13, 16, 16
    torch.manual_seed(1)
    pg = {"a": torch.randn(C, LF, LH, LW), "b": torch.randn(C, LF, LH, LW)}
    d = rule(pg, torch.randn(C, LF, LH, LW), {w: 0.5 for w in pg})
    check_true("aggregate keeps the latent shape", tuple(d.shape) == (C, LF, LH, LW))
    check_true("aggregate is finite", bool(torch.isfinite(d).all()))
    check_true("aggregate is non-zero", float(d.abs().sum()) > 0)
    # rescale=True matches ||d|| to the UNIT-NORMED mean's norm, regardless of `nounit`.
    # This is exactly why the A&E learning rate has to be re-calibrated for this arm.
    g0_ref = sum(g / g.norm() for g in pg.values()) / len(pg)
    check("||d|| is rescaled to the unit-norm mean", d.norm(), g0_ref.norm(), tol=1e-3)
    check_true("...which is NOT the raw A&E gradient scale (lr must be recalibrated)",
               abs(float(d.norm()) - float(sum(pg.values()).norm() / 2)) > 1.0)
    # A single surviving subject must degrade gracefully (the all-but-one-satisfied case).
    d1 = rule({"b": pg["b"]}, None, {"b": 0.5})
    check_true("single-subject case is finite and unit-norm",
               bool(torch.isfinite(d1).all()) and abs(float(d1.norm()) - 1.0) < 1e-4)

    # The scale-matched arm: rescale to the argmax-hinge subject's gradient norm, which for a
    # max-over-subjects objective IS the step plain A&E takes. Direction-only change => A&E's
    # own lr carries over untouched, with none of the 250x rescale the "ref" arm needs.
    print("\nScale-matched CAGrad (direction-only; A&E keeps its own step size):")
    rule_ae = GRAD_AGGREGATE_REGISTRY["cagrad_patch_nounit_aescale_c8.0"]
    # Make the hinges unequal and the gradient norms very unequal, so matching the argmax
    # subject is distinguishable from matching the mean or the other subject.
    pg2 = {"a": torch.randn(C, LF, LH, LW) * 10.0, "b": torch.randn(C, LF, LH, LW) * 0.1}
    sc = {"a": 0.2, "b": 0.9}                       # "b" is the most neglected -> A&E picks it
    d_ae = rule_ae(pg2, None, sc)
    check("||d|| matches the ARGMAX-hinge subject's gradient", d_ae.norm(), pg2["b"].norm(),
          tol=1e-2)
    check_true("...which is NOT the other subject's norm",
               abs(float(d_ae.norm()) - float(pg2["a"].norm())) > 1.0)
    g0_ref2 = sum(g / g.norm() for g in pg2.values()) / len(pg2)
    check_true("...and NOT the unit-norm mean the default arm uses",
               abs(float(d_ae.norm()) - float(g0_ref2.norm())) > 1e-3)
    check_true("scale-matched aggregate is finite", bool(torch.isfinite(d_ae).all()))
    # Direction must still be CAGrad's, not a copy of the argmax subject's gradient.
    cos = float((d_ae.flatten() @ pg2["b"].flatten())
                / (d_ae.norm() * pg2["b"].norm() + 1e-12))
    check_true("direction is the rule's, not a rescaled copy of the argmax gradient",
               abs(cos) < 0.999)
    # c=0.0 is the matched-mean control: same step length, no trust-region correction.
    d_c0 = GRAD_AGGREGATE_REGISTRY["cagrad_patch_nounit_aescale_c0.0"](pg2, None, sc)
    check("the c=0 control takes the SAME step length", d_c0.norm(), d_ae.norm(), tol=1e-2)
    check_true("...but a different direction from c=8",
               float((d_c0 - d_ae).abs().sum()) > 1e-6)
    # Regression: the default "ref" arm must be untouched by the new option.
    d_ref = rule(pg2, None, sc)
    check("default scale='ref' still matches the unit-norm mean", d_ref.norm(),
          g0_ref2.norm(), tol=1e-3)

    # --------------------------------------------------------------------- schedule
    print("\nSchedule (their max_iter_to_alter / scale_range / thresholds):")
    ms = build_model(frame_reduce="global", max_step=25,
                     thresholds={0: 0.05, 10: 0.5, 20: 0.8}, max_refinement_steps=20)
    check_true("guides the first 25 steps", all(ms.should_steer(i) for i in range(25)))
    check_true("stops guiding at step 25", not any(ms.should_steer(i) for i in (25, 30, 49)))
    check("step 0 gets 1 + 20 iterations", ms.iters_for_step(0), 21)
    check("a plain step gets 1 iteration", ms.iters_for_step(5), 1)
    check("step 20 is a refinement step", ms.iters_for_step(20), 21)
    check("lr multiplier at step 0 = sqrt(1.0)", ms.lr_multiplier(0, 50), 1.0)
    check("lr multiplier at the last step = sqrt(0.5)", ms.lr_multiplier(49, 50), math.sqrt(0.5))
    check("lr multiplier decays monotonically",
          float(ms.lr_multiplier(10, 50) > ms.lr_multiplier(20, 50)), 1.0)

    # Threshold logic: at step 10 the target is 1 - 0.5 = 0.5.
    check_true("iteration 0 always applies an update",
               not ms.stop_iteration(10, {"ae_loss": 0.9}, 0))
    check_true("refinement continues while the subject is below threshold",
               not ms.stop_iteration(10, {"ae_loss": 0.9}, 1))
    check_true("refinement stops once the threshold is cleared",
               ms.stop_iteration(10, {"ae_loss": 0.4}, 1))
    check_true("non-threshold steps never early-exit",
               not ms.stop_iteration(5, {"ae_loss": 0.0}, 1))

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("All Attend-and-Excite self-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
