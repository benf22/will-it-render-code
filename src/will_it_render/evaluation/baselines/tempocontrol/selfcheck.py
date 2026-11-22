"""CPU self-check of the TempoControl loss terms — no GPU, no model, seconds to run.

    python -m baselines.tempocontrol.selfcheck

Builds synthetic captures with a known attention map and checks each term against a
hand-computed value, plus the two properties the all-frames adaptation rests on:
a constant control signal makes the Pearson term exactly zero *and* gradient-free, and
the loss stays differentiable w.r.t. the captured attention.
"""
from __future__ import annotations

import sys

import torch

from will_it_render.evaluation.baselines.tempocontrol.loss import TempoControlSteeringModel
from will_it_render.common.probe_types import BlockCapture, CaptureType

N_BLOCKS, N_HEADS, T, H, W = 3, 2, 5, 4, 4
LK = 7
TARGET = [2, 3]  # two subword tokens of one target word

_failures: list[str] = []


def _scalar(x) -> float:
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def check(name: str, got, want, tol: float = 1e-5) -> None:
    got, want = _scalar(got), _scalar(want)
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got:.6f}, want {want:.6f}")
    if not ok:
        _failures.append(name)


def check_true(name: str, ok: bool) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        _failures.append(name)


def make_captures(attn_row: torch.Tensor) -> list[BlockCapture]:
    """Captures whose token/head/layer-averaged map is exactly ``attn_row`` [T,H,W].

    Every block and head carries the same values, and the target tokens carry them
    identically, so all the means the model takes are identity operations.
    """
    flat = attn_row.reshape(-1)  # [Lq]
    attn = torch.zeros(1, N_HEADS, flat.numel(), LK, dtype=torch.float64)
    for t in TARGET:
        attn[0, :, :, t] = flat
    attn.requires_grad_(True)
    key = torch.zeros(1, LK, N_HEADS, 8, dtype=torch.float64)
    return [
        BlockCapture(attn_map=attn, text_key=key, capture_type=CaptureType.VIDEO_CA)
        for _ in range(N_BLOCKS)
    ], attn


def build(**weights) -> TempoControlSteeringModel:
    kwargs = dict(
        energy_weight=0.0, entropy_weight=0.0, pearson_weight=0.0, weakest_weight=0.0
    )
    kwargs.update(weights)
    return TempoControlSteeringModel(
        device=torch.device("cpu"),
        agg_kwargs={"extract_keys": False, "spatial_downsample": 1, "latent_hw": (H, W)},
        diffusion_steps_range=[0],
        **kwargs,
    )


def main() -> int:
    torch.manual_seed(0)
    # Per-frame energies deliberately uneven so mean != min.
    attn_row = torch.rand(T, H, W, dtype=torch.float64) * torch.linspace(
        0.2, 1.0, T, dtype=torch.float64
    ).view(T, 1, 1)
    captures, attn_leaf = make_captures(attn_row)
    words = {"dog": TARGET}

    flat = attn_row.reshape(T, -1)
    energies = flat.sum(dim=1)
    probs = flat / flat.sum(dim=1, keepdim=True)
    entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

    print("attention map recovery + term values (all-ones control signal)")
    model = build(energy_weight=1.0)
    recovered = model._attention_map(captures, TARGET)
    check("attention map max-abs error", (recovered - attn_row).abs().max(), 0.0, tol=1e-9)

    loss, m = model.loss(captures, 0, target_words_indices=words)
    check("energy term (loss)", loss, -energies.mean())
    check("reported energy metric", m["energy"], energies.mean())
    check("reported min_energy metric", m["min_energy"], energies.min())

    loss, _ = build(entropy_weight=1.0).loss(captures, 0, target_words_indices=words)
    check("entropy term (loss)", loss, entropies.mean())

    loss, _ = build(weakest_weight=1.0).loss(captures, 0, target_words_indices=words)
    check("weakest-frame term (loss)", loss, -energies.min())

    print("\nall-ones control signal ⇒ Pearson term is inert")
    loss_p, m_p = build(pearson_weight=1.0).loss(captures, 0, target_words_indices=words)
    check("pearson-only loss", loss_p, 0.0, tol=0.0)
    check("reported correlation", m_p["pearson"], 0.0, tol=0.0)
    grad = torch.autograd.grad(loss_p, attn_leaf, allow_unused=True)[0]
    check_true(
        "pearson-only loss has no gradient w.r.t. the attention",
        grad is None or float(grad.abs().max()) == 0.0,
    )
    # Weighting the Pearson term therefore cannot change the all-frames objective.
    l_off, _ = build(energy_weight=0.3, entropy_weight=10.0).loss(
        captures, 0, target_words_indices=words
    )
    l_on, _ = build(energy_weight=0.3, entropy_weight=10.0, pearson_weight=1.0).loss(
        captures, 0, target_words_indices=words
    )
    check("pearson_weight does not change the all-ones loss", l_on, l_off, tol=0.0)

    print("\nnon-constant control signal ⇒ Pearson is live (their temporal task)")
    ramp = [0] * (T - 2) + [1, 1]
    model_ramp = build(pearson_weight=1.0, control_signal=ramp)
    loss_r, m_r = model_ramp.loss(captures, 0, target_words_indices=words)
    check_true("correlation is non-zero", abs(m_r["pearson"]) > 1e-6)
    grad_r = torch.autograd.grad(loss_r, attn_leaf, allow_unused=True)[0]
    check_true(
        "pearson term has a gradient", grad_r is not None and float(grad_r.abs().max()) > 0.0
    )
    # Frames outside the signal are penalised by the negative-energy term (eq. 6).
    _, m_neg = build(energy_weight=1.0, control_signal=ramp).loss(
        captures, 0, target_words_indices=words
    )
    check("neg_energy over inactive frames", m_neg["neg_energy"], energies[: T - 2].mean())
    check("energy over active frames", m_neg["energy"], energies[T - 2 :].mean())

    print("\ncorrelation-based early stopping + per-word control")
    # pearson_primary tracks the FIRST word (control_signal1) — single word ⇒ == pearson.
    check("pearson_primary == first word's correlation", m_r["pearson_primary"], m_r["pearson"], tol=0.0)
    check_true("early-stop on by default (τ=0.9)", model_ramp.early_stop_threshold == 0.9)
    check_true("early-stop metric is pearson_primary", model_ramp.early_stop_metric == "pearson_primary")
    off = build(pearson_weight=1.0, control_signal=ramp, correlation_based_steps=False)
    check_true("flag off ⇒ no early stop", off.early_stop_threshold is None)
    # Per-word control: temp word → ramp (non-zero corr), static word → all-ones (0 corr).
    # pearson_primary (temp) then differs from the two-word mean, proving distinct routing.
    mw = build(pearson_weight=1.0, energy_weight=1.0)
    mw.set_prompt_timelines({"dog": ramp, "cat": [1] * T})
    _, mm = mw.loss(captures, 0, target_words_indices={"dog": TARGET, "cat": TARGET})
    check_true("per-word control routes distinct signals per word",
               abs(mm["pearson_primary"]) > 1e-6 and abs(mm["pearson_primary"] - mm["pearson"]) > 1e-6)

    print("\ncontrol-signal resampling + multi-word averaging")
    resampled = build(energy_weight=1.0, control_signal=[0] * 20 + [1] * 20)
    resampled.loss(captures, 0, target_words_indices=words)
    check("resampled signal length", resampled._control_signal.numel(), T)
    two_words, _ = build(energy_weight=1.0).loss(
        captures, 0, target_words_indices={"dog": TARGET, "cat": TARGET}
    )
    check("two identical words average to one word's loss", two_words, -energies.mean())

    print("\ndifferentiability of the full objective")
    full, _ = build(energy_weight=0.3, entropy_weight=10.0, weakest_weight=1.0).loss(
        captures, 0, target_words_indices=words
    )
    g = torch.autograd.grad(full, attn_leaf)[0]
    check_true("full loss has a non-zero gradient", float(g.abs().max()) > 0.0)

    print("\nrandom_window control signal (Pareto on-fraction knob)")
    m = TempoControlSteeringModel(
        device=torch.device("cpu"),
        agg_kwargs={"extract_keys": False, "spatial_downsample": 1, "latent_hw": (H, W)},
        diffusion_steps_range=[0], energy_weight=1.0,
        control_signal="random_window", control_on_frames=1, control_window_seed=0,
    )
    m.set_prompt_context(prompt="a dog and a cat")
    w1 = m._random_window(T, torch.device("cpu"))
    check("1-frame window has exactly 1 active frame", w1.sum(), 1.0, tol=0.0)
    check_true("window is contiguous", int((w1 > 0).sum()) == 1)
    # deterministic across rebuilds for same prompt+seed
    m.set_prompt_context(prompt="a dog and a cat")
    check_true("same prompt+seed → same window", bool(torch.equal(w1, m._random_window(T, torch.device("cpu")))))
    # different prompts → (generally) different placement
    starts = set()
    for p in ["p one", "p two", "p three", "p four", "p five", "p six"]:
        m.set_prompt_context(prompt=p)
        w = m._random_window(T, torch.device("cpu"))
        starts.add(int((w > 0).nonzero().flatten()[0]))
    check_true("different prompts spread across the timeline", len(starts) >= 2)
    # 3-frame window is a contiguous block of 3
    m3 = TempoControlSteeringModel(device=torch.device("cpu"),
        agg_kwargs={"extract_keys": False, "spatial_downsample": 1, "latent_hw": (H, W)},
        diffusion_steps_range=[0], control_signal="random_window", control_on_frames=3)
    m3.set_prompt_context(prompt="x")
    w3 = m3._random_window(T, torch.device("cpu"))
    on = (w3 > 0).nonzero().flatten().tolist()
    check("3-frame window active count", w3.sum(), 3.0, tol=0.0)
    check_true("3-frame window contiguous", on == list(range(on[0], on[0] + 3)))
    # sparse signal makes neg-energy term active (suppression outside window)
    _, mm = m.loss(captures, 0, target_words_indices=words)
    check_true("sparse signal activates neg-energy suppression", mm["neg_energy"] > 0)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
