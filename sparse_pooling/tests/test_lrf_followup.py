"""Regression checks of the A3mean follow-up (docs/experiments/a3mean_long_horizon_batch_scaling.md): CSV identity, canonical streams,
device-side gamma, gradient accumulation, measurement accounting, resume.  The kernels themselves are covered by test_learned_reduction.py.

Tolerances were fixed before the comparisons were run.  Direct and accumulated execution are not required to be bitwise identical:
FP32 atomics reorder sums and FP32 accumulates in a different grouping.
"""
import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import learned_reduction_report as REPORT  # noqa: E402
import learned_reduction_tasks as T  # noqa: E402
import lrf_perf as PERF  # noqa: E402
import lrf_stream as S  # noqa: E402
import lrf_train as TR  # noqa: E402

from conftest import requires_cuda_backend  # noqa: E402
from sparse_pooling import CoverTopK16Layer  # noqa: E402
from sparse_pooling.experimental import reduction_reference as RR  # noqa: E402
from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_  # noqa: E402
from sparse_pooling.experimental.learned_reduction import ReducerConfig, ReductionModel  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

# Direct vs accumulated, same operands, before the optimizer (relative L2).
TOL_SPARSE_GRAD = 1e-4       # FP32 main_grad with a different summation grouping: the split-gradient kernel tolerance
TOL_DENSE_GRAD = 1e-4        # V / c / d / head, FP32 BLAS sums
TOL_INPUT_GRAD = 2e-3        # dx passes through BF16 (the training path casts it) before the projection's gradient: half a BF16 ulp, 2 ** -9
TOL_LOSS = 1e-5              # FP32 mean of the same values
TOL_ONE_UPDATE_DENSE = 1e-4  # FP32 parameters after one update with matched states
TOL_ONE_UPDATE_BF16 = 4e-3   # BF16 parameters: one stochastic rounding with the same seed; a gradient change can cross an ulp boundary (ulp 2 ** -8)
TOL_ORACLE = 1e-4
PRIMARY_MODELS = ["A0", "A2", "A3mean"]
SMALL_H, SMALL_S = 4096, 64


def relerr(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def small_base(H=SMALL_H, G=8, D=256, seed=1):
    L = KLayout(D, H, G, 64, 256)
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(L.weight_shape, device="cuda", generator=g) / 8).bfloat16()
    b = (0.1 * torch.randn(L.bias_shape, device="cuda", generator=g)).bfloat16()
    zero_padding_(L, w, b)
    return CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G, connectivity=ConnectivityPlan(64, 2, seed=seed),
                                          normalization="energy", dx_variant="mt2_nw4_wg2")


class SmallTask:
    """A teacher with a small input dimension; D0 matches the small layers."""
    name, D0, identity_proj = "small", 256, False

    def __init__(self):
        g = torch.Generator(device="cpu").manual_seed(5)
        self.A = (torch.randn(T.OUT, 256, generator=g) / 16).to("cuda")

    def __call__(self, x0):
        return torch.tanh(x0 @ self.A.t())


def ns(model="A3mean", m_phys=512, accum=1, updates=8, seed=0, s=SMALL_S, **kw):
    d = dict(root="", H=SMALL_H, s=s, model=model, task="small", seed=seed, m_phys=m_phys, accum=accum, updates=updates, eval_every=4, grad_every=0, full_every=0,
             milestones=[], lr_w=1e-3, lr_small=1e-3, p_mult=1.0, lr_multiplier=1.0, verbose=0, stage="test")
    d.update(kw)
    return argparse.Namespace(**d)


def trainer(model="A3mean", base_seed=1, on_block=None, **kw):
    a = ns(model=model, **kw)
    return TR.Trainer(a, base=small_base(seed=base_seed), task=SmallTask(), output_dim=32, on_block=on_block)


# ------------------------------------------------------------------------------------- CSV identity
def perf_record(dx, dx_ms):
    return dict(H=10, model="A0", batch=8192, dx=dx, config=dict(epilogue="mma_split", dp_mode="fused"), wpt=2, dx_mt=2, dx_nw=4, dx_wg=2, dx_reload=0, tf32=False,
                median_ms=dict(dx=dx_ms, step=10.0 + dx_ms), memory=dict(peak_full_step_bytes=1, note="x"), last_info=dict(kgen_dx=dict(variant=dict(name=dx))))


def test_perf_csv_keeps_dx_request_and_timing(tmp_path):
    rows = [REPORT.perf_row(perf_record("mt2_nw4_wg2", 100.0)), REPORT.perf_row(perf_record("mt4_nw8_wg4", 120.0))]
    keys = sorted({k for d in rows for k in d})
    f = tmp_path / "perf.csv"
    REPORT.write_csv(f, rows, keys)
    with open(f) as h:
        back = list(csv.DictReader(h))
    assert "dx" not in back[0], "no column may mix the request and the timing"
    groups = {r["dx_request"]: float(r["dx_kernel_ms"]) for r in back}
    assert groups == {"mt2_nw4_wg2": 100.0, "mt4_nw8_wg4": 120.0}
    assert {r["dx_resolved_variant"] for r in back} == {"mt2_nw4_wg2", "mt4_nw8_wg4"}


def test_historical_perf_records_recover_request_and_timing():
    p = os.path.join(HERE, "..", "results", "learned_sliding_window_reduction", "raw_perf.jsonl.gz")
    with gzip.open(p, "rt") as h:
        rs = [json.loads(ln) for ln in h]
    rows = [REPORT.perf_row(r) for r in rs]
    assert len({r["dx_request"] for r in rows}) > 1
    for row, r in zip(rows, rs):
        assert row["dx_request"] == r["dx"] and row["dx_kernel_ms"] == r["median_ms"]["dx"]


# ------------------------------------------------------------------------------------- streams
def test_block_seed_is_stable_and_process_independent():
    code = ("import sys; sys.path.insert(0, %r); import lrf_stream as S; print(S.block_seed('train', 1, 7, 256))" % os.path.join(HERE, "..", "benchmarks"))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=str(i))).stdout.strip() for i in (1, 2)}
    assert outs == {str(S.block_seed("train", 1, 7, 256))}
    assert S.block_seed("train", 0, 0, 256) == 3160962896428375711


def test_splits_do_not_overlap():
    seeds = {}
    for D0 in (256, 1024):
        for sd in (0, 1, 2):
            for b in (*range(64), 16_383, 16_384, 200_000):
                seeds.setdefault(S.block_seed("train", sd, b, D0), ("train", sd, b, D0))
            seeds.setdefault(S.block_seed("calibration", sd, 0, D0), ("calibration", sd, 0, D0))
        for i in S.validation_ids():
            seeds.setdefault(S.block_seed("validation", 0, i, D0), ("validation", i, D0))
        for i in S.test_ids():
            seeds.setdefault(S.block_seed("test", 0, i, D0), ("test", i, D0))
    assert len(seeds) == 2 * (3 * 67 + 3 + S.N_VALIDATION + S.N_TEST)
    assert S.N_VALIDATION >= 8 and S.N_TEST >= 16


def test_direct_batch_is_exactly_the_accumulated_blocks_and_never_wraps():
    x = S.blocks("train", 2, 32, 16, 256)
    assert x.shape == (8192, 256)
    for i in range(16):
        assert torch.equal(x[i * 512:(i + 1) * 512], S.block("train", 2, 32 + i, 256))
    assert S.checksum(S.block("train", 2, 8191, 256)) != S.checksum(S.block("train", 2, 8192 + 8191, 256))
    assert S.checksum(S.block("train", 2, 0, 256)) != S.checksum(S.block("train", 2, 4096, 256))


@requires_cuda_backend
def test_batch_modes_consume_identical_block_sequences():
    logs = {}
    for name, kw in dict(direct=dict(m_phys=8192, accum=1), accumulated=dict(m_phys=512, accum=16), small=dict(m_phys=512, accum=1, updates=48)).items():
        seen = []
        kw = dict(kw)
        t = trainer(on_block=lambda s, b, seen=seen: seen.append((s, b)), **kw)
        for _ in range(kw.get("updates", 3)):
            t.effective_update()
        logs[name] = [b for s, b in seen if s == "train"]
        assert not [s for s, b in seen if s == "test"], "training must not read the test split"
        assert t.samples_seen == 48 * 512
        assert t.next_block == 48
    assert logs["direct"] == logs["accumulated"] == logs["small"] == list(range(48))


@requires_cuda_backend
def test_seen_diagnostics_never_read_future_positions():
    seen = []
    t = trainer(on_block=lambda s, b: seen.append((s, b)))
    t.seen_diagnostics()
    assert not [x for x in seen if x[0] == "train"], "nothing is consumed yet: no training diagnostic may read the stream"
    for _ in range(3):
        t.effective_update()
    seen.clear()
    d = t.seen_diagnostics()
    assert d["recent_seen"] is not None and d["fixed_seen"] is None
    assert max(b for s, b in seen if s == "train") < t.next_block
    for _ in range(7):
        t.effective_update()
    seen.clear()
    d = t.seen_diagnostics()
    assert d["fixed_seen"] is not None
    assert max(b for s, b in seen if s == "train") < t.next_block
    assert sorted({b for s, b in seen}) == list(range(10)), "recent 2..9 and fixed 0..7"


# ------------------------------------------------------------------------------------- gamma
@requires_cuda_backend
@pytest.mark.parametrize("model", PRIMARY_MODELS)
def test_forward_does_not_read_gamma_on_the_host(model, monkeypatch):
    t = trainer(model=model)
    x = torch.randn(64, 256, device="cuda").to(torch.bfloat16)
    gid = id(t.model.gamma)
    real = torch.Tensor.__float__

    def guard(self):
        if id(self) == gid:
            raise AssertionError("gamma read on the host in forward")
        return real(self)

    monkeypatch.setattr(torch.Tensor, "__float__", guard)
    t.model(x)


@requires_cuda_backend
@pytest.mark.parametrize("gamma", [1.0, 3.25])
def test_gamma_output_gradient_and_checkpoint(gamma):
    base = small_base()
    cfg = ReducerConfig("A3mean", "learned", L=SMALL_S, s=SMALL_S, init="mean")
    m = ReductionModel(base, cfg, output_dim=32, seed=3, gamma=gamma)
    x = torch.randn(64, 256, device="cuda").to(torch.bfloat16)
    U = m.features(x) + m.c
    y = m(x)
    want = (U * gamma) @ m.V + m.d
    assert relerr(y.detach(), want.detach()) < 1e-5
    if gamma == 1.0:
        assert torch.equal((U * m.gamma).detach(), U.detach()), "gamma = 1 is an exact identity in FP32"
    gy = torch.randn_like(y)
    g_new = torch.autograd.grad(y, m.V, gy)[0]
    g_ref = torch.autograd.grad((m.features(x) + m.c) * gamma @ m.V + m.d, m.V, gy)[0]
    assert relerr(g_new, g_ref) < 1e-5
    st = {k: v.clone() for k, v in m.state_dict().items() if k == "gamma"}
    assert float(st["gamma"]) == pytest.approx(gamma)
    m.gamma.fill_(9.0)
    m.load_state_dict(st, strict=False)
    assert float(m.gamma) == pytest.approx(gamma)
    m.zero_main_grad()


# ------------------------------------------------------------------------------------- accumulation
def test_accumulation_algebra_fp64():
    """Independent FP64 reference: 16 x backward(mean / 16) over equal microbatches = backward(mean over all rows)."""
    torch.manual_seed(0)
    x = torch.randn(8192 // 64, 6, dtype=torch.float64)
    tgt = torch.randn(x.shape[0], 3, dtype=torch.float64)
    W = torch.randn(6, 7, dtype=torch.float64, requires_grad=True)
    V = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)

    def f(a, t):
        return (torch.relu(a @ W) @ V - t).square().mean()

    f(x, tgt).backward()
    gw, gv = W.grad.clone(), V.grad.clone()
    W.grad, V.grad = None, None
    for a, t in zip(x.chunk(16), tgt.chunk(16)):
        assert a.shape[0] == x.shape[0] // 16
        (f(a, t) / 16).backward()
    assert relerr(W.grad, gw) < 1e-13 and relerr(V.grad, gv) < 1e-13
    # the double-average bug this protects against is detectable
    assert relerr(W.grad / 16, gw) > 0.9


def collect(t):
    m = t.model
    d = dict(W=m.sparse.weight.main_grad.clone(), b=m.sparse.bias.main_grad.clone(), head=t.head.weight.grad.clone(), head_b=t.head.bias.grad.clone(),
             proj=t.proj.weight.grad.clone(), proj_b=t.proj.bias.grad.clone())
    if m.reducer is not None:
        d["P"] = m.reducer.P.main_grad.clone()
    for k in ("c", "V", "d"):
        p = getattr(m, k)
        if p is not None:
            d[k] = p.grad.clone()
    return d


def paired_trainers(model, **kw):
    a = trainer(model=model, m_phys=8192, accum=1, **kw)
    b = trainer(model=model, m_phys=512, accum=16, **kw)
    for (ka, pa), (kb, pb) in zip(a.model.named_parameters(), b.model.named_parameters()):
        assert ka == kb and torch.equal(pa.detach().float(), pb.detach().float())
    assert torch.equal(a.proj.weight, b.proj.weight) and torch.equal(a.head.weight, b.head.weight)
    assert float(a.model.gamma) == float(b.model.gamma), "one canonical calibration: gamma does not depend on the execution mode"
    return a, b


@requires_cuda_backend
@pytest.mark.parametrize("model", PRIMARY_MODELS)
def test_direct_and_accumulated_gradients_agree_before_the_optimizer(model):
    a, b = paired_trainers(model)
    la, lb = a.backward_calls(), b.backward_calls()
    assert abs(la - lb) / abs(la) < TOL_LOSS
    ga, gb = collect(a), collect(b)
    for k in ga:
        tol = TOL_INPUT_GRAD if k.startswith("proj") else (TOL_SPARSE_GRAD if k in ("W", "b", "P") else TOL_DENSE_GRAD)
        assert relerr(gb[k], ga[k]) < tol, f"{model} {k}: {relerr(gb[k], ga[k]):.2e}"
    assert a.fused_calls == b.fused_calls == 0
    assert not torch.equal(ga["W"], torch.zeros_like(ga["W"]))


@requires_cuda_backend
def test_accumulated_a3_gradients_match_the_fp64_oracle():
    """The accumulated main_grad of W / b / P against a materializing FP64 oracle of the whole 8,192-row mean."""
    t = trainer(model="A3mean", m_phys=512, accum=16, s=SMALL_S)
    red, sp = t.model.reducer, t.model.sparse
    t.backward_calls()
    x0 = S.blocks("train", 0, 0, 16, 256).to("cuda")
    tgt = t.task(x0)
    xq = t.proj(x0).detach().to(torch.bfloat16).double()
    W = sp.weight.detach().double().requires_grad_(True)
    b = sp.bias.detach().double().requires_grad_(True)
    Pl = red.routing.unpack(red.P.detach()).double().requires_grad_(True)
    starts = torch.arange(red.spec.J) * red.spec.stride
    h = RR.hidden(xq, sp.indices, W, b, sp.counts)
    U = RR.window_outputs(h, Pl, starts) + t.model.c.detach().double()
    y = (U * float(t.model.gamma)) @ t.model.V.detach().double() + t.model.d.detach().double()
    loss = (y.float().double() @ t.head.weight.detach().double().t() + t.head.bias.detach().double() - tgt.double()).square().mean()
    dw, db, dp = torch.autograd.grad(loss, [W, b, Pl])
    assert relerr(sp.weight.main_grad, dw) < TOL_ORACLE
    assert relerr(sp.bias.main_grad, db) < TOL_ORACLE
    assert relerr(red.P.main_grad, red.routing.pack(dp)) < TOL_ORACLE


@requires_cuda_backend
@pytest.mark.parametrize("model", PRIMARY_MODELS)
def test_one_matched_update_and_counters(model):
    a, b = paired_trainers(model)
    a.effective_update()
    b.effective_update()
    assert a.fused_calls == b.fused_calls == a.small_calls == b.small_calls == 1, "one optimizer step per effective update"
    assert a.fused._seed == b.fused._seed
    assert a.update_count == b.update_count == 1
    assert a.next_block == b.next_block == 16
    assert a.samples_seen == b.samples_seen == 8192
    for (k, pa), (_, pb) in zip(a.model.named_parameters(), b.model.named_parameters()):
        tol = TOL_ONE_UPDATE_DENSE if pa.dtype == torch.float32 else TOL_ONE_UPDATE_BF16
        assert relerr(pb.detach().float(), pa.detach().float()) < tol, f"{model} {k}: {relerr(pb.detach().float(), pa.detach().float()):.2e}"
    for t in (a, b):
        assert float(t.model.sparse.weight.main_grad.abs().max()) == 0.0, "the fused optimizer consumes main_grad only at the update"
        assert t.head.weight.grad is None
        for g in t.fused.param_groups:
            for p in g["params"]:
                assert t.fused.state[p]["step"] == 1


@requires_cuda_backend
def test_microbatches_do_not_update_parameters():
    t = trainer(model="A3mean", m_phys=512, accum=16)
    w0, p0 = t.model.sparse.weight.detach().clone(), t.model.reducer.P.detach().clone()
    t.backward_calls()
    assert torch.equal(t.model.sparse.weight.detach(), w0) and torch.equal(t.model.reducer.P.detach(), p0)
    assert float(t.model.reducer.P.main_grad.abs().max()) > 0
    t.optimizer_step()
    assert not torch.equal(t.model.reducer.P.detach(), p0)


@requires_cuda_backend
@pytest.mark.parametrize("model", PRIMARY_MODELS)
def test_several_effective_updates_stay_close_and_finite(model):
    a, b = paired_trainers(model)
    for _ in range(4):
        la, lb = a.effective_update(), b.effective_update()
        assert math.isfinite(la) and math.isfinite(lb)
    assert a.update_count == b.update_count == 4 and a.samples_seen == b.samples_seen == 4 * 8192
    a.check_position()
    b.check_position()
    assert abs(a.validation() - b.validation()) / a.validation() < 1e-2


def test_score_uses_the_last_four_evaluations_and_marks_divergence():
    curve = [dict(val=v) for v in (9.0, 4.0, 3.0, 2.0, 1.0)]
    assert TR.score_of(curve, True) == pytest.approx(2.5)
    assert TR.score_of(curve, False) == float("inf")
    assert TR.score_of(curve[:3], True) == float("inf")
    assert TR.score_of([dict(val=float("nan"))] * 4, True) == float("inf")


# ------------------------------------------------------------------------------------- accounting
@requires_cuda_backend
@pytest.mark.parametrize("model", PRIMARY_MODELS)
def test_storage_categories_deduplicate_aliases(model):
    t = trainer(model=model)
    t.effective_update()
    cats = PERF.storage_categories(t.model, t.proj, t.head, t.fused, t.small, [])
    L = t.model.sparse
    assert cats["model_parameters"] >= L.weight.untyped_storage().nbytes() + L.bias.untyped_storage().nbytes()
    assert cats["main_grad"] == sum(p.main_grad.untyped_storage().nbytes() for g in t.fused.param_groups for p in g["params"])
    assert cats["fused_optimizer_state"] == sum(2 * p.numel() * 2 for g in t.fused.param_groups for p in g["params"])
    if model == "A2":
        base_ptr = L.weight.untyped_storage().data_ptr()
        assert t.model.sparse.weight.untyped_storage().data_ptr() == base_ptr
    assert sum(cats.values()) <= torch.cuda.memory_allocated()


# ------------------------------------------------------------------------------------- resume and test lock
def trained_state(t):
    return {k: v.float().clone() for k, v in t.parameter_state().items()}


def worst_difference(a, b):
    return max(relerr(a[k], b[k]) for k in a)


@requires_cuda_backend
@pytest.mark.parametrize("model", ["A0", "A3mean"])
def test_resume_consumes_the_next_blocks_and_matches_within_nondeterminism(model, tmp_path):
    U, half = 12, 6
    runs = {}
    for name in ("a", "b"):
        seen = []
        t = trainer(model=model, updates=U, on_block=lambda s, b, seen=seen: seen.append((s, b)))
        t.train()
        runs[name] = (trained_state(t), t.curve[-1]["val"], [b for s, b in seen if s == "train"])
    # the same run, interrupted
    first = trainer(model=model, updates=half)
    first.train()
    first.save_full(tmp_path)
    ck = sorted(tmp_path.glob("full_u*.pt"))[-1]
    seen = []
    c = trainer(model=model, updates=U, on_block=lambda s, b: seen.append((s, b)))
    gamma0 = float(c.model.gamma)
    c.model.gamma.fill_(123.0)
    c.load_full(ck)
    assert float(c.model.gamma) == gamma0, "gamma comes from the checkpoint"
    assert (c.update_count, c.next_block) == (half, half)
    assert c.fused._seed == first.fused._seed
    consumed = []
    orig = c.next_microbatch

    def spy():
        consumed.append(c.next_block)
        return orig()

    c.next_microbatch = spy
    c.train()
    assert consumed == list(range(half, U)), "the resumed run consumes exactly the next blocks"
    resumed = trained_state(c)
    noise = worst_difference(runs["a"][0], runs["b"][0])
    diff = worst_difference(runs["a"][0], resumed)
    assert diff <= max(10 * noise, 1e-2), f"resume difference {diff:.2e}, floating-point baseline {noise:.2e}"
    assert abs(c.curve[-1]["val"] - runs["a"][1]) / runs["a"][1] < 1e-2
    assert [e["update"] for e in c.curve] == sorted({e["update"] for e in c.curve})


@requires_cuda_backend
def test_checkpoint_rejects_another_identity_and_rotates(tmp_path):
    t = trainer(model="A3mean", updates=2, eval_every=1)
    t.train()
    for _ in range(4):
        t.effective_update()
        t.save_full(tmp_path, keep=2)
    assert len(list(tmp_path.glob("full_u*.pt"))) == 2
    other = trainer(model="A2", updates=2)
    with pytest.raises((ValueError, KeyError)):
        other.load_full(sorted(tmp_path.glob("full_u*.pt"))[-1])


def test_evaluate_refuses_a_locked_test_split(tmp_path):
    a = argparse.Namespace(unlock=str(tmp_path / "missing.json"), record="", root="", verify_sha=0, out="")
    with pytest.raises(SystemExit, match="locked"):
        TR.evaluate(a)


# ------------------------------------------------------------------------------------- analysis
def test_paired_statistics_and_holm():
    import lrf_report as RP

    r = RP.paired({0: 1.0, 1: 2.0, 2: 3.0}, {0: 2.0, 1: 3.0, 2: 4.5})
    assert r["mean"] == pytest.approx(-7 / 6)
    assert r["pct"] == pytest.approx(100 * (-7 / 6) / (9.5 / 3))
    assert r["ci_low"] < r["mean"] < r["ci_high"]
    # scipy.stats.t.sf(4.303, 2) * 2 = 0.05
    assert RP.t2_p(4.303) == pytest.approx(0.05, abs=2e-4)
    assert RP.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert RP.holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])
    assert RP.holm([0.03, 0.04]) == pytest.approx([0.06, 0.06])


def test_curve_interpolation_never_extrapolates():
    import lrf_report as RP

    r = dict(curve=[dict(train_s=0.0, val=4.0), dict(train_s=10.0, val=2.0)])
    assert RP.curve_at(r, "train_s", 5.0) == pytest.approx(3.0)
    assert RP.curve_at(r, "train_s", 10.5) is None
