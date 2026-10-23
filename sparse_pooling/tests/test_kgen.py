"""EXPERIMENTAL generalized fan-in / logical C extensions (sparse_pooling.experimental) of the K/C/S/R sweep.

Correctness gate before any full-scale timing (results/RANDOM_KCSR_COVERAGE_PERFORMANCE.md section 4):
generator equivalence with the production generator, float64 reference comparisons of forward / dX / dW / db
(ragged H, partial C blocks, C % 16 == 8 tails, token tails, cancellation, near-zero pre-activations, full-K
activation decisions), non-finite propagation against the production kernels on the same graph, the matched
K = 8 backend control, the logical C = 128 / 256 physical lowering (forward, backward, round trips,
checkpoints), FusedAdamW / main_grad, unsupported requests, and a composed multi-step training run.
"""
import math

import pytest
import torch

from conftest import rel_err, requires_cuda_backend
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.config import PhysicalPlan
from sparse_pooling.layout import SparseLayout
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.supergroups import ConnectivityPlan, generate_indices

pytestmark = requires_cuda_backend


def _kg():
    from sparse_pooling.experimental import kgen_graph, kgen_layer, lowering, reference

    return kgen_graph, kgen_layer, lowering, reference


def kgen_layer(D, H, G, K, C, S, R, seed=3, prec="split", bias_std=0.3, act="relu", init_seed=5):
    _, KL, _, _ = _kg()
    pool = "mean" if act == "relu" else {"type": "mean", "activation": "identity"}
    return KL.KgenLayer(D, H, G, K, C, ConnectivityPlan(S, R, seed=seed), pooling=pool, device="cuda", seed=init_seed,
                        bias_std=bias_std, plan=PhysicalPlan(grad_precision=prec))


def run(lay, x, dy):
    """(y, dX, dW, db) with FP32 main_grad (cleared first) and an FP32 input (dX returned in FP32)."""
    lay.enable_main_grad()
    lay.zero_main_grad()
    x = x.detach().clone().requires_grad_(True)
    y = lay(x)
    y.backward(dy)
    return y.detach(), x.grad.detach(), lay.weight.main_grad.clone(), lay.bias.main_grad.clone()


def reference(lay, x, dy, act="relu"):
    _, _, _, REF = _kg()
    return REF.grads(x.double(), lay.indices.long(), lay.weight.detach().double(), lay.bias.detach().double(),
                     lay.counts, dy.double(), act)


def maxrel(a, r):
    a, r = a.double(), r.double()
    return float((a - r).abs().max() / (r.abs().max() + 1e-300))


def inputs(B, D, G, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, D, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(B, G, device="cuda", generator=g)
    return x, dy


# --------------------------------------------------------------------------------- generator
@pytest.mark.parametrize("H,C,S,R,seed", [(100003, 8, 32, 8, 0), (77777, 16, 16, 4, 1), (123457, 32, 64, 2, 2),
                                          (99991, 64, 128, 3, 0), (5000, 8, 8, 2, 4)])
def test_generalized_generator_reproduces_production_at_k8(H, C, S, R, seed):
    KG, _, _, _ = _kg()
    L = SparseLayout(1024, H, 1024, 8, C, True, packed=False)
    plan = ConnectivityPlan(S, R, seed=seed)
    idx, graph, _ = generate_indices(L, plan, "cuda")
    idx2, feat2, bk2, _ = KG.generate(1024, L.counts_tensor("cuda"), C, 8, L.blocks_per_group, L.unit_blocks, plan, "cuda")
    assert torch.equal(idx, idx2) and torch.equal(graph.feat, feat2) and torch.equal(graph.buckets, bk2)


@pytest.mark.parametrize("K,C,S,R,H", [(16, 16, 32, 4, 30011), (32, 24, 64, 3, 9001), (64, 256, 64, 4, 20000),
                                       (64, 8, 128, 2, 7777), (16, 136, 16, 8, 30000)])
def test_generalized_generator_structure(K, C, S, R, H):
    KG, _, _, _ = _kg()
    lay = kgen_layer(256, H, 16, K, C, S, R)
    rep = lay.kcsr_validate()
    assert rep["H"] == H and rep["logical_records"] == int(lay.layout.valid_blocks().sum())
    # local passes: inside every supergroup, consecutive S / K records of its stream partition the set
    tab = lay.kcsr_table()
    feat = lay.sg_feat.cpu().long()
    for q in range(min(feat.shape[0], 6)):
        rq = tab["records"][tab["rec_q"] == q].long()
        n_pass = S // K
        for p0 in range(0, (len(rq) // n_pass) * n_pass, n_pass):
            got = rq[p0:p0 + n_pass].reshape(-1).sort().values
            assert torch.equal(got, feat[q].sort().values)
    # K = S: every record of a supergroup is the whole set
    if K == S:
        assert all(torch.equal(r.long().sort().values, feat[q].sort().values)
                   for r, q in zip(tab["records"][:200], tab["rec_q"][:200].long()))


def test_generator_seeds_and_determinism():
    a = kgen_layer(256, 20000, 8, 16, 32, 64, 4, seed=1)
    b = kgen_layer(256, 20000, 8, 16, 32, 64, 4, seed=1)
    c = kgen_layer(256, 20000, 8, 16, 32, 64, 4, seed=2)
    assert torch.equal(a.indices, b.indices) and torch.equal(a.kg_buckets, b.kg_buckets)
    assert not torch.equal(a.indices, c.indices)


# ------------------------------------------------------------------------- vs float64 reference
CASES = [  # (K, C, S, R, H, G, D, B, precision)
    (8, 8, 32, 2, 999, 8, 128, 40, "split"), (8, 128, 32, 4, 5003, 8, 256, 56, "split"),
    (8, 256, 32, 8, 9000, 8, 256, 24, "bf16"), (16, 8, 64, 4, 1001, 8, 256, 64, "split"),
    (16, 16, 32, 2, 1000, 8, 128, 64, "split"), (16, 136, 16, 4, 3000, 4, 64, 24, "split"),
    (16, 256, 16, 3, 7000, 4, 128, 72, "bf16"), (32, 24, 64, 3, 1500, 8, 256, 72, "split"),
    (32, 64, 64, 2, 4097, 8, 256, 8, "split"), (32, 256, 128, 2, 9000, 4, 256, 32, "split"),
    (64, 8, 64, 3, 803, 8, 256, 48, "split"), (64, 40, 64, 2, 1203, 8, 256, 56, "bf16"),
    (64, 256, 128, 2, 5000, 4, 256, 48, "split"), (64, 64, 64, 1, 2050, 2, 128, 136, "split"),
]


@pytest.mark.parametrize("K,C,S,R,H,G,D,B,prec", CASES)
@pytest.mark.parametrize("act", ["relu", "identity"])
def test_against_float64_reference(K, C, S, R, H, G, D, B, prec, act):
    lay = kgen_layer(D, H, G, K, C, S, R, prec=prec, act=act)
    x, dy = inputs(B, D, G)
    y, dx, dw, db = run(lay, x, dy)
    yr, dxr, dwr, dbr = reference(lay, x, dy, act)
    tol = 2e-5 if prec == "split" else 2e-2
    assert maxrel(y, yr) < 1e-6
    assert maxrel(dx, dxr) < tol and maxrel(dw, dwr) < tol
    assert maxrel(db, dbr) < 1e-6  # db = sum of FP32 dZ (no BF16 operand)
    inv = ~lay.layout.valid_mask("cuda")
    assert bool((dw[inv] == 0).all()) and bool((db[inv] == 0).all())  # padded neurons never receive gradient


def test_cancellation_and_near_zero_preactivation():
    """z = sum of K terms that cancel to (near) zero: one activation decision on the complete sum."""
    K, C, S, R, H, G, D, B = 32, 16, 64, 2, 4096, 8, 256, 64
    lay = kgen_layer(D, H, G, K, C, S, R, bias_std=0.0)
    with torch.no_grad():  # weights +a on the first half of the fan-in, -a on the second
        a = torch.randn(lay.weight.shape[:-1], device="cuda").abs()[..., None].bfloat16()
        sign = torch.ones(K, device="cuda").bfloat16()
        sign[K // 2:] = -1
        lay.weight.copy_(a * sign)
        lay.weight.mul_(lay.layout.valid_mask("cuda")[..., None])
    x = torch.ones(B, D, device="cuda")
    x[B // 2:] = (torch.randn(B // 2, D, device="cuda") * 1e-3).bfloat16().float()  # BF16-exact (the layer consumes BF16)
    dy = torch.randn(B, G, device="cuda")
    y, dx, dw, db = run(lay, x, dy)
    yr, dxr, dwr, dbr = reference(lay, x, dy)
    assert torch.equal(y[: B // 2], torch.zeros_like(y[: B // 2]))  # exact cancellation -> z = 0 -> inactive
    assert maxrel(y, yr) < 1e-5 and maxrel(dx, dxr) < 2e-5 and maxrel(dw, dwr) < 2e-5 and maxrel(db, dbr) < 1e-6


@pytest.mark.parametrize("K", [16, 32, 64])
def test_full_k_activation_decision(K):
    """Partial fan-in sums of opposite sign to the full sum: a K8 split (two independently activated sub-neurons)
    would differ; the layer must use the single complete-sum decision in forward and every gradient."""
    C, S, R, H, G, D, B = 16, 64, 2, 2048, 4, 128, 32
    lay = kgen_layer(D, H, G, K, C, S, R, bias_std=0.0)
    with torch.no_grad():
        w = torch.zeros(lay.weight.shape, device="cuda")
        w[..., :8] = 1.0          # first K8 tile: +8 x
        w[..., 8:16] = -1.25      # second tile: -10 x  -> full z = -2 x < 0 for x > 0
        lay.weight.copy_(w.bfloat16() * lay.layout.valid_mask("cuda")[..., None])
    x = torch.ones(B, D, device="cuda")
    dy = torch.randn(B, G, device="cuda")
    y, dx, dw, db = run(lay, x, dy)
    assert torch.equal(y, torch.zeros_like(y)) and torch.equal(dw, torch.zeros_like(dw)) and torch.equal(dx, torch.zeros_like(dx))
    yr, dxr, dwr, dbr = reference(lay, x, dy)
    assert float(yr.abs().max()) == 0.0
    # the K8-split model (each 8-input tile activated and pooled separately) is active on the first tile
    _, _, _, REF = _kg()
    idx, wd, bd = lay.indices.long(), lay.weight.detach().double(), lay.bias.detach().double()
    y_split = sum(REF.forward(x.double(), idx[..., t:t + 8], wd[..., t:t + 8], bd * (t == 0), lay.counts)
                  for t in range(0, K, 8))
    assert float(y_split.min()) > 1.0


def test_token_tails_and_tiny_batches():
    for B in (8, 24, 40, 136, 520):
        lay = kgen_layer(256, 3001, 8, 32, 24, 64, 2)
        x, dy = inputs(B, 256, 8, seed=B)
        y, dx, dw, db = run(lay, x, dy)
        yr, dxr, dwr, dbr = reference(lay, x, dy)
        assert maxrel(y, yr) < 1e-6 and maxrel(dx, dxr) < 2e-5 and maxrel(dw, dwr) < 2e-5


def test_unsupported_requests_raise():
    _, KL, _, _ = _kg()
    with pytest.raises(ValueError):
        kgen_layer(256, 2000, 8, 12, 16, 32, 2)  # K not compiled
    with pytest.raises(ValueError):
        kgen_layer(256, 2000, 8, 16, 264, 32, 2)  # C > 256
    with pytest.raises(ValueError):
        kgen_layer(256, 2000, 8, 64, 16, 32, 2)  # K > S
    lay = kgen_layer(256, 2000, 8, 16, 16, 32, 2)
    x, dy = inputs(64, 256, 8)
    lay.plan = PhysicalPlan(reduction="deterministic")
    with pytest.raises(ValueError, match="deterministic"):
        lay(x)
    lay.plan = PhysicalPlan()
    with pytest.raises(ValueError, match="feature-major"):
        lay(x[:12])  # B % 8 != 0
    with pytest.raises(ValueError):
        KL.KgenLayer(256, 2000, 8, 16, 16, ConnectivityPlan(32, 2), pooling="max", device="cuda")(x)
    # the production layer still rejects K != 8 and C > 64
    with pytest.raises(ValueError, match="fan-in 8"):
        SparseLayout(256, 2000, 8, 16, 16, True)
    with pytest.raises(ValueError):
        SparseLayout(256, 2000, 8, 8, 128, True)


# ------------------------------------------------------------- matched K8 control / non-finite
def native_sg(D, H, G, C, S, R, prec="split", seed=0, act="relu"):
    pool = "mean" if act == "relu" else {"type": "mean", "activation": "identity"}
    lay = SparsePoolingLayer(D, H, G, shared_neurons=C, pooling=pool, device="cuda", seed=seed, bias_std=0.3,
                             connectivity=ConnectivityPlan(S, R, seed=seed), plan=PhysicalPlan(grad_precision=prec))
    return lay


def run_native(lay, x, dy):
    lay.enable_main_grad()
    lay.zero_main_grad()
    x = x.detach().clone().requires_grad_(True)
    y = lay(x)
    y.backward(dy)
    L = lay.layout
    _, dw, db = L.to_canonical(lay.indices, lay.weight.main_grad, lay.bias.main_grad)
    nb = -(-L.max_group // L.block)
    return y.detach(), x.grad, dw[:, :nb], db[:, :nb]


@pytest.mark.parametrize("C,S,R", [(8, 32, 4), (16, 16, 8), (64, 64, 2), (32, 128, 2)])
@pytest.mark.parametrize("prec", ["split", "bf16"])
def test_matched_k8_control_same_function_as_production(C, S, R, prec):
    _, KL, _, _ = _kg()
    D, H, G, B = 512, 40_003, 16, 64
    nat = native_sg(D, H, G, C, S, R, prec)
    kg = KL.from_native(nat)
    x, dy = inputs(B, D, G)
    yn, dxn, dwn, dbn = run_native(nat, x, dy)
    yk, dxk, dwk, dbk = run(kg, x, dy)
    tol = 2e-5 if prec == "split" else 1e-2
    assert rel_err(yk, yn) < 1e-6 and rel_err(dxk, dxn) < tol and rel_err(dwk, dwn) < tol and rel_err(dbk, dbn) < 1e-6


@pytest.mark.parametrize("case", ["inf", "nan", "huge", "subnormal", "zero_boundary"])
@pytest.mark.parametrize("prec", ["split", "bf16"])
def test_nonfinite_propagation_matches_production(case, prec):
    """kgen uses the production generic dZ expression: NaN / inf patterns of every gradient equal the production
    kernels' on the same graph (the production specialised epilogues fall back to that same generic path)."""
    _, KL, _, _ = _kg()
    D, H, G, B = 256, 20_000, 8, 64
    nat = native_sg(D, H, G, 8, 32, 8, prec)
    kg = KL.from_native(nat)
    x, dy = inputs(B, D, G)
    if case == "inf":
        dy[3, 2] = float("inf")
        dy[5, 1] = -float("inf")
    elif case == "nan":
        dy[7, 0] = float("nan")
    elif case == "huge":
        dy = dy * 1e30
    elif case == "subnormal":
        dy = dy * 1e-36
    else:
        with torch.no_grad():
            nat.bias.zero_()
            kg.bias.zero_()
        x[::2] = 0.0
    yn, dxn, dwn, dbn = run_native(nat, x, dy)
    yk, dxk, dwk, dbk = run(kg, x, dy)
    for a, b in ((dxk, dxn), (dwk, dwn), (dbk, dbn)):
        assert torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.isinf(a), torch.isinf(b))
        fin = torch.isfinite(b)
        assert rel_err(a[fin], b[fin]) < (2e-5 if prec == "split" else 1e-2)
    assert torch.equal(torch.isnan(yk), torch.isnan(yn)) and rel_err(yk, yn) < 1e-6


# --------------------------------------------------------------------------- logical C lowering
@pytest.mark.parametrize("C", [128, 256])
@pytest.mark.parametrize("prec", ["split", "bf16"])
def test_lowering_logical_c_to_native_c64(C, prec):
    KG, KL, LW, _ = _kg()
    D, H, G, B, S, R = 512, 50_017, 8, 64, 32, 4
    plan = ConnectivityPlan(S, R, seed=1)
    L, idx, w, b, bk, feat, spec, _ = LW.build_logical(D, H, G, C, plan, "cuda", seed=7, bias_std=0.3)
    KG.validate(D, L.counts_tensor("cuda"), C, 8, idx, feat, bk, R)
    low = LW.lower(L, idx, w, b, bk, feat, spec, plan=PhysicalPlan(grad_precision=prec), device="cuda")
    kg = KL.KgenLayer(D, H, G, 8, C, plan, device="cuda", plan=PhysicalPlan(grad_precision=prec),
                      graph=(idx, feat, bk, spec), weights=(w, b))
    # native structural validation of the physical C = 64 graph
    from sparse_pooling.supergroups import SupergroupGraph, validate

    Lp = low.layout
    validate(Lp, Lp.unpack_indices(low.indices), SupergroupGraph(low.connectivity, low.sg_feat, low.sg_buckets))
    # same logical graph and parameters in both representations
    import sys
    sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/benchmarks")
    import kcsr_graph

    assert kcsr_graph.fingerprint(kcsr_graph.native_table(low, logical_C=C)) == kcsr_graph.fingerprint(kg.kcsr_table())
    assert kcsr_graph.params_sha(low) == kcsr_graph.params_sha(kg)
    # round trip
    i2, w2, b2, bk2 = LW.unlower(low)
    assert torch.equal(i2, idx) and torch.equal(w2, w) and torch.equal(b2, b) and torch.equal(bk2, bk)
    # same function: lowered native vs kgen vs float64 reference
    x, dy = inputs(B, D, G)
    yl, dxl, _, _ = run_native(low, x, dy)
    _, dwl_p, dbl_p = Lp.to_canonical(low.indices, low.weight.main_grad, low.bias.main_grad)
    yk, dxk, dwk, dbk = run(kg, x, dy)
    yr, dxr, dwr, dbr = reference(kg, x, dy)
    tol = 2e-5 if prec == "split" else 2e-2
    for y, dx in ((yl, dxl), (yk, dxk)):
        assert maxrel(y, yr) < 1e-6 and maxrel(dx, dxr) < tol
    # physical main_grad maps one to one onto the logical parameters
    f = C // 64
    dwl = dwl_p[:, :f * L.blocks_per_group].reshape(w.shape)
    dbl = dbl_p[:, :f * L.blocks_per_group].reshape(b.shape)
    assert maxrel(dwl, dwr) < tol and maxrel(dbl, dbr) < 1e-6 and maxrel(dwk, dwr) < tol
    # checkpoint round trip of the lowered layer
    sd = {k: v.clone() for k, v in low.state_dict().items()}
    low2 = LW.lower(L, idx, torch.zeros_like(w), torch.zeros_like(b), bk, feat, spec, device="cuda")
    low2.load_state_dict(sd)
    assert all(torch.equal(v, low2.state_dict()[k]) for k, v in sd.items())
    assert rel_err(low2(x.bfloat16()), low(x.bfloat16())) < 1e-6  # fast mode: FP32-atomic order varies


def test_lowering_partial_logical_tail():
    """Outputs whose last logical C = 256 block holds <= 64, 65..128 and > 192 valid neurons."""
    KG, KL, LW, _ = _kg()
    D, G, S, R = 256, 4, 32, 2
    for H in (4 * 256 * 5 + 4 * 30, 4 * 256 * 5 + 4 * 100, 4 * 256 * 5 + 4 * 250, 4 * 256 * 5 + 3):
        plan = ConnectivityPlan(S, R, seed=2)
        L, idx, w, b, bk, feat, spec, _ = LW.build_logical(D, H, G, 256, plan, "cuda", seed=7, bias_std=0.3)
        low = LW.lower(L, idx, w, b, bk, feat, spec, device="cuda")
        x, dy = inputs(32, D, G)
        yl, dxl, _, _ = run_native(low, x, dy)
        kg = KL.KgenLayer(D, H, G, 8, 256, plan, device="cuda", graph=(idx, feat, bk, spec), weights=(w, b))
        yr, dxr, _, _ = reference(kg, x, dy)
        assert maxrel(yl, yr) < 1e-6 and maxrel(dxl, dxr) < 2e-5


# ------------------------------------------------------------------- optimizer / training state
def test_main_grad_accumulation_and_fused_adamw():
    lay = kgen_layer(256, 9001, 8, 16, 24, 64, 2)
    lay.enable_main_grad()
    xs = [inputs(64, 256, 8, seed=s) for s in (1, 2, 3)]
    for x, dy in xs:
        (lay(x) * dy).sum().backward()
    acc_w, acc_b = lay.weight.main_grad.clone(), lay.bias.main_grad.clone()
    ref_w = sum(reference(lay, x, dy)[2] for x, dy in xs)
    assert maxrel(acc_w, ref_w) < 2e-5
    w0 = lay.weight.detach().clone()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-2, moments="fp32", master_weights=True)
    opt.step()
    st = opt.state[lay.weight]
    assert st["step"] == 1 and torch.equal(lay.weight.main_grad, torch.zeros_like(acc_w))
    assert torch.equal(lay.bias.main_grad, torch.zeros_like(acc_b))
    m_ref = 0.1 * acc_w
    assert maxrel(st["m"], m_ref) < 1e-6
    inv = ~lay.layout.valid_mask("cuda")
    assert bool((lay.weight[inv] == 0).all()) and not torch.equal(lay.weight.detach(), w0)
    # second step: moments / counters advance; gradients were cleared
    (lay(xs[0][0]) * xs[0][1]).sum().backward()
    opt.step()
    assert opt.state[lay.weight]["step"] == 2


def test_checkpoint_round_trip():
    _, KL, _, _ = _kg()
    a = kgen_layer(256, 9001, 8, 32, 32, 64, 2)
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    b = kgen_layer(256, 9001, 8, 32, 32, 64, 2, init_seed=99)
    b.load_state_dict(sd)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in sd.items())
    x, _ = inputs(64, 256, 8)
    assert rel_err(a(x), b(x)) < 1e-6  # fast mode: FP32-atomic summation order varies run to run
    c = kgen_layer(256, 9001, 8, 32, 32, 64, 2, seed=7)  # different graph spec seed -> refused
    with pytest.raises(ValueError, match="graph spec"):
        c.load_state_dict(sd)


@pytest.mark.parametrize("K,C,S,R", [(16, 32, 64, 4), (64, 8, 64, 2), (8, 256, 32, 2)])
def test_composed_model_multi_step_training(K, C, S, R):
    """x0 -> Linear(D0, D) -> kgen layer -> Linear(G, G2) -> MSE: gradients reach the upstream projection; the
    first step matches float64 autograd through the reference layer; the loss decreases over 30 FusedAdamW steps."""
    _, _, _, REF = _kg()
    torch.manual_seed(0)
    D0, D, H, G, G2, B = 64, 256, 8192, 16, 8, 256
    lay = kgen_layer(D, H, G, K, C, S, R, bias_std=0.2)
    lay.enable_main_grad()
    proj = torch.nn.Linear(D0, D, device="cuda")
    head = torch.nn.Linear(G, G2, device="cuda")
    x0 = torch.randn(B, D0, device="cuda")
    tgt = 0.5 * torch.randn(B, G2, device="cuda")
    # step-1 gradients vs float64 (projection output rounded to BF16 as the layer consumes it)
    h = proj(x0)
    hb = h.bfloat16().float()
    hb.retain_grad()
    loss = ((torch.tanh(head(lay(hb))) - tgt) ** 2).mean()
    loss.backward()
    gproj = proj.weight.grad.clone()
    hd = hb.detach().double().requires_grad_(True)
    y64 = REF.forward(hd, lay.indices.long(), lay.weight.detach().double(), lay.bias.detach().double(), lay.counts)
    l64 = ((torch.tanh(y64 @ head.weight.double().t() + head.bias.double()) - tgt.double()) ** 2).mean()
    gh, = torch.autograd.grad(l64, [hd])
    assert maxrel(hb.grad, gh) < 3e-5
    assert float(gproj.abs().max()) > 0
    lay.zero_main_grad()
    proj.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    opt = FusedAdamW([lay.weight, lay.bias], lr=3e-3)
    opt2 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=3e-3)
    losses = []
    for _ in range(30):
        loss = ((torch.tanh(head(lay(proj(x0).bfloat16()).float())) - tgt) ** 2).mean()
        loss.backward()
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()
        losses.append(float(loss))
    assert all(math.isfinite(v) for v in losses) and losses[-1] < 0.9 * losses[0]
