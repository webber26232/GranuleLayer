"""K16 dX tiling variants of the kgen bundle (experimental/kgen_dx_variants.py, kernels/.../kgen_k16_dxv.cu;
results/INTERACTION_COVERAGE_K16_DX.md section 5.3).

Gates: every screened tiling equals the FP64 reference and the legacy launch (forward unchanged; dX within the
existing tolerance; dW / db bitwise unchanged since only the dX launch differs) on K16 / C16 with full and partial
groups, B 8 / 24 / 40 / 64 / 512 (token tails), zero inputs / weights, near-ReLU-boundary pre-activations; launch
legality and capability-based rejection; resolution / cache keys / generic default; optimizer updates, moments,
repeated steps and canonical restore with SR off, SR-on training separately; grouped Mix32 changes only its K16
component with a single dX cast.
"""
import dataclasses
import os
import sys

import pytest
import torch

from conftest import requires_cuda_backend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, os.path.dirname(__file__))

SCREEN_NAMES = ["mt4_nw8_wg2", "mt4_nw8_wg4", "mt4_nw8_wg8", "mt2_nw8_wg1", "mt2_nw8_wg2", "mt4_nw4_wg1", "mt4_nw4_wg2",
                "mt2_nw4_wg1"]


def _case(H=8 * 431 + 4, D=256, policy="Cover-TopK", bias_scale=None, zero_w=False):
    from test_weighted_mixed_supports import _layers, base_layer, masked
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental.kgx_onetile import lower_local

    Cn, GX, KX, L1, REF = _layers()
    base = base_layer(H, D=D)
    if bias_scale is not None:
        with torch.no_grad():
            base.bias.mul_(bias_scale)
    if zero_w:
        with torch.no_grad():
            base.weight.zero_()
    L = base.layout
    w_sel = base.weight.detach() if not zero_w else torch.randn(L.weight_shape, device="cuda").bfloat16()
    s = LS.build(policy, 16, L, base.kg_buckets, w=w_sel, Q=base.sg_feat.shape[0], device="cuda")
    em = s.element_mask(L)
    dm = masked(base, em)
    lo = lower_local(dm, s)
    return Cn, REF, dm, em, lo, s


def _run(lo, x, dy, variant):
    lo.dx_variant = variant
    lo.enable_main_grad()
    lo.zero_main_grad()
    xx = x.clone().requires_grad_(True)
    y = lo(xx)
    y.backward(dy)
    return y.detach(), xx.grad.detach(), lo.weight.main_grad.clone(), lo.bias.main_grad.clone()


def maxrel(a, r):
    a, r = a.double(), r.double()
    return float((a - r).abs().max() / (r.abs().max() + 1e-300))


@requires_cuda_backend
def test_all_screened_tilings_equal_reference_and_legacy():
    Cn, REF, dm, em, lo, s = _case()
    L = dm.layout
    gg, nn = Cn.neurons(dm.counts, 0, dm.counts.numel())
    sl = Cn.dense_slots(dm, gg, nn)
    for B in (8, 24, 40, 64, 512):
        g = torch.Generator(device="cuda").manual_seed(B)
        x = torch.randn(B, L.input_dim, device="cuda", generator=g).bfloat16().float()
        dy = torch.randn(B, L.groups, device="cuda", generator=g)
        yr, dxr, dwr, dbr = REF.grads(x.double(), dm.logical_indices().long(), dm.weight.detach().double(),
                                      dm.bias.detach().double(), dm.counts, dy.double())
        y0, dx0, dw0, db0 = _run(lo, x, dy, "legacy")
        assert maxrel(dx0, dxr) < 2e-5
        for v in SCREEN_NAMES:
            y, dx, dw, db = _run(lo, x, dy, v)
            assert torch.equal(y, y0), v  # forward untouched
            assert torch.equal(dw, dw0) and torch.equal(db, db0), v  # dW / db launch untouched
            assert maxrel(dx, dxr) < 2e-5, (v, B, maxrel(dx, dxr))
            assert maxrel(dx, dx0) < 2e-5, (v, B)
            info = lo.last_info["kgen_dx"]
            assert info["variant"]["name"] == v and info["tokens"] == 16 * int(v[2])


@requires_cuda_backend
def test_zero_inputs_weights_and_relu_boundary():
    Cn, REF, dm, em, lo, s = _case(zero_w=True)
    L = dm.layout
    x = torch.zeros(40, L.input_dim, device="cuda")
    dy = torch.randn(40, L.groups, device="cuda")
    for v in ["legacy"] + SCREEN_NAMES:
        _, dx, dw, db = _run(lo, x, dy, v)
        assert int((dx != 0).sum()) == 0 or float(dx.abs().max()) == 0.0
    # near-boundary pre-activations: differences only through flipped near-zero decisions
    Cn, REF, dm, em, lo, s = _case(bias_scale=0.01)
    g = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(64, L.input_dim, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(64, L.groups, device="cuda", generator=g)
    yr, dxr, _, _ = REF.grads(x.double(), dm.logical_indices().long(), dm.weight.detach().double(),
                              dm.bias.detach().double(), dm.counts, dy.double())
    for v in SCREEN_NAMES:
        _, dx, _, _ = _run(lo, x, dy, v)
        assert maxrel(dx, dxr) < 5e-5, v


@requires_cuda_backend
def test_launch_legality_and_capability_rejection():
    from sparse_pooling.experimental import kgen_dx_variants as DXV

    Cn, REF, dm, em, lo, s = _case(D=1024)
    rt = lo._runtime()
    assert DXV.smem_bytes(1024, 64, 4, 8, 2) == 75776 and DXV.smem_bytes(1024, 64, 4, 8, 4) == 43008
    assert [DXV.smem_bytes(1024, 64, m, n, g) // 1024 for m, n, g in DXV.SCREEN] == [74, 42, 26, 70, 38, 74, 42, 38]
    with pytest.raises(ValueError, match="shared memory"):
        lo.dx_variant = "mt4_nw8_wg1"  # 138 KiB > 99 KiB opt-in
        lo.dx_variant_resolved()
    for bad in ("mt8_nw8_wg2", "mt4_nw8_wg3", "nonsense"):
        lo.dx_variant = bad
        with pytest.raises(ValueError):
            lo.dx_variant_resolved()
    small = dataclasses.replace(rt.caps, smem_per_block_optin=48 * 1024) if dataclasses.is_dataclass(rt.caps) else None
    if small is not None:
        assert "shared memory" in DXV.reason(small, rt.backend, 1024, 64, 16, 4, 8, 2)
        # auto on a non-3090 capability record -> the generic legacy launch
        v = DXV.resolve("auto", dataclasses.replace(rt.caps, name="Some Other GPU"), rt.backend, 1024, 64, 16, 16)
        assert v.legacy
    lo.dx_variant = "legacy"
    assert lo.dx_variant_resolved().legacy and lo.dx_variant_resolved().wg == rt.dx_group_warps == 2
    # every screened tiling is compiled and legal at D 1024 / S 64 on this device
    for m, n, g in DXV.SCREEN:
        assert DXV.reason(rt.caps, rt.backend, 1024, 64, 16, m, n, g) == ""
    # compiled attributes are readable
    at = rt.backend.dxv_attributes(4, 8, 0, 1, 1, DXV.smem_bytes(1024, 64, 4, 8, 4))
    assert at["registers_per_thread"] > 0 and at["active_ctas_per_sm"] >= 1


@requires_cuda_backend
def test_optimizer_moments_repeated_steps_and_restore_sr_off():
    from test_weighted_mixed_supports import _layers
    from train_state import TrainState

    Cn, GX, KX, L1, REF = _layers()
    results = {}
    for v in ("legacy", "mt4_nw8_wg4", "mt2_nw4_wg1"):
        Cn, REF, dm, em, lo, s = _case()
        lo.dx_variant = v
        lo.enable_main_grad()
        opt = L1.L1AdamW([lo.weight, lo.bias], lr=1e-3, stochastic_rounding=False)
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(64, dm.layout.input_dim, device="cuda", generator=g).bfloat16()
        dy = torch.randn(64, dm.layout.groups, device="cuda", generator=g)
        for _ in range(3):
            xx = x.clone().requires_grad_(True)
            lo(xx).backward(dy)
            opt.step()
        results[v] = (lo.weight.detach().clone(), opt.state[lo.weight]["m"].clone(), opt.state[lo.weight]["v"].clone())
        ident = Cn.identity(lo, opt)
        st = TrainState([lo.weight, lo.bias], opt).capture()
        fp0 = st.fingerprint()
        for _ in range(2):
            lo(x.clone().requires_grad_(True)).backward(dy)
            opt.step()
        st.restore()
        assert st.fingerprint() == fp0 and Cn.identity(lo, opt) == ident
    # dW / db launches are identical and x is not trained: the parameter / moment trajectories are bitwise equal
    for v in ("mt4_nw8_wg4", "mt2_nw4_wg1"):
        for a, b in zip(results["legacy"], results[v]):
            assert torch.equal(a, b), v


@requires_cuda_backend
def test_sr_on_training_runs_and_stays_finite():
    from test_weighted_mixed_supports import _layers

    Cn, GX, KX, L1, REF = _layers()
    Cn, REF, dm, em, lo, s = _case()
    lo.dx_variant = "mt4_nw8_wg4"
    lo.enable_main_grad()
    proj = torch.nn.Linear(32, dm.layout.input_dim, device="cuda")
    opt = L1.L1AdamW([lo.weight, lo.bias], lr=1e-3)
    opt2 = torch.optim.AdamW(proj.parameters(), lr=1e-3)
    live = lo.weight.detach() != 0
    g = torch.Generator(device="cuda").manual_seed(1)
    losses = []
    for i in range(5):
        x0 = torch.randn(64, 32, device="cuda", generator=g)
        y = lo(proj(x0).bfloat16())
        loss = (y.float() ** 2).mean()
        loss.backward()
        opt2.step()
        opt2.zero_grad()
        opt.step()
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(losses))) and proj.weight.grad is None
    assert int((lo.weight.detach()[~live] != 0).sum()) == 0  # nothing leaves the support


@requires_cuda_backend
def test_grouped_mix32_changes_only_k16_component():
    from test_weighted_mixed_supports import _layers, base_layer, masked
    from sparse_pooling.experimental import mix32_supports as M32

    Cn, GX, KX, L1, REF = _layers()
    base = base_layer(8 * 431 + 4)
    L = base.layout
    s = M32.build("Cover-Mix32", 1, L, base.kg_buckets, w=base.weight.detach(), graph_seed=1, device="cuda",
                  Q=base.sg_feat.shape[0])
    em = s.element_mask(L)
    gl = GX.lower_mixed(masked(base, em), s)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(40, L.input_dim, device="cuda", generator=g).bfloat16()
    dy = torch.randn(40, L.groups, device="cuda", generator=g)
    outs = {}
    for v in ("legacy", "mt4_nw8_wg4", "mt2_nw8_wg1"):
        gl.dx_variant = v
        gl.enable_main_grad()
        gl.zero_main_grad()
        xx = x.float().requires_grad_(True)
        y = gl(xx)
        y.backward(dy)
        b16 = x.clone().requires_grad_(True)
        gl.zero_main_grad()
        gl(b16).backward(dy)
        assert torch.equal(b16.grad, xx.grad.to(torch.bfloat16))  # one final cast of the shared FP32 dX
        dv = gl.last_info["dx_variants"]
        assert dv["K32"]["legacy_entry"] and dv["K8"]["legacy_entry"]
        assert dv["K16"]["name"] == v
        outs[v] = (y.detach(), xx.grad.detach(), gl.weight.main_grad.clone())
    for v in ("mt4_nw8_wg4", "mt2_nw8_wg1"):
        # forward: three types add into one FP32 ysum with atomics (order not deterministic); dW: exclusive, unchanged
        assert maxrel(outs[v][0], outs["legacy"][0]) < 1e-6 and torch.equal(outs[v][2], outs["legacy"][2])
        assert maxrel(outs[v][1], outs["legacy"][1]) < 2e-5


@requires_cuda_backend
def test_interleaved_legacy_and_variant_launches_keep_their_shared_memory():
    """Regression: the (MT 4, NW 8) variant is the legacy kernel symbol; interleaving the legacy launch and the variant
    launch at different D (different dynamic shared memory) in one process must never lower the other's attribute."""
    from sparse_pooling.experimental import kgen_dx_variants as DXV

    small = _case(D=256)
    big = _case(D=1024, H=4 * 431 + 4)
    rt = big[4]._runtime()
    rt.backend.dxv_attributes(4, 8, 0, 1, 1, DXV.smem_bytes(1024, 64, 4, 8, 8))  # a small request: must not lower
    seq = [(big, "mt4_nw8_wg8"), (small, "legacy"), (big, "mt4_nw8_wg2"), (big, "legacy"), (small, "mt4_nw8_wg4"),
           (big, "legacy"), (small, "legacy"), (big, "mt4_nw8_wg4")]
    for (Cn, REF, dm, em, lo, s), v in seq:
        L = dm.layout
        x = torch.randn(40, L.input_dim, device="cuda").bfloat16().float()
        dy = torch.randn(40, L.groups, device="cuda")
        _, dx, _, _ = _run(lo, x, dy, v)
        _, dxr, _, _ = REF.grads(x.double(), dm.logical_indices().long(), dm.weight.detach().double(),
                                 dm.bias.detach().double(), dm.counts, dy.double())
        assert maxrel(dx, dxr) < 2e-5, v
