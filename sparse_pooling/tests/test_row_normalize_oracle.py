"""Row normalization (experimental/row_normalize.py) and the teacher-mapped oracle (benchmarks/rsn_oracle.py) of the
random-support study (results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md sections 5-6), plus the indexed K16
path's rejection / non-finite behaviour.

Normalization: tiny rows against hand-computed scales (fan-in sqrt(64 / k), energy sqrt(E_dense / E_kept) with the
cap and the relative epsilon), zero / disconnected rows stay zero, all-zero retained rows clipped without new
edges, the support (nonzero pattern) unchanged, a hybrid backbone + tile row scaled once, W-and-bias scaling
keeps every ReLU decision.  Oracle: the FP64 algebra (mean pooling denominator, analytic readout) reproduces the
targets exactly for T = 1 / 16 including duplicated terms; on the GPU the error against BF16-rounded-input
targets is at FP32 rounding while FP32-input targets show the input quantization floor.
"""
import math
import os
import sys

import pytest
import torch

from conftest import requires_cuda_backend
from sparse_pooling.supergroups import ConnectivityPlan

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))


def RN():
    from sparse_pooling.experimental import row_normalize

    return row_normalize


def test_scales_on_tiny_rows():
    rn = RN()
    w = torch.zeros(5, 64)
    w[0] = 1.0  # E_dense 64
    w[1, :4] = 2.0  # E_dense 16
    w[2, :] = 0.5
    w[3, :] = 1.0
    w[4, :] = 3.0
    m = torch.zeros(5, 64, dtype=torch.uint8)
    m[0, :16] = 1  # k 16, E_kept 16 -> energy s = 2, fanin s = 2
    m[1, :2] = 1  # k 2, E_kept 8 -> energy sqrt(2), fanin sqrt(32)
    m[2, 63] = 1  # k 1, E_kept 0.25 vs 16 -> sqrt(64) = 8 = cap (clipped), fanin 8
    m[3, :] = 0  # disconnected -> 0
    m[4, 10:12] = 1
    valid = torch.ones(5, dtype=torch.bool)
    valid[4] = False  # padded row: never scaled
    s, k, ed, ek = rn.row_scales(w.bfloat16(), m, valid, "energy")
    assert torch.allclose(s, torch.tensor([2.0, math.sqrt(2.0), 8.0, 0.0, 0.0]))
    s, *_ = rn.row_scales(w.bfloat16(), m, valid, "fanin")
    assert torch.allclose(s, torch.tensor([2.0, math.sqrt(32.0), 8.0, 0.0, 0.0]))
    s, *_ = rn.row_scales(w.bfloat16(), m, valid, "none")
    assert torch.equal(s, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    # cap and relative epsilon: a connected row with all retained weights zero gets s_max, stays all-zero
    w2 = w.clone()
    w2[0, :16] = 0.0
    s, *_ = rn.row_scales(w2.bfloat16(), m, valid, "energy", s_max=5.0)
    assert float(s[0]) == 5.0


def test_apply_keeps_support_and_scales_each_weight_once():
    rn = RN()
    g = torch.Generator().manual_seed(0)
    w = torch.randn(2, 3, 16, 64, generator=g).bfloat16()
    mask = (torch.rand(2, 3, 16, 64, generator=g) < 0.3).to(torch.uint8)
    mask[0, 0, 0] = 0  # disconnected row
    w[0, 0, 1][mask[0, 0, 1].bool()] = 0  # connected, all retained weights zero
    valid = torch.ones(2, 3, 16, dtype=torch.bool)
    valid[1, 2, 10:] = False
    mask[~valid] = 0
    b = torch.randn(2, 3, 16, generator=g).bfloat16()
    for tr in ("none", "fanin", "energy"):
        wn, bn, rec = rn.apply(w, b, mask, valid, tr)
        kept = (w * mask) != 0
        assert torch.equal(wn != 0, kept), tr  # support unchanged (no new edges, nothing flushed)
        s, *_ = rn.row_scales(w, mask, valid, tr)
        want = (w.float() * mask * s.view(2, 3, 16, 1)).bfloat16()
        assert torch.equal(wn, want) and torch.equal(bn, b)
        assert int((wn[0, 0, 0] != 0).sum()) == 0
        if tr == "energy":
            assert rec["connected_rows_all_zero"] == 1 and rec["clipped"] >= 1
            ea = rec["energy_ratio_after"]
            assert ea["median"] == pytest.approx(1.0, abs=0.02)
    wn, bn, rec = rn.apply(w, b, mask, valid, "energy", scale_bias=True)
    s, *_ = rn.row_scales(w, mask, valid, "energy")
    assert torch.equal(bn, (b.float() * s.view(2, 3, 16)).bfloat16())


def test_joint_positive_scaling_keeps_relu_decisions():
    rn = RN()
    g = torch.Generator().manual_seed(1)
    w = torch.randn(1, 1, 16, 64, generator=g).bfloat16()
    mask = (torch.rand(1, 1, 16, 64, generator=g) < 0.25).to(torch.uint8)
    valid = torch.ones(1, 1, 16, dtype=torch.bool)
    b = (0.3 * torch.randn(1, 1, 16, generator=g)).bfloat16()
    x = torch.randn(200, 64, generator=g).double()
    wn, bn, _ = rn.apply(w, b, mask, valid, "energy", scale_bias=True)
    z0 = x @ (w.double() * mask)[0, 0].t() + b.double()[0, 0]
    z1 = x @ wn.double()[0, 0].t() + bn.double()[0, 0]
    assert torch.equal(z0 > 0, z1 > 0) or float(((z0 > 0) != (z1 > 0)).double().mean()) < 0.01  # BF16 rounding


@requires_cuda_backend
def test_hybrid_backbone_row_scaled_once():
    """H2: the dense masked row holds backbone and tile coordinates once (overlap counted once); the scale applies
    to every retained coordinate exactly once and the energy of the scaled row matches E_dense up to rounding."""
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgx_layer import KgxLayer

    rn = RN()
    lay = KgxLayer(256, 3003, 8, 64, ConnectivityPlan(64, 2, seed=1), variant="idf", tile=128, device="cuda", seed=3)
    ms = WM.build("H2", 0.15, lay.weight.detach(), lay.layout)
    assert ms.stats["backbone_tile_overlaps"] > 0
    em = ms.element_mask(lay.layout)
    valid = lay.layout.valid_mask("cuda")
    wn, bn, rec = rn.apply(lay.weight.detach(), lay.bias.detach(), em, valid, "energy")
    assert torch.equal(wn != 0, (lay.weight.detach() * em) != 0)
    assert int((em.reshape(-1, 64).sum(1) * valid.reshape(-1)).sum()) == ms.enabled_slots()
    s, k, ed, ek = rn.row_scales(lay.weight.detach(), em, valid, "energy")
    ratio = (wn.float() ** 2).reshape(-1, 64).sum(1)[valid.reshape(-1)] / ed[valid.reshape(-1)]
    unclipped = (s[valid.reshape(-1)] < 8.0)
    assert float((ratio[unclipped] - 1).abs().max()) < 0.03


# ------------------------------------------------------------------------------------------ oracle
def _fp64_oracle_check(T, dup=False):
    """Pure FP64 algebra on a tiny 'record': 256 neurons, mean pooling, analytic readout."""
    import wps_interaction as WI

    g = torch.Generator().manual_seed(T)
    n_g, k = 256, 3
    mu, sd = 0.37, 0.61  # any constants: the construction is exact for the constants it uses
    x = torch.randn(1000, 64, generator=g, dtype=torch.float64)
    pos = torch.stack([torch.randperm(64, generator=g)[:k] for _ in range(T)])
    if dup and T > 1:
        pos[1] = pos[0]
    tgt = sum((torch.relu(x[:, pos[t]].sum(1) - 1.0) - mu) for t in range(T)) / (math.sqrt(T) * sd)
    W = torch.zeros(n_g, 64, dtype=torch.float64)
    bias = torch.zeros(n_g, dtype=torch.float64)
    for t in range(T):
        W[t, pos[t]] = 1.0
        bias[t] = -1.0
    y = torch.relu(x @ W.t() + bias).mean(1)  # mean over the n_g valid neurons
    yhat = n_g / (math.sqrt(T) * sd) * y - math.sqrt(T) * mu / sd
    return float((yhat - tgt).abs().max())


@pytest.mark.parametrize("T,dup", [(1, False), (16, False), (16, True)])
def test_oracle_algebra_fp64(T, dup):
    assert _fp64_oracle_check(T, dup) < 1e-12


@requires_cuda_backend
@pytest.mark.parametrize("terms", [1, 16])
def test_gpu_oracle_reproduces_the_targets(terms):
    """Reduced geometry (D = G = 1024? no: D 1024, G 64, H = 64 x 256 -> one record per output): the anchor oracle
    of an A8 support and of the dense model; error vs BF16-rounded-input targets at FP32 rounding, vs FP32 targets
    at the input quantization floor; order-1 outputs not applicable; the mask is respected."""
    import rsn_oracle as RO
    import wps_interaction as WI
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgx_layer import KgxLayer

    G, C = 64, 256
    lay = KgxLayer(1024, G * C, G, C, ConnectivityPlan(64, 256, seed=0), variant="idf", tile=128, device="cuda",
                   seed=17, keep_indices=False)
    ids, rv, order, loc = WI.target_inputs(lay, terms)
    sig = WI.sigmas("cuda")
    ms = WM.build("A8", 0.25, lay.weight.detach(), lay.layout)
    em = ms.element_mask(lay.layout)
    for mask, sel in ((em, ms), (None, None)):
        w, b, sc, off, st = RO.build(lay, mask, sel, loc, order, rv, sig, terms, "anchor")
        assert st["terms_unmatched"] == 0 and st["records"] == int((order > 1).sum())
        if mask is not None:
            assert not bool((w.bool() & ~mask.bool()).any())
        with torch.no_grad():
            lay.weight.copy_(w)
            lay.bias.copy_(b)
        gx = torch.Generator().manual_seed(5)
        x = torch.randn(512, 1024, generator=gx).cuda()
        pred = sc * lay(x.bfloat16()).float() + off
        for xin, tol in ((x.bfloat16().float(), 1e-4), (x, None)):
            t = WI.targets(xin, ids, rv, order, sig)
            sel_o = order > 1
            nm = float(((pred - t)[:, sel_o] ** 2).sum() / (t[:, sel_o] ** 2).sum())
            if tol is not None:
                assert nm < 1e-9
            else:
                assert 1e-9 < nm < 1e-3  # the BF16 input quantization floor
    m2 = WM.build("U16", 0.25, lay.weight.detach(), lay.layout)
    with pytest.raises(ValueError):
        RO.build(lay, m2.element_mask(lay.layout), m2, loc, order, rv, sig, terms, "anchor")
    w, b, sc, off, st = RO.build(lay, m2.element_mask(lay.layout), m2, loc, order, rv, sig, terms, "match")
    assert st["terms_matched"] + st["terms_unmatched"] == st["terms_total"]


# ------------------------------------------------------------------------------------------ indexed path
@requires_cuda_backend
def test_indexed_path_rejections_and_nonfinite():
    from sparse_pooling.config import PhysicalPlan
    from sparse_pooling.experimental import kgx_onetile as OT
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental.kgx_layer import KgxLayer

    lay = KgxLayer(256, 3003, 8, 64, ConnectivityPlan(64, 2, seed=1), variant="idf", tile=128, device="cuda", seed=3,
                   keep_indices=True)
    s = LS.build("RP-row", 16, lay.layout, lay.kg_buckets, layout_seed=0, device="cuda")
    with torch.no_grad():
        lay.weight.mul_(s.element_mask(lay.layout))
    lo = OT.lower_local(lay, s)
    with pytest.raises(ValueError, match="B % 8"):  # documented capability: kgen tokens in multiples of 8
        lo(torch.randn(33, 256, device="cuda").bfloat16())
    x = torch.randn(40, 256, device="cuda").bfloat16()
    y = lo(x)  # token tail (40: not a multiple of the 16-token tile)
    assert y.shape == (40, 8)
    det = OT.lower_local(lay, s, plan=PhysicalPlan(grad_precision="split", reduction="deterministic"))
    with pytest.raises(ValueError):
        det(x)
    x2 = x.clone()
    x2[3, 7] = float("inf")
    ref = lay(x2).float()
    got = lo(x2).float()
    assert torch.equal(torch.isnan(ref), torch.isnan(got)) and torch.equal(torch.isinf(ref), torch.isinf(got))
    bad = s.env.clone()
    bad[s.nv > 0][:, 1] = bad[s.nv > 0][:, 0]
    bad[torch.nonzero(s.nv > 0)[0, 0], 1] = bad[torch.nonzero(s.nv > 0)[0, 0], 0]
    with pytest.raises(ValueError):
        OT.lower_env(lay, bad)
