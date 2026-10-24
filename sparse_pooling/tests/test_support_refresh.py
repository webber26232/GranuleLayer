"""Budget-exact support refresh (experimental/support_refresh.py; results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md
section 2): first fit against a scalar loop, 1,000-refresh sequences on small geometries with partial row groups,
anchors and backbone overlap costs (enabled <= B0 and shortfall < one legal tile after EVERY refresh, mandatory
support intact, overlap never double-counted), the legacy freed-only rule drifting below B0 on the same
sequences, incompatible replacement costs (reduced prune, explicit infeasible record), and the state remap of
the learning driver (surviving values / moments unchanged, grown slots at zero)."""
import os
import sys

import pytest
import torch

from conftest import requires_cuda_backend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))


def _m():
    from sparse_pooling.experimental import support_refresh as SR
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_

    return SR, WM, KLayout, zero_padding_


def setup(policy, rho, H=3003, C=64, seed=0):
    SR, WM, KLayout, zp = _m()
    L = KLayout(256, H, 8, 64, C)
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(L.weight_shape, generator=g).bfloat16()
    zp(L, w)
    return L, WM.build(policy, rho, w, L), g


def test_first_fit_matches_scalar_loop():
    SR = _m()[0]
    g = torch.Generator().manual_seed(1)
    for trial in range(300):
        n = int(torch.randint(0, 40, (1,), generator=g))
        costs = torch.randint(1, 20, (n,), generator=g) * torch.randint(1, 4, (n,), generator=g)
        rem = int(torch.randint(0, 200, (1,), generator=g))
        take = SR.first_fit(costs, rem).tolist()
        left, want = rem, []
        for c in costs.tolist():
            want.append(c <= left)
            if c <= left:
                left -= c
        assert take == want


@pytest.mark.parametrize("policy,rho", [("U16", 0.3), ("P16", 0.4), ("P8", 0.3), ("A8", 0.4), ("A16", 0.5),
                                        ("H2", 0.1), ("H2A", 0.3)])
def test_thousand_refreshes_keep_the_budget(policy, rho):
    SR = _m()[0]
    L, ms, g = setup(policy, rho)
    B0 = ms.stats["budget_slots"]
    mand0, bb0 = ms.mandatory.clone(), None if ms.bb is None else ms.bb.clone()
    cmax = SR.max_tile_cost(ms)
    assert ms.stats["partial_groups"] > 0
    nv = ms.nv
    for i in range(1000):
        ps, gs = torch.rand(ms.tiles.shape, generator=g), torch.rand(ms.tiles.shape, generator=g)
        r = SR.refresh(ms, ps, gs, B0, 0.1)
        e = ms.enabled_slots()
        assert r["B0"] == B0 and r["enabled_after"] == e and e <= B0 and B0 - e < cmax and not r["infeasible"]
        assert bool((ms.tiles | ~mand0).all()), "mandatory tile lost"
        assert bool((ms.tiles[nv == 0] == 0).all()), "padded group got a tile"
        if i % 97 == 0:  # the element mask counts every enabled valid slot exactly once (backbone overlap)
            assert int(ms.element_mask(L).sum()) == e
    assert torch.equal(ms.mandatory, mand0)
    if bb0 is not None:
        assert torch.equal(ms.bb, bb0)


@pytest.mark.parametrize("policy,rho", [("U16", 0.3), ("P8", 0.3), ("H2A", 0.3)])
def test_legacy_rule_accumulates_a_deficit(policy, rho):
    """The parent's freed-only rule loses every unfilled remainder: on the same score sequence its deficit grows
    past one legal tile (the bug), while the budget-exact rule stays within one tile."""
    SR = _m()[0]
    L, ms, g = setup(policy, rho)
    B0 = ms.stats["budget_slots"]
    for _ in range(200):
        SR.refresh_freed_only(ms, torch.rand(ms.tiles.shape, generator=g), torch.rand(ms.tiles.shape, generator=g), 0.1)
    assert B0 - ms.enabled_slots() > SR.max_tile_cost(ms)


def test_carry_is_recovered():
    """A remainder left by one refresh is regrown by the next one (remaining = B0 - enabled, not the freed slots)."""
    SR = _m()[0]
    L, ms, g = setup("U16", 0.3)
    B0 = ms.stats["budget_slots"]
    # take away two extra tiles by hand: a deficit the next refresh must recover
    extra = torch.nonzero((ms.tiles & ~ms.mandatory).reshape(-1)).squeeze(1)[:2]
    ms.tiles.view(-1)[extra] = False
    before = ms.enabled_slots()
    assert B0 - before >= SR.max_tile_cost(ms)
    r = SR.refresh(ms, torch.rand(ms.tiles.shape, generator=g), torch.rand(ms.tiles.shape, generator=g), B0, 0.1)
    assert r["carry_in"] == B0 - before and r["shortfall"] < SR.max_tile_cost(ms) and r["carry_recovered"] > 0


def test_incompatible_costs_reduce_the_prune_or_record_infeasible():
    SR, WM, KLayout, zp = _m()
    L, ms, g = setup("U16", 1.0)  # every tile enabled: the only candidates are the just-pruned tiles
    B0 = ms.stats["budget_slots"]
    tiles0 = ms.tiles.clone()
    r = SR.refresh(ms, torch.rand(ms.tiles.shape, generator=g), torch.rand(ms.tiles.shape, generator=g), B0, 0.1,
                   exclude_pruned=True)
    assert r["prune_requested"] > 0 and r["prune_rolled_back"] == r["prune_requested"] and r["pruned"] == 0
    assert r["bound_ok"] and torch.equal(ms.tiles, tiles0) and ms.enabled_slots() == B0
    # an unreachable target (more than every tile) is recorded, never silently lowered
    r = SR.refresh(ms, torch.rand(ms.tiles.shape, generator=g), torch.rand(ms.tiles.shape, generator=g),
                   B0 + 10 * SR.max_tile_cost(ms), 0.1)
    assert r["infeasible"] and r["B0"] == B0 + 10 * SR.max_tile_cost(ms) and r["within_budget"]
    # partial-group candidates cost less than the pruned full tile: grown by first fit, shortfall within a tile
    L, ms, g = setup("U16", 0.3, H=3003 - 8 * 7)
    B0 = ms.stats["budget_slots"]
    for _ in range(50):
        r = SR.refresh(ms, torch.rand(ms.tiles.shape, generator=g), torch.rand(ms.tiles.shape, generator=g), B0, 0.2)
        assert r["shortfall"] < SR.max_tile_cost(ms)


@requires_cuda_backend
def test_state_remap_preserves_survivors_and_regrows_at_zero():
    import wps_refresh as WR
    from sparse_pooling.experimental import kgx_sparse as SPX
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan

    SR, WM, _, _ = _m()
    lay = KgxLayer(256, 3003, 8, 64, ConnectivityPlan(64, 2, seed=1), variant="idf", tile=128, device="cuda", seed=3)
    lay.enable_main_grad()
    ms = WM.build("A8", 0.4, lay.weight.detach(), lay.layout)
    mask = ms.element_mask(lay.layout)
    opt = L1.L1AdamW([lay.weight, lay.bias], lr=1e-3)
    L1.apply_mask_(opt, lay.weight, mask)
    valid = SPX.valid_slots(lay)
    g = torch.Generator(device="cuda").manual_seed(0)
    for step in range(6):
        x = torch.randn(64, 256, device="cuda", generator=g).bfloat16()
        (lay(x).float() ** 2).mean().backward()
        gr = lay.weight.main_grad.clone()
        opt.step()
        w_before, st = lay.weight.detach().clone(), opt.state[lay.weight]
        m_before, v_before = st["m"].clone(), st["v"].clone()
        r = SR.public(SR.refresh(ms, WR.tile_l1(lay.weight.detach(), ms, valid), WR.tile_l1(gr, ms, valid),
                                 ms.stats["budget_slots"], 0.2))
        mask_old = mask
        mask = WR.remap_state(opt, lay, ms, mask_old, r, valid)
        rm = r["state_remap"]
        assert rm["surviving_preserved"] and rm["grown_zero"] and rm["nonzero_outside_mask"] == 0
        keep = (mask_old != 0) & (mask != 0)
        assert torch.equal(lay.weight.detach()[keep.bool()], w_before[keep.bool()])
        assert torch.equal(st["m"][keep.bool()], m_before[keep.bool()]) and torch.equal(st["v"][keep.bool()],
                                                                                       v_before[keep.bool()])
        assert r["shortfall"] < SR.max_tile_cost(ms) and r["grown"] > 0
