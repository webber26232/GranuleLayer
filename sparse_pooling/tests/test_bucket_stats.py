"""Canonical bucket / chunk statistics (supergroups.bucket_statistics, chunk_statistics): exact
definitions on hand-built tables, generated C = 8 / 64 graphs, the matched C64 -> C8 conversion,
reconstruction from the plan alone, the dispatch descriptor and its denominator."""
import math
from types import SimpleNamespace

import pytest
import torch

from conftest import requires_cuda_backend
from sparse_pooling import PhysicalPlan
from sparse_pooling.layout import SparseLayout, init_parameters
from sparse_pooling.supergroups import (ConnectivityPlan, SgExec, bucket_statistics, bucket_table, chunk_statistics,
                                        generate_indices, matched_c8_from_c64, nearest_rank_quantiles, unit_list,
                                        work_items)


def lay8(D=128, H=4000, G=4):
    return SparseLayout(D, H, G, 8, 8, True, packed=True)


def table(rows):
    return torch.tensor(rows, dtype=torch.int32).reshape(-1, 4)


def test_hand_built_definitions():
    L = lay8(G=4)
    # (q, g, first unit, units); Q = 3 supergroups, G = 4 outputs -> 12 possible buckets, 4 nonempty
    b = table([[0, 0, 0, 2], [0, 2, 0, 1], [1, 0, 2, 3], [2, 3, 0, 1]])
    st = bucket_statistics(L, b, 3)
    assert st["total_work_units"] == 7 and st["nonempty_buckets"] == 4
    assert st["total_possible_buckets"] == 12 and st["empty_buckets"] == 8
    assert st["bucket_occupancy"] == 4 / 12
    assert st["mean_units_per_nonempty_bucket"] == 7 / 4
    assert st["mean_units_per_possible_bucket"] == 7 / 12
    assert math.isclose(st["mean_units_per_possible_bucket"],
                        st["bucket_occupancy"] * st["mean_units_per_nonempty_bucket"], rel_tol=1e-15)
    assert (st["bucket_units_min"], st["bucket_units_max"]) == (1, 3)
    # nearest rank on [1, 1, 2, 3]: p 0, .1, .5, .9, .99, 1 -> ranks 1, 1, 2, 4, 4, 4
    assert st["bucket_units_quantiles"] == [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
    assert st["neurons_per_work_unit"] == L.unit_blocks * 8 == 32


def test_all_nonempty_and_zero_records():
    L = lay8(G=2)
    b = table([[0, 0, 0, 3], [0, 1, 0, 2], [1, 0, 3, 1], [1, 1, 2, 4]])
    st = bucket_statistics(L, b, 2)
    assert st["bucket_occupancy"] == 1.0
    assert st["mean_units_per_nonempty_bucket"] == st["mean_units_per_possible_bucket"] == 2.5
    z = bucket_statistics(L, table([[0, 0, 0, 3], [1, 1, 0, 0]]), 2)  # a zero-unit record is an empty bucket
    assert z["nonempty_buckets"] == 1 and z["zero_unit_records"] == 1 and z["mean_units_per_nonempty_bucket"] == 3.0


def test_zero_work_and_malformed_tables():
    L = lay8(G=2)
    st = bucket_statistics(L, table([]), 4)
    assert st["total_work_units"] == 0 and st["nonempty_buckets"] == 0 and st["empty_buckets"] == 8
    assert st["mean_units_per_nonempty_bucket"] is None and st["bucket_units_quantiles"] is None
    assert st["mean_units_per_possible_bucket"] == 0.0
    ex = SgExec(L, torch.zeros((4, 16), dtype=torch.int16), table([]))
    assert ex.mean_bucket_units() == 0.0  # the documented zero-work dispatch value
    with pytest.raises(ValueError, match="duplicate"):
        bucket_statistics(L, table([[0, 0, 0, 1], [0, 0, 1, 1]]), 2)
    with pytest.raises(ValueError, match="negative"):
        bucket_statistics(L, table([[0, 0, 0, -1]]), 2)
    with pytest.raises(ValueError, match="range"):
        bucket_statistics(L, table([[2, 0, 0, 1]]), 2)


def test_quantile_convention():
    assert nearest_rank_quantiles(torch.tensor([5.0])) == [5.0] * 6
    v = torch.arange(1, 101).float()
    assert nearest_rank_quantiles(v) == [1.0, 10.0, 50.0, 90.0, 99.0, 100.0]


@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("H,G,S,R", [(1_000_000, 1024, 64, 16), (10_000_000, 1024, 64, 16), (77_777, 64, 32, 8),
                                     (20_003, 8, 16, 4), (300_000, 16, 16, 1), (64 * 8 + 3, 8, 16, 4)])
def test_generated_graphs(C, H, G, S, R):
    """Reconstruction from the plan alone (bucket_table) = the generator's table; identities; the
    nonempty mean of positive integer units is >= 1; units counted from the records."""
    L = SparseLayout(1024 if S == 64 else 256, H, G, 8, C, True, packed=True)
    plan = ConnectivityPlan(S, R, seed=5)
    feat, b = bucket_table(L.canonical, plan)
    st = bucket_statistics(L, b, plan.num_supergroups(L.input_dim), S, plan)
    un = L.unit_blocks * L.block
    nvu = (L.counts_tensor("cpu").long() + un - 1) // un
    assert st["total_work_units"] == int(nvu.sum())
    assert st["mean_units_per_nonempty_bucket"] >= 1.0
    assert math.isclose(st["mean_units_per_possible_bucket"],
                        st["bucket_occupancy"] * st["mean_units_per_nonempty_bucket"], rel_tol=1e-12)
    assert st["valid_neurons"] == H and st["padded_neuron_capacity"] >= H
    if H <= 400_000:
        idx, graph, _ = generate_indices(L.canonical, plan, "cpu")
        assert torch.equal(graph.buckets.cpu(), b) and torch.equal(graph.feat.cpu(), feat)


def test_historical_h1m_values():
    """The H = 1M S = 64 crossover row (SCALE_DISPATCH_FUSED_BACKWARD.md 4.2): 31,744 C = 8 units
    in 31,744 nonempty buckets of 262,144 possible -> nonempty mean 1.0, possible mean 0.121."""
    L = SparseLayout(1024, 1_000_000, 1024, 8, 8, True, packed=True)
    plan = ConnectivityPlan(64, 16, seed=0)
    _, b = bucket_table(L.canonical, plan)
    st = bucket_statistics(L, b, 256, 64, plan)
    assert st["total_work_units"] == 31_744 and st["nonempty_buckets"] == 31_744
    assert st["total_possible_buckets"] == 262_144
    assert st["mean_units_per_nonempty_bucket"] == 1.0
    assert round(st["mean_units_per_possible_bucket"], 2) == 0.12


def test_matched_c64_to_c8():
    L64 = SparseLayout(256, 100_003, 16, 8, 64, True, packed=False)
    plan = ConnectivityPlan(16, 4, seed=1)
    idx, g64, _ = generate_indices(L64, plan, "cpu")
    _, w, bb = init_parameters(L64, "cpu", seed=0)
    L8, _, _, _, g8 = matched_c8_from_c64(L64, idx, w, bb, g64)
    s64 = bucket_statistics(L64, g64.buckets, g64.Q, 16, plan)
    s8 = bucket_statistics(L8, g8.buckets, g8.Q, 16, plan)
    # a C = 64 unit (64 neurons) = two C = 8 units (32 neurons): 2 U minus fully padded halves
    nvu8 = (L8.counts_tensor("cpu").long() + 31) // 32
    assert s8["total_work_units"] == int(nvu8.sum()) <= 2 * s64["total_work_units"]
    assert s8["nonempty_buckets"] == s64["nonempty_buckets"]
    assert s64["neurons_per_work_unit"] == 64 and s8["neurons_per_work_unit"] == 32


def test_chunk_statistics_distinct_from_buckets():
    L = SparseLayout(256, 300_000, 16, 8, 8, True, packed=True)
    plan = ConnectivityPlan(16, 4, seed=0)
    _, b = bucket_table(L.canonical, plan)
    from sparse_pooling.supergroups import SupergroupGraph, chunk_table
    g = SupergroupGraph(plan, torch.zeros((64, 16), dtype=torch.int16), b)
    items, q, _ = work_items(L, g)
    ch = chunk_table(q, items[:, 2], 64, 40)
    cs = chunk_statistics(items, ch, 64, 40)
    assert cs["chunks"] == ch.shape[0] > 64 and cs["items"] == items.shape[0]
    upc = [int(items[lo:hi, 2].sum()) for _, lo, hi in ch.tolist()]
    assert cs["units_per_chunk_min"] == min(upc) and cs["units_per_chunk_max"] == max(upc)
    assert math.isclose(cs["units_per_chunk_mean"], sum(upc) / len(upc))
    assert sum(upc) == bucket_statistics(L, b, 64)["total_work_units"] == len(unit_list(L, g)[0])


# ------------------------------------------------------------------ dispatch descriptor
def fake_rt(buckets, Q=8, G=4, precision="split", B=8192):
    L = SparseLayout(128, 4 * 8 * 32 * 10, G, 8, 8, True, packed=True)
    sg = SgExec(L, torch.zeros((Q, 64), dtype=torch.int16), buckets)
    return SimpleNamespace(plan=PhysicalPlan(grad_precision=precision), layout=L, sg=sg,
                           level=SimpleNamespace(activation="relu"), idx=torch.zeros(1, device="cuda"), det=False), B


@requires_cuda_backend
def test_dispatch_reads_the_nonempty_mean():
    """Two buckets of 5 units per output over Q = 8: nonempty mean 5.0 (>= 4: private_t32)
    while the possible-bucket mean is 1.25 (< 4): a denominator regression flips the plan."""
    from sparse_pooling.kernels.cuda_ops import CudaBackend
    rows = [[q, g, 5 * (q % 2), 5] for g in range(4) for q in (2 * g, 2 * g + 1)]
    rows.sort()
    rt, B = fake_rt(table(rows))
    st = rt.sg.bucket_stats
    assert st["mean_units_per_nonempty_bucket"] == 5.0 and st["mean_units_per_possible_bucket"] == 1.25
    be = CudaBackend.get()
    plan, why = be.sg_dx_plan(rt, B)
    assert plan == "private_t32" and "mean units per nonempty bucket 5.00 >= 4.0" in why
    assert "possible-bucket mean 1.25" in why
    # BF16 threshold 2.0: 1.5 nonempty mean (3 units in two buckets) -> shared_t64, 2.0 -> private
    rt, B = fake_rt(table(sorted([[q, g, 0, 1 + (q % 2)] for g in range(4) for q in (2 * g, 2 * g + 1)])), precision="bf16")
    assert rt.sg.mean_bucket_units() == 1.5 and be.sg_dx_plan(rt, B)[0] == "shared_t64"
    rt, B = fake_rt(table(sorted([[q, g, 0, 2] for g in range(4) for q in (2 * g, 2 * g + 1)])), precision="bf16")
    assert rt.sg.mean_bucket_units() == 2.0 and be.sg_dx_plan(rt, B)[0] == "private_t32"
    # explicit plans still take precedence over the rule
    rt.plan = PhysicalPlan(grad_precision="bf16", cuda_grouped_dx_plan="shared_t64")
    assert be.sg_dx_plan(rt, B) == ("shared_t64", "explicit cuda_grouped_dx_plan")
    rt.plan = PhysicalPlan(grad_precision="bf16", cuda_grouped_dx_tokens=64)
    assert be.sg_dx_plan(rt, B)[0] == "tiles"


@requires_cuda_backend
def test_runtime_and_benchmark_fields_use_the_canonical_function():
    from test_input_supergroups import inputs, run, sg_layer
    lay = sg_layer(1024, 300_000, 64, 8, 64, 16, plan=PhysicalPlan())
    x, dy = inputs(64, 1024, 64)
    info = run(lay, x, dy)[4]
    sg = lay._rt.sg
    canon = bucket_statistics(lay.layout, lay.sg_buckets, sg.Q, sg.S, lay.connectivity)
    gd = info["grouped_dx"]
    for k in ("mean_units_per_nonempty_bucket", "mean_units_per_possible_bucket", "bucket_occupancy"):
        assert gd[k] == canon[k]
    d = sg.item_chunk_stats(gd["chunk_units"])
    for k, v in canon.items():
        assert d[k] == v, k
    assert d["units_per_bucket_mean"] == canon["mean_units_per_nonempty_bucket"]  # schema-1 alias
    assert d["chunks"] == gd["chunks"]
