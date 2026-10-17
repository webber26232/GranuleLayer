"""Layout, partition, packing and the raw sparse projection."""
import math

import pytest
import torch

from conftest import make_layer, rel_err, requires_cuda, requires_cuda_backend
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.layout import SparseLayout, init_parameters
from sparse_pooling.reference import pytorch_reference as ref


@pytest.mark.parametrize("H,G,C", [(100_000_000, 1024, 8), (100_000_000, 1024, 64), (5000, 16, 8), (1000, 7, 16)])
def test_partition_enumerates_exactly_H(H, G, C):
    L = SparseLayout(1024, H, G, 8, C, True)
    counts = L.counts_tensor("cpu").long()
    offs = L.offsets_tensor("cpu")
    assert int(counts.sum()) == H
    assert int(offs[0]) == 0
    assert torch.equal(offs[1:], offs[:-1] + counts[:-1])  # contiguous, no gaps / overlaps
    assert int(counts.max() - counts.min()) <= 1
    assert L.slots_per_group >= L.max_group
    assert L.blocks_per_group % 4 == 0
    if H < 10**6:
        ids = L.global_ids("cpu")
        valid = ids[ids >= 0]
        assert torch.equal(torch.sort(valid).values, torch.arange(H))


def test_spec_layout_100M():
    L = SparseLayout(1024, 100_000_000, 1024, 8, 8, True)
    # spec: indices [1024, 12208, 8], weights [1024, 12208, 8, 8], bias [1024, 12208, 8]
    assert L.index_shape == (1024, 12208, 8)
    assert L.weight_shape == (1024, 12208, 8, 8)
    assert L.bias_shape == (1024, 12208, 8)
    assert L.padding_fraction() < 1e-3


@requires_cuda
@pytest.mark.parametrize("shared", [True, False])
def test_indices_distinct_sorted_and_padding_zero(shared):
    L = SparseLayout(64, 3001, 8, 8, 16, shared)
    idx, w, b = init_parameters(L, "cuda", seed=1, bias_std=0.5)
    flat = idx.reshape(-1, 8).long()
    inval = ~L.valid_mask("cuda")
    if not shared:  # Mode A: padded neurons carry zeroed indices by design
        flat = flat[~inval.reshape(-1)]
    assert int(flat.min()) >= 0 and int(flat.max()) < 64
    assert bool((flat[:, 1:] > flat[:, :-1]).all()), "indices must be distinct and sorted"
    assert bool((w.reshape(8, -1, 8)[inval] == 0).all())
    assert bool((b.reshape(8, -1)[inval] == 0).all())


@requires_cuda
@pytest.mark.parametrize("C", [8, 16, 32, 64])
def test_pack_unpack_roundtrip(C):
    Lp = SparseLayout(64, 7000, 8, 8, C, True, packed=True)
    L = Lp.canonical
    i, w, b = init_parameters(L, "cuda", seed=2, bias_std=0.1)
    ip, wp, bp = init_parameters(Lp, "cuda", seed=2, bias_std=0.1)
    assert torch.equal(Lp.unpack_weights(wp), w)
    assert torch.equal(Lp.unpack_bias(bp), b)
    assert torch.equal(Lp.unpack_indices(ip), i)
    assert torch.equal(Lp.pack_weights(w), wp)
    assert wp.shape == Lp.weight_shape and bp.shape == Lp.bias_shape and ip.shape == Lp.index_shape


def _projection_check(backend, C, shared, weight_layout, B=37, D=128):
    # one neuron per group + Mean(identity): the pooled output IS the projection
    G = 64
    lay, w, b = make_layer(D, G, G, C if shared else None, {"type": "mean", "activation": "identity"},
                           "minimal_stats", backend, weight_layout=weight_layout)
    x = torch.randn(B, D, device="cuda").bfloat16().float()
    y = lay(x)
    z = ref.hidden_preactivations(x, lay.indices, w, b, lay.layout)[:, :, 0]
    assert rel_err(y, z) < 1e-6


@requires_cuda
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("shared", [True, False])
def test_projection_triton(C, shared):
    _projection_check("triton", C, shared, "auto")


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 16, 32, 64])
@pytest.mark.parametrize("weight_layout", ["packed", "canonical"])
def test_projection_cuda(C, weight_layout):
    _projection_check("cuda", C, True, weight_layout)
