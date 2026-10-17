"""Physical (execution-only) vs semantic (model-changing) pooling hierarchies.

Associative reductions split across programs must equal the flat operator;
nonlinear multi-level models must equal an explicit hierarchical reference and
must NOT be confused with the flat operator."""
import math

import pytest
import torch

from conftest import make_layer, rel_err, requires_cuda, requires_cuda_backend, run_both
from sparse_pooling import PhysicalPlan
from sparse_pooling.autograd import Level1Runtime
from sparse_pooling.config import LevelSpec, PoolingConfig
from sparse_pooling.kernels import ops
from sparse_pooling.layout import SparseLayout, init_parameters
from sparse_pooling.reference import pytorch_reference as ref


# ------------------------------------------------------------ physical: equal to flat
@requires_cuda
@pytest.mark.parametrize("spec,exact", [
    (LevelSpec("max"), True),                         # Max  -> Max
    (LevelSpec("mean", activation="identity"), False),  # Sum  -> Sum (with counts)
    (LevelSpec("mean"), False),                       # Mean with counts -> Mean with counts
    (LevelSpec("logmeanexp", temperature=0.5), False),  # LSE state -> LSE state
    (LevelSpec("topk_mean", k=8), True),              # Top-K -> Top-K merge
    (LevelSpec("softmax_weighted", beta=2.0), False),
    (LevelSpec("power_mean", p=4), False),
])
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_physical_hierarchy_equals_flat(spec, exact, backend):
    if backend == "cuda":
        from conftest import cuda_backend_available

        if not cuda_backend_available():
            pytest.skip("no CUDA backend")
        from sparse_pooling.kernels.cuda_ops import CudaBackend

        be = CudaBackend.get()
    else:
        be = None
    L = SparseLayout(1024, 50000, 8, 8, 8, True, packed=(backend == "cuda"))
    idx, w, b = init_parameters(L, "cuda", seed=4, bias_std=0.2)
    x = torch.randn(48, 1024, device="cuda").bfloat16()
    outs = []
    for ps in (1, 2, 5, 16):
        rt = Level1Runtime(L, idx, spec, PhysicalPlan(phys_splits=ps, block_m=32), None, 1 << 30, be)
        outs.append(rt.forward(x, w, b, spec.type in ("max", "topk_mean")))
    for o in outs[1:]:
        if exact:  # selection is exact; a mean of the selected values may round differently
            if spec.type == "max":
                assert torch.equal(o["pre"], outs[0]["pre"])
            else:
                assert rel_err(o["pre"], outs[0]["pre"]) < 1e-6
            if "winner" in o:
                assert torch.equal(o["winner"], outs[0]["winner"])
            if "topk_slot" in o:
                assert torch.equal(torch.sort(o["topk_slot"], -1).values, torch.sort(outs[0]["topk_slot"], -1).values)
        else:
            assert rel_err(o["pre"], outs[0]["pre"]) < 1e-5


# ------------------------------------------------------------ semantic: distinct models
SEMANTIC = [
    # (level1, level2, collapses_to_flat_max)
    ({"type": "max", "neurons_per_shard": 256}, {"type": "max"}, True),  # Max -> Max == global Max
    ({"type": "max", "neurons_per_shard": 256, "out_activation": "identity"}, {"type": "mean"}, False),
    ({"type": "mean", "neurons_per_shard": 256}, {"type": "max"}, False),
    ({"type": "topk_mean", "k": 4, "neurons_per_shard": 256}, {"type": "logmeanexp", "temperature": 0.5}, False),
    ({"type": "max", "neurons_per_shard": 256, "transform": "affine"}, {"type": "max"}, False),
    ({"type": "topk_mean", "k": 4, "neurons_per_shard": 256, "transform": "gate"}, {"type": "weighted_sum"}, False),
    ({"type": "mean", "neurons_per_shard": 256}, {"type": "logmeanexp", "temperature": 1.0}, False),
    ({"type": "logmeanexp", "temperature": 0.5, "neurons_per_shard": 256}, {"type": "mean", "activation": "identity"},
     False),
]


@requires_cuda
@pytest.mark.parametrize("l1,l2,collapses", SEMANTIC)
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_semantic_hierarchy_matches_explicit_reference(l1, l2, collapses, backend):
    if backend == "cuda":
        from conftest import cuda_backend_available

        if not cuda_backend_available():
            pytest.skip("no CUDA backend")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = {"levels": [l1, l2]}
        lay, w, b = make_layer(64, 3000, 8, 8, cfg, "auto", backend, seed=1)
    # make learned transforms non-trivial
    with torch.no_grad():
        for n in lay._tnames:
            getattr(lay, "t_" + n).add_(0.3 * torch.randn_like(getattr(lay, "t_" + n)))
    r = run_both(lay, w, b, 32, 64)
    for k, (a, e) in r.items():
        assert rel_err(a, e) < 2e-5, f"{k}: {rel_err(a, e):.2e}"
    # relation to the flat (single-level) operator over the whole group
    x = torch.randn(16, 64, device="cuda").bfloat16().float()
    z = ref.hidden_preactivations(x, lay.indices, w, b, lay.layout)
    flat = ref.pool(z, lay.layout.valid_mask("cuda"), LevelSpec("max"))
    y = lay(x).detach()
    if collapses:
        assert torch.allclose(y, flat), "Max -> Max must equal global Max"
    else:
        assert not torch.allclose(y, flat, atol=1e-3)


@requires_cuda
def test_semantic_mean_mean_is_not_global_mean_with_partial_shard():
    """Mean of shard means equals the global mean only if every shard has the
    same size; with a partial last shard it is a different model."""
    lay, w, b = make_layer(64, 8 * 600, 8, 8, {"levels": [{"type": "mean", "activation": "identity",
                                                           "neurons_per_shard": 256},
                                                          {"type": "mean", "activation": "identity"}]},
                           "auto", "triton")
    x = torch.randn(8, 64, device="cuda").bfloat16().float()
    z = ref.hidden_preactivations(x, lay.indices, w, b, lay.layout)
    flat = ref.pool(z, lay.layout.valid_mask("cuda"), LevelSpec("mean", activation="identity"))
    assert not torch.allclose(lay(x).detach(), flat, atol=1e-4)


@requires_cuda
def test_api_distinguishes_physical_and_semantic():
    with pytest.raises(ValueError):
        PoolingConfig([LevelSpec("max", neurons_per_shard=256)])  # single level must not shard semantically
    with pytest.raises(ValueError):
        PoolingConfig([LevelSpec("max"), LevelSpec("mean")])  # level 2 needs level-1 shards
    with pytest.warns(UserWarning):
        PoolingConfig([LevelSpec("max", neurons_per_shard=256), LevelSpec("max")])
