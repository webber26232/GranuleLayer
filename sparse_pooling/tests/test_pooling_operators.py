"""Forward correctness of every fused pooling operator (both backends)."""
import pytest
import torch

from conftest import requires_cuda, requires_cuda_backend
from sparse_pooling.autograd import Level1Runtime
from sparse_pooling.config import LevelSpec, PhysicalPlan
from sparse_pooling.kernels import ops
from sparse_pooling.layout import SparseLayout, init_parameters
from sparse_pooling.reference import pytorch_reference as ref

SPECS = [
    LevelSpec("max"),
    LevelSpec("max", activation="gelu"),
    LevelSpec("topk_mean", k=2),
    LevelSpec("topk_mean", k=8),
    LevelSpec("topk_mean", k=32, activation="relu"),
    LevelSpec("mean"),
    LevelSpec("mean", activation="identity"),
    LevelSpec("mean", activation="gelu"),
    LevelSpec("logmeanexp", temperature=0.1),
    LevelSpec("logmeanexp", temperature=2.0),
    LevelSpec("power_mean", p=1),
    LevelSpec("power_mean", p=4),
    LevelSpec("power_mean", p=2.5),
    LevelSpec("softmax_weighted", beta=0.5),
    LevelSpec("softmax_weighted", beta=4.0),
]


def _run(L, idx, w, b, x, spec, plan, backend, track):
    rt = Level1Runtime(L, idx, spec, plan, None, 1 << 30, backend)
    out = rt.forward(x, w, b, track)
    return out, rt


def _check(out, spec, zr, mask, track):
    ex = {}
    yr = ref.pool(zr, mask, spec, ex)
    y = ops.act_torch(out["pre"], spec.out_act_id)
    # relative to max(|y|, typical neuron magnitude ~0.1): near-zero pooled
    # values (e.g. Mean(identity)) only carry absolute FP32 summation noise
    err = ((y - yr).abs() / (yr.abs() + 0.1)).max().item()
    assert err < 2e-5, f"{spec.describe()}: rel err {err:.2e}"
    if spec.type == "max" and track:
        assert torch.equal(out["winner"].long(), ex["winner"]), "winner ids differ"
    if spec.type == "topk_mean" and track:
        a = torch.sort(out["topk_slot"].long(), -1).values
        r = torch.sort(ex["topk"], -1).values
        assert torch.equal(a, r), "top-k index sets differ"


@requires_cuda
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("C", [8, 16, 64])
@pytest.mark.parametrize("B", [1, 2, 37, 128])
@pytest.mark.parametrize("splits", [1, 3])
def test_triton_forward(shared, C, B, splits):
    L = SparseLayout(64, 4099, 16, 8, C, shared)
    idx, w, b = init_parameters(L, "cuda", seed=1, bias_std=0.3)
    x = torch.randn(B, 64, device="cuda").bfloat16()
    zr = ref.hidden_preactivations(x, idx, w, b, L)
    mask = L.valid_mask("cuda")
    plan = PhysicalPlan(phys_splits=splits, block_m=32)
    for spec in SPECS:
        out, _ = _run(L, idx, w, b, x, spec, plan, None, spec.type in ("max", "topk_mean"))
        _check(out, spec, zr, mask, spec.type in ("max", "topk_mean"))


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 16, 32, 64])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("cfg", [(32, 16, 1, "feature_major", 64), (16, 8, 3, "feature_major", 40),
                                 (32, 8, 0, "token_major", 33)])
def test_cuda_forward(C, packed, cfg):
    from sparse_pooling.kernels.cuda_ops import CudaBackend

    bm, nw, ps, lay, B = cfg
    L = SparseLayout(1024, 30011, 16, 8, C, True, packed=packed)
    idx, w, b = init_parameters(L, "cuda", seed=1, bias_std=0.3)
    x = torch.randn(B, 1024, device="cuda").bfloat16()
    zr = ref.hidden_preactivations(x, idx, w, b, L)
    mask = L.valid_mask("cuda")
    plan = PhysicalPlan(phys_splits=ps, cuda_block_m=bm, cuda_warps=nw, input_layout=lay)
    be = CudaBackend.get()
    for spec in SPECS:
        for track in ([False, True] if spec.type in ("max", "topk_mean") else [False]):
            out, rt = _run(L, idx, w, b, x, spec, plan, be, track)
            _check(out, spec, zr, mask, track)


@requires_cuda
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_deterministic_ties(backend):
    """All neurons identical -> Max picks the smallest neuron id, Top-K the
    k smallest ids, in every group, with any split count."""
    if backend == "cuda":
        from conftest import cuda_backend_available

        if not cuda_backend_available():
            pytest.skip("no CUDA backend")
        from sparse_pooling.kernels.cuda_ops import CudaBackend

        be = CudaBackend.get()
    else:
        be = None
    L = SparseLayout(1024, 9000, 8, 8, 8, True, packed=(backend == "cuda"))
    idx, w, b = init_parameters(L, "cuda", seed=0)
    w.zero_()
    b.fill_(0.5)
    from sparse_pooling.layout import zero_padding_

    zero_padding_(L, weights=w, bias=b)
    x = torch.randn(32, 1024, device="cuda").bfloat16()
    for ps in (1, 4):
        plan = PhysicalPlan(phys_splits=ps, block_m=32)
        out, _ = _run(L, idx, w, b, x, LevelSpec("max"), plan, be, True)
        assert bool((out["winner"] == 0).all())
        assert torch.allclose(out["pre"], torch.full_like(out["pre"], 0.5))
        out, rt = _run(L, idx, w, b, x, LevelSpec("topk_mean", k=8), plan, be, True)
        assert rt.last_info["forward_backend"] == ("cuda" if be is not None else "triton")
        assert torch.equal(torch.sort(out["topk_slot"].long(), -1).values,
                           torch.arange(8, device="cuda").expand_as(out["topk_slot"]))


@requires_cuda
def test_merge_states_independently():
    """Per-split state merging (torch side) against explicit two-stage references."""
    torch.manual_seed(0)
    B, U, N, PS = 5, 3, 64, 4
    z = torch.randn(B, U, N, device="cuda") * 3
    mask = torch.ones(U, N, dtype=torch.bool, device="cuda")
    for spec in [LevelSpec("max", out_activation="identity"), LevelSpec("mean"),
                 LevelSpec("logmeanexp", temperature=0.3), LevelSpec("topk_mean", k=4)]:
        flat = ref.pool(z, mask, spec)
        hier = ref.reference_physical_hierarchy(z, mask, spec, PS)
        assert torch.allclose(flat, hier, rtol=1e-5, atol=1e-6), spec.describe()


@requires_cuda_backend
@pytest.mark.parametrize("k", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("C", [8, 16, 64])
def test_cuda_topk_two_phase(k, C):
    """Groups large enough that the CUDA Top-K kernel seeds its threshold from
    an exact merge of the first 1/16 of the blocks (phase A) before phase B."""
    from sparse_pooling.kernels.cuda_ops import CudaBackend

    L = SparseLayout(1024, 4 * 21000, 4, 8, C, True, packed=True)
    idx, w, b = init_parameters(L, "cuda", seed=7, bias_std=0.2)
    x = torch.randn(48, 1024, device="cuda").bfloat16()
    zr = ref.hidden_preactivations(x, idx, w, b, L)
    mask = L.valid_mask("cuda")
    spec = LevelSpec("topk_mean", k=k, activation="relu")
    for bm in (16, 32):
        plan = PhysicalPlan(phys_splits=1, cuda_block_m=bm)
        out, rt = _run(L, idx, w, b, x, spec, plan, CudaBackend.get(), True)
        # k = 32 is routed to Triton (register budget; measured faster there)
        assert rt.last_info["forward_backend"] == ("cuda" if k <= 16 else "triton")
        _check(out, spec, zr, mask, True)
