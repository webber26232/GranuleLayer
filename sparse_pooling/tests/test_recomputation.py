"""Recomputation: same function, consistent policies, no hidden-tensor storage."""
import pytest
import torch

from conftest import make_layer, rel_err, requires_cuda, requires_cuda_backend, run_both
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.autograd import Level1Runtime
from sparse_pooling.config import LevelSpec
from sparse_pooling.layout import SparseLayout, init_parameters


@requires_cuda
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_recompute_is_bitwise_deterministic(backend):
    be = None
    if backend == "cuda":
        from conftest import cuda_backend_available

        if not cuda_backend_available():
            pytest.skip("no CUDA backend")
        from sparse_pooling.kernels.cuda_ops import CudaBackend

        be = CudaBackend.get()
    L = SparseLayout(1024, 40000, 16, 8, 8, True, packed=(backend == "cuda"))
    idx, w, b = init_parameters(L, "cuda", seed=5, bias_std=0.1)
    x = torch.randn(64, 1024, device="cuda").bfloat16()
    for spec in (LevelSpec("max"), LevelSpec("logmeanexp", temperature=0.5), LevelSpec("softmax_weighted"),
                 LevelSpec("mean")):
        rt = Level1Runtime(L, idx, spec, PhysicalPlan(phys_splits=1), None, 1 << 30, be)
        a = rt.forward(x, w, b, True)
        c = rt.forward(x, w, b, True)
        for k in a:
            assert torch.equal(a[k], c[k]), f"{spec.describe()} {k} not reproducible"


@requires_cuda
@pytest.mark.parametrize("backend", ["triton", "cuda"])
@pytest.mark.parametrize("pooling,policies", [
    ("max", ["save_winners", "minimal_stats", "recompute_stats"]),
    ({"type": "topk_mean", "k": 8}, ["save_winners", "recompute_stats"]),
    ({"type": "logmeanexp", "temperature": 0.5}, ["minimal_stats", "recompute_stats"]),
    ({"type": "softmax_weighted", "beta": 2.0}, ["minimal_stats", "recompute_stats"]),
    ({"type": "power_mean", "p": 4}, ["minimal_stats", "recompute_stats"]),
])
def test_policies_give_identical_gradients(backend, pooling, policies):
    if backend == "cuda":
        from conftest import cuda_backend_available

        if not cuda_backend_available():
            pytest.skip("no CUDA backend")
    grads = []
    for pol in policies:
        lay, w, b = make_layer(1024 if backend == "cuda" else 64, 20000, 8, 8, pooling, pol, backend)
        torch.manual_seed(0)
        x = torch.randn(32, lay.layout.input_dim, device="cuda").bfloat16().float().requires_grad_(True)
        lay(x).square().sum().backward()
        grads.append((x.grad, lay.weight.grad, lay.bias.grad))
        assert lay.last_info["policy"] == pol
        expect = pol == "recompute_stats" or (pooling == "max" and pol == "minimal_stats")
        assert lay.last_info["recomputed_stats"] == expect
    for g in grads[1:]:
        for a, e in zip(g, grads[0]):
            # atomics make FP32 summation order differ run to run
            assert rel_err(a, e) < 1e-5


@requires_cuda
@pytest.mark.parametrize("pooling", ["max", "mean", {"type": "logmeanexp", "temperature": 0.5}])
def test_no_hidden_activation_is_materialized(pooling):
    """[B, H] in FP32 would be 2 GiB here; forward+backward must stay far below."""
    B, H = 256, 2_000_000
    lay = SparsePoolingLayer(1024, H, 1024, shared_neurons=8, pooling=pooling, checkpoint_policy="auto",
                             device="cuda")
    lay.enable_main_grad()
    x = torch.randn(B, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = lay(x)
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    hidden_bytes = B * H * 4
    assert extra < hidden_bytes / 20, f"peak extra {extra / 2**20:.0f} MiB suggests materialization"
    assert lay.last_info["checkpoint_bytes"] <= B * 1024 * 8 + 1024


@requires_cuda
def test_inplace_param_update_between_fwd_and_bwd_is_rejected():
    lay, w, b = make_layer(64, 3000, 8, 8, "mean", "minimal_stats", "triton")
    x = torch.randn(4, 64, device="cuda").bfloat16().float()
    y = lay(x)
    with torch.no_grad():
        lay.weight.add_(1.0)  # would make the recomputation see different weights
    with pytest.raises(RuntimeError):
        y.sum().backward()


@requires_cuda
def test_recomputed_level1_features_match_saved():
    cfg = {"levels": [{"type": "topk_mean", "k": 4, "neurons_per_shard": 256}, {"type": "logmeanexp"}]}
    res = []
    for ck in ("save", "recompute"):
        lay, w, b = make_layer(64, 4000, 8, 8, dict(cfg, level1_checkpoint=ck), "save_winners", "triton")
        torch.manual_seed(1)
        x = torch.randn(16, 64, device="cuda").bfloat16().float().requires_grad_(True)
        lay(x).sum().backward()
        res.append((x.grad, lay.weight.grad, lay.last_info["checkpoint_bytes"]))
    assert rel_err(res[0][0], res[1][0]) < 1e-6 and rel_err(res[0][1], res[1][1]) < 1e-6
    assert res[1][2] == 0 and res[0][2] > 0  # recompute keeps no level-1 features


@requires_cuda
def test_chunked_winner_recomputation_matches(monkeypatch):
    """When saved ids would exceed the checkpoint budget, winners are recomputed
    in token chunks; gradients must equal the unchunked path."""
    import sparse_pooling.autograd as ag

    cfg = {"levels": [{"type": "topk_mean", "k": 4, "neurons_per_shard": 256}, {"type": "logmeanexp"}]}
    grads = []
    for budget in (1 << 40, 4096):
        monkeypatch.setattr(ag, "CHECKPOINT_BUDGET", budget)
        lay, w, b = make_layer(64, 4000, 8, 8, cfg, "save_winners", "triton")
        torch.manual_seed(2)
        x = torch.randn(96, 64, device="cuda").bfloat16().float().requires_grad_(True)
        lay(x).sum().backward()
        grads.append((x.grad, lay.weight.grad, lay.bias.grad, dict(lay.last_info)))
    assert "policy_fallback" in grads[1][3] and grads[1][3]["backward_token_chunks"] > 1
    for a, e in zip(grads[1][:3], grads[0][:3]):
        assert rel_err(a, e) < 1e-6
