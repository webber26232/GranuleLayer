"""Runtime behaviour on real hardware: the generic profile forced on this GPU, profile forcing vs
legality, the explanation API, streams, device checks, ABI checks, multi-device routing."""
import ctypes

import pytest
import torch

from conftest import rel_err, requires_cuda_backend
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.supergroups import ConnectivityPlan
from test_input_supergroups import inputs, reference, run, sg_layer


def gen(**kw):
    return PhysicalPlan(tuning_profile="generic_cuda", **kw)


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("S", [16, 64])
@pytest.mark.parametrize("precision", ["split", "bf16"])
@pytest.mark.parametrize("reduction", ["fast", "deterministic"])
def test_generic_profile_matches_reference(C, S, precision, reduction):
    lay = sg_layer(1024, 200_000, 64, C, S, S // 4, plan=gen(grad_precision=precision, reduction=reduction))
    x, dy = inputs(200, 1024, 64)
    y, dx, dw, db, info = run(lay, x, dy)
    yr, dxr, dwr, dbr = reference(lay, x, dy)
    tol = 2e-5 if precision == "split" else 4e-3
    assert rel_err(y, yr) < 1e-5 and rel_err(dx, dxr) < tol and rel_err(dw, dwr) < tol and rel_err(db, dbr) < tol
    ex = info["execution"]
    assert ex["profile"]["name"] == "generic_cuda" and ex["profile"]["forced"]
    assert ex["profile"]["status"] == "numerics-validated, performance-unmeasured"
    assert info["reduction"]["mode"] == reduction
    if info["dx_kernel"] == "grouped":
        assert info["grouped_dx"]["scatter"] == "baseline"  # no measured scatter table
    assert info["dw_staging"] == "full" and info["dx_epilogue"] == "generic"


@requires_cuda_backend
def test_generic_profile_deterministic_repeatable():
    def once():
        lay = sg_layer(1024, 200_000, 64, 8, 64, 16, plan=gen(reduction="deterministic"))
        x, dy = inputs(256, 1024, 64)
        return run(lay, x, dy)[:4]

    a, b = once(), once()
    assert all(torch.equal(u, v) for u, v in zip(a, b))


@requires_cuda_backend
def test_profile_forcing_cannot_bypass_legality():
    # the RTX 3090 profile forced on whatever this device is: an illegal explicit request still raises
    lay = sg_layer(1024, 200_000, 64, 8, 64, 16, plan=PhysicalPlan(tuning_profile="rtx3090", reduction="deterministic",
                                                                   cuda_grouped_dx_plan="shared_t64"))
    x, dy = inputs(64, 1024, 64)
    with pytest.raises(ValueError, match="shared_t64"):
        run(lay, x, dy)
    lay.plan = PhysicalPlan(tuning_profile="generic_cuda", cuda_input_supergroup_backend="grouped",
                            cuda_grouped_dx_scatter="feature_owner", cuda_grouped_dx_tokens=64)  # S = 64 > 32
    with pytest.raises(ValueError, match="feature_owner"):
        run(lay, x, dy)
    with pytest.raises(ValueError, match="unknown tuning profile"):
        lay.plan = PhysicalPlan(tuning_profile="h100")
        run(lay, x, dy)


@requires_cuda_backend
def test_explain_and_execution_metadata():
    lay = sg_layer(1024, 200_000, 64, 8, 64, 16, plan=PhysicalPlan())
    ex = lay.explain(1024)
    for k in ("backend", "device", "profile", "build", "reduction", "forward", "backward"):
        assert k in ex, k
    assert ex["build"]["kind"] == "production" and ex["build"]["target"].startswith("sm_")
    assert ex["device"]["source"] == "runtime" and ex["profile"]["match_reason"]
    assert ex["backward"]["dx"]["kernel"] in ("grouped", "generic")
    x, dy = inputs(1024, 1024, 64)
    info = run(lay, x, dy)[4]
    assert info["execution"]["backward"]["dx"]["scatter"] == info["grouped_dx"]["scatter"]
    assert info["execution"]["forward"]["kernel"] == info["forward_kernel"]
    assert info["execution"]["build"]["abi"] == 3


@requires_cuda_backend
def test_rtx3090_profile_reported_only_on_the_measured_board():
    from sparse_pooling.backends.capabilities import query
    from sparse_pooling.backends.profiles.rtx3090 import is_rtx3090

    lay = sg_layer(1024, 200_000, 64, 8, 64, 16, plan=PhysicalPlan())
    prof = lay.explain(1024)["profile"]
    if is_rtx3090(query("cuda"))[0]:
        assert prof["name"] == "rtx3090" and prof["status"] == "hardware-validated"
    else:
        assert prof["name"] == "generic_cuda" and prof["status"] != "hardware-validated"


@requires_cuda_backend
@pytest.mark.parametrize("reduction", ["fast", "deterministic"])
def test_nondefault_stream(reduction):
    """Launches follow the current stream of the tensors' device; results equal the default stream."""
    lay = sg_layer(1024, 200_000, 64, 8, 16, 4, plan=PhysicalPlan(reduction=reduction))
    x, dy = inputs(256, 1024, 64)
    ref = run(lay, x, dy)[:4]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        lay.weight.grad = lay.bias.grad = None
        got = run(lay, x, dy)[:4]
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    if reduction == "deterministic":
        assert all(torch.equal(a, b) for a, b in zip(ref, got))
    else:
        assert all(rel_err(a, b) < 1e-5 for a, b in zip(ref, got))


@requires_cuda_backend
def test_mixed_devices_rejected_early():
    lay = sg_layer(1024, 200_000, 64, 8, 16, 4, plan=PhysicalPlan())
    with pytest.raises(ValueError, match="parameters are on"):
        lay(torch.randn(64, 1024).bfloat16())


@requires_cuda_backend
def test_device_migration_rebuilds_runtime():
    lay = SparsePoolingLayer(1024, 200_000, 64, shared_neurons=8, pooling="mean", device="cpu",
                             connectivity=ConnectivityPlan(16, 4))
    assert "CPU" in lay.explain()["backend"]
    lay = lay.cuda()
    y = lay(torch.randn(64, 1024, device="cuda").bfloat16())
    assert y.is_cuda and lay.last_info["execution"]["device"]["index"] == torch.cuda.current_device()
    assert lay._rt.sg.feat.is_cuda and lay._rt.caps.backend == "cuda"


@requires_cuda_backend
def test_abi_check_rejects_mismatched_layouts(monkeypatch):
    from sparse_pooling.backends.cuda import runtime as R

    be = R.backend_for("cuda")
    R.check_abi(be.lib, "production", "production", be.native.path)  # the loaded library passes

    class Shifted(ctypes.Structure):
        _fields_ = [("X", ctypes.c_void_p), ("sxf", ctypes.c_int)]

    monkeypatch.setattr(R, "STRUCTS", (Shifted,) + R.STRUCTS[1:])
    with pytest.raises(RuntimeError, match="layout mismatch"):
        R.check_abi(be.lib, "production", "production", be.native.path)
    monkeypatch.setattr(R, "STRUCTS", R.STRUCTS)
    with pytest.raises(RuntimeError, match="expected experimental"):
        R.check_abi(be.lib, "experimental", "production", be.native.path)


@requires_cuda_backend
def test_registry_shares_code_per_target_not_per_call():
    from sparse_pooling.backends.cuda import runtime as R

    a, b = R.backend_for("cuda"), R.backend_for(torch.device("cuda", torch.cuda.current_device()))
    assert a is b
    from sparse_pooling.kernels.cuda_ops import CudaBackend

    assert CudaBackend.get() is a


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs (real per-device routing)")
def test_two_devices():
    out = []
    for d in (0, 1):
        lay = sg_layer(1024, 200_000, 64, 8, 16, 4, plan=PhysicalPlan(reduction="deterministic"))
        lay = lay.to(f"cuda:{d}")
        g = torch.Generator(device=f"cuda:{d}").manual_seed(0)
        x = torch.randn(128, 1024, device=f"cuda:{d}", generator=g).bfloat16().requires_grad_(True)
        y = lay(x)
        y.float().square().sum().backward()
        assert lay.last_info["execution"]["device"]["index"] == d
        out.append((y.cpu(), x.grad.cpu()))
    assert rel_err(out[0][0], out[1][0]) < 1e-5
