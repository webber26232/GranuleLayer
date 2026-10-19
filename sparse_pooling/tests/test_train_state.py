"""benchmarks/train_state.TrainState: complete snapshot / in-place restore of a training state
(parameters, main_grad / .grad, AdamW moments and steps, hyperparameters, the stochastic-
rounding counter FusedAdamW._seed, torch RNG states) used by the controlled benchmarks."""
import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_cuda_backend
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.supergroups import ConnectivityPlan

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from train_state import TrainState, tensor_fingerprint  # noqa: E402


def setup(backend="generic", sr=True, master=False, moments="bf16"):
    lay = SparsePoolingLayer(256, 50_000, 16, shared_neurons=8, pooling="mean", device="cuda", seed=0,
                             plan=PhysicalPlan(cuda_input_supergroup_backend=backend),
                             connectivity=ConnectivityPlan(16, 4, seed=1))
    lay.enable_main_grad()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-3, weight_decay=0.01, moments=moments, master_weights=master,
                     stochastic_rounding=sr)
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    xs = [torch.randn(64, 256, device="cuda", generator=g).bfloat16() for _ in range(4)]
    dys = [torch.randn(64, 16, device="cuda", generator=g) for _ in range(4)]
    return lay, opt, xs, dys


def train(lay, opt, xs, dys, n):
    for i in range(n):
        (lay(xs[i % len(xs)]).float() * dys[i % len(dys)]).sum().backward()
        opt.step()


def snapshot_values(lay, opt):
    out = {"w": lay.weight.detach().clone(), "b": lay.bias.detach().clone(),
           "gw": lay.weight.main_grad.clone(), "seed": opt._seed,
           "rng": torch.cuda.get_rng_state().clone(), "cpu_rng": torch.get_rng_state().clone()}
    for i, p in enumerate((lay.weight, lay.bias)):
        st = opt.state[p]
        out[f"m{i}"], out[f"v{i}"], out[f"step{i}"] = st["m"].clone(), st["v"].clone(), st["step"]
        if "master" in st:
            out[f"master{i}"] = st["master"].clone()
    out["lr"] = opt.param_groups[0]["lr"]
    return out


def assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@requires_cuda_backend
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("master,moments", [(False, "bf16"), (True, "fp32")])
def test_restore_is_complete(device, master, moments):
    lay, opt, xs, dys = setup(master=master, moments=moments)
    train(lay, opt, xs, dys, 3)  # lazy AdamW state exists, moments / steps / seed advanced
    ref = snapshot_values(lay, opt)
    st = TrainState([lay.weight, lay.bias], opt, device=device).capture()
    fp = st.fingerprint()
    train(lay, opt, xs, dys, 2)
    opt.param_groups[0]["lr"] = 5.0  # hyperparameters are restored too
    torch.cuda.manual_seed(1234)
    torch.rand(10, device="cuda")
    (lay(xs[0]).float() * dys[0]).sum().backward()  # a pending (unapplied) main_grad
    assert st.fingerprint() != fp
    st.restore()
    assert st.fingerprint() == fp
    assert_same(snapshot_values(lay, opt), ref)
    # restoration is in place: the tensors are the same objects / storage
    assert opt.state[lay.weight]["m"].data_ptr() == st.opt.state[lay.weight]["m"].data_ptr()


@requires_cuda_backend
@pytest.mark.parametrize("sr", [True, False])
def test_restored_trajectories_repeat_bitwise(sr):
    """Generic backend with a fixed upstream gradient: deterministic dW / db and AdamW, so two
    runs from the restored state must agree bitwise, including stochastic rounding (seed)."""
    lay, opt, xs, dys = setup(sr=sr)
    train(lay, opt, xs, dys, 2)
    st = TrainState([lay.weight, lay.bias], opt).capture()
    train(lay, opt, xs, dys, 3)
    a = snapshot_values(lay, opt)
    st.restore()
    train(lay, opt, xs, dys, 3)
    b = snapshot_values(lay, opt)
    a.pop("rng"); b.pop("rng")
    assert_same(a, b)


@requires_cuda_backend
def test_weight_only_reset_is_incomplete():
    """The historical benchmark reset (weights / bias only) leaves moments, steps and the SR
    counter advanced: the next update differs from the reference trajectory."""
    lay, opt, xs, dys = setup()
    train(lay, opt, xs, dys, 2)
    w0, b0 = lay.weight.detach().clone(), lay.bias.detach().clone()
    st = TrainState([lay.weight, lay.bias], opt).capture()
    train(lay, opt, xs, dys, 1)
    ref = snapshot_values(lay, opt)
    with torch.no_grad():  # historical: weights / bias only
        lay.weight.copy_(w0)
        lay.bias.copy_(b0)
    train(lay, opt, xs, dys, 1)
    hist = snapshot_values(lay, opt)
    assert hist["step0"] == ref["step0"] + 1 and hist["seed"] != ref["seed"]
    assert not torch.equal(hist["w"], ref["w"])
    st.restore()
    train(lay, opt, xs, dys, 1)
    full = snapshot_values(lay, opt)
    full.pop("rng"); ref.pop("rng")
    assert_same(full, ref)


def test_fingerprint_is_exact():
    t = torch.zeros(1000, dtype=torch.bfloat16)
    f0 = tensor_fingerprint(t)
    t[500] = torch.finfo(torch.bfloat16).tiny  # one bit
    assert tensor_fingerprint(t) != f0
    u = torch.zeros(1000, dtype=torch.bfloat16)
    u[500] = -0.0
    assert tensor_fingerprint(u) != f0  # signed zero


def test_fingerprint_chunked_matches_and_is_bounded():
    import train_state
    t = torch.randint(-30000, 30000, (3 * train_state.FP_CHUNK + 12345,), dtype=torch.int16, device="cuda")
    b = t.view(torch.int16)
    want = (int(b.sum(dtype=torch.int64)), int(b[1::3].sum(dtype=torch.int64)))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    assert tensor_fingerprint(t) == want
    assert torch.cuda.max_memory_allocated() - base <= 8 * train_state.FP_CHUNK + (1 << 20)
