"""Fused AdamW vs torch.optim.AdamW."""
import pytest
import torch

from conftest import requires_cuda
from sparse_pooling.optim import FusedAdamW


@requires_cuda
@pytest.mark.parametrize("moments,master", [("fp32", True), ("bf16", True), ("fp32", False), ("bf16", False)])
def test_fused_adamw_matches_torch(moments, master):
    torch.manual_seed(0)
    w0 = torch.randn(10_000, device="cuda").bfloat16()
    p = torch.nn.Parameter(w0.clone())
    p.main_grad = torch.zeros(p.shape, device="cuda")
    q = torch.nn.Parameter(w0.float().clone())
    opt = FusedAdamW([p], lr=1e-2, weight_decay=0.1, moments=moments, master_weights=master)
    ref = torch.optim.AdamW([q], lr=1e-2, weight_decay=0.1)
    for _ in range(5):
        g = torch.randn(p.shape, device="cuda")
        p.main_grad.copy_(g)
        q.grad = g.clone()
        opt.step()
        ref.step()
        assert float(p.main_grad.abs().max()) == 0.0  # fused zero_grad
    err = ((p.float() - q).abs().max() / q.abs().max()).item()
    tol = {("fp32", True): 1e-2, ("bf16", True): 2e-2, ("fp32", False): 3e-2, ("bf16", False): 3e-2}[(moments, master)]
    assert err < tol  # p is stored in BF16 (2^-8 relative), everything else is rounding


@requires_cuda
def test_stochastic_rounding_is_unbiased():
    p = torch.nn.Parameter(torch.ones(1_000_000, device="cuda").bfloat16())
    p.main_grad = torch.full(p.shape, 1.0, device="cuda")
    opt = FusedAdamW([p], lr=1e-4, betas=(0.0, 0.0), eps=0.0, moments="fp32", stochastic_rounding=True)
    opt.step()  # update = -lr * sign(g) = -1e-4, far below the BF16 ulp at 1.0 (2^-7)
    mean = p.float().mean().item()
    assert abs(mean - (1 - 1e-4)) < 2e-5


def _moment_reference_violations(n, master, steps=8, seed=0):
    """Per step: elements whose new first moment is not the rounding of b1 m + (1 - b1) g of the
    kernel's own previous state (a gradient read after another thread's fused zeroing gives
    m = b1 m exactly), and the final state bits."""
    torch.manual_seed(seed)
    p = torch.nn.Parameter((torch.randn(n, device="cuda") * 0.1).bfloat16())
    p.main_grad = torch.zeros(n, device="cuda")
    opt = FusedAdamW([p], lr=1e-3, moments="fp32" if master else "bf16", master_weights=master,
                     stochastic_rounding=False)
    gs = [torch.randn(n, device="cuda") * 1e-3 for _ in range(steps)]
    bad = 0
    for g in gs:
        st = opt.state.get(p, {})
        m0 = st["m"].float().clone() if st else torch.zeros(n, device="cuda")
        p.main_grad.copy_(g)
        opt.step()
        m = opt.state[p]["m"]
        want = (0.9 * m0 + 0.1 * g).to(m.dtype)
        # rounding scale of the operands (the result may cancel), one ulp of the moment dtype
        mag = torch.maximum((0.9 * m0).abs(), (0.1 * g).abs())
        ulp = (mag * (2.0**-7 if m.dtype == torch.bfloat16 else 2.0**-22)).clamp_min(1e-30)
        bad += int(((m.float() - want.float()).abs() > 2 * ulp).sum())
    return bad, (p.detach().view(torch.int16).clone(), opt.state[p]["m"].clone())


@requires_cuda
@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("n", [812_345, 819_200])
def test_in_place_update_is_race_free(n, master):
    """Regression: every buffer is updated in place and Triton loads / stores G (and MASTER)
    in different thread layouts; before the barrier in _adamw_kernel some elements read the
    fused zeroing of another thread (40 identical runs: 9 / 4 distinct final states at these
    sizes).  Every run must follow the reference recurrence and repeat bitwise."""
    finals = []
    for r in range(12):
        bad, fin = _moment_reference_violations(n, master)
        assert bad == 0, f"run {r}: {bad} first-moment elements do not follow the AdamW recurrence"
        finals.append(fin)
    for f in finals[1:]:
        assert torch.equal(f[0], finals[0][0]) and torch.equal(f[1], finals[0][1])
