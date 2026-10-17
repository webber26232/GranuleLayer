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
