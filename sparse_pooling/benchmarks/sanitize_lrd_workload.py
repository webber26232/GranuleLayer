"""Small boundary-rich workload over the learned sliding-window reduction kernels for NVIDIA Compute Sanitizer
(run_sanitizer.sh ... lrd): overlap, no overlap, multi-channel, one window, non-multiple L / s, uneven H / G with ragged
C16 tails, token tails B = 8 / 40 / 136, every forward epilogue, every compiled dX tiling, separate and fused dP at both
task counts, main_grad accumulation, non-finite upstream gradients, FusedAdamW, the stride view."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import CoverTopK16Layer  # noqa: E402
from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_  # noqa: E402
from sparse_pooling.experimental.learned_reduction import ReducerConfig, ReductionModel, clone_base  # noqa: E402
from sparse_pooling.optim import FusedAdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402


def base_of(H, G=8, D=256, seed=1):
    L = KLayout(D, H, G, 64, 256)
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(L.weight_shape, device="cuda", generator=g) / 8).bfloat16()
    b = (0.1 * torch.randn(L.bias_shape, device="cuda", generator=g)).bfloat16()
    zero_padding_(L, w, b)
    return CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G,
                                          connectivity=ConnectivityPlan(64, 2, seed=seed), normalization="energy")


GEOMS = [(4096, 320, 64, 1), (4096, 64, 64, 1), (3840, 320, 320, 5), (4096, 4096, 4096, 1), (4004, 100, 64, 1),
         (4020, 372, 64, 1), (4096, 40, 8, 1)]
for H, L, s, ch in GEOMS:
    base = base_of(H)
    for epi in ("mma_split", "mma_bf16", "simt"):
        m = ReductionModel(clone_base(base), ReducerConfig("t", "learned", L=L, s=s, channels=ch, epilogue=epi), output_dim=32)
        opt = FusedAdamW(m.sparse_parameters() + m.reducer_parameters(), lr=1e-3)
        for B in (8, 40, 136):
            x = torch.randn(B, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for rep in range(2):
                U = m.reducer(x)
                torch.autograd.backward(U, torch.randn_like(U))
            opt.step()
            torch.cuda.synchronize()
        print(H, L, s, ch, epi, flush=True)
    for tiling, wg in (((4, 8, 0), 4), ((4, 8, 1), 2), ((2, 8, 0), 4), ((2, 8, 1), 1), ((2, 4, 0), 2), ((2, 4, 1), 1)):
        for mode, wpt in (("separate", 4), ("fused", 4), ("separate", 2), ("fused", 2)):
            m = ReductionModel(clone_base(base), ReducerConfig("t", "learned", L=L, s=s, channels=ch, dp_mode=mode), output_dim=32,
                               dx_tiling=tiling, dx_group_warps=wg, wpt=wpt)
            for B, nonfinite in ((40, False), (136, True)):
                x = torch.randn(B, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                U = m.reducer(x)
                g = torch.randn_like(U)
                if nonfinite:
                    g[0, 0] = float("inf")
                    g[1, g.shape[1] - 1] = float("nan")
                torch.autograd.backward(U, g)
                torch.cuda.synchronize()
        print(H, L, s, ch, tiling, wg, flush=True)
H, G, L, s = 16 * 320, 16, 320, 64
base = base_of(H, G)
for cid, cfg in dict(A2=ReducerConfig("A2", "block_mean", L=s, s=s), A5=ReducerConfig("A5", "block_mean", L=L, s=s),
                     A4=ReducerConfig("A4", "learned", L=L, s=s), TS=ReducerConfig("TS", "learned", L=L, s=s, second=(16, 4, "gelu"))).items():
    m = ReductionModel(clone_base(base), cfg, output_dim=32)
    opt = FusedAdamW(m.sparse_parameters() + m.reducer_parameters(), lr=1e-3)
    x = torch.randn(40, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # sparse and reduction kernels + the compact tiled second stage; the dense torch.matmul readout is not exercised here
    # (cuBLAS initialization itself trips a host-side cuCtxGetLimit report under memcheck, independent of this code)
    U = m.features(x)
    if m.second is not None:
        U = m.second(U)
    U.float().square().mean().backward()
    opt.step()
    torch.cuda.synchronize()
print("WORKLOAD COMPLETE lrd", flush=True)
