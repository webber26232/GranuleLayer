"""Small workload over the accepted Cover-TopK/16 path for NVIDIA Compute Sanitizer (run_sanitizer.sh ... cover):
CoverTopK16Layer (indexed K16 / C16 kgen) built from K64 / C256 envelopes at D 1024 (the validated shared-memory
footprint) and D 256, ragged H (partial 16-neuron groups, 175 / 176-neuron-style tail records), the legacy dX launch,
mt4_nw8_wg4 selected through the profile ("auto" with the RTX 3090 scope widened in-process to this small case when
the device is the RTX 3090; explicit otherwise), every other compiled tiling explicitly, token tails B = 8 / 40 /
136 / 512, non-finite upstream gradients, FP32 main_grad accumulation over two backward passes, FusedAdamW updates
(BF16 moments, stochastic rounding) and a packed save / load round trip."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import CoverTopK16Layer  # noqa: E402
from sparse_pooling.backends.profiles import rtx3090  # noqa: E402
from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_  # noqa: E402
from sparse_pooling.optim import FusedAdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

G = 8
n = 0
for D, H in ((1024, 8 * 431 + 4), (256, 8 * 320 + 3)):
    L = KLayout(D, H, G, 64, 256)
    g = torch.Generator(device="cuda").manual_seed(D)
    w = (torch.randn(L.weight_shape, device="cuda", generator=g) / 8).bfloat16()
    b = (0.1 * torch.randn(L.bias_shape, device="cuda", generator=g)).bfloat16()
    zero_padding_(L, w, b)
    lay = CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G,
                                         connectivity=ConnectivityPlan(64, 2, seed=1), normalization="energy")
    lay.enable_main_grad()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-3)
    c = lay.kgen_dx_context(40)
    is3090 = rtx3090.is_rtx3090(lay._runtime().caps)[0]
    if is3090:  # route WG4 through the profile / dispatch path for this small case
        rtx3090.KGEN_DX_AUTO = (dict(rtx3090.KGEN_DX_AUTO[0], D=c.D, G=c.G, H=c.H, R=c.R, batches=(8, 40, 136, 512)),)
    for v in ("legacy", "auto" if is3090 else "mt4_nw8_wg4", "mt4_nw8_wg8", "mt2_nw8_wg2", "mt2_nw4_wg1",
              "mt4_nw4_wg2"):
        lay.dx_variant = v
        for B, nonfinite in ((8, False), (40, True), (136, False), (512, False)):
            x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for rep in range(2):  # main_grad accumulates over two backward passes
                y = lay(x)
                gy = torch.randn_like(y.float())
                if nonfinite and rep == 0:
                    gy[0, 0] = float("inf")
                    gy[1, 1] = float("nan")
                (y.float() * gy).sum().backward()
            if not nonfinite:
                opt.step()
            else:
                lay.zero_main_grad()
            torch.cuda.synchronize()
            n += 1
        print(D, v, lay.last_info["kgen_dx"]["variant"]["name"], flush=True)
    with tempfile.TemporaryDirectory() as td:
        lay.save_packed(os.path.join(td, "c.pt"))
        lay2 = CoverTopK16Layer.load_packed(os.path.join(td, "c.pt"))
        lay2.dx_variant = "legacy"
        x = torch.randn(40, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        (lay2(x).float() * torch.randn(40, G, device="cuda")).sum().backward()
        torch.cuda.synchronize()
        n += 1
print(f"{n} Cover-TopK/16 forward + backward (+ optimizer) calls", flush=True)
print("WORKLOAD COMPLETE cover", flush=True)
