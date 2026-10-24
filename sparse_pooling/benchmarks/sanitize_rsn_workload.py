"""Small workload over the indexed local-support path of the random-support study for NVIDIA Compute Sanitizer
(run_sanitizer.sh ... rsn): every local support policy (FP-mag, FP-balanced, RP-record, RP-row, BP-row) at fan-in
16 and 8, lowered to the existing kgen K16 / K8, C16 kernels with explicit (non-contiguous, sorted) local records
(experimental/kgx_onetile.lower_local), at logical C = 64 / 128 / 256 with partial row groups and partial logical
tails, split / BF16 dZ, token tails (B = 8, 40, 136), non-finite upstream gradients, main_grad accumulation and the
L1AdamW update; the fixed-partition supports also on the index-free 16 x 16 tile kernels (kgx_wide.local_as_maskset)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling.config import PhysicalPlan  # noqa: E402
from sparse_pooling.experimental import local_supports as LS  # noqa: E402
from sparse_pooling.experimental.kgx_layer import KgxLayer  # noqa: E402
from sparse_pooling.experimental.kgx_onetile import lower_local  # noqa: E402
from sparse_pooling.experimental.kgx_wide import KgxWideLayer, local_as_maskset  # noqa: E402
from sparse_pooling.experimental.l1 import L1AdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

D, G = 256, 8
n = 0


def call(lay, B, nonfinite=False, steps=1):
    global n
    lay.enable_main_grad()
    for _ in range(steps):
        x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = lay(x)
        g = torch.randn_like(y)
        if nonfinite:
            g[0, 0] = float("inf")
            g[min(1, B - 1), 1] = float("nan")
        (y.float() * g).sum().backward()
    torch.cuda.synchronize()
    n += 1


for C, H, R in ((256, 8 * 300 + 5, 2), (128, 3003, 2), (64, 2999, 1)):
    for prec, B, nonfinite in (("split", 40, False), ("bf16", 8, False), ("split", 136, True)):
        base = KgxLayer(D, H, G, C, ConnectivityPlan(64, R, seed=1), variant="idf", tile=128, device="cuda", seed=2,
                        plan=PhysicalPlan(grad_precision=prec), keep_indices=False)
        with torch.no_grad():
            base.bias.normal_(0, 0.2)
        L = base.layout
        for k in (16, 8):
            for pol in LS.POLICIES:
                s = LS.build(pol, k, L, base.kg_buckets, w=base.weight.detach(), graph_seed=1, layout_seed=3,
                             device="cuda")
                em = s.element_mask(L)
                dm = KgxLayer(D, H, G, C, base.connectivity, variant="idf", tile=128, device="cuda",
                              plan=PhysicalPlan(grad_precision=prec),
                              weights=(base.weight.detach() * em, base.bias.detach().clone()), keep_indices=False)
                lo = lower_local(dm, s)
                call(lo, B, nonfinite, steps=2)
                if B == 40:
                    o = L1AdamW([lo.weight, lo.bias], lr=1e-3)
                    o.step()
                    torch.cuda.synchronize()
                    n += 1
                if pol.startswith("FP") and k == 16:
                    call(KgxWideLayer(dm, local_as_maskset(s)), B, nonfinite)
                del lo, dm
torch.cuda.synchronize()
print(f"{n} rsn forward + backward / optimizer calls", flush=True)
print("WORKLOAD COMPLETE rsn", flush=True)
