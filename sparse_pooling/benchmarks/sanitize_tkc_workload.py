"""Small workload over the supports new in the TopK/16 / Cover-TopK round for NVIDIA Compute Sanitizer
(run_sanitizer.sh ... tkc): Cover-TopK (TopK + per-record coverage repair) and TopK supports at k 8 / 16 on the
UNCHANGED indexed kgen K8 / K16 path (kgx_onetile.lower_local), logical C256 with n175 / n176 record tails (11 row
groups, the last partial), anisotropic weights (many repaired inputs), split / BF16 dZ, token tails (B = 8, 40, 136),
non-finite upstream gradients, main_grad accumulation over two backward passes and the L1AdamW update."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling.config import PhysicalPlan  # noqa: E402
from sparse_pooling.experimental import local_supports as LS  # noqa: E402
from sparse_pooling.experimental.kgx_layer import KgxLayer  # noqa: E402
from sparse_pooling.experimental.kgx_onetile import lower_local  # noqa: E402
from sparse_pooling.experimental.l1 import L1AdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

D, G = 256, 8
H = G * 431 + 4  # outputs 0..3: 256 + 176 neurons, 4..7: 256 + 175
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


for prec, B, nonfinite in (("split", 40, False), ("bf16", 8, False), ("split", 136, True)):
    base = KgxLayer(D, H, G, 256, ConnectivityPlan(64, 2, seed=1), variant="idf", tile=128, device="cuda", seed=2,
                    plan=PhysicalPlan(grad_precision=prec), keep_indices=False)
    with torch.no_grad():
        base.bias.normal_(0, 0.2)
        sc = torch.ones(64, device="cuda")
        sc[:12] = 6.0  # TopK concentrates on 12 coordinates: the repair swaps many inputs
        base.weight.mul_(sc.to(base.weight.dtype))
    L = base.layout
    Q = base.sg_feat.shape[0]

    def masked(em):
        return KgxLayer(D, H, G, 256, base.connectivity, variant="idf", tile=128, device="cuda",
                        plan=PhysicalPlan(grad_precision=prec),
                        weights=(base.weight.detach() * em, base.bias.detach().clone()), keep_indices=False)

    for k in (8, 16):
        for pol in ("Cover-TopK", "TopK"):
            s = LS.build(pol, k, L, base.kg_buckets, w=base.weight.detach(), device="cuda", Q=Q)
            if pol == "Cover-TopK":
                assert s.stats["repair"]["swaps"] > 0
            lo = lower_local(masked(s.element_mask(L)), s)
            call(lo, B, nonfinite, steps=2)
            if B == 40:
                o = L1AdamW([lo.weight, lo.bias], lr=1e-3)
                o.step()
                torch.cuda.synchronize()
                n += 1
            del lo
torch.cuda.synchronize()
print(f"{n} tkc forward + backward / optimizer calls", flush=True)
print("WORKLOAD COMPLETE tkc", flush=True)
