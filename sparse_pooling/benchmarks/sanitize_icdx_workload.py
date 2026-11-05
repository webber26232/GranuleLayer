"""Small workload over the paths new in the interaction-coverage / K16 dX round for NVIDIA Compute Sanitizer
(run_sanitizer.sh ... icdx): every compiled K16 dX tiling variant (legacy entry and the seven screened variants; WG 1
private and WG > 1 shared CAS copies) on the indexed K16 / C16 path of Balanced-TopK/16 and Cover-TopK/16, logical
C256 with n175 / n176 record tails (partial row groups), D 1024 (the actual shared-memory footprint) and D 256; the
grouped Cover-Mix32 p1 / p2 (K32 / K16 / K8) layer with legacy and variant K16 dX; split / BF16 dZ, token tails
(B 8, 40, 136), non-finite upstream gradients, main_grad accumulation over two backward passes, L1AdamW update."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling.config import PhysicalPlan  # noqa: E402
from sparse_pooling.experimental import local_supports as LS  # noqa: E402
from sparse_pooling.experimental import mix32_supports as M32  # noqa: E402
from sparse_pooling.experimental.kgx_grouped import lower_mixed  # noqa: E402
from sparse_pooling.experimental.kgx_layer import KgxLayer  # noqa: E402
from sparse_pooling.experimental.kgx_onetile import lower_local  # noqa: E402
from sparse_pooling.experimental.l1 import L1AdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

G = 8
H = G * 431 + 4  # outputs 0..3: 256 + 176 neurons, 4..7: 256 + 175
VARIANTS = ["legacy", "mt4_nw8_wg2", "mt4_nw8_wg4", "mt4_nw8_wg8", "mt2_nw8_wg1", "mt2_nw8_wg2", "mt4_nw4_wg1",
            "mt4_nw4_wg2", "mt2_nw4_wg1"]
n = 0


def call(lay, D, B, nonfinite=False, steps=1):
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


for D, prec, B, nonfinite in ((1024, "split", 40, False), (256, "bf16", 8, False), (256, "split", 136, True)):
    base = KgxLayer(D, H, G, 256, ConnectivityPlan(64, 2, seed=1), variant="idf", tile=128, device="cuda", seed=2,
                    plan=PhysicalPlan(grad_precision=prec), keep_indices=False)
    with torch.no_grad():
        base.bias.normal_(0, 0.2)
    L = base.layout
    Q = base.sg_feat.shape[0]

    def masked(em):
        return KgxLayer(D, H, G, 256, base.connectivity, variant="idf", tile=128, device="cuda",
                        plan=PhysicalPlan(grad_precision=prec),
                        weights=(base.weight.detach() * em, base.bias.detach().clone()), keep_indices=False)

    for pol in ("Balanced-TopK", "Cover-TopK"):
        s = LS.build(pol, 16, L, base.kg_buckets, w=base.weight.detach(), device="cuda", Q=Q)
        lo = lower_local(masked(s.element_mask(L)), s)
        for v in (VARIANTS if pol == "Balanced-TopK" else ["legacy", "mt4_nw8_wg4", "mt2_nw4_wg1"]):
            lo.dx_variant = v
            call(lo, D, B, nonfinite, steps=2)
        if B == 40:
            o = L1AdamW([lo.weight, lo.bias], lr=1e-3)
            o.step()
            torch.cuda.synchronize()
            n += 1
        del lo
    for p in (1, 2):
        s = M32.build("Cover-Mix32", p, L, base.kg_buckets, w=base.weight.detach(), device="cuda", Q=Q)
        gl = lower_mixed(masked(s.element_mask(L)), s)
        for v in ("legacy", "mt4_nw8_wg4", "mt2_nw8_wg1"):
            gl.dx_variant = v
            call(gl, D, B, nonfinite, steps=2)
        if B == 40:
            o = L1AdamW([gl.weight, gl.bias], lr=1e-3)
            o.step()
            torch.cuda.synchronize()
            n += 1
        del gl
torch.cuda.synchronize()
print(f"{n} icdx forward + backward / optimizer calls", flush=True)
print("WORKLOAD COMPLETE icdx", flush=True)
