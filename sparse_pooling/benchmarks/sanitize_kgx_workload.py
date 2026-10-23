"""Small workload over the EXPERIMENTAL K = 64 kgx kernels (bundle "kgx") and the L1AdamW update for NVIDIA
Compute Sanitizer (run_sanitizer.sh ... kgx): every dense variant (direct with S = 64 / 128, idf_rmw, idf; the
kgen / lowered reference routes), physical tiles 64 / 128 / 256 incl. split large-C items (C = 512, 2048),
partial logical tails, split / BF16, token tails (B = 8, 40, 136), private and shared (CAS) dX slab copies,
non-finite coefficients; tile-sparse in both item modes (standard / runs of kept groups) and CSR with static
dX windows, bias-only rows, all-zero and all-one masks; L1AdamW none / coupled / prox / masked."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling.backends.cuda import variants as V  # noqa: E402
from sparse_pooling.backends.dispatch import smem_limit  # noqa: E402
from sparse_pooling.config import PhysicalPlan  # noqa: E402
from sparse_pooling.experimental import kgx_sparse as SP  # noqa: E402
from sparse_pooling.experimental.kgx_layer import KgxLayer  # noqa: E402
from sparse_pooling.experimental.l1 import L1AdamW, apply_mask_  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

D, G = 256, 8
n = 0


def call(lay, B, nonfinite=False):
    global n
    lay.enable_main_grad()
    x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = lay(x)
    g = torch.randn_like(y)
    if nonfinite:
        g[0, 0] = float("inf")
        g[min(1, B - 1), 1] = float("nan")
    (y.float() * g).sum().backward()
    torch.cuda.synchronize()
    n += 1


for C, S, R, H, tile in ((64, 64, 2, 3001, 64), (512, 64, 2, 9001, 128), (2048, 64, 1, 9001, 256), (256, 128, 2, 6000, 128),
                         (1024, 128, 1, 9000, 64)):
    for prec, B, nonfinite in (("split", 40, False), ("bf16", 8, False), ("split", 136, True)):
        lay = KgxLayer(D, H, G, C, ConnectivityPlan(S, R, seed=1), variant="direct", tile=tile, device="cuda", seed=2,
                       plan=PhysicalPlan(grad_precision=prec), keep_indices=True)
        with torch.no_grad():
            lay.bias.normal_(0, 0.1)
        for v in lay.bench_variants():
            lay.select_variant(v)
            if v in ("direct", "idf_rmw"):
                rt = lay._runtime()
                for wg in (1, 2, rt.cfg["dx_warps"]):
                    if V.kgx_smem("dx", v, D, S, rt.cfg, wg) > smem_limit(rt.caps):
                        continue
                    rt.dx_group_warps = wg
                    call(lay, B, nonfinite)
            else:
                call(lay, B, nonfinite)
# sparse formats
base = KgxLayer(D, 9001, G, 128, ConnectivityPlan(64, 2, seed=1), variant="idf", tile=64, device="cuda", seed=2,
                keep_indices=False)
with torch.no_grad():
    base.bias.normal_(0, 0.2)
valid = SP.valid_slots(base)
w0 = base.weight.detach().clone()
for dens in (0.0, 0.05, 0.6, 1.0):
    with torch.no_grad():
        base.weight.copy_(w0)
    tm = SP.tile_mask(base.weight.detach(), valid, dens)
    em = SP.tiles_to_elements(tm, valid)
    with torch.no_grad():
        base.weight.mul_(em.bfloat16())
    for B in (8, 136):
        call(SP.KgxTileLayer(base, tm), B)
        call(SP.KgxCsrLayer(base, SP.random_mask(valid, dens, 3) & em if dens < 1 else em), B)
# L1AdamW
p = torch.nn.Parameter(torch.randn(100_003, device="cuda").bfloat16())
for mode in ("none", "coupled", "prox"):
    p.main_grad = torch.randn(100_003, device="cuda")
    o = L1AdamW([p], lr=1e-3, l1=mode, mu=1e-3)
    o.step()
    n += 1
mask = (torch.rand(100_003, device="cuda") < 0.3).to(torch.uint8)
apply_mask_(o, p, mask)
p.main_grad.normal_()
o.step()
torch.cuda.synchronize()
print(f"{n} kgx forward + backward / optimizer calls", flush=True)
print("WORKLOAD COMPLETE kgx", flush=True)
