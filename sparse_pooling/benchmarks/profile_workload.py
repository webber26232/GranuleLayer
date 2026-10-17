"""One forward + backward step for profilers (ncu / nsys / torch profiler)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import SparsePoolingLayer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="max")
ap.add_argument("--C", type=int, default=8)
ap.add_argument("--hidden", type=int, default=10_000_000)
ap.add_argument("--batch", type=int, default=8192)
ap.add_argument("--policy", default="auto")
args = ap.parse_args()
pool = {"type": args.pool}
if args.pool == "topk_mean":
    pool["k"] = 8
lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=args.C, pooling=pool, device="cuda",
                         checkpoint_policy=args.policy)
lay.enable_main_grad()
x = torch.randn(args.batch, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
for _ in range(2):  # first iteration compiles / autotunes
    lay(x).float().square().mean().backward()
torch.cuda.synchronize()
print(lay.last_info)
