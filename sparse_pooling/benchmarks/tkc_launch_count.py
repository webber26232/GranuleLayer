"""Actual GPU kernel-launch totals of one complete training step (forward, backward, optimizer) per execution format
(results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md section 2.2): the SAME layer / optimizer factories as the timing runs
(k64_factory:wide, k64_opt:l1, M 8192, split dZ, BF16), two warm-up steps, then one step under torch.profiler.
Counted: every CUDA kernel / memset / memcpy event of the step, grouped by name; separate from the adapter's backend
calls (``last_info.launches_fwd / _bwd`` of the timing records).  Never used for timing (profiler overhead).

    python tkc_launch_count.py --specs Mix-TopK#0:grouped,TopK/16:indexed,... --graph-seed 0 --out X.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import common  # noqa: F401
import kwa_jobs as KJ
import rsn_cases as RC
import wps_cases as WC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import k64_factory
    import k64_opt
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    out = dict(schema="granule.tkc_launch_count/1", graph_seed=a.graph_seed, batch=WC.M,
               provenance=provenance(Path(common.__file__).resolve().parent.parent), runs={})
    for item in a.specs.split(","):
        spec, fmt = item.rsplit(":", 1)
        args = SimpleNamespace(input_dim=WC.D, hidden=WC.H, groups=WC.G, C=RC.C, grad_precision="split",
                               factory=KJ.factory_of(spec, fmt, a.graph_seed), opt=dict(l1="none"), init_seed=WC.INIT_SEED)
        conn = ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed)
        lay = k64_factory.wide(args, conn)
        lay.enable_main_grad()
        opt = k64_opt.l1(args, lay)
        gx = torch.Generator(device="cuda").manual_seed(WC.INPUT_SEED)
        x = torch.randn(WC.M, WC.D, device="cuda", generator=gx).to(torch.bfloat16).requires_grad_(True)

        def step():
            y = lay(x)
            y.float().square().mean().backward()
            opt.step()
            x.grad = None

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            step()
            torch.cuda.synchronize()
        kern, other = collections.Counter(), collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                n = e.name
                if n.lower().startswith(("memset", "memcpy")):
                    other[n.split(" ")[0]] += 1
                else:
                    kern[n[:90]] += 1
        info = getattr(lay, "last_info", None) or {}
        out["runs"][f"{spec}:{fmt}"] = dict(kernels_total=sum(kern.values()), memset_memcpy=dict(other),
                                            kernels_by_name=dict(kern.most_common()),
                                            backend_calls=dict(fwd=info.get("launches_fwd"), bwd=info.get("launches_bwd")))
        print(f"{spec} {fmt}: {sum(kern.values())} kernels, {sum(other.values())} memset / memcpy", flush=True)
        del lay, opt, x, prof
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
