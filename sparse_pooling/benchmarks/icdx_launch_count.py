"""Kernel launches of one complete training step per execution format and dX variant (results/INTERACTION_COVERAGE_
K16_DX.md section 2.3): the SAME layer / optimizer factories as the timing runs (k64_factory:wide, k64_opt:l1, M 8192,
split dZ, BF16), two warm-up steps, then STEPS steps under torch.profiler (CUPTI activity trace).

Three separate counts per step: adapter BACKEND calls (last_info launches_fwd / _bwd; indexed = 1 / 2), CUDA KERNEL
launches, and memset / memcpy operations.  A trace is COMPLETE only if every counted name occurs a multiple of STEPS
times and every expected layer kernel (kgen fwd / dX / dW per fan-in type) occurs exactly STEPS times; otherwise it is
reported as incomplete (lower bounds), never as an exact total.  Outside every timing session.

    python icdx_launch_count.py --specs "Cover-TopK/16:indexed:legacy,Cover-Mix32:p1#0:grouped:mt4_nw8_wg4" --out X.json
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

STEPS = 4


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
    from torch.profiler import ProfilerActivity, profile

    out = dict(schema="granule.icdx_launch_count/1", profiled_steps=STEPS, graph_seed=a.graph_seed, batch=WC.M,
               provenance=provenance(Path(common.__file__).resolve().parent.parent), runs={})
    for item in a.specs.split(","):
        spec, fmt, dxv = item.rsplit(":", 2)
        f = KJ.factory_of(spec, fmt, a.graph_seed)
        if dxv != "legacy":
            f["dx_variant"] = dxv
        args = SimpleNamespace(input_dim=WC.D, hidden=WC.H, groups=WC.G, C=RC.C, grad_precision="split", factory=f,
                               opt=dict(l1="none"), init_seed=WC.INIT_SEED)
        lay = k64_factory.wide(args, ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed))
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
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(STEPS):
                step()
            torch.cuda.synchronize()
        kern, other = collections.Counter(), collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                n = e.name
                if "#" in n or n.startswith("Optimizer."):
                    continue
                if n.lower().startswith(("memset", "memcpy")):
                    other[n.split(" ")[0]] += 1
                else:
                    kern[n[:110]] += 1
        info = getattr(lay, "last_info", None) or {}
        types = [pk.K for pk in lay.packs] if hasattr(lay, "packs") else ([lay.layout.fan_in] if fmt == "indexed" else [])
        expected = {}
        for K in types:
            for kind in ("kg_fwd_kernel", "kg_dx_kernel", "kg_dw_kernel"):
                expected[f"{kind}<{K}"] = sum(c for nme, c in kern.items() if kind in nme and f"<{K}," in nme)
        missing = [k for k, c in expected.items() if c != STEPS]
        fractional = [k for k, c in list(kern.items()) + list(other.items()) if c % STEPS]
        complete = not missing and not fractional
        bc = dict(fwd=info.get("launches_fwd", 1 if fmt == "indexed" else None),
                  bwd=info.get("launches_bwd", 2 if fmt == "indexed" else None))
        out["runs"][item] = dict(backend_calls_per_step=bc, cuda_kernels_per_step=sum(kern.values()) / STEPS,
                                 memset_memcpy_per_step={k: v / STEPS for k, v in other.items()},
                                 kernels_by_name={k: v / STEPS for k, v in sorted(kern.items(), key=lambda kv: -kv[1])},
                                 expected_layer_kernels={k: v / STEPS for k, v in expected.items()},
                                 complete=complete, missing_expected=missing, fractional=fractional,
                                 label="exact" if complete else "INCOMPLETE trace (lower bounds)",
                                 dx_variant=(lay.graph_stats.get("dx_variant") if hasattr(lay, "graph_stats") else None))
        print(f"{item}: {sum(kern.values()) / STEPS:g} kernels, {sum(other.values()) / STEPS:g} memset / memcpy, "
              f"complete {complete}", flush=True)
        del lay, opt, x, prof
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
