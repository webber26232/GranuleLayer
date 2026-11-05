"""Profile of the restored Cover-TopK/16 indexed K16 baseline and of every screened K16 dX tiling, OUTSIDE any timing
session (results/INTERACTION_COVERAGE_K16_DX.md section 5.1).

Recorded (only what the installed tools permit; nothing inferred is labelled as measured):

* compiled attributes per tiling (cudaFuncGetAttributes: registers / thread, local bytes / thread = stack or spill,
  static shared memory), dynamic shared memory, theoretical active CTAs / SM (cudaOccupancyMaxActiveBlocksPerMultiprocessor)
  and the resulting theoretical warps / SM, the device limits, the grid (CTAs, waves);
* per-kernel GPU durations from CUPTI activity tracing (torch.profiler; kernel trace needs no counter permission):
  forward / dX / dW / optimizer kernels of complete training steps at M 8192 (2 warm-up + 6 profiled steps per tiling);
* analytic work per dX launch (staged X bytes, weight bytes, FP32 global atomics issued, shared-memory slab bytes) --
  ESTIMATES from the kernel structure, labelled as such;
* Nsight Compute / Nsight Systems availability: the versions and the exact result of one counter-collection attempt
  (GPU performance counters require administrator rights unless NVreg_RestrictProfilingToAdminUsers=0).

    python icdx_profile.py --support Cover-TopK/16 --out X.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import common  # noqa: F401
import kwa_jobs as KJ
import rsn_cases as RC
import wps_cases as WC

STEPS = 6


def tool_probe() -> dict:
    out = {}
    for t in ("ncu", "nsys"):
        p = shutil.which(t) or f"/usr/local/cuda/bin/{t}"
        try:
            v = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
            out[t] = dict(path=p, version=v[-1] if v else "")
        except Exception as e:  # noqa: BLE001
            out[t] = dict(path=p, error=repr(e))
    try:
        r = Path("/proc/driver/nvidia/params").read_text()
        out["RmProfilingAdminOnly"] = [ln for ln in r.splitlines() if "Profiling" in ln]
    except OSError:
        pass
    if "path" in out.get("ncu", {}):
        code = "import torch; a=torch.ones(1<<20, device='cuda'); (a*2).sum().item()"
        r = subprocess.run([out["ncu"]["path"], "--metrics", "sm__cycles_elapsed.avg", sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=600)
        txt = (r.stdout + r.stderr)
        out["ncu_counter_attempt"] = dict(exit=r.returncode, permission_denied="ERR_NVGPUCTRPERM" in txt,
                                          tail=txt.strip().splitlines()[-6:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", default="Cover-TopK/16")
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import k64_factory
    import k64_opt
    from sparse_pooling.backends.capabilities import query
    from sparse_pooling.backends.dispatch import smem_limit
    from sparse_pooling.experimental import kgen_dx_variants as DXV
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    res = dict(schema="granule.icdx_profile/1", support=a.support, graph_seed=a.graph_seed, batch=WC.M,
               provenance=provenance(Path(common.__file__).resolve().parent.parent), tools=tool_probe())
    args = SimpleNamespace(input_dim=WC.D, hidden=WC.H, groups=WC.G, C=RC.C, grad_precision="split",
                           factory=KJ.factory_of(a.support, "indexed", a.graph_seed), opt=dict(l1="none"),
                           init_seed=WC.INIT_SEED)
    lay = k64_factory.wide(args, ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed))
    lay.enable_main_grad()
    opt = k64_opt.l1(args, lay)
    caps = query(torch.device("cuda"))
    rt = lay._runtime()
    be = rt.backend
    res["device"] = dict(name=caps.name, cc=list(caps.cc), sm_count=caps.sm_count,
                         smem_per_block_optin=caps.smem_per_block_optin, smem_limit_used=smem_limit(caps),
                         smem_per_sm=torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
                         if hasattr(torch.cuda.get_device_properties(0), "shared_memory_per_multiprocessor") else None,
                         regs_per_sm=65536, max_warps_per_sm=48)
    res["work_lists"] = dict(rt.sg.stats)
    n_chunks = rt.sg.chunks.shape[0]
    items = rt.sg.items.cpu()
    blocks = int(items[:, 2].sum())
    res["build"] = be.build_info()
    gx = torch.Generator(device="cuda").manual_seed(WC.INPUT_SEED)
    x = torch.randn(WC.M, WC.D, device="cuda", generator=gx).to(torch.bfloat16).requires_grad_(True)
    w0, b0 = lay.weight.detach().clone(), lay.bias.detach().clone()

    def step():
        y = lay(x)
        y.float().square().mean().backward()
        opt.step()
        x.grad = None

    variants = ["legacy"] + [f"mt{m}_nw{n}_wg{g}" for m, n, g in DXV.SCREEN if (m, n, g) != (4, 8, 2)]
    res["tilings"] = {}
    from torch.profiler import ProfilerActivity, profile

    for v in variants:
        lay.dx_variant = v
        dv = lay.dx_variant_resolved()
        smem = DXV.smem_bytes(WC.D, WC.S, dv.mt, dv.nw, dv.wg)
        at = be.dxv_attributes(dv.mt, dv.nw, 0, 1, 1, smem)
        n_tiles = math.ceil(WC.M / (16 * dv.mt))
        ctas = n_chunks * n_tiles
        act = at.get("active_ctas_per_sm", -1)
        bm = 16 * dv.mt
        with torch.no_grad():
            lay.weight.copy_(w0)
            lay.bias.copy_(b0)
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(STEPS):
                step()
            torch.cuda.synchronize()
        dur = collections.defaultdict(float)
        cnt = collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA and "#" not in e.name:
                key = ("kg_dx" if "kg_dx_kernel" in e.name else "kg_fwd" if "kg_fwd_kernel" in e.name else
                       "kg_dw" if "kg_dw_kernel" in e.name else "optimizer" if "l1" in e.name.lower() or "adam" in
                       e.name.lower() else "other")
                dur[key] += e.device_time / 1000.0  # ms
                cnt[key] += 1
        res["tilings"][v] = dict(
            resolved=dv.describe(), dynamic_smem_bytes=smem, attributes=at, ctas=ctas, token_tiles=n_tiles,
            chunks=n_chunks, theoretical_warps_per_sm=(act * dv.nw) if act and act > 0 else None,
            waves=(ctas / (act * caps.sm_count)) if act and act > 0 else None,
            kernel_ms_per_step={k: dur[k] / STEPS for k in dur}, kernel_count_per_step={k: cnt[k] / STEPS for k in cnt},
            analytic_per_dx_launch=dict(
                staged_x_bytes=ctas * WC.S * bm * 2, weight_bytes_read=n_tiles * blocks * 16 * 16 * 2,
                slab_bytes_per_cta=(dv.nw // dv.wg) * WC.S * bm * 4, global_fp32_atomics_upper=ctas * WC.S * bm,
                shared_cas_copies=dv.nw // dv.wg, note="estimates from the kernel structure (not counters)"))
        print(f"{v}: dX {res['tilings'][v]['kernel_ms_per_step'].get('kg_dx', float('nan')):.1f} ms, regs "
              f"{at.get('registers_per_thread')}, local {at.get('local_bytes_per_thread')} B, CTAs/SM {act}", flush=True)
    res["host"] = dict(note="host enqueue / step wall are recorded by the timing jobs (bench_supergroups); support "
                            "construction cost by icdx_construct")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
