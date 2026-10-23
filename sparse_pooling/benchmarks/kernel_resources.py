"""Registers / spills (from the compiled library) and theoretical residency of the
native forward, k_dx, k_dw and k_dw_idx kernels on sm_86.

Residency uses the CUDA occupancy rules for compute capability 8.6: 64K
registers per SM allocated per warp in units of 256, 100 KiB shared memory per
SM with 1 KiB reserved per CTA, at most 48 warps and 16 CTAs per SM.  These are
computed limits, not measured achieved occupancy (counters are admin-only).

    python sparse_pooling/benchmarks/kernel_resources.py [--out results/traffic/kernel_resources.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import common  # noqa: F401  (puts the repository root on sys.path)
from sparse_pooling.kernels.cuda_ops import build_library

D = 1024
POOLS = {0: "max", 2: "mean", 3: "lme", 4: "pow", 5: "sw"}


def smem_bytes(kind, t):
    if kind == "fwd_mma_kernel":  # <POOL, ACT, C, MT, NW, TRACK, PI, PACKED>
        mt, nw = t[3], t[4]
        return max(D * 16 * mt * 2, 4 * nw * 16 * mt * 4)
    if kind == "k_dx":
        return D * 16 * 6
    if kind == "k_dw":
        return D * 16 * 2 * 2
    if kind == "k_dw_idx":  # <POOL, ACT, C, NW, WPW, ST, SPLIT, PI, MINB>
        C, nw, wpw, st = t[2], t[3], t[4], t[5]
        ub = 1 if C // 8 >= 4 else 4 // (C // 8)
        return nw * st * wpw * ub * 8 * 16 * 2
    return 0


def residency(regs, threads, smem):
    warps = threads // 32
    reg_per_warp = -(-regs * 32 // 256) * 256
    by_regs = 65536 // (warps * reg_per_warp)
    by_smem = 102400 // (smem + 1024)
    by_warps = 48 // warps
    ctas = min(by_regs, by_smem, by_warps, 16)
    limiter = min((by_regs, "registers"), (by_smem, "shared memory"), (by_warps, "warps"))[1]
    return ctas, ctas * warps, limiter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=r"^(k_dw_idx|k_dw|k_dx)<2, 1, 64|^fwd_mma_kernel<2, 1, 64, 2, 16")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lib = build_library()
    txt = subprocess.run(["cuobjdump", "-res-usage", str(lib)], capture_output=True, text=True).stdout
    rows = []
    for m in re.finditer(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:\d+ LOCAL:(\d+)", txt):
        name = subprocess.run(["cu++filt", m.group(1)], capture_output=True, text=True).stdout.strip()
        name = re.sub(r"\(int\)|\(bool\)|void |\(BwdArgs\)|\(FwdArgs\)", "", name)
        if not re.search(args.filter, name):
            continue
        kind = name.split("<")[0]
        t = [int(v) for v in name[name.index("<") + 1:-1].split(",")]
        nw = t[4] if kind == "fwd_mma_kernel" else t[3]
        smem = smem_bytes(kind, t)
        ctas, warps, lim = residency(int(m.group(2)), nw * 32, smem)
        rows.append({"kernel": name, "regs": int(m.group(2)), "stack": int(m.group(3)), "local": int(m.group(4)),
                     "threads": nw * 32, "smem_bytes": smem, "ctas_per_sm": ctas, "warps_per_sm": warps,
                     "limiter": lim})
    for r in sorted(rows, key=lambda r: r["kernel"]):
        print(f"{r['kernel']:46s} regs {r['regs']:3d} stack {r['stack']:2d}  thr {r['threads']:3d} "
              f"smem {r['smem_bytes'] / 1024:5.1f} KiB -> {r['ctas_per_sm']} CTA/SM ({r['warps_per_sm']:2d} warps, "
              f"{r['limiter']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
