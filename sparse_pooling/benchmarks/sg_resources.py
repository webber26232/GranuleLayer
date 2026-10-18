"""Registers / stack / spills of the input-supergroup kernels (cuobjdump of the built library)
and computed sm_86 residency for given S and launch shapes (computed limits, not measured
occupancy; counters are admin-only on this machine).

    python sparse_pooling/benchmarks/sg_resources.py [--S 64] [--filter sg_dx] [--out results/input_supergroups/kernel_resources.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import common  # noqa: F401
from kernel_resources import residency
from sparse_pooling.kernels.cuda_ops import build_library

D = 1024
DW_CFGS = {0: (8, 2, 1, 3, 2), 1: (8, 4, 1, 3, 1), 2: (8, 2, 4, 2, 2), 3: (8, 4, 4, 2, 1)}


def smem_bytes(kind, t, S, dx_group_warps=1):
    m = (D * 2 + 15) // 16 * 16
    if kind == "sg_fwd_kernel":  # <ACT, C, MT, NW>
        return m + S * 16 * t[2] * 2
    if kind == "sg_dx_kernel":  # <ACT, C, MT, NW, SPLIT, EPI>
        bm = 16 * t[2]
        return m + S * bm * 2 + (t[3] // dx_group_warps) * S * bm * 4
    if kind == "sg_dw_kernel":  # <ACT, C, NW, WPW, TS, ST, SPLIT, MINB, EPI>
        return m + t[5] * t[4] * S * 16 * 2
    return 0


def kernels(lib, pattern):
    txt = subprocess.run(["cuobjdump", "-res-usage", str(lib)], capture_output=True, text=True).stdout
    for m in re.finditer(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:\d+ LOCAL:(\d+)", txt):
        name = subprocess.run(["cu++filt", m.group(1)], capture_output=True, text=True).stdout.strip()
        name = re.sub(r"\((int|bool)\)", "", name.removeprefix("void ")).replace("(SgArgs)", "")
        if re.search(pattern, name):
            yield name, int(m.group(2)), int(m.group(3)), int(m.group(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--filter", default=r"^sg_")
    ap.add_argument("--dx-group-warps", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lib = build_library()
    rows = []
    for name, regs, stack, local in kernels(lib, args.filter):
        kind = name.split("<")[0]
        t = [int(v) if v.lstrip("-").isdigit() else (1 if v == "true" else 0)
             for v in re.findall(r"[-\w]+", name.split("<", 1)[1].split(">")[0])]
        threads = 32 * (t[3] if kind in ("sg_fwd_kernel", "sg_dx_kernel") else t[2])
        smem = smem_bytes(kind, t, args.S, args.dx_group_warps)
        ctas, warps, lim = residency(regs, threads, smem) if smem <= 99 * 1024 else (0, 0, "smem > 99 KiB")
        rows.append(dict(kernel=name, regs=regs, stack=stack, local=local, threads=threads, S=args.S,
                         smem=smem, ctas_per_sm=ctas, warps_per_sm=warps, limiter=lim))
        print(f"{name:70s} regs {regs:3d} stack {stack:4d} local {local:4d} smem {smem / 1024:6.1f} KiB "
              f"-> {ctas} CTA/SM {warps} warps ({lim})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
