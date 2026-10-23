"""Confirm tensor-core usage from machine code.

* native CUDA kernels: count HMMA / LDSM / MOVM / ATOMS / MUFU per kernel in
  the SASS of the built library (cuobjdump), plus registers / spills;
* Triton kernels: count mma.sync instructions in the generated PTX.

    python sparse_pooling/benchmarks/inspect_sass.py
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys

import torch

from common import save
from sparse_pooling.kernels.cuda_ops import build_library

PATTERNS = ["HMMA.1688.F32.BF16", "HMMA.16816.F32.BF16", "LDSM.16.MT88", "LDSM.16.M88", "MOVM", "ATOMS.CAST.SPIN",
            "ATOMS.CAST.SPIN.64", "RED.E.ADD.F32", "MUFU.EX2", "LDG.E.128", "LDG.E.64", "LDGSTS", "BAR.SYNC"]


def sass_counts(obj):
    out = subprocess.run(["cuobjdump", "-sass", str(obj)], capture_output=True, text=True).stdout
    res = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            cur = m.group(1)
            res[cur] = collections.Counter()
            continue
        if cur is None:
            continue
        for p in PATTERNS:
            if re.search(r"\b" + re.escape(p) + r"\b", line):
                res[cur][p] += 1
    usage = subprocess.run(["cuobjdump", "-res-usage", str(obj)], capture_output=True, text=True).stdout
    regs = {}
    fn = None
    for line in usage.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            fn = m.group(1)
        m = re.search(r"REG:(\d+) STACK:(\d+).*LOCAL:(\d+)", line)
        if m and fn:
            regs[fn] = tuple(int(x) for x in m.groups())
    return res, regs


def demangle(n):
    try:
        return subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip()
    except Exception:
        return n


def triton_mma():
    import triton

    from sparse_pooling.config import LevelSpec, PhysicalPlan
    from sparse_pooling.kernels import ops
    from sparse_pooling.layout import SparseLayout, init_parameters

    L = SparseLayout(64, 4096, 8, 8, 8, True)
    idx, w, b = init_parameters(L, "cuda")
    x = torch.randn(64, 64, device="cuda").bfloat16()
    units = ops.make_units(L, None, "cuda")
    plan = PhysicalPlan(phys_splits=1, block_m=64)
    xo, sf, sb = ops.input_operand(x, plan)
    ops.fused_forward_states(xo, sf, sb, 64, idx, w, b, L.counts_tensor("cuda"), L, LevelSpec("max"), units, plan,
                             False, 1)
    from sparse_pooling.kernels import triton_fused as tk

    found = {}
    for kern in (tk.fwd_pool_kernel,):
        for dev_cache in kern.device_caches.values():
            for k in dev_cache[0].values():
                ptx = k.asm.get("ptx", "")
                found[kern.fn.__name__] = sorted({l.strip().split()[0] for l in ptx.splitlines() if "mma.sync" in l})
    return found


def main():
    lib = build_library()  # the production library of the current device (its own build identity)
    report = {}
    for obj in [lib]:
        counts, regs = sass_counts(obj)
        for fn, c in counts.items():
            name = demangle(fn)
            short = name.split("(")[0]
            if not any(k in short for k in ("fwd_mma_kernel", "fwd_topk_kernel", "k_dx", "k_dw")):
                continue
            r = regs.get(fn, (None, None, None))
            report[name] = {"counts": dict(c), "regs": r[0], "stack": r[1], "local": r[2]}
    # print a compact sample (one instantiation per family)
    shown = set()
    for name, rec in sorted(report.items()):
        fam = name.split("<")[0]
        key = (fam, "8," in name.split("<")[1][:12] if "<" in name else "")
        if key in shown:
            continue
        shown.add(key)
        print(f"{name[:110]}\n    regs={rec['regs']} stack={rec['stack']} {rec['counts']}")
    tri = triton_mma()
    print("Triton PTX mma instructions:", tri)
    save("sass_inspection", {"cuda": report, "triton_ptx_mma": tri})


if __name__ == "__main__":
    main()
