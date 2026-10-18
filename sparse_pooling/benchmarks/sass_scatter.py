"""Static SASS of the grouped C = 8 dX hot loops (baseline sg_dx_kernel and the scatter variants
of sparse_pool_sg_dxs.cuh): instruction classes per loop body, shared-memory access widths,
and an excerpt around the slab read-modify-write.  Static counts of one loop-body iteration
(not executed counts); a kernel holds two bodies (specialised epilogue, generic fallback):
the first HMMA loop in address order is the specialised one.

    python sass_scatter.py --kernel 'sg_dx_kernel<1, 8, 4, 8, 1, 2>' \\
        --kernel 'sgx_dx_kernel<1, 8, 4, 8, 1, 2, 2, 1>' --out results/c8_dx_scatter/sass.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess

import common  # noqa: F401
from sass_loops import classify, loops, parse
from sparse_pooling.kernels.cuda_ops import build_library


def mangle(name: str) -> str:
    base, args = name.split("<", 1)
    vals = [v.strip() for v in args.rstrip(">").split(",")]
    kinds = {"sg_dx_kernel": "iiiibi", "sgx_dx_kernel": "iiiibiii"}[base]
    enc = "".join(f"Lb{v}E" if k == "b" else f"Li{v}E" for k, v in zip(kinds, vals))
    return f"_Z{len(base)}{base}I{enc}Ev6SgArgs"


def sass_of(lib, mangled: str) -> str:
    out = subprocess.run(["cuobjdump", "-sass", "-fun", mangled, str(lib)], capture_output=True, text=True).stdout
    if "Function" not in out:
        raise SystemExit(f"{mangled} not found")
    return out


def shared_widths(body):
    c = collections.Counter()
    for _, op, _ in body:
        b = op.split(".")[0]
        if b in ("LDS", "STS", "ATOMS", "LDSM"):
            w = re.search(r"\.(64|128|U8|U16|S8|S16)\b", op)
            c[f"{b}.{w.group(1) if w else '32'}" if b != "LDSM" else op] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", required=True)
    ap.add_argument("--excerpt", type=int, default=0, help="print N instructions around the first STS of the hot loop")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lib = build_library()
    rep = {}
    for k in args.kernel:
        txt = sass_of(lib, mangle(k))
        ins = parse(txt)
        hot = [(lo, hi, body) for lo, hi, body in loops(ins) if any(op.startswith("HMMA") for _, op, _ in body)]
        # innermost HMMA loops (no other HMMA loop strictly inside)
        inner = [h for h in hot if not any(o is not h and h[0] <= o[0] and o[1] <= h[1] and o[1] - o[0] < h[1] - h[0]
                                           for o in hot)]
        out = {"total_instructions": len(ins), "loops": []}
        for lo, hi, body in inner:
            c = collections.Counter(classify(op) for _, op, _ in body)
            out["loops"].append({"start": hex(lo), "end": hex(hi), "instructions": len(body), "classes": dict(c),
                                 "shared": shared_widths(body)})
        rep[k] = out
        print(f"== {k} ({mangle(k)}): {len(ins)} instructions")
        for L in out["loops"]:
            top = ", ".join(f"{n} {v}" for n, v in sorted(L["classes"].items(), key=lambda kv: -kv[1]))
            print(f"  loop {L['start']}-{L['end']}: {L['instructions']} instr; shared {L['shared']}\n    {top}")
        if args.excerpt and inner:
            lo, hi, body = inner[0]
            lines = [ln for ln in txt.splitlines() if re.search(r"/\*([0-9a-f]{4,})\*/", ln)]
            sel = [ln for ln in lines if lo <= int(re.search(r"/\*([0-9a-f]{4,})\*/", ln).group(1), 16) <= hi]
            idx = next((i for i, ln in enumerate(sel) if re.search(r"\bSTS", ln)), 0)
            a, b = max(0, idx - args.excerpt), idx + args.excerpt // 2
            out["excerpt"] = [re.sub(r"\s+", " ", ln.split(";")[0]).strip() + ";" for ln in sel[a:b]]
            print("\n".join("    " + s for s in out["excerpt"]))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
