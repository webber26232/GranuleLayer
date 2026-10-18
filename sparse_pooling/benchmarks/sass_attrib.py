"""MOV attribution in the hot loops of the native kernels (static SASS).

For every register move (MOV, IMAD.MOV) inside an HMMA-carrying loop, find the
first instruction that reads the moved register (scanning forward, wrapping
around the loop) and classify the move by that consumer:

  hmma_c   C (accumulator) operand of an HMMA  -> accumulator / bias fragment preparation
  hmma_a   A operand of an HMMA                -> operand packing (dW: dZ fragments)
  hmma_b   B operand of an HMMA
  other    any other instruction (index math, stores, ...)
  dead     overwritten before any read inside the loop

Counts are static (one copy of each instruction in the loop body), not dynamic
execution counts: the loops contain masked/unmasked and per-work-unit copies of
the same block code, only one of which runs per block.

    python sparse_pooling/benchmarks/sass_attrib.py --kernel 'k_dw_idx<2, 1, 64, 8, 2, 2, 1, 0, 2, 1>'
"""
from __future__ import annotations

import argparse
import collections
import json
import re

import common  # noqa: F401
from sass_loops import kernel_sass, loops, parse
from sparse_pooling.kernels.cuda_ops import build_library

REG = re.compile(r"\bR(\d+)(\.64)?\b")


def operands(args):
    return [a.strip() for a in args.split(",")]


def regs_of(opnd, width=1):
    m = REG.search(opnd)
    if not m or "RZ" == opnd.strip():
        return set()
    r = int(m.group(1))
    w = 2 if m.group(2) or ".64]" in opnd else width
    return set(range(r, r + w))


def dest_width(op):
    if op.startswith("HMMA") or ".128" in op or op.startswith("LDSM.16.MT88.4") or op.startswith("LDSM.16.M88.4"):
        return 4
    if ".64" in op or op.startswith("LDSM.16.MT88.2") or op.startswith("LDSM.16.M88.2") or op.startswith("CS2R"):
        return 2
    return 1


def src_roles(op, ops):
    """[(role, regs)] for the source operands of one instruction."""
    if op.startswith("HMMA"):
        k16 = "16816" in op
        return [("hmma_a", regs_of(ops[1], 4 if k16 else 2)), ("hmma_b", regs_of(ops[2], 2 if k16 else 1)),
                ("hmma_c", regs_of(ops[3], 4))]
    if op.startswith(("ST", "RED", "ATOM")):  # no register destination
        return [("other", regs_of(o, 4 if ".128" in op else (2 if ".64" in op else 1))) for o in ops]
    return [("other", regs_of(o)) for o in ops[1:]]


def writes(op, ops):
    if op.startswith(("ST", "RED", "BRA", "BAR", "WARPSYNC", "DEPBAR", "NOP", "BSYNC", "BSSY")) or not ops:
        return set()
    if op.startswith("ATOM"):
        return regs_of(ops[0])
    return regs_of(ops[0], dest_width(op))


def attribute(body):
    rows = [(a, op, operands(args)) for a, op, args in body]
    res = collections.Counter()
    detail = []
    n = len(rows)
    for i, (addr, op, ops) in enumerate(rows):
        if not (op == "MOV" or op.startswith("IMAD.MOV")):
            continue
        d = regs_of(ops[0])
        role = "dead"
        cons = None
        for k in range(1, n):
            a2, op2, ops2 = rows[(i + k) % n]
            hit = [r for r, rs in src_roles(op2, ops2) if rs & d]
            if hit:
                role, cons = hit[0], (hex(a2), op2)
                break
            if writes(op2, ops2) & d:
                break
        res[role] += 1
        detail.append((hex(addr), ops[0], role, cons))
    return res, detail


def mix(body):
    c = collections.Counter()
    for _, op, _ in body:
        base = "MOV" if op.startswith("IMAD.MOV") else op.split(".")[0]
        c[base] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", required=True)
    ap.add_argument("--lib", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()
    lib = args.lib or build_library()
    report = {}
    for kname in args.kernel:
        ins = parse(kernel_sass(lib, kname))
        rep = []
        for lo, hi, body in loops(ins):
            m = mix(body)
            if m.get("HMMA", 0) == 0:
                continue
            att, det = attribute(body)
            rep.append({"start": hex(lo), "end": hex(hi), "instructions": len(body), "mix": dict(m),
                        "mov_by_consumer": dict(att)})
            print(f"== {kname} loop {hex(lo)}-{hex(hi)}: {len(body)} instr, HMMA {m['HMMA']}, MOV {m.get('MOV', 0)}"
                  f" -> {dict(att)}")
            if args.detail:
                for d in det:
                    print("   ", *d)
        report[kname] = rep
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
