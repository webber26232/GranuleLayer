"""Static SASS instruction mix of the hot loops of the native backward kernels.

Finds every loop (backward branch) of the selected kernels in the built library
and counts instruction classes inside its body.  Static counts: one loop-body
iteration, no dynamic execution frequencies.

    python sparse_pooling/benchmarks/sass_loops.py --kernel 'k_dw_idx<2, 1, 64, 8, 2, 2, 1, 0, 2>'
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess

import common  # noqa: F401  (puts the repository root on sys.path)
from sparse_pooling.kernels.cuda_ops import build_library

CLASSES = {
    "HMMA": ["HMMA"], "LDSM": ["LDSM"], "LDGSTS": ["LDGSTS"], "LDG": ["LDG"], "LDS": ["LDS"], "STS": ["STS"],
    "ATOMS": ["ATOMS"], "RED/ATOMG": ["RED", "ATOMG", "ATOM"], "STG": ["STG"],
    "F2FP (cvt->bf16x2)": ["F2FP"], "PRMT": ["PRMT"], "SHF/SHL": ["SHF", "SHL"], "LOP3": ["LOP3"],
    "FADD": ["FADD"], "FMUL": ["FMUL"], "FFMA": ["FFMA"], "FSEL": ["FSEL"], "FSETP": ["FSETP"], "FMNMX": ["FMNMX"],
    "MUFU": ["MUFU"], "IMAD/IADD/LEA": ["IMAD", "IADD3", "LEA", "IABS"], "ISETP": ["ISETP"], "SEL": ["SEL"],
    "MOV": ["MOV", "IMAD.MOV"], "SHFL": ["SHFL"], "BAR/WARPSYNC/DEPBAR": ["BAR", "WARPSYNC", "DEPBAR", "LDGDEPBAR"],
    "BRA/BSSY/BSYNC": ["BRA", "BSSY", "BSYNC"], "MOVM": ["MOVM"], "P2R/R2P/PLOP3": ["P2R", "R2P", "PLOP3"],
}


def classify(op):
    if op.startswith("IMAD.MOV"):
        return "MOV"
    base = op.split(".")[0]
    for k, v in CLASSES.items():
        if base in v:
            return k
    return "other"


def kernel_sass(lib, demangled):
    out = subprocess.run(["cuobjdump", "-sass", str(lib)], capture_output=True, text=True).stdout
    chunks = re.split(r"\n\s+Function : ", out)[1:]
    names = [c.split("\n", 1)[0].strip() for c in chunks]
    dem = subprocess.run(["cu++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    for chunk, d in zip(chunks, dem):
        d = re.sub(r"\(int\)|\(bool\)|void |\(BwdArgs\)|\(FwdArgs\)", "", d.strip())
        if d == demangled:
            return chunk
    raise SystemExit(f"kernel {demangled!r} not found")


def parse(chunk):
    ins = []
    for m in re.finditer(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P\w+\s+)?)([A-Z][A-Z0-9_.]*)([^;]*);", chunk):
        addr = int(m.group(1), 16)
        ins.append((addr, m.group(3), m.group(4)))
    return ins


def loops(ins):
    res = []
    for addr, op, args in ins:
        if op.startswith("BRA"):
            t = re.search(r"0x([0-9a-f]+)", args)
            if t and int(t.group(1), 16) < addr:
                lo = int(t.group(1), 16)
                body = [i for i in ins if lo <= i[0] <= addr]
                res.append((lo, addr, body))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", required=True, help="demangled name, e.g. 'k_dx<2, 1, 64, 32, 1, 0>'")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lib = build_library()
    report = {}
    for k in args.kernel:
        ins = parse(kernel_sass(lib, k))
        rep = {"total_instructions": len(ins), "loops": []}
        for lo, hi, body in loops(ins):
            c = collections.Counter(classify(op) for _, op, _ in body)
            if c.get("HMMA", 0) == 0:
                continue
            rep["loops"].append({"start": hex(lo), "end": hex(hi), "instructions": len(body), "classes": dict(c)})
        report[k] = rep
        print(f"== {k}: {len(ins)} instructions")
        for L in rep["loops"]:
            c = L["classes"]
            top = ", ".join(f"{n} {v}" for n, v in sorted(c.items(), key=lambda kv: -kv[1]))
            print(f"  loop {L['start']}-{L['end']}: {L['instructions']} instr: {top}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
