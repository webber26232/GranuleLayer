"""Static SASS / resource evidence of the compact C = 8 dW projection (Track C).

For every grouped C = 8 dW kernel symbol (sg_dw_kernel<ACT, 8, NW, WPW, TS, ST, SPLIT, MINB, EPI,
PROJ>) of the built library: registers, stack, spill stores / loads (cuobjdump -res-usage) and
static instruction counts (cuobjdump -sass): projection / gradient MMAs, MOVM (movmatrix), the
FTZ compares that produce the ReLU masks (FSET.BF.GT.FTZ / FSETP.GT.FTZ: the production decision
must keep .ftz), PRMT / SHF / LOP3, shuffles, barriers, total.  Static counts cover every unrolled
copy (units per warp x blocks per unit x full / ragged paths); the per-block-tile expectation is
padded 2 x HMMA.1688 + 1 x HMMA.16816, compact 1 x HMMA.1688 + 1 x HMMA.16816 + 2 x MOVM.

    python dw_projection_sass.py --out ../results/determinism_metrics_dw_projection/dw_projection_sass
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

import common  # noqa: F401
from sparse_pooling.kernels.cuda_ops import SRC_DIR, _unit_hash, build_library

OPS = {"HMMA.1688": r"HMMA\.1688\.F32\.BF16", "HMMA.16816": r"HMMA\.16816\.F32\.BF16", "MOVM": r"\bMOVM\b",
       "FSET.GT.FTZ": r"FSET\.BF\.GT\.FTZ", "FSETP.GT.FTZ": r"FSETP\.GT\.FTZ", "FSET/FSETP.GT (no FTZ)": r"FSETP?(\.BF)?\.GT\.AND(?!.*FTZ)",
       "PRMT": r"\bPRMT\b", "SHF": r"\bSHF\b", "LOP3": r"\bLOP3\b", "SHFL": r"\bSHFL\b", "BAR": r"\bBAR\b",
       "LDSM": r"\bLDSM\b", "STL": r"\bSTL\b", "LDL": r"\bLDL\b", "FADD": r"\bFADD\b", "FSEL": r"\bFSEL\b"}


def demangle(names):
    r = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return r.stdout.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_library()
    src = SRC_DIR / "sg_dw_c8.cu"
    obj = SRC_DIR / "build" / "objs" / f"{src.stem}-{_unit_hash(src)}.o"
    sass = subprocess.run(["cuobjdump", "-sass", str(obj)], capture_output=True, text=True).stdout
    res = subprocess.run(["cuobjdump", "-res-usage", str(obj)], capture_output=True, text=True).stdout
    funcs = {}
    cur = None
    for ln in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", ln)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur and re.match(r"\s*/\*[0-9a-f]{4,}\*/", ln):
            funcs[cur].append(ln)
    usage = {}
    lines = res.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"\s*Function (\S+):", ln)
        if m and i + 1 < len(lines):
            u = lines[i + 1]
            g = lambda k: int(re.search(k + r":(\d+)", u).group(1)) if re.search(k + r":(\d+)", u) else None
            usage[m.group(1)] = dict(registers=g("REG"), stack=g("STACK"), shared=g("SHARED"), local=g("LOCAL"))
    mangled = list(funcs)
    rows = []
    for mg, dm in zip(mangled, demangle(mangled)):
        t = re.search(r"sg_dw_kernel<(.*)>\(SgArgs\)", dm)
        if not t:
            continue
        targs = [x.strip() for x in t.group(1).split(",")]
        cnt = collections.Counter()
        for ln in funcs[mg]:
            for k, pat in OPS.items():
                if re.search(pat, ln):
                    cnt[k] += 1
        rows.append(dict(symbol=dm, template=targs, total_instructions=len(funcs[mg]), counts=dict(cnt),
                         **usage.get(mg, {})))
    key = lambda r: tuple(r["template"])
    rows.sort(key=key)
    out = dict(object=str(obj), library=str(build_library()), kernels=rows,
               note="static counts over the whole kernel (all unrolled copies); not dynamic instruction counts")
    Path(args.out + ".json").write_text(json.dumps(out, indent=1))
    hdr = "| template <ACT, C, NW, WPW, TS, ST, SPLIT, MINB, EPI, PROJ> | regs | stack | local | HMMA.1688 | HMMA.16816 | MOVM | FSET(P).GT.FTZ | non-FTZ GT | PRMT | total |"
    md = [hdr, "|" + "---|" * 11]
    for r in rows:
        c = r["counts"]
        md.append(f"| {', '.join(r['template'])} | {r.get('registers')} | {r.get('stack')} | {r.get('local')} | "
                  f"{c.get('HMMA.1688', 0)} | {c.get('HMMA.16816', 0)} | {c.get('MOVM', 0)} | "
                  f"{c.get('FSET.GT.FTZ', 0) + c.get('FSETP.GT.FTZ', 0)} | {c.get('FSET/FSETP.GT (no FTZ)', 0)} | "
                  f"{c.get('PRMT', 0)} | {r['total_instructions']} |")
    Path(args.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
