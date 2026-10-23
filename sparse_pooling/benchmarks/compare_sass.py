"""Compare the machine code of two native builds kernel by kernel (cuobjdump; no GPU needed).

Used to show that a host-side / build-system refactor left the device code unchanged: for every
kernel (mangled name) present in both builds, the SASS text (addresses + encodings stripped of the
line-info comments) and the resource usage (registers, stack, static shared / local memory) are
compared.  A kernel with identical SASS has identical arithmetic, summation order and FTZ behaviour
and the same resource-limited occupancy, so its numerics and speed are unchanged by construction.

    python compare_sass.py --old <old .so> [...] --new <new .so> [...] --out summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def _run(*a) -> str:
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout


def sass_by_function(lib: str) -> dict:
    out, name, buf = {}, None, []
    for line in _run("cuobjdump", "-sass", lib).splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if name:
                out[name] = "\n".join(buf)
            name, buf = m.group(1), []
            continue
        if name and (re.match(r"\s*/\*[0-9a-f]{4,}\*/", line) or re.match(r"\s*/\* 0x[0-9a-f]+ \*/", line)):
            # instruction + both encoding words (no line-info comments); cuobjdump pads columns per
            # cubin, so whitespace is normalised
            buf.append(re.sub(r"\s+", " ", line.strip()))
    if name:
        out[name] = "\n".join(buf)
    return out


def resources(lib: str) -> dict:
    out = {}
    for line in _run("cuobjdump", "-res-usage", lib).splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if m:
            out[cur] = dict(zip(("reg", "stack", "shared", "local"), map(int, m.groups())))
    return out


def load(libs):
    sass, res = {}, {}
    for lib in libs:
        sass.update(sass_by_function(lib))
        res.update(resources(lib))
    return sass, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    so, ro = load(a.old)
    sn, rn = load(a.new)
    common = sorted(set(so) & set(sn))
    same = [k for k in common if so[k] == sn[k]]
    diff = [k for k in common if so[k] != sn[k]]
    res_diff = {k: (ro.get(k), rn.get(k)) for k in common if ro.get(k) != rn.get(k)}
    demangle = lambda names: _run("c++filt", *names).splitlines() if names else []  # noqa: E731
    summary = dict(old=a.old, new=a.new, kernels_old=len(so), kernels_new=len(sn), common=len(common),
                   identical_sass=len(same), differing_sass=demangle(diff[:50]), n_differing=len(diff),
                   resource_differences={k: v for k, v in list(res_diff.items())[:50]}, n_resource_differences=len(res_diff),
                   only_old=demangle(sorted(set(so) - set(sn))[:200]), n_only_old=len(set(so) - set(sn)),
                   only_new=demangle(sorted(set(sn) - set(so))[:200]), n_only_new=len(set(sn) - set(so)),
                   sass_digest_new=hashlib.sha256("".join(sn[k] for k in sorted(sn)).encode()).hexdigest())
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(f"common {len(common)}: identical {len(same)}, differing {len(diff)}, resource differences {len(res_diff)}; "
          f"only old {summary['n_only_old']}, only new {summary['n_only_new']}")


if __name__ == "__main__":
    main()
