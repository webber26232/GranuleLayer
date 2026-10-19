"""Layer-level numerics of the reduction modes (Track A, report §5.1): for S / R = (16, 4), (32, 8),
(64, 16) and R = 1, split and BF16 gradients, on shapes small enough for an FP64 reference:

* deterministic: 10 identical forward / backward calls in this process, byte-for-byte equal;
  with --records, the bits of the first call are saved (role write) or compared (role compare)
  so two process launches can be checked against each other;
* deterministic vs fast and each vs the FP64 reference: max abs error, normwise relative error,
  RMS error, non-finite counts, and near-zero handling (elements whose reference magnitude is
  below 1e-30: count, and how many of them differ from the reference by more than 1e-30);
* fast run-to-run: 10 calls, number of calls whose bits differ from the first.

    python det_numerics.py --records DIR --role write   --out ../results/.../det_numerics_p1
    python det_numerics.py --records DIR --role compare --out ../results/.../det_numerics_p2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

import common  # noqa: F401
from sparse_pooling import PhysicalPlan

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_input_supergroups import (BOUNDARY_CASE, boundary_reference, inputs, reference, run,  # noqa: E402
                                    sg_layer)

# (D, H, G, S, R, B, extra plan fields)
SHAPES = [
    (1024, 300_000, 64, 16, 4, 200, {}),
    (1024, 300_000, 64, 32, 8, 136, {}),
    (1024, 300_000, 64, 64, 16, 256, {}),
    (256, 300_000, 16, 16, 4, 72, dict(cuda_grouped_dx_chunk_units=48, cuda_grouped_fwd_chunk_units=40)),
    (512, 200_003, 32, 32, 8, 136, dict(cuda_grouped_dx_chunk_units=64, cuda_grouped_fwd_chunk_units=24)),
    (256, 20_011, 8, 16, 1, 64, {}),
]
NAMES = ["y", "dX", "dW", "db"]


def bits(t):
    return t.detach().contiguous().view(torch.int32)


def err(a, r):
    a, r = a.double(), r.double()
    d = a - r
    fin = torch.isfinite(d)
    near = r.abs() < 1e-30
    return dict(max_abs=float(d[fin].abs().max()) if fin.any() else 0.0,
                rel_l2=float(d[fin].norm() / r[fin].norm().clamp_min(1e-300)),
                rms=float(d[fin].square().mean().sqrt()) if fin.any() else 0.0,
                nonfinite=int((~torch.isfinite(a)).sum()), nonfinite_ref=int((~torch.isfinite(r)).sum()),
                near_zero=int(near.sum()), near_zero_off=int((near & (d.abs() > 1e-30)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--role", choices=["write", "compare"], required=True)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.records, exist_ok=True)
    rows = []
    for D, H, G, S, R, B, kw in SHAPES:
        for precision in ("split", "bf16"):
            name = f"D{D}_H{H}_G{G}_S{S}_R{R}_B{B}_{precision}"
            lay = sg_layer(D, H, G, 8, S, R, plan=PhysicalPlan(grad_precision=precision, reduction="deterministic",
                                                                **kw))
            x, dy = inputs(B, D, G)
            det = [run(lay, x, dy) for _ in range(args.reps)]
            same = all(torch.equal(bits(a), bits(b)) for r in det[1:] for a, b in zip(det[0][:4], r[:4]))
            rec = Path(args.records) / f"{name}.pt"
            if args.role == "write":
                torch.save([t.cpu() for t in det[0][:4]], rec)
                cross = None
            else:
                ref0 = torch.load(rec)
                cross = all(torch.equal(bits(a.cuda()), bits(b)) for a, b in zip(ref0, det[0][:4]))
            info = det[0][4]
            lay.plan = PhysicalPlan(grad_precision=precision, reduction="fast", **kw)
            fast = [run(lay, x, dy) for _ in range(args.reps)]
            fast_diff = sum(any(not torch.equal(bits(a), bits(b)) for a, b in zip(fast[0][:4], r[:4])) for r in fast[1:])
            boundary = (S, R, (D, H, G, B), precision) == (BOUNDARY_CASE["S"], BOUNDARY_CASE["R"], BOUNDARY_CASE["shape"],
                                                           BOUNDARY_CASE["precision"])
            lay.plan = PhysicalPlan(grad_precision=precision, reduction="deterministic", **kw)
            refs = (boundary_reference(lay, x, dy) if boundary else reference(lay, x, dy, torch.float64))
            row = dict(case=name, D=D, H=H, G=G, S=S, R=R, B=B, precision=precision, plan=kw,
                       det_reps=args.reps, det_repeat_bitwise=same, det_cross_process_bitwise=cross,
                       fast_reps=args.reps, fast_runs_differing_from_first=fast_diff,
                       dx_chunks=info["grouped_dx"].get("chunks"), fwd_chunks=info["grouped_fwd"].get("chunks"),
                       det_reduction=info["grouped_dx"]["reduction"], boundary_reference=boundary)
            for i, n in enumerate(NAMES):
                row[f"det_vs_fast.{n}"] = err(det[0][i], fast[0][i])
                row[f"det_vs_ref.{n}"] = err(det[0][i], refs[i])
                row[f"fast_vs_ref.{n}"] = err(fast[0][i], refs[i])
            rows.append(row)
            print(f"{name}: det x{args.reps} bitwise {same} cross-process {cross} | fast runs differing {fast_diff}/"
                  f"{args.reps - 1} | det-fast rel dX {row['det_vs_fast.dX']['rel_l2']:.1e} dW "
                  f"{row['det_vs_fast.dW']['rel_l2']:.1e} | det-ref rel dX {row['det_vs_ref.dX']['rel_l2']:.1e} "
                  f"fast-ref {row['fast_vs_ref.dX']['rel_l2']:.1e}", flush=True)
    Path(args.out + ".json").write_text(json.dumps(rows, indent=1))
    md = ["| Case | det x10 bitwise | det 2 processes | fast runs ≠ first | quantity | det−fast max abs / rel L2 / RMS | "
          "det−ref rel L2 | fast−ref rel L2 | non-finite (det/fast/ref) | near-zero ref (n / off) |",
          "|---|---|---|---:|---|---|---:|---:|---|---|"]
    for r in rows:
        for n in NAMES:
            a, b, c = r[f"det_vs_fast.{n}"], r[f"det_vs_ref.{n}"], r[f"fast_vs_ref.{n}"]
            md.append(f"| {r['case']} | {r['det_repeat_bitwise']} | {r['det_cross_process_bitwise']} | "
                      f"{r['fast_runs_differing_from_first']}/{r['fast_reps'] - 1} | {n} | {a['max_abs']:.2e} / "
                      f"{a['rel_l2']:.2e} / {a['rms']:.2e} | {b['rel_l2']:.2e} | {c['rel_l2']:.2e} | "
                      f"{a['nonfinite']}/{c['nonfinite']}/{c['nonfinite_ref']} | {b['near_zero']} / {b['near_zero_off']} |")
    Path(args.out + ".md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
