"""Nonlinear-loss / multi-step training validation of the grouped dX scatter variants
(training_validation.py: models, lockstep runs, predeclared checks C1-C3).

For each graph, model and optimizer mode: REPEATS baseline runs (envelope of the forward /
dX atomic-order variation), REPEATS - 1 runs of every candidate scatter (same grouped
forward / dW), one generic-backend run (numerical context, no check), all from the same
initial state over the same pre-generated batches.

    python numerics_training.py --out results/c8_dx_scatter/training_validation
"""
from __future__ import annotations

import argparse
import time

import torch

import common  # noqa: F401
from common import save
from sparse_pooling.kernels.cuda_ops import SG_DX_SCATTERS
from training_validation import batches, check, lockstep


def dx_plan(T, nw, wg, scatter):
    return dict(cuda_input_supergroup_backend="auto", cuda_grouped_dx="grouped", cuda_grouped_dx_tokens=T,
                cuda_grouped_dx_warps=nw, cuda_grouped_dx_reduction=wg, cuda_grouped_dx_scatter=scatter)


GRAPHS = {  # name: (graph, dX tile, candidate scatters, steps)
    "C8_S16_R4": (dict(C=8, S=16, R=4), (64, 8, 1), ("vectorized", "vectorized_transpose", "swizzle4",
                                                     "feature_owner"), 16),
    "C8_S32_R8": (dict(C=8, S=32, R=8), (32, 8, 1), ("vectorized", "vectorized_transpose", "swizzle4",
                                                     "feature_owner"), 16),
    "C8_S16_R16": (dict(C=8, S=16, R=16), (64, 8, 1), ("vectorized", "vectorized_transpose", "feature_owner"), 8),
    "C8_S64_R16": (dict(C=8, S=64, R=16), (32, 8, 1), ("vectorized", "vectorized_transpose", "swizzle4"), 8),
    "C8_S64_R16_shared": (dict(C=8, S=64, R=16), (64, 8, 8), ("swizzle4",), 8),
    "C64_S16_R4": (dict(C=64, S=16, R=4), None, (), 8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--groups", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    ap.add_argument("--models", default="mse,square,compose")
    ap.add_argument("--opt-modes", default="det,sr,fp32")
    ap.add_argument("--precisions", default="split")
    ap.add_argument("--candidates", default="", help="override the candidate scatters of every C = 8 graph")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = []
    for gname in args.graphs.split(","):
        graph, tile, cands, steps = GRAPHS[gname]
        if args.candidates and tile is not None:
            cands = tuple(c for c in args.candidates.split(",")
                          if not (c == "feature_owner" and graph["S"] > 32)
                          and not (c != "swizzle4" and not c.startswith("swizzle4") and tile[2] != 1)
                          and not (c.startswith("swizzle4") and tile[2] == 1 and c != "swizzle4"))
        for prec in args.precisions.split(","):
            for model in args.models.split(","):
                for om in args.opt_modes.split(","):
                    cfg = dict(graph, D=1024, H=args.hidden, G=args.groups, B=args.batch, model=model, precision=prec,
                               D0=256, G2=64, lr=1e-3)
                    if tile is None:
                        variants = {"baseline": (dict(cuda_input_supergroup_backend="auto"), None),
                                    "baseline_null": (dict(cuda_input_supergroup_backend="auto"), "same")}
                    else:
                        variants = {"baseline": (dx_plan(*tile, "baseline"), None),
                                    "baseline_null": (dx_plan(*tile, "baseline"), "same")}  # null control
                        for c in cands:
                            variants[c] = (dx_plan(*tile, c), SG_DX_SCATTERS[c]["order"])
                    variants["generic_backend"] = (dict(cuda_input_supergroup_backend="generic"), None)
                    data = batches(cfg, steps)
                    t0 = time.perf_counter()
                    r = lockstep(cfg, variants, om, data, args.repeats)
                    env = r["series"]["baseline"]
                    row = {"graph": gname, "cfg": cfg, "opt_mode": om, "steps": steps, "repeats": args.repeats,
                           "kernels": r["kernels"], "finite": r["finite"], "losses": r["losses"],
                           "baseline_envelope": env, "variants": {}}
                    for name, (_, order) in variants.items():
                        if name == "baseline":
                            continue
                        v = {"series": r["series"][name], "order": order}
                        if order is not None:
                            v["check"] = check(env, r["series"][name], order)
                        row["variants"][name] = v
                    results.append(row)
                    last = env[-1]
                    print(f"{gname} {prec} {model} {om}: {time.perf_counter() - t0:.1f}s finite {all(r['finite'].values())}"
                          f" | base-vs-base step {steps}: W {last.get('W', 0):.2e} dX {last.get('dX', 0):.2e} "
                          f"loss {last['loss']:.2e}", flush=True)
                    for name, v in row["variants"].items():
                        lv = v["series"][-1]
                        ck = v.get("check")
                        print(f"    {name:22s} W {lv.get('W', 0):.2e} dX {lv.get('dX', 0):.2e} loss {lv['loss']:.2e}"
                              + (f"  check x{ck['factor']:.0f}: {'PASS' if ck['pass'] else 'FAIL'} "
                                 f"(worst {ck['worst_ratio_to_limit']:.2f} of limit)" if ck else ""), flush=True)
                    torch.cuda.empty_cache()
    save(args.out, {"args": vars(args), "results": results})


if __name__ == "__main__":
    main()
