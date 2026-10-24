"""Explicit infeasibility check of the width-preserving study: a requested (policy, rho, C) whose mandatory
support exceeds the exact valid-slot budget must be REJECTED by width_masks.build (no silent density increase,
no smaller H).  The check runs on the full-size KLayout (no checkpoint values are needed before the rejection:
the weights are a meta tensor)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import common  # noqa: F401
import wps_cases as WC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgen_layout import KLayout
    from provenance import provenance

    L = KLayout(WC.D, WC.H, WC.G, WC.K, a.C)
    f = WM.feasibility(WM.POLICIES[a.policy], a.rho, L)
    w = torch.empty(L.weight_shape, dtype=torch.bfloat16, device="meta")
    res = dict(policy=a.policy, rho=a.rho, C=a.C, feasibility={k: v for k, v in f.items()},
               provenance=provenance(Path(common.__file__).resolve().parent.parent))
    try:
        WM.build(a.policy, a.rho, w, L)
        res["outcome"] = "BUILT (NOT REJECTED)"
    except WM.Infeasible as e:
        res["outcome"] = "rejected"
        res["message"] = str(e)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(("REJECTED: " + res["message"]) if res["outcome"] == "rejected" else res["outcome"], flush=True)


if __name__ == "__main__":
    main()
