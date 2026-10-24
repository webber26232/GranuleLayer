"""Support-maintenance costs of the width-preserving formats (full size, outside every timed training step):
the components of one support refresh U for (policy, rho) from the actual alpha = 0 checkpoint, CUDA-
synchronized wall clock, median of --reps:

  select      width_masks.build (scores, protected / anchor / backbone selection, budget fill)
  emask       element mask of the support (dense masked reference / coverage)
  compress    dense values + support -> KgxWideLayer (tile values, tile metadata, backbone coordinates and
              values, tile value mask)
  work_lists  kgx work lists of the compact layer (items / chunks / tasks / bias-only rows; host work incl.)
  state_remap both BF16 Adam moments gathered from the dense layout into the compact order (tiles + backbone)
  densify     compact -> dense W_eff (needed to rescore against a dense candidate set)

U = select + compress + work_lists + state_remap.  A regrowing refresh additionally needs one dense scoring
step for the absent edges' gradients (charged by the report from the measured dense step).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import common  # noqa: F401
import wps_cases as WC


def timed(fn, reps):
    out, ts = None, []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return out, dict(median_s=statistics.median(ts), min_s=min(ts), samples_s=ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.experimental.kgx_wide import KgxWideLayer
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import hardware_state, provenance

    rec, w, b, _ = SP.load_checkpoint(WC.ckpt(a.C))
    lay = KgxLayer(WC.D, WC.H, WC.G, a.C, ConnectivityPlan(WC.S, WC.R, seed=0), variant="idf", tile=128,
                   device="cuda", weights=(w, b), keep_indices=False)
    del w, b
    L = lay.layout
    res = dict(policy=a.policy, rho=a.rho, C=a.C, checkpoint_weight_sha256=rec["weight_sha256"],
               provenance=provenance(Path(common.__file__).resolve().parent.parent), hardware_state=hardware_state())
    ms, res["select"] = timed(lambda: WM.build(a.policy, a.rho, lay.weight.detach(), L), a.reps)
    em, res["emask"] = timed(lambda: ms.element_mask(L), a.reps)
    with torch.no_grad():
        lay.weight.mul_(em.to(lay.weight.dtype))
    del em
    torch.cuda.empty_cache()
    sp, res["compress"] = timed(lambda: KgxWideLayer(lay, ms), a.reps)

    def wl():
        sp._rt = None
        return sp._runtime()

    rt, res["work_lists"] = timed(wl, a.reps)
    res["work_list_stats"] = dict(rt.sg.stats)
    m_dense = torch.zeros(lay.weight.shape, dtype=torch.bfloat16, device="cuda")
    n16 = m_dense.numel() // 1024

    def remap():
        out = []
        for _ in range(2):
            v = m_dense.view(n16, 16, ms.policy.nkt, ms.tb).permute(0, 2, 1, 3)[ms.tiles].contiguous()
            if ms.bb is not None:
                wd = m_dense.view(n16, 16, 64)
                ar = torch.arange(n16, device="cuda")
                v = (v, torch.stack([wd[ar, :, ms.bb[:, j]] for j in range(2)], -1).contiguous())
            out.append(v)
        return out

    _, res["state_remap"] = timed(remap, a.reps)
    _, res["densify"] = timed(lambda: sp.dense_params(), a.reps)
    res["bytes"] = dict(sp.metadata_bytes(), work_lists=int(rt.sg.stats["bytes"]))
    res["U_seconds"] = sum(res[k]["median_s"] for k in ("select", "compress", "work_lists", "state_remap"))
    res["U_components"] = ["select", "compress", "work_lists", "state_remap"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"{a.policy} {a.rho:g}: U = {res['U_seconds']:.3f} s (select {res['select']['median_s']:.3f}, compress "
          f"{res['compress']['median_s']:.3f}, work lists {res['work_lists']['median_s']:.3f}, remap "
          f"{res['state_remap']['median_s']:.3f}); emask {res['emask']['median_s']:.3f}, densify "
          f"{res['densify']['median_s']:.3f}", flush=True)


if __name__ == "__main__":
    main()
