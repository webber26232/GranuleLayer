"""Support-maintenance costs of the K64 frozen-mask formats (full size, outside any timed training step).

For a checkpoint and a format, times (CUDA-synchronized wall clock, median of --reps) every component of a
mask refresh U:

  select        mask selection from the dense values (learned support / magnitude / tile ranking)
  compress      dense values + mask -> compact representation (tile values + tile map, or CSR values / inputs /
                row pointers): the format's constructor
  work_lists    kgx work lists of the compact layer (items / chunks / tasks / bias-only rows, CSR dX windows;
                host work included)
  state_remap   optimizer moments gathered from the dense layout into the compact order (2 BF16 tensors)
  densify       compact values -> dense layout (needed to reselect against a dense candidate)

and records the resulting bytes.  The per-step amortization base + U / T for T in {1, 16, 128} is computed
by the report from these components and the measured ordinary steps (a periodic reselection schedule; a
regrowing dynamic method additionally needs dense gradients on refresh steps: the report charges one dense
step per refresh for it).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import common  # noqa: F401
import k64_cases as KC


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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--format", choices=["tile", "csr"], required=True)
    ap.add_argument("--mask", choices=["learned", "magnitude", "tile", "random"], required=True)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import hardware_state, provenance

    rec, w, b, _ = SP.load_checkpoint(a.checkpoint)
    lay = KgxLayer(KC.D, KC.H, KC.G, a.C, ConnectivityPlan(64, a.R, seed=KC.GRAPH_SEED), variant="idf", tile=128,
                   device="cuda", weights=(w, b), keep_indices=False)
    del w, b
    valid = SP.valid_slots(lay)
    res = dict(checkpoint=a.checkpoint, checkpoint_rec=rec, C=a.C, R=a.R, format=a.format, mask=a.mask,
               density_target=a.density, provenance=provenance(Path(common.__file__).resolve().parent.parent),
               hardware_state=hardware_state())
    tm = None
    if a.mask == "learned":
        mask, res["select"] = timed(lambda: SP.learned_mask(lay.weight.detach(), valid), a.reps)
    elif a.mask == "magnitude":
        mask, res["select"] = timed(lambda: SP.magnitude_mask(lay.weight.detach(), valid, a.density), a.reps)
    elif a.mask == "random":
        mask, res["select"] = timed(lambda: SP.random_mask(valid, a.density, 0), a.reps)
    else:
        tm, res["select"] = timed(lambda: SP.tile_mask(lay.weight.detach(), valid, a.density), a.reps)
        mask = SP.tiles_to_elements(tm, valid)
    with torch.no_grad():
        lay.weight.data.mul_(mask.to(lay.weight.dtype))
    res["density"] = int(mask.sum(dtype=torch.int64)) / (int(valid.sum()) * 64)
    if a.format == "tile":
        if tm is None:
            raise SystemExit("format tile needs mask tile")
        sp, res["compress"] = timed(lambda: SP.KgxTileLayer(lay, tm), a.reps)
    else:
        sp, res["compress"] = timed(lambda: SP.KgxCsrLayer(lay, mask), a.reps)

    def wl():
        sp._rt = None
        rt = sp._runtime()
        if a.format == "csr":
            sp._win_key = None
            sp.dx_windows(rt)
        return rt

    rt, res["work_lists"] = timed(wl, a.reps)
    res["work_list_stats"] = dict(rt.sg.stats)
    # optimizer state remap: dense BF16 moments -> compact order
    m_dense = torch.zeros(lay.weight.shape, dtype=torch.bfloat16, device="cuda")
    if a.format == "tile":
        n16 = lay.weight.numel() // 1024

        def remap():
            return [m_dense.view(n16, 16, 4, 16).permute(0, 2, 1, 3)[tm].contiguous() for _ in range(2)]
    else:
        mv, dv = mask.view(-1, 64), m_dense.view(-1, 64)

        def remap():  # chunked over neuron slots (a whole-tensor boolean gather materialises 8 B per element)
            out = []
            for _ in range(2):
                out.append(torch.cat([dv[s:s + (1 << 21)][mv[s:s + (1 << 21)].bool()]
                                      for s in range(0, dv.shape[0], 1 << 21)]))
            return out
    _, res["state_remap"] = timed(remap, a.reps)
    _, res["densify"] = timed(lambda: sp.dense_params(), a.reps)
    res["bytes"] = dict(sp.metadata_bytes(), work_lists=int(rt.sg.stats["bytes"]), dense_values=lay.weight.numel() * 2,
                        mask_uint8=mask.numel())
    res["U_seconds"] = sum(res[k]["median_s"] for k in ("select", "compress", "work_lists", "state_remap"))
    res["U_components"] = ["select", "compress", "work_lists", "state_remap"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"{a.format} {a.mask} C{a.C} density {res['density']:.4f}: U = {res['U_seconds']:.2f} s "
          f"(select {res['select']['median_s']:.2f}, compress {res['compress']['median_s']:.2f}, "
          f"work lists {res['work_lists']['median_s']:.2f}, remap {res['state_remap']['median_s']:.2f}); "
          f"densify {res['densify']['median_s']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
