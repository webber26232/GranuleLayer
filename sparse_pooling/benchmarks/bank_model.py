"""COMPUTED (not measured) shared-memory wavefronts of the grouped C = 8 dX slab read-modify-write
for each layout, from the device offsets (CudaBackend.sgx_layout) and random 8-feature records
drawn from S local features (the balanced generator's records are uniformly random subsets).
A 64-bit access is served per half-warp, a 128-bit access per quarter-warp; a phase costs as
many wavefronts as the largest number of distinct addresses in one bank.  Hardware counters
are admin-only on this machine, so this is an analytical estimate.

    python bank_model.py --out results/c8_dx_scatter/bank_model.json
"""
from __future__ import annotations

import argparse
import json

import torch

import common  # noqa: F401
from sparse_pooling.kernels import cuda_ops

LAYOUTS = {  # name: (sc for the layout export, vector width, transposed lane mapping)
    "baseline": (0, 2, False), "swizzle4": (6, 2, False), "vectorized_parity": (3, 4, False),
    "vectorized": (2, 4, False), "vectorized_transpose": (7, 4, True)}


def wavefronts(off, rec, width, trans, blk):
    """Wavefronts of the loads (= stores) of one block for one m16 tile (float2) or tile pair
    (float4), both column parities."""
    total = 0
    phase = 16 if width == 2 else 8
    for e in (0, 1):
        words = {}
        for lane in range(32):
            t, c = lane >> 2, lane & 3
            if trans:
                f, tok = rec[2 * (lane >> 3) + e], lane & 7
            else:
                f, tok = rec[2 * c + e], t
            if width == 2:
                ws = [off[f][16 * blk + tok], off[f][16 * blk + tok + 8]]
            else:
                b = 32 * blk
                ws = [off[f][b + tok], off[f][b + tok + 8], off[f][b + 16 + tok], off[f][b + 24 + tok]]
            words[lane] = ws
        for p0 in range(0, 32, phase):
            banks = {}
            for lane in range(p0, p0 + phase):
                for w in words[lane]:
                    banks.setdefault(w % 32, set()).add(w)
            total += max(len(v) for v in banks.values())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    be = cuda_ops.CudaBackend.get()
    g = torch.Generator().manual_seed(0)
    res = {}
    for S in (16, 32, 64):
        for name, (sc, width, trans) in LAYOUTS.items():
            T = 64
            off = be.sgx_layout(sc, T, S).cpu().tolist()
            tot = 0
            for _ in range(args.samples):
                rec = torch.randperm(S, generator=g)[:8].tolist()
                tot += wavefronts(off, rec, width, trans, 0)
            per_tile = tot / args.samples / (1 if width == 2 else 2)  # loads of one m16 tile, both parities
            ideal = 4  # 2 x 256 B per tile and parity -> 2 instr x 2 wavefronts (64-bit) or 1 x 4 (128-bit) / 2 tiles
            res[f"S{S}_{name}"] = {"S": S, "layout": name, "width_bits": 32 * width,
                                   "load_wavefronts_per_tile": per_tile, "rmw_wavefronts_per_tile": 2 * per_tile,
                                   "conflict_free_rmw_wavefronts_per_tile": 2 * ideal,
                                   "instructions_per_tile": (4 if width == 2 else 2)}
            print(f"S={S:3d} {name:22s} {32 * width:3d}-bit: load wavefronts / m16 tile {per_tile:5.2f} "
                  f"(conflict-free {ideal}), RMW {2 * per_tile:5.2f}, shared instr / tile {(4 if width == 2 else 2)}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
