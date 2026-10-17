"""Source-derived logical memory-request model for the native forward, k_dx and k_dw.

Counts the bytes requested by the kernels' load/store/cp.async instructions as
the code in kernels/cuda/*.cuh issues them, for the launch shapes chosen by
kernels/cuda_ops.py.  These are LOGICAL request bytes (level A).  They are not
L2 or DRAM traffic: the comments say which requests are expected to hit L1/L2.
No GPU is used.

    python sparse_pooling/benchmarks/traffic_model.py --hidden 100000000 --C 64
"""
from __future__ import annotations

import argparse
import json
import math

import common  # noqa: F401  (puts the repository root on sys.path)
from sparse_pooling.config import PhysicalPlan
from sparse_pooling.kernels.cuda_ops import DW_CFG, DW_CFGS
from sparse_pooling.layout import SparseLayout

GiB = 2**30


def model(H, C, B, D=1024, G=1024, grad_precision="split", pool="mean"):
    L = SparseLayout(input_dim=D, hidden_dim=H, groups=G, fan_in=8, block=C, shared_indices=True, packed=True)
    plan = PhysicalPlan()
    NB = L.blocks_per_group
    UB = L.unit_blocks
    U = G  # single level, one unit per group
    w_bytes = G * NB * C * 8 * 2  # padded BF16 weights
    b_bytes = G * NB * C * 2
    i_bytes = G * NB * 8 * 2
    p_bytes = w_bytes + b_bytes + i_bytes
    out = {"H": H, "C": C, "B": B, "NB_per_group": NB, "unit_blocks": UB,
           "padded_neurons": G * NB * C, "param_bytes": p_bytes}

    # ---- forward: CTA = (BM tokens, unit); 1 split at B = 8192
    BM = plan.cuda_block_m
    n_tiles = math.ceil(B / BM)
    ctas = n_tiles * U
    out["forward"] = {
        "ctas": ctas, "threads": plan.cuda_warps * 32, "tokens_per_cta": BM,
        "smem_bytes": max(D * BM * 2, 4 * plan.cuda_warps * BM * 4),
        # every CTA loads every parameter record of its unit once (ld.global.nc, 16 B per lane)
        "param_requests_bytes": n_tiles * p_bytes,
        # cp.async of the [D, BM] BF16 slab
        "x_staging_bytes": ctas * D * BM * 2,
        # one compact state per (token, unit): st_m/st_s (mean: s only + m) FP32
        "state_write_bytes": B * U * 4 * (2 if pool != "max" else 1),
        "distinct_param_bytes": p_bytes, "distinct_x_bytes": B * D * 2,
    }

    # ---- k_dx: CTA = (16 tokens, units_per_cta units); 16 warps
    t16 = math.ceil(B / 16)
    gy = max(1, min(U, math.ceil(4096 / t16)))
    upc = math.ceil(U / gy)
    gy = math.ceil(U / upc)
    blocks_total = G * NB
    passes = 2 if grad_precision == "split" else 1
    out["dx"] = {
        "ctas": t16 * gy, "threads_baseline": 16 * 32, "threads_default": (32 if C == 64 else 16) * 32, "tokens_per_cta": 16, "units_per_cta": upc,
        "smem_bytes": D * 16 * (2 + 4),
        "param_requests_bytes": t16 * p_bytes,  # every token tile re-reads all parameters
        "x_staging_bytes": t16 * gy * D * 16 * 2,
        "coef_requests_bytes": t16 * gy * upc * (32 * 2 * 4),  # RowCoef: 2 rows per lane, c1 only for Mean
        # dx_block: 2 x 64-bit CAS per lane per (block, 16-token tile) = 64 per warp-block
        "smem_cas64_ops": t16 * blocks_total * 64,
        "smem_cas_bytes_rmw": t16 * blocks_total * 64 * 8 * 2,
        "global_atomics": t16 * gy * D * 16,  # flush: <= one FP32 atomicAdd per (feature, token) per CTA
        "global_atomic_bytes": t16 * gy * D * 16 * 4,
        "mma_m16n8k8_recompute": t16 * blocks_total * (C // 8),
        # dA = dZ W^T: C = 8 one m16n8k8 per pass, else NT/2 m16n8k16 per pass
        "mma_grad_products": t16 * blocks_total * (passes if C == 8 else (C // 16) * passes),
    }

    # ---- k_dw: CTA = (NW * WPW work units of one unit u), loops over all tokens
    out["dw"] = {}
    for cfg, (nw, wpw) in DW_CFGS.items():
        wu_per_cta = nw * wpw
        n_chunks = math.ceil(math.ceil(NB / UB) / wu_per_cta)
        ctas = n_chunks * U
        rows = wu_per_cta * UB * 8  # indexed feature-row occurrences per CTA
        tiles = math.ceil(B / 16)
        out["dw"][f"cfg{cfg}_{nw}x{wpw}"] = {
            "ctas": ctas, "threads": nw * 32, "work_units_per_cta": wu_per_cta, "index_rows_per_cta": rows,
            "smem_full_bytes": D * 16 * 2 * 2,
            "smem_indexed_bytes_2stage": 2 * rows * 16 * 2,
            "x_staging_full_bytes": ctas * tiles * D * 16 * 2,
            "x_staging_indexed_bytes": ctas * tiles * rows * 16 * 2,
            "param_requests_bytes": p_bytes,  # exclusive ownership: each record loaded once
            "grad_rmw_bytes": 2 * (G * NB * C * 8 * 4 + G * NB * C * 4),  # FP32 main_grad read + write
            # ColCoef: 4 tokens x (c1[, c2, c3]) scalar loads per lane per tile (L1-resident)
            "coef_requests_bytes": ctas * tiles * nw * 32 * 4 * 4 * (1 if pool == "mean" else 3),
            # dw_block: NP = ceil(NT/2) row tiles x 2 token halves of m16n8k8 recompute,
            # NP m16n8k16 per pass (C = 8 folds hi/lo into one MMA)
            "mma_m16n8k8_recompute": tiles * blocks_total * ((C // 8 + 1) // 2) * 2,
            "mma_m16n8k16_grad": tiles * blocks_total * ((C // 8 + 1) // 2) * (1 if C == 8 else passes),
        }
    if DW_CFG.get(C) is not None:
        nw, wpw = DW_CFGS[DW_CFG[C]]
        out["dw_default"] = f"cfg{DW_CFG[C]}_{nw}x{wpw}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[1_000_000, 10_000_000, 100_000_000])
    ap.add_argument("--C", type=int, nargs="+", default=[8, 64])
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = [model(H, C, args.batch) for H in args.hidden for C in args.C]
    for r in res:
        f, dx, dw = r["forward"], r["dx"], r["dw"][r["dw_default"]]
        print(f"H={r['H']:>11,} C={r['C']:2d}: fwd CTAs {f['ctas']:>7,} params {f['param_requests_bytes'] / GiB:7.1f} GiB "
              f"X {f['x_staging_bytes'] / GiB:6.1f} | dX CTAs {dx['ctas']:>5,} params {dx['param_requests_bytes'] / GiB:7.1f} GiB "
              f"CAS64 {dx['smem_cas64_ops'] / 1e9:6.2f}e9 | dW CTAs {dw['ctas']:>6,} X full {dw['x_staging_full_bytes'] / GiB:7.1f} "
              f"indexed {dw['x_staging_indexed_bytes'] / GiB:6.1f} GiB coef {dw['coef_requests_bytes'] / GiB:6.1f} GiB")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
