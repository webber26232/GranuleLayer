"""Ownership map of every reduction stage, counted from the actual work lists (Track A).

For a graph (built from its plan on the host: supergroups.bucket_table, the generator's bucket
construction) and the resolved kernel configurations, per destination:
  forward   fast: FP32 atomics per (output, token) = items (buckets) of the output (one per
            supergroup touching it) x token tiles: writer multiplicity; deterministic: partial
            rows per output (= the same items), each written once by one CTA, summed in item order;
  dX        fast: CTAs flushing into one (feature, token) = chunks of the supergroups containing
            the feature (R partition rounds x chunks per supergroup), plus the shared-copy CAS
            inside a CTA when the plan shares copies (warps per copy > 1); deterministic: partial
            rows (chunk, local feature) per feature, coverage of every row exactly once;
  dW / db   units per CTA chunk and coverage of every valid unit by exactly one CTA (sg_dw:
            register accumulators over all tokens, plain RMW write-back);
  optimizer FusedAdamW programs of BLOCK elements: disjoint element ranges (one program owns an
            element: loads, barrier, stores).
    python ownership_map.py --out ../results/determinism_metrics_dw_projection/ownership_map
"""
from __future__ import annotations

import argparse
import json
import math

import torch

import common  # noqa: F401
from sparse_pooling.config import DX_PLANS
from sparse_pooling.kernels.cuda_ops import SG_CHUNK_UNITS, SG_DW_CFGS
from sparse_pooling.layout import SparseLayout
from sparse_pooling.supergroups import ConnectivityPlan, SgExec, bucket_table

CASES = [(16, 4, 64, 8, 1), (32, 8, 32, 8, 1), (64, 16, 64, 8, 8)]  # S, R, fast dX tokens, warps, warps per copy


def stats(t):
    t = t.double()
    return dict(min=int(t.min()), max=int(t.max()), mean=float(t.mean()))


def one(H, M, S, R, T_fast, wg_fast, G=1024, D=1024, seed=0):
    L = SparseLayout(D, H, G, 8, 8, True, packed=True)
    plan = ConnectivityPlan(S, R, seed=seed)
    feat, b = bucket_table(L.canonical, plan)
    sg = SgExec(L, feat, b, plan)
    Q = sg.Q
    out = dict(H=H, M=M, S=S, R=R, Q=Q, G=G, D=D, bucket_stats=sg.bucket_stats)
    # ---- forward (sg_fwd: chunk_units 2048, 128-token tiles)
    items, chf = sg.item_chunks(SG_CHUNK_UNITS["fwd"])
    per_out = torch.bincount(items[:, 0].long(), minlength=G)
    off, rec = sg.det_rows("fwd", SG_CHUNK_UNITS["fwd"])
    cover = torch.bincount(rec.long(), minlength=items.shape[0])
    out["forward"] = dict(
        chunks=chf.shape[0], items=items.shape[0], token_tiles=math.ceil(M / 128),
        fast_atomics_per_output_token=stats(per_out), fast_atomics_total=int(items.shape[0]) * M,
        det_partial_rows_per_output=stats(off[1:] - off[:-1]), det_rows_written_once=bool((cover == 1).all()),
        det_order="items of the output in ascending item index (= supergroup order), from +0.0f",
        det_partial_bytes_full_M=items.shape[0] * M * 4)
    # ---- dX (chunk_units 4096)
    _, chx = sg.item_chunks(SG_CHUNK_UNITS["dx"])
    qc = chx[:, 0].long()
    fq = feat.long()[qc]  # [chunks, S] features of each chunk
    writers = torch.bincount(fq.reshape(-1), minlength=D)
    per_q = torch.bincount(qc, minlength=Q)
    offx, recx = sg.det_rows("dx", SG_CHUNK_UNITS["dx"])
    coverx = torch.bincount(recx.long(), minlength=chx.shape[0] * S)
    member = torch.bincount(feat.long().reshape(-1), minlength=D)
    p = DX_PLANS["private_t32"]
    out["dx"] = dict(
        chunks=chx.shape[0], chunks_per_supergroup=stats(per_q), supergroups_per_feature=stats(member),
        fast_flush_ctas_per_feature_token=stats(writers),
        fast_in_cta=("shared FP32 copy, 64-bit CAS by %d warps" % wg_fast) if wg_fast > 1 else "warp-private copies",
        fast_tokens=T_fast,
        det_tokens=p["tokens"] if S == 64 else T_fast, det_in_cta="warp-private copies, summed in copy order 0..7",
        det_partial_rows_per_feature=stats(offx[1:] - offx[:-1]),
        det_rows_equal_fast_writers=bool(torch.equal((offx[1:] - offx[:-1]).long(), writers)),
        det_rows_written_once=bool((coverx == 1).all()),
        det_order="chunks containing the feature in ascending chunk index (supergroup, then item range), from +0.0f",
        det_partial_bytes_full_M=chx.shape[0] * S * M * 4)
    # ---- dW / db (cfg 2: 8 warps x 2 units)
    nw, wpw = SG_DW_CFGS[2][:2]
    units, chw = sg.unit_chunks(nw * wpw)
    ucov = torch.bincount(units.long(), minlength=int(units.max()) + 1)
    out["dw"] = dict(ctas=chw.shape[0], units=units.shape[0], units_per_cta_max=int((chw[:, 2] - chw[:, 1]).max()),
                     every_unit_one_cta=bool((ucov[units.long()] == 1).all()),
                     order="per unit: register accumulators over 16-token tiles in token order, one write-back")
    out["optimizer"] = dict(block=2048, programs_weight=math.ceil(L.groups * L.blocks_per_group * 64 / 2048),
                            ownership="program p owns elements [2048 p, 2048 p + 2048): all loads, barrier, stores")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", default="100000000,10000000,1000000")
    args = ap.parse_args()
    res = []
    shapes = [(H, 8192, 1024, 1024) for H in map(int, args.hidden.split(","))]
    shapes.append((20_000_000, 1024, 64, 256))  # det_training.py medium cases (Q = 64, several chunks)
    for H, M, G, D in shapes:
        for S, R, T, nw, wg in CASES:
            r = one(H, M, S, R, T, wg, G=G, D=D)
            res.append(r)
            f, x = r["forward"], r["dx"]
            print(f"H={H:,} D={D} G={G} S={S} R={R}: fwd items {f['items']} atomics/output {f['fast_atomics_per_output_token']} "
                  f"det rows once {f['det_rows_written_once']} | dX chunks {x['chunks']} flush CTAs/feature "
                  f"{x['fast_flush_ctas_per_feature_token']} (chunks/q {x['chunks_per_supergroup']['max']}) rows==writers {x['det_rows_equal_fast_writers']} once "
                  f"{x['det_rows_written_once']} | dW one CTA/unit {r['dw']['every_unit_one_cta']}", flush=True)
    json.dump(res, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
