"""Resource / traffic model of shared-projection (fused) backward schedules (Track F0,
results/SCALE_DISPATCH_FUSED_BACKWARD.md section 7).  Exact counts from the actual metadata of
the H = 100M graphs (valid units / blocks per supergroup, dX chunks, dW unit chunks); every
quantity is a CALCULATION with stated units, not a measurement.  Separate ledgers:
projection MMAs, gradient MMAs, dZ / packing, parameter and input delivery (logical bytes:
requests, not DRAM traffic), local dX updates, global dX flushes, dW / db write-back, scratch,
launch / CTA structure and register / shared-memory budgets (compiled values for existing
kernels, labelled lower-bound estimates for proposed ones).

Instruction counts per block-tile (8 neurons x 16 tokens, C = 8, split gradients, Mean(ReLU),
mask dX epilogue, fast dW epilogue; from the source of sg_dx_tile / dw_block and the SASS
counts in results/c8_dx_scatter/sass_*):
  dX : projection 1 HMMA.1688 (16 tok x 8 neurons, k = 8); gradient 2 HMMA.1688 (hi, lo)
  dW : projection 2 HMMA.1688 (one per 8-token half; rows 8-15 of the m16 tile are padding at
       C = 8); gradient 1 HMMA.16816 (hi in rows 0-7, lo in rows 8-15)
  fused (dX-layout projection kept): 1 + 2 HMMA.1688 + 1 HMMA.16816 + 2 MOVM (transpose of
       the 16-bit mask pairs into the dW A layout, tokens 0-7 / 8-15)

    python fusion_model.py --out results/scale_dispatch_fused_backward/fusion_model
"""
from __future__ import annotations

import argparse
import math

import torch

import common  # noqa: F401
from common import save
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.kernels import cuda_ops
from sparse_pooling.supergroups import ConnectivityPlan

HMMA1688_FLOP = 2 * 16 * 8 * 8
HMMA16816_FLOP = 2 * 16 * 8 * 16
MEASURED_TFLOPS = 69.0  # report C8_DX_SCATTER_OPTIMIZATION.md section 11 (illustration only)
UNIT_PARAM_BYTES = 32 * 8 * 2 + 32 * 2 + 4 * 8 * 2  # W (32 neurons x 8, BF16) + bias + indices per unit
UNIT_GRAD_BYTES = (32 * 8 + 32) * 4  # FP32 main_grad dW + db per unit


def graph(S, R, H, M):
    lay = SparsePoolingLayer(1024, H, 1024, shared_neurons=8, pooling="mean", device="cuda",
                             plan=PhysicalPlan(cuda_input_supergroup_backend="grouped"),
                             connectivity=ConnectivityPlan(S, R, seed=0))
    sg = lay._runtime().sg
    counts = lay._runtime().units.counts.long().cpu()
    b = sg.graph.buckets.cpu().long()
    n_q = torch.bincount(b[:, 0], weights=b[:, 3].double(), minlength=sg.Q).long()
    _, ch = sg.item_chunks(cuda_ops.SG_CHUNK_UNITS["dx"])
    out = dict(S=S, R=R, H=H, M=M, Q=sg.Q, units=int(n_q.sum()), valid_neurons=int(counts.sum()),
               valid_blocks=int(((counts + 7) // 8).sum()), dx_chunks=int(ch.shape[0]), n_q=n_q,
               dw_ctas={upc: int(((n_q + upc - 1) // upc).sum()) for upc in (16, 32, 64)},
               param_bytes=lay.weight.numel() * 2 + lay.bias.numel() * 2 + lay.indices.numel() * 2,
               grad_bytes=(lay.weight.numel() + lay.bias.numel()) * 4)
    del lay
    torch.cuda.empty_cache()
    return out


def schedules(g, regs):
    S, M = g["S"], g["M"]
    bt = g["valid_blocks"] * math.ceil(M / 16)  # block-tiles
    useful_proj = 2 * g["valid_neurons"] * M * 8
    dxT = {16: 64, 32: 32, 64: 32}[S]
    out = []

    def add(name, **kw):
        kw["name"] = name
        kw["hmma1688"] = kw["proj_hmma1688"] + kw["grad_hmma1688"]
        kw["issued_TFLOP"] = (kw["hmma1688"] * HMMA1688_FLOP + kw["grad_hmma16816"] * HMMA16816_FLOP) / 1e12
        kw["projection_evaluations_per_neuron_token"] = kw.pop("proj_evals")
        kw["mma_time_at_69TFLOPS_s"] = kw["issued_TFLOP"] / MEASURED_TFLOPS  # illustration, not a prediction
        out.append(kw)

    chunks = g["dx_chunks"]
    # ---- current separate backward (selected dX plan + grouped dW cfg 2)
    add("separate: dX (T %d, 8 warps, private copies) + dW (cfg 2: 8 warps x 2 units)" % dxT,
        proj_evals=2, proj_hmma1688=bt * (1 + 2), grad_hmma1688=bt * 2, grad_hmma16816=bt * 1,
        useful_projection_TFLOP=2 * useful_proj / 1e12,
        ctas={"dx": chunks * math.ceil(M / dxT), "dw": g["dw_ctas"][16]},
        dx_flush_atomics=chunks * S * M, dx_local_update_bytes=bt * 16 * 8 * 4 * 2,
        param_logical_bytes=g["param_bytes"] * (math.ceil(M / dxT) + 1),
        input_staging_logical_bytes=chunks * S * M * 2 + g["dw_ctas"][16] * S * M * 2,
        dw_rmw_bytes=2 * g["grad_bytes"], scratch_bytes=0,
        smem_KiB={"dx": round((2048 + S * dxT * 2 + 8 * S * dxT * 4) / 1024, 1), "dw": round((2048 + 2 * 4 * S * 16 * 2) / 1024, 1)},
        registers={"dx": regs.get("dx"), "dw": regs.get("dw")}, launches=3, note="measured baseline")
    # ---- F1: dW-owned CTAs stream all tokens and flush dX per window
    for upc, nwarp, wpw in ((16, 8, 2), (32, 8, 4), (64, 16, 4)):
        for Tw in (16, 64):
            add(f"F1 dW-owned {upc} units/CTA ({nwarp} warps x {wpw}), window {Tw}",
                proj_evals=1, proj_hmma1688=bt, grad_hmma1688=bt * 2, grad_hmma16816=bt, movm_per_block_tile=2,
                useful_projection_TFLOP=useful_proj / 1e12,
                ctas={"fused": g["dw_ctas"][upc]}, windows_per_cta=math.ceil(M / Tw),
                dx_flush_atomics=g["dw_ctas"][upc] * S * M, dx_local_update_bytes=bt * 16 * 8 * 4 * 2,
                param_logical_bytes=g["param_bytes"], input_staging_logical_bytes=g["dw_ctas"][upc] * S * M * 2,
                dw_rmw_bytes=2 * g["grad_bytes"], scratch_bytes=0,
                smem_KiB={"fused": round((2048 + 2 * 4 * S * 16 * 2 + nwarp * S * Tw * 4) / 1024, 1)},
                registers={"fused": f">= {regs.get('dw') if wpw == 2 else regs.get('dw4')} (dW cfg) + dX state"},
                dw_accumulator_regs_per_thread=24 * wpw, launches=2,
                note="flush count = CTAs x S x M independent of the window; dW owner keeps accumulators over all M")
    # ---- F2: coarse 4096-unit chunks, serial token windows, dW partial RMW per window
    for Tw in (256, 512, 1024):
        nw = math.ceil(M / Tw)
        smem = 2048 + S * Tw * 2 + S * Tw * 4
        add(f"F2 coarse chunks (4096 units), serial windows {Tw}",
            proj_evals=1, proj_hmma1688=bt, grad_hmma1688=bt * 2, grad_hmma16816=bt, movm_per_block_tile=2,
            useful_projection_TFLOP=useful_proj / 1e12, ctas={"fused": chunks}, windows_per_cta=nw,
            dx_flush_atomics=chunks * S * M, dx_local_update_bytes=bt * 16 * 8 * 4 * 2,
            dx_local_update_kind="one shared copy per CTA: 64-bit CAS (warps share features)",
            param_logical_bytes=g["param_bytes"] * nw, input_staging_logical_bytes=chunks * S * M * 2,
            dw_rmw_bytes=2 * g["grad_bytes"] * nw, scratch_bytes=0,
            smem_KiB={"fused": round(smem / 1024, 1)}, fits_99KiB=smem <= 99 * 1024,
            registers={"fused": ">= 24 dW accumulators + dX tile state (one unit per warp at a time)"},
            dw_accumulator_regs_per_thread=24, launches=2,
            note="dW / db summed into main_grad once per window (order differs from the separate dW)")
    # ---- F3: bounded mask exchange (one-bit ReLU masks, 64 MiB chunks)
    chunk_bits = 64 * 2**20 * 8
    neurons_per_chunk = chunk_bits // M
    nchunks = math.ceil(g["valid_neurons"] / neurons_per_chunk)
    mask_bytes = g["valid_neurons"] * M / 8
    add("F3 one-bit mask exchange (64 MiB chunks, producer dW -> consumer dX)",
        proj_evals=1, proj_hmma1688=bt * 2, grad_hmma1688=bt * 2, grad_hmma16816=bt,
        useful_projection_TFLOP=useful_proj / 1e12, ctas={"pairs_of_launches": nchunks},
        dx_flush_atomics=None, mask_write_plus_read_bytes=2 * mask_bytes, mask_total_bytes=mask_bytes,
        neurons_per_chunk=neurons_per_chunk, chunks=nchunks, launches=2 * nchunks + 1,
        param_logical_bytes=None, input_staging_logical_bytes=None, dw_rmw_bytes=2 * g["grad_bytes"],
        dx_local_update_bytes=bt * 16 * 8 * 4 * 2, scratch_bytes=64 * 2**20,
        note="the dW kernel keeps its projection (it needs z^T); dX reads masks instead of recomputing")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=100_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    regs = {16: {"dx": 86, "dw": 127, "dw4": 229}, 32: {"dx": 78, "dw": 127, "dw4": 229}}  # resources_S*_selected.json
    res = []
    for S, R in ((16, 4), (32, 8)):
        g = graph(S, R, args.hidden, args.batch)
        sc = schedules(g, regs[S])
        g.pop("n_q")
        res.append({"graph": g, "schedules": sc})
        print(f"== S = {S}, R = {R}: units {g['units']:,} valid blocks {g['valid_blocks']:,} dX chunks {g['dx_chunks']} "
              f"dW CTAs (16 units) {g['dw_ctas'][16]:,}")
        for s in sc:
            fl, pb = s.get("dx_flush_atomics"), s.get("param_logical_bytes")
            fls = "-" if fl is None else f"{fl / 1e9:7.2f}G"
            pbs = "-" if pb is None else f"{pb / 1e9:6.1f} GB"
            print(f"  {s['name'][:78]:78s} proj evals {s['projection_evaluations_per_neuron_token']} "
                  f"HMMA {s['hmma1688'] / 1e9:6.1f}G+{s['grad_hmma16816'] / 1e9:4.1f}G flush {fls} "
                  f"dW RMW {s['dw_rmw_bytes'] / 1e9:6.1f} GB param {pbs}")
    save(args.out, {"args": vars(args), "assumptions": {
        "hmma_flop": {"HMMA.1688": HMMA1688_FLOP, "HMMA.16816": HMMA16816_FLOP},
        "measured_TFLOPS_illustration": MEASURED_TFLOPS, "unit_param_bytes": UNIT_PARAM_BYTES,
        "unit_grad_bytes": UNIT_GRAD_BYTES,
        "logical_bytes": "requests issued by the schedule; cache hits / DRAM traffic not modelled"}, "results": res})


if __name__ == "__main__":
    main()
