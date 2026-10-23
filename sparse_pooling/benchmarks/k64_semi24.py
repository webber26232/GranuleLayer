"""Ampere 2:4 feasibility / control of the K64 study: a BOUNDED PANEL MICROBENCHMARK, not a training measurement.

Audit (installed stack, no upgrade): PyTorch semi-structured sparsity (torch.sparse.to_sparse_semi_structured,
cuSPARSELt / CUTLASS backends) for BF16 on this sm_86 device; compression time of both operand orientations.

Panel = one logical record: X [M, 64] (the set's inputs, M = 8192 tokens) and the record's weights W [C, 64].
  forward z panel   Z = X W^T     (reduction over the 64 inputs: needs W 2:4 along the input dimension)
  dX panel          dA = dZ W     (reduction over the C neurons: needs W^T 2:4 along the neuron dimension)
  dW panel          dW = dZ^T X   (does not multiply by W: weight sparsity does not make it a sparse GEMM)
A mask that is 2:4 along the inputs is generally NOT 2:4 along the neurons; ``bidirectional_mask`` builds,
per 4 x 4 block, the best of the 90 patterns with two nonzeros in every row AND column (magnitude score), so
one numerical matrix W serves both orientations (never two differently pruned matrices).  The real layer
also needs the gather of each record's inputs, ReLU, the pooling sum and full-z recomputation, which a library
GEMM does not fuse (a fused version would materialize [M, C] per panel); none of that is measured here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

import common  # noqa: F401


def patterns_2of4_rows_cols():
    out = []
    rows = [r for r in itertools.product([0, 1], repeat=4) if sum(r) == 2]
    for m in itertools.product(rows, repeat=4):
        t = torch.tensor(m)
        if bool((t.sum(0) == 2).all()):
            out.append(t)
    return torch.stack(out)  # [90, 4, 4]


@torch.no_grad()
def two_four_mask(w: torch.Tensor) -> torch.Tensor:
    """2:4 along the last (input) dimension: keep the 2 largest |w| of every 4 consecutive inputs."""
    a = w.float().abs().view(*w.shape[:-1], -1, 4)
    idx = a.topk(2, dim=-1).indices
    m = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, idx, True)
    return m.view(w.shape)


@torch.no_grad()
def bidirectional_mask(w: torch.Tensor) -> torch.Tensor:
    """[C, 64] mask, 2:4 along inputs AND along neurons (per 4 x 4 block: best row/column-2:4 pattern)."""
    P = patterns_2of4_rows_cols().to(w.device).float()  # [90, 4, 4]
    C, K = w.shape
    blk = w.float().abs().view(C // 4, 4, K // 4, 4).permute(0, 2, 1, 3)  # [C/4, K/4, 4, 4]
    score = torch.einsum("abij,pij->abp", blk, P)
    best = score.argmax(-1)
    m = P[best].bool()  # [C/4, K/4, 4, 4]
    return m.permute(0, 2, 1, 3).reshape(C, K)


def is_24(m: torch.Tensor, dim: int) -> bool:
    mm = m.transpose(dim, -1) if dim != -1 else m
    return bool((mm.reshape(*mm.shape[:-1], -1, 4).sum(-1) <= 2).all())


def bench(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return dict(median_ms=statistics.median(ts), min_ms=min(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8192)
    ap.add_argument("--Cs", default="64,256,2048")
    ap.add_argument("--panels", type=int, default=64, help="independent panels per timed call (batched loop)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from provenance import hardware_state, provenance

    dev = torch.device("cuda")
    res = dict(provenance=provenance(Path(common.__file__).resolve().parent.parent), hardware_state=hardware_state(),
               device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
               torch=torch.__version__, cuda_runtime=torch.version.cuda, status="panel microbenchmark only",
               cusparselt_available=bool(torch.backends.cusparselt.is_available()),
               cusparselt_version=torch.backends.cusparselt.version() if torch.backends.cusparselt.is_available()
               else None, panels=[])
    try:
        from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

        res["semi_structured_api"] = "torch.sparse.to_sparse_semi_structured"
        res["force_cutlass"] = bool(getattr(SparseSemiStructuredTensor, "_FORCE_CUTLASS", None))
        res["backend_note"] = ("default backend of this PyTorch build: cuSPARSELt when available and _FORCE_CUTLASS is "
                               "False, else CUTLASS")
    except Exception as e:  # noqa: BLE001
        res["semi_structured_api"] = f"unavailable: {e!r}"
        Path(a.out).write_text(json.dumps(res, indent=1))
        return
    torch.manual_seed(0)
    # transpose-eligibility audit on random K64 weight panels
    w = torch.randn(2048, 64, device=dev).bfloat16()
    m_in = two_four_mask(w)
    res["audit"] = dict(input_24_mask_is_24_along_inputs=is_24(m_in, -1), input_24_mask_is_24_along_neurons=is_24(m_in, 0),
                        neuron_quads_violating_fraction=float((m_in.t().reshape(64, -1, 4).sum(-1) > 2).float().mean()))
    m_bi = bidirectional_mask(w)
    res["audit"].update(bidirectional_is_24_inputs=is_24(m_bi, -1), bidirectional_is_24_neurons=is_24(m_bi, 0),
                        magnitude_kept_input_24=float((w.float().abs() * m_in).sum() / w.float().abs().sum()),
                        magnitude_kept_bidirectional=float((w.float().abs() * m_bi).sum() / w.float().abs().sum()),
                        bidirectional_patterns=int(patterns_2of4_rows_cols().shape[0]))
    for C in [int(c) for c in a.Cs.split(",")]:
        row = dict(C=C, M=a.M, panels=a.panels)
        X = torch.randn(a.panels, a.M, 64, device=dev).bfloat16()
        Wd = torch.randn(a.panels, C, 64, device=dev).bfloat16()
        dZ = torch.randn(a.panels, a.M, C, device=dev).bfloat16()
        Wb = torch.stack([Wd[i] * bidirectional_mask(Wd[i]) for i in range(a.panels)])
        try:
            t0 = time.perf_counter()
            Ws = [to_sparse_semi_structured(Wb[i]) for i in range(a.panels)]  # forward orientation (inputs)
            torch.cuda.synchronize()
            row["compress_fwd_ms_per_panel"] = (time.perf_counter() - t0) * 1e3 / a.panels
            t0 = time.perf_counter()
            WTs = [to_sparse_semi_structured(Wb[i].t().contiguous()) for i in range(a.panels)]  # dX orientation
            torch.cuda.synchronize()
            row["compress_dx_ms_per_panel"] = (time.perf_counter() - t0) * 1e3 / a.panels
            # correctness of both orientations against the dense masked product
            z_s = torch.nn.functional.linear(X[0], Ws[0])
            z_d = X[0].float() @ Wb[0].float().t()
            row["fwd_max_rel_err"] = float((z_s.float() - z_d).abs().max() / z_d.abs().max())
            da_s = torch.nn.functional.linear(dZ[0], WTs[0])
            da_d = dZ[0].float() @ Wb[0].float()
            row["dx_max_rel_err"] = float((da_s.float() - da_d).abs().max() / da_d.abs().max())
            row["dense_fwd"] = bench(lambda: [X[i] @ Wb[i].t() for i in range(a.panels)])
            row["sparse_fwd"] = bench(lambda: [torch.nn.functional.linear(X[i], Ws[i]) for i in range(a.panels)])
            row["dense_dx"] = bench(lambda: [dZ[i] @ Wb[i] for i in range(a.panels)])
            row["sparse_dx"] = bench(lambda: [torch.nn.functional.linear(dZ[i], WTs[i]) for i in range(a.panels)])
            row["dense_dw"] = bench(lambda: [dZ[i].t() @ X[i] for i in range(a.panels)])
            row["status"] = "measured"
        except Exception as e:  # noqa: BLE001
            row["status"] = "unsupported"
            row["reason"] = repr(e)[:500]
        res["panels"].append(row)
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, dict)}), flush=True)
        for k in ("dense_fwd", "sparse_fwd", "dense_dx", "sparse_dx", "dense_dw"):
            if k in row:
                print(f"  C{C} {k}: {row[k]['median_ms']:.3f} ms / {a.panels} panels", flush=True)
        del X, Wd, dZ, Wb
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
