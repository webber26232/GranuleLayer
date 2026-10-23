"""Dispatch decision matrix: what the CUDA backend selects for a grid of synthetic workloads,
without building or launching kernels (host-only; no GPU needed).

The matrix covers the grouped (input-supergroup) phases, the deterministic reduction, the generic
kernels' tilings / epilogues / staging and representative explicit overrides.  Each decision is
recorded as a value or as the exception type it raises.  tests/test_dispatch.py compares the
resolver under the RTX 3090 profile against the fixture tests/data/plan_matrix_rtx3090.json, which
was generated from the reviewed implementation before the dispatch refactor:

    python plan_matrix.py --impl-root ../GranuleLayer-det --impl legacy --out plan_matrix_rtx3090.json

``--impl legacy`` drives the CudaBackend methods of the pre-refactor kernels/cuda_ops.py with a mocked
device capability (8, 6); ``--impl resolver`` drives backends.dispatch with a synthetic RTX 3090
capability record (tests import ``decisions`` directly).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# (C, S, mean units per nonempty bucket, M) of the grouped matrix; D = 1024, G = 1024 unless noted
GROUPED_C = (8, 64)
GROUPED_S = (8, 16, 32, 64, 128, 256, 512)
BUCKET_UNITS = (0.5, 1.0, 1.9, 2.5, 3.5, 6.0, 14.0)
BATCHES = (64, 1000, 1024, 8192)
OVERRIDES = [
    {},
    dict(cuda_grouped_dx_plan="tiles"),
    dict(cuda_grouped_dx_plan="shared_t64"),
    dict(cuda_grouped_dx_plan="private_t32"),
    dict(cuda_grouped_dx_tokens=64),
    dict(cuda_grouped_dx_tokens=128, cuda_grouped_dx_warps=16),
    dict(cuda_grouped_dx_reduction=8),
    dict(cuda_grouped_dx_reduction=1),
    dict(cuda_grouped_dx_scatter="baseline"),
    dict(cuda_grouped_dx_scatter="vectorized"),
    dict(cuda_grouped_dx_scatter="swizzle4_capped"),
    dict(cuda_grouped_dx_scatter="feature_owner", cuda_grouped_dx_tokens=64),
    dict(cuda_grouped_fwd_tokens=32, cuda_grouped_fwd_warps=4),
    dict(cuda_grouped_dw_cfg=1),
    dict(cuda_grouped_dw_projection="compact"),
    dict(cuda_grouped_dw_projection="padded"),
    dict(cuda_input_supergroup_backend="grouped"),
    dict(cuda_input_supergroup_backend="generic"),
    dict(cuda_grouped_dx="generic"),
    dict(cuda_dx_epilogue="generic", cuda_dw_epilogue="generic"),
    dict(cuda_dx_epilogue="fast"),
    dict(cuda_dw_epilogue="mask"),
    dict(cuda_grouped_backward="fused"),
    dict(deterministic_workspace_bytes=4 << 20),
]
# generic (non-supergroup) kernels: pooling x activation x C
GENERIC_POOLS = [("mean", "relu"), ("mean", "identity"), ("logmeanexp", "identity"), ("power_mean", "relu"),
                 ("softmax_weighted", "identity"), ("max", "identity"), ("topk_mean", "identity")]
GENERIC_C = (8, 16, 32, 64)
GENERIC_H = (1_000_000, 2_000_000, 100_000_000)
GENERIC_OVERRIDES = [{}, dict(cuda_dw_staging="full"), dict(cuda_dw_staging="indexed"), dict(cuda_dw_cfg=2),
                     dict(cuda_dx_warps=8), dict(input_layout="token_major"), dict(cuda_block_m=16),
                     dict(grad_precision="bf16")]


def _call(fn, *a):
    try:
        v = fn(*a)
        return list(v) if isinstance(v, tuple) else v
    except Exception as e:  # noqa: BLE001
        return {"raises": type(e).__name__}


def grouped_cases(quick=False):
    cs = GROUPED_C
    ss = GROUPED_S if not quick else (16, 64)
    us = BUCKET_UNITS if not quick else (1.0, 6.0)
    for C, S, ub, B, prec, act, det in itertools.product(cs, ss, us, BATCHES, ("split", "bf16"), ("relu", "identity"),
                                                          (False, True)):
        for ov in (OVERRIDES if (B == 8192 and ub in (1.0, 6.0) and S in (16, 64) and act == "relu") else [{}]):
            yield dict(C=C, S=S, ub=ub, B=B, D=1024, G=1024, H=100_000_000, precision=prec, act=act, det=det,
                       pool="mean", plan=ov)
    for D, S in ((256, 16), (256, 64), (4096, 64), (16384, 16)):  # other input widths (shared-memory fits)
        for C, det in itertools.product(cs, (False, True)):
            yield dict(C=C, S=S, ub=6.0, B=1024, D=D, G=64, H=20_000_000, precision="split", act="relu", det=det,
                       pool="mean", plan={})


def generic_cases():
    for (pool, act), C, H in itertools.product(GENERIC_POOLS, GENERIC_C, GENERIC_H):
        for ov in GENERIC_OVERRIDES:
            yield dict(C=C, S=0, ub=0.0, B=8192, D=1024, G=1024, H=H, precision=ov.get("grad_precision", "split"),
                       act=act, det=False, pool=pool, plan={k: v for k, v in ov.items() if k != "grad_precision"})


def make_rt(case, sp):
    """A runtime stand-in with the fields the dispatch reads (no tensors, no kernels)."""
    from sparse_pooling.config import LevelSpec, PhysicalPlan
    from sparse_pooling.layout import SparseLayout

    L = SparseLayout(case["D"], case["H"], case["G"], 8, case["C"], True, packed=True)
    kw = dict(case["plan"])
    if case["det"]:
        kw["reduction"] = "deterministic"
    lv = LevelSpec(type=case["pool"], activation=case["act"]) if case["pool"] != "topk_mean" else \
        LevelSpec(type="topk_mean", k=8)
    plan = PhysicalPlan(grad_precision=case["precision"], **kw)
    sg = None
    if case["S"]:
        ub = case["ub"]
        bs = dict(mean_units_per_nonempty_bucket=ub, mean_units_per_possible_bucket=ub / 2, bucket_occupancy=0.5)
        sg = SimpleNamespace(S=case["S"], bucket_stats=bs, mean_bucket_units=lambda ub=ub: ub)
    return SimpleNamespace(layout=L, level=lv, plan=plan, sg=sg, det=case["det"], last_info={},
                           units=SimpleNamespace(S=1, BPS=L.blocks_per_group, U=L.groups),
                           idx=SimpleNamespace(device="cuda:0"))


def legacy_decisions(case):
    """Decisions of the pre-refactor kernels/cuda_ops.CudaBackend (device capability mocked)."""
    import torch
    from sparse_pooling.kernels import cuda_ops

    torch.cuda.get_device_capability = lambda *a, **k: (8, 6)
    be = object.__new__(cuda_ops.CudaBackend)
    rt = make_rt(case, None)
    B = case["B"]
    out = dict(supports_forward=_call(be.supports_forward, rt, False), supports_backward=_call(be.supports_backward, rt))
    sxb = 1 if (rt.plan.input_layout == "feature_major" and B % 8 == 0) else case["D"]
    staging = _call(cuda_ops.CudaBackend.dw_staging, rt.plan, rt.layout, "cuda:0", sxb, None)
    out["dw_staging"] = staging
    if rt.sg is not None:
        out["det_reason_empty"] = _call(lambda: be.det_reason(rt, B) == "")
        for ph in ("fwd", "dx", "dw"):
            out[f"use_{ph}"] = _call(be.sg_use, rt, ph, B)
        out["fwd_cfg"] = _call(be.sg_fwd_cfg, rt)
        out["dx_plan"] = _call(lambda: be.sg_dx_plan(rt, B)[0])
        out["dx_cfg"] = _call(be.sg_dx_cfg, rt, B)
        if isinstance(out["dx_cfg"], list) and not isinstance(out["dx_plan"], dict):
            T, nw, wg, _ = out["dx_cfg"]
            out["dx_scatter"] = _call(lambda: be.sg_dx_scatter(rt, T, nw, wg, out["dx_plan"])[0])
        out["dw_cfg"] = _call(be.sg_dw_cfg, rt)
    sg_dw = out.get("use_dw") is True
    sg_any = sg_dw or out.get("use_dx") is True
    dx_epi = _call(cuda_ops._epilogue, rt.plan.cuda_dx_epilogue, rt.level, rt.plan.grad_precision, "dx")
    dw_epi = (_call(cuda_ops._epilogue, rt.plan.cuda_dw_epilogue, rt.level, rt.plan.grad_precision, "dw")
              if (staging == "indexed" or sg_dw) else "generic")
    out.update(dx_epilogue=dx_epi, dw_epilogue=dw_epi)
    if rt.sg is not None and sg_dw and isinstance(dw_epi, str):
        out["dw_projection"] = _call(lambda: be.sg_dw_projection(rt, dw_epi)[0])
    if rt.sg is not None and rt.plan.cuda_grouped_backward in ("fused", "fused_recompute") and sg_any:
        out["fused_ok"] = _call(lambda: be.sg_fused_reason(rt, B, dx_epi, dw_epi) == "")
    plan, L, lv = rt.plan, rt.layout, rt.level
    cfg = plan.cuda_dw_cfg if plan.cuda_dw_cfg >= 0 else cuda_ops.DW_CFG[L.block]
    if plan.cuda_dw_cfg < 0 and staging == "indexed":
        cfg = cuda_ops.DW_CFG_INDEXED[lv.type] if lv.type in cuda_ops.DW_CFG_INDEXED else cfg
    if plan.cuda_dw_cfg < 0 and L.block == 8 and lv.type != "mean":
        cfg = 0
    if cfg == 3 and staging == "full":
        cfg = 0
    out["generic_dw_cfg"] = cfg
    out["dx_warps"] = plan.cuda_dx_warps or cuda_ops.dx_warps_default(L.block, rt.units.BPS, L.unit_blocks)
    if rt.sg is not None and case["det"] and out.get("det_reason_empty") is True:
        out["det_window"] = _call(be.det_window, rt, 1024, 64, B, "dx")
    return out


def decisions(case, impl="resolver", caps=None, profile=None):
    if impl == "legacy":
        return legacy_decisions(case)
    from sparse_pooling.backends.dispatch import matrix_decisions

    return matrix_decisions(make_rt(case, None), case["B"], caps, profile)


def all_cases(quick=False):
    return list(grouped_cases(quick)) + list(generic_cases())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl-root", default=str(Path(__file__).resolve().parents[2]),
                    help="repository root whose sparse_pooling package is examined")
    ap.add_argument("--impl", default="resolver", choices=["legacy", "resolver"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.impl_root).resolve()))
    import subprocess

    import sparse_pooling

    root = Path(sparse_pooling.__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = [dict(case=c, decisions=decisions(c, args.impl)) for c in all_cases(args.quick)]
    Path(args.out).write_text(json.dumps(dict(schema="granule.plan_matrix/1", impl=args.impl, commit=commit,
                                              device="RTX 3090 (sm_86, capability mocked)", rows=rows), indent=0))
    print(f"{len(rows)} cases -> {args.out} ({args.impl} @ {commit[:12]})")


if __name__ == "__main__":
    main()
