"""Case manifest of the width-preserving sparsity study (results/WIDTH_PRESERVING_SPARSITY.md): fixed model,
policies x retained densities, random-score controls, the low-density anchor test at C2048, explicit
infeasibility checks, checkpoints and seeds.  ``manifest()`` computes the exact feasibility / skeleton of
EVERY requested case from the KLayout (no GPU) and is written before any job runs.
"""
from __future__ import annotations

import math
from pathlib import Path

D, G, M = 1024, 1024, 8192
K, S, R = 64, 64, 256
H = 12_500_000
W0 = H * K
GRAPH_SEEDS = (0, 1, 2)
INIT_SEED, INPUT_SEED = 17, 23
CKPT_DIR = "/home/evga/Projects/pamachine/k64l1_ckpt"  # K64 study checkpoints (outside git)

PRIMARY = ([("U16", d) for d in (0.05, 0.10, 0.125, 0.1875, 0.25, 0.375, 0.50)] +
           [("P16", d) for d in (0.25, 0.375, 0.50)] +
           [("U8", d) for d in (0.125, 0.1875, 0.25)] +
           [("P8", d) for d in (0.125, 0.1875, 0.25)] +
           [("A16", d) for d in (0.375, 0.50)] +
           [("A8", d) for d in (0.1875, 0.25)] +
           [("H2", 0.05), ("H2", 0.10), ("H2A", 0.10)])
RANDOM_CONTROLS = [("P16", 0.25), ("A8", 0.1875), ("A8", 0.25)]  # identical constraints / budget, random scores
INFEASIBLE_CHECKS = [("P16", 0.10, 256), ("P8", 0.05, 256), ("H2A", 0.05, 256)]  # must be rejected
C2048_TEST = [("H2A", 0.05), ("H2", 0.05), ("U16", 0.05)]  # + the dense control, same graph / checkpoint
ONE_TILE = {("P16", 0.25): 16, ("P8", 0.125): 8}  # exactly one tile per row group: K16 / K8 lowered control
RANDOM_SEED = 1  # score seed of the random-score controls


def ckpt(C: int, seed: int = 0) -> str:
    """The actual alpha = 0 checkpoint of the K64 study for (C, graph seed) (R256, step 128)."""
    return str(Path(CKPT_DIR) / (f"C{C}_R256_prox_a0" + (f"_g{seed}" if seed else "") + "_step128.pt"))


def formats(policy: str, rho: float):
    """Execution formats of one mask: the new compact kernels, the identical-mask dense masked reference, the
    measured 16 x 16 tile kernels (b16 without backbone) and the lowered K16 / K8 control (one-tile cases)."""
    from sparse_pooling.experimental.width_masks import POLICIES

    P = POLICIES[policy]
    out = ["wide", "dense_masked"]
    if P.tb == 16 and not P.backbone:
        out.append("tile")
    if (policy, rho) in ONE_TILE:
        out.append("lowered")
    return out


def case_id(policy, rho, C=256, score="magnitude", seed=0):
    return f"{policy}_r{rho:g}_C{C}" + ("_rnd" if score == "random" else "") + (f"_g{seed}" if seed else "")


def manifest() -> dict:
    """Machine-readable manifest: every requested (policy, rho, C) with exact budget / skeleton / feasibility."""
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgen_layout import KLayout

    rows = []
    counts = {}
    for C in (64, 256, 2048):
        L = KLayout(D, H, G, K, C)
        c = WM.layout_counts(L, "cpu")
        counts[C] = {k: v for k, v in c.items() if k != "nv"}
        counts[C]["Q"] = R * D // S
    req = ([(p, d, 256, "magnitude", "primary") for p, d in PRIMARY] +
           [(p, d, 256, "random", "random-score control") for p, d in RANDOM_CONTROLS] +
           [(p, d, C, "magnitude", "explicit infeasibility check") for p, d, C in INFEASIBLE_CHECKS] +
           [(p, d, 2048, "magnitude", "C2048 low-density anchor test") for p, d in C2048_TEST])
    for p, d, C, score, role in req:
        L = KLayout(D, H, G, K, C)
        f = WM.feasibility(WM.POLICIES[p], d, L, counts=dict(counts[C], nv=None))
        rows.append(dict(id=case_id(p, d, C, score), policy=p, rho_target=d, C=C, score=score, role=role,
                         feasible=f["feasible"], reason=f.get("reason"), budget_slots=f["budget_slots"],
                         skeleton_slots=f["skeleton_slots"], floor_density=f["floor_density"],
                         anchor_neurons=f["anchor_neurons"], bias_only_lower_bound=f.get("bias_only_lower_bound"),
                         formats=formats(p, d) if f["feasible"] else [],
                         lowered_control=(p, d) in ONE_TILE))
    bounds = {f"b{b}": WM.describe_bounds(R * D // S, b, D) for b in (16, 8)}
    sk = {}
    for C in (64, 256, 2048):
        L = KLayout(D, H, G, K, C)
        sk[C] = {p: WM.skeleton(WM.POLICIES[p], L, counts=dict(counts[C], nv=None)) for p in WM.POLICIES}
    return dict(schema="granule.wps_manifest/1", model=dict(D=D, G=G, K=K, S=S, R=R, H=H, W0=W0, M=M,
                                                            Q=R * D // S, logical_C_screen=256),
                seeds=dict(graph=list(GRAPH_SEEDS), init=INIT_SEED, input=INPUT_SEED, random_score=RANDOM_SEED),
                checkpoints={f"C{C}_g{s}": ckpt(C, s) for C in (256, 2048) for s in GRAPH_SEEDS} | {"C64_g0": ckpt(64)},
                layout_counts={str(k): v for k, v in counts.items()}, skeletons={str(k): v for k, v in sk.items()},
                one_tile_coverage_bounds=bounds,
                granularity_bias_only_bounds={f"b{b}": {f"{d:g}": max(0.0, 1 - 64 * d / b)
                                                        for d in (0.05, 0.10, 0.125, 0.1875, 0.25, 0.375, 0.5)}
                                              for b in (16, 8)},
                cases=rows)


def comb(n, t):
    return math.comb(n, t)
