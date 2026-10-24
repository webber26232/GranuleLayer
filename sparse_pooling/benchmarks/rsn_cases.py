"""Cases of the random-support / normalization / optimization-control study
(results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md): support specifications, seeds, checkpoints and the
machine-readable manifest (``manifest()``, no GPU; written before any job runs).

Support spec strings (``support(spec, ...)``):

* ``dense``                       the unpruned K64 model;
* ``<width policy>@<rho>``        experimental/width_masks policies of the parent study (U16@0.25, A8@0.25,
                                  H2@0.05, H2A@0.1, P16@0.25 = FP-mag, ...), magnitude scores;
* ``<local policy>[/k][#ls]``     experimental/local_supports policies (FP-mag, FP-balanced, RP-record, RP-row,
                                  BP-row), fan-in k (default 16), layout seed ls (default 0).

Seeds are separate: ``graph_seed`` (the K64 envelope graph and its checkpoint), ``layout_seed`` (local subsets
only), init / checkpoint seed, data seed and training (stochastic-rounding) seed of each driver.
"""
from __future__ import annotations

import math

import torch

import wps_cases as WC

D, G, M, K, S, R, H = WC.D, WC.G, WC.M, WC.K, WC.S, WC.R, WC.H
W0 = WC.W0
C = 256
LOCAL = ("FP-mag", "FP-balanced", "RP-record", "RP-row", "BP-row")
RANDOMIZED = ("RP-record", "RP-row", "BP-row")
SCREEN_LAYOUT_SEEDS = (0, 1, 2)
CONFIRM_GRAPH_SEEDS, CONFIRM_LAYOUT_SEEDS = (0, 1, 2), (0, 1)
CONTROLS = ("FP-mag", "FP-balanced", "U16@0.25", "A8@0.25", "dense")
OPTIONAL = ("A8@0.1875",)
NORM_FIRST = ("BP-row", "H2@0.05")  # all three treatments
NORM_SET = ("FP-mag", "U16@0.25", "<best randomized>", "A8@0.25", "H2@0.05", "H2A@0.1", "dense")


def parse(spec: str) -> dict:
    if spec == "dense":
        return dict(kind="dense", spec=spec)
    if "@" in spec:
        p, rho = spec.split("@")
        return dict(kind="width", policy=p, rho=float(rho), spec=spec)
    body, _, ls = spec.partition("#")
    p, _, k = body.partition("/")
    if p not in LOCAL:
        raise ValueError(f"support spec {spec!r}")
    return dict(kind="local", policy=p, k=int(k or 16), layout_seed=int(ls or 0), spec=spec)


def spec_id(spec: str) -> str:
    return spec.replace("@", "_r").replace("/", "_k").replace("#", "_ls")


@torch.no_grad()
def support(spec: str, lay, graph_seed: int = 0):
    """(element mask [G, NB, C, 64] uint8 or None for dense, MaskSet / LocalSupport / None, record) of ``spec`` on a
    KgxLayer ``lay`` holding the PRE-pruning weights (selection is from these values; nothing is modified)."""
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental import width_masks as WM

    c = parse(spec)
    L = lay.layout
    if c["kind"] == "dense":
        return None, None, dict(c, rho_mask=1.0)
    if c["kind"] == "width":
        ms = WM.build(c["policy"], c["rho"], lay.weight.detach(), L)
        em = ms.element_mask(L)
        st = {k: v for k, v in ms.stats.items() if "hist" not in k}
        return em, ms, dict(c, rho_mask=ms.stats["rho_mask"], mask_sha256=SP.mask_fingerprint(em), mask_stats=st)
    sup = LS.build(c["policy"], c["k"], L, lay.kg_buckets, w=lay.weight.detach(), graph_seed=graph_seed,
                   layout_seed=c["layout_seed"], device=lay.weight.device)
    em = sup.element_mask(L)
    st = {k: v for k, v in sup.stats.items() if k != "selection"}
    return em, sup, dict(c, rho_mask=sup.enabled_slots() / (64 * L.hidden_dim), mask_sha256=SP.mask_fingerprint(em),
                         support_fingerprint=sup.fingerprint(), support_stats=st)


def manifest() -> dict:
    """Every requested study case with its role and status placeholder (statuses are appended by the job runner)."""
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.experimental import width_masks as WM

    L = KLayout(D, H, G, K, C)
    cnt = WM.layout_counts(L, "cpu")
    Q = R * D // S
    groups, records = cnt["groups"], cnt["records"]
    ideal = {p: {f"t{t}": LS.ideal_prediction(p, t, 16, groups, Q, records) for t in (2, 3, 4)}
             for p in ("RP-record", "RP-row", "BP-row", "FP-balanced")}
    n_rec_out = math.ceil(max(int(c) for c in L.counts_tensor("cpu")) / 16)
    cases = []
    for ls in SCREEN_LAYOUT_SEEDS:
        for p in RANDOMIZED:
            cases.append(dict(stage="screen", support=f"{p}#{ls}", graph_seed=0, formats=["indexed_k16"]))
    cases += [dict(stage="screen", support="FP-mag", graph_seed=0, formats=["indexed_k16", "tile", "wide"]),
              dict(stage="screen", support="FP-balanced", graph_seed=0, formats=["indexed_k16", "tile", "wide"]),
              dict(stage="screen", support="U16@0.25", graph_seed=0, formats=["wide", "tile"]),
              dict(stage="screen", support="A8@0.25", graph_seed=0, formats=["wide"]),
              dict(stage="screen", support="A8@0.1875", graph_seed=0, formats=["wide"], optional=True),
              dict(stage="screen", support="dense", graph_seed=0, formats=["dense"])]
    return dict(schema="granule.rsn_manifest/1",
                model=dict(D=D, G=G, K=K, S=S, R=R, H=H, W0=W0, M=M, Q=Q, C_envelope=C,
                           retained=dict(k=16, rho=0.25, slots=16 * H, sharing="16-neuron row groups")),
                geometry=dict(valid_groups=groups, records=records, partial_groups=cnt["partial_groups"],
                              records_per_output_k16=n_rec_out),
                seeds=dict(graph_screen=0, graph_confirm=list(CONFIRM_GRAPH_SEEDS),
                           layout_screen=list(SCREEN_LAYOUT_SEEDS), layout_confirm=list(CONFIRM_LAYOUT_SEEDS),
                           init=WC.INIT_SEED, input=WC.INPUT_SEED),
                checkpoints={f"C256_g{s}": WC.ckpt(256, s) for s in CONFIRM_GRAPH_SEEDS},
                ideal_predictions=ideal,
                per_output_union_bound={f"t{t}": LS.per_output_union_bound(n_rec_out, t, 16) for t in (2, 3, 4)},
                one_tile_union_bound={f"t{t}": Q * 4 * math.comb(16, t) / math.comb(D, t) for t in (2, 3, 4)},
                normalization=dict(first=list(NORM_FIRST), treatments=["none", "fanin", "energy"],
                                   bounded_set=list(NORM_SET), extra="H2@0.05 W-and-bias energy control"),
                cases=cases)
