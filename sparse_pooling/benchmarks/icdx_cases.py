"""Cases of the interaction-coverage / K16 dX round (results/INTERACTION_COVERAGE_K16_DX.md): fixed workload, seeds,
test protocols, bounds, preregistered contrasts, selection rules and the machine-readable manifest (``manifest()``,
no GPU; written before any job) with the seed collision check (``seed_collisions()``).

Support specs are rsn_cases specs.  New candidates: ``Balanced-TopK/16#<quota layout>``,
``Cover-Mix32:p1#<class layout>``, ``Cover-Mix32:p2#<class layout>``; controls ``Cover-TopK/16``, ``BP-row#0``,
``U16@0.25``, ``Mix-TopK#0``, ``dense``.  Backends: ``legacy`` (the original K16 dX launch) or a K16 dX tiling variant
``mt<MT>_nw<NW>_wg<WG>`` (experimental/kgen_dx_variants.py).
"""
from __future__ import annotations

import math

import tkc_cases as TC
import wps_cases as WC

D, G, M, K, S, R, H = WC.D, WC.G, WC.M, WC.K, WC.S, WC.R, WC.H
C = 256
Q = R * D // S
W0 = WC.W0
GRAPH_SEEDS = (0, 1, 2)
DEV_SEEDS = (0, 1, 2)
FINAL_SEEDS = (12, 13, 14, 15, 16, 17)  # fresh data / SR seeds (6..11 used by the previous round)
GRAPH_CHECK_SEED = 12
TEST_PROTOCOL = "icdx39000"
TEST_BATCH_SEEDS = tuple(range(39_000, 39_008))
INTERACTION_TEST_PROTOCOL = "icdx89000"
INTERACTION_TEST_SEEDS = tuple(range(89_000, 89_008))
ANALYSIS_SEEDS = tuple(range(97_000, 97_064))
INTERACTION_INIT_SR = ((30, 0), (31, 1), (32, 2))
INTERACTION_LR, INTERACTION_LR_READOUT = 1e-2, 1e-2
INTERACTION_STEPS, INTERACTION_LONG = 1000, 4000
ANOVA_PER_ORDER, ANOVA_SAMPLE_SEED = 32, 20261105
ORDER_SEED = 20261105
MARGIN = 0.003
FAMILY_ALPHA = 0.05
FAMILY_SIZE = 4
NEW = ("Balanced-TopK/16#0", "Cover-Mix32:p1#0", "Cover-Mix32:p2#0")
MIXED = ("Cover-Mix32:p1#0", "Cover-Mix32:p2#0")
CONTROLS = ("Cover-TopK/16", "BP-row#0", "U16@0.25", "Mix-TopK#0", "dense")
CONTROL_TREATMENT = {"Cover-TopK/16": "energy", "BP-row#0": "energy", "U16@0.25": "energy", "Mix-TopK#0": "energy",
                     "dense": "none"}
CORE8 = NEW + CONTROLS
CONTINUITY_REFERENCE = {  # 32-update seed-0 records of the controls in earlier rounds (continuity check)
    "Cover-TopK/16": "topk16_coverage_constrained_topk/dev32/learn_Cover-TopK_k16_g0_energy_s32_d0.json",
    "BP-row#0": "weighted_random_supports_anchors/m25_screen/learn_BP-row_ls0_g0_energy_s32_d0.json",
    "U16@0.25": "weighted_random_supports_anchors/m25_screen/learn_U16_r0.25_g0_energy_s32_d0.json",
    "Mix-TopK#0": "weighted_random_supports_anchors/m25_screen/learn_Mix-TopK_ls0_g0_energy_s32_d0.json",
    "dense": "weighted_random_supports_anchors/m25_screen/learn_dense_g0_none_s32_d0.json"}
REUSABLE_DEV256 = {  # earlier 256-update development records (seeds 0 / 1 / 2) reusable after continuity
    "Cover-TopK/16": "topk16_coverage_constrained_topk/dev256/learn_Cover-TopK_k16_g0_energy_s256_d{d}.json",
    "Mix-TopK#0": "weighted_random_supports_anchors/m25_dev/learn_Mix-TopK_ls0_g0_energy_s256_d{d}.json"}
FORMAT = {"Cover-TopK/16": "indexed", "Balanced-TopK/16#0": "indexed", "Balanced-TopK/16#1": "indexed",
          "BP-row#0": "indexed", "BP-row#1": "indexed", "Cover-Mix32:p1#0": "grouped", "Cover-Mix32:p2#0": "grouped",
          "Cover-Mix32:p1#1": "grouped", "Cover-Mix32:p2#1": "grouped", "U16@0.25": "wide", "Mix-TopK#0": "grouped",
          "TopK/16": "indexed", "dense": "dense"}
POPULATIONS = {"Cover-Mix32:p1": dict(H32=786_432, H16=10_140_704, H8=1_572_864),
               "Cover-Mix32:p2": dict(H32=1_572_864, H16=7_781_408, H8=3_145_728)}
RECORD_CLASSES = {"Uniform K16": {16: 16}, "Mix32 p1": {32: 1, 16: 13, 8: 2}, "Mix32 p2": {32: 2, 16: 10, 8: 4}}
DX_LEGACY = "legacy"
DX_SCREEN_SUPPORTS = ("Cover-TopK/16", "BP-row#0")
SCREEN_ROUNDS, CONFIRM_ROUNDS = (2, 2, 2), (4, 2, 3)
SMALL_BATCH_GUARD = 64


def incidence(classes: dict, t: int) -> int:
    return sum(n * math.comb(k, t) for k, n in classes.items())


def union_bounds_763() -> dict:
    """Per-output triple union bounds at the maximum 763 row groups per output (47 full records + one 11-group
    tail record)."""
    out = {}
    for name, cls in RECORD_CLASSES.items():
        p = cls.get(32, 0)
        tail = {32: p, 16: 11 - 3 * p, 8: 2 * p} if p else {16: 11}
        out[name] = {f"t{t}": min(1.0, (47 * incidence(cls, t) + incidence(tail, t)) / math.comb(D, t)) for t in (2, 3, 4)}
    return out


def seed_collisions() -> dict:
    """Every reserved batch-seed set of this round against every other seed set of the code and earlier plans (the
    full-size batch(seed) and reduced xbatch(seed) generators share torch CPU seeds, so all sets are checked jointly)."""
    import rsn_interaction as RI
    import rsn_learn as RL
    import wps_interaction as WI
    from k64_learn import REG, WARMUP

    sets = {}
    sets["full_train_d0_17_s256"] = {10_000 + WARMUP + REG + i + 1_000_000 * d for d in range(18) for i in range(256)}
    sets["full_checkpoint_construction"] = set(range(10_000, 10_000 + WARMUP + REG + 32))
    sets["full_validation"] = set(range(9_000, 9_004))
    sets["full_calibration"] = set(range(8_000, 8_004))
    for name, s in RL.TEST_PROTOCOLS.items():
        sets[f"full_test_{name}"] = set(s)
    sets["reduced_train_0_3999"] = {WI.TRAIN_SEED + i for i in range(INTERACTION_LONG)}
    sets["reduced_heldout"] = set(WI.HELD_SEEDS)
    sets["reduced_calibration"] = {WI.TRAIN_SEED - 1}
    for name, s in RI.TEST_PROTOCOLS.items():
        sets[f"reduced_test_{name}"] = set(s)
    sets["reduced_analysis"] = set(RI.ANALYSIS_SEEDS)
    reserved = ("full_test_icdx39000", "reduced_test_icdx89000", "reduced_analysis")
    rows = []
    for r in reserved:
        for name, s in sets.items():
            if name == r:
                continue
            c = sets[r] & s
            rows.append(dict(reserved=r, other=name, collisions=sorted(c)[:10], n=len(c)))
    bad = [x for x in rows if x["n"]]
    if bad:
        raise SystemExit(f"seed collisions: {bad}")
    assert tuple(RL.TEST_PROTOCOLS[TEST_PROTOCOL]) == TEST_BATCH_SEEDS
    assert tuple(RI.TEST_PROTOCOLS[INTERACTION_TEST_PROTOCOL]) == INTERACTION_TEST_SEEDS
    assert tuple(RI.ANALYSIS_SEEDS) == ANALYSIS_SEEDS
    return dict(checked_pairs=len(rows), collisions=0, sets={k: [min(v), max(v), len(v)] for k, v in sets.items()},
                old_protocols_unchanged=dict(kwa19000=list(RL.TEST_PROTOCOLS["kwa19000"]),
                                             tkc29000=list(RL.TEST_PROTOCOLS["tkc29000"]),
                                             kwa59000=list(RI.TEST_PROTOCOLS["kwa59000"]),
                                             tkc69000=list(RI.TEST_PROTOCOLS["tkc69000"])))


PRIMARY = (("Balanced-TopK/16#0", "Cover-TopK/16"), ("<selected Mix32>", "Cover-TopK/16"),
           ("<selected Mix32>", "Mix-TopK#0"), ("Cover-TopK/16@indexed:<chosen dX>", "Cover-TopK/16@indexed:legacy"))


def manifest() -> dict:
    from sparse_pooling.experimental import kgen_dx_variants as DXV

    return dict(schema="granule.icdx_manifest/1",
                model=dict(D=D, G=G, K_envelope=K, C_envelope=C, S=S, R=R, Q_total=Q, H=H, W0=W0, M_perf=M, M_learn=512,
                           records=49_152, full_records=48_128, tail_records=1_024, valid_row_groups=781_312,
                           pooling="Mean(ReLU), original per-output denominator",
                           numerics="BF16 inputs / parameters / moments, FP32 accumulation / main_grad, split dZ, "
                                    "stochastic rounding on, no L1, wd 0"),
                budget=dict(valid_slots=16 * H, biases=H),
                candidates=dict(new=list(NEW), controls=list(CONTROLS), control_treatment=CONTROL_TREATMENT,
                                continuity_controls=["TopK/16", "Cover-TopK/8"],
                                populations=POPULATIONS,
                                record_tuple_incidences={n: {f"t{t}": incidence(c, t) for t in (2, 3, 4)}
                                                         for n, c in RECORD_CLASSES.items()},
                                per_output_union_bounds_763=union_bounds_763()),
                seeds=dict(graph=list(GRAPH_SEEDS), development=list(DEV_SEEDS), final=list(FINAL_SEEDS),
                           graph_check=GRAPH_CHECK_SEED, quota_layout_primary=0, quota_layout_sensitivity=1,
                           class_layout_primary=0, class_layout_structural=1,
                           validation_batches=[9_000, 9_001, 9_002, 9_003],
                           final_test=dict(protocol=TEST_PROTOCOL, batches=list(TEST_BATCH_SEEDS)),
                           interaction=dict(init_sr=[list(p) for p in INTERACTION_INIT_SR],
                                            development_batches=[49_000, 49_001, 49_002, 49_003],
                                            train_batches=[50_000, 53_999],
                                            final_test=dict(protocol=INTERACTION_TEST_PROTOCOL,
                                                            batches=list(INTERACTION_TEST_SEEDS)),
                                            analysis_batches=[ANALYSIS_SEEDS[0], ANALYSIS_SEEDS[-1]],
                                            anova_sample=dict(per_order=ANOVA_PER_ORDER, seed=ANOVA_SAMPLE_SEED)),
                           historical=dict(tkc29000="previous locked test (historical)", tkc69000="historical",
                                           kwa19000="historical", kwa59000="historical"),
                           timing_order_seed=ORDER_SEED),
                checkpoints={f"C256_g{s}": WC.ckpt(256, s) for s in GRAPH_SEEDS},
                primary_contrasts=[list(p) for p in PRIMARY],
                decision_rule=dict(margin=MARGIN, family_alpha=FAMILY_ALPHA, family_size=FAMILY_SIZE,
                                   upper="mean(delta) + t.ppf(1 - 0.05 / 4, n - 1) sd(delta) / sqrt(n) (paired, keyed by "
                                         "seed; n / df from the validated cohort; t = 3.16338145 at df 5)",
                                   improvement="upper < 0", noninferior="0 <= upper < margin",
                                   otherwise="inconclusive", not_run="contrast 4 not estimable if no backend qualifies "
                                   "(alpha not recycled)", exploratory="U16 and other unregistered contrasts",
                                   descriptive="pointwise two-sided 95 % intervals separately"),
                dx_search=dict(screen=[list(x) for x in DXV.SCREEN], legacy="(4, 8, 2) through the original entry",
                               screen_supports=list(DX_SCREEN_SUPPORTS), screen_rounds=list(SCREEN_ROUNDS),
                               confirm_rounds=list(CONFIRM_ROUNDS), max_finalists=2, max_extra_implementations=2,
                               selection="complete step time, not dX time alone"),
                backend_rule=dict(correctness="all gates pass", cover_m8192_gain=">= 5 % lower graph-mean complete step",
                                  regressions="<= 3 % graph-mean complete-step regression on BP-row/16 and the selected "
                                              "Mix32 at M8192 and on Cover-TopK/16 at M512",
                                  otherwise="keep legacy, report the negative result"),
                prior_decision=dict(comparison="TopK/16 - U16 (tkc29000)", mean=0.00216211, adjusted_upper=0.00306761,
                                    margin=0.003, verdict="noninferiority not established; not extended this round"),
                spec_grammar=__doc__)
