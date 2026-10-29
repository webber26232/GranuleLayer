"""Cases of the TopK/16 confirmation / coverage-constrained TopK round (results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md):
fixed workload, seeds, test protocols, bounds, preregistered comparisons and the machine-readable manifest
(``manifest()``, no GPU; written before any job).

Support specs are rsn_cases specs: ``TopK/k``, ``Cover-TopK/k`` (k 8 / 16; deterministic, no layout seed),
``BP-row/8#ls``, ``BP-row#ls`` (K16), ``U8@0.125``, ``U16@0.25``, ``A8@0.25``, ``Mix-TopK#0`` (structural / performance
reference and a member of the new cohort), ``dense``.  Seeds: graph seed (envelope graph + its alpha-0 checkpoint),
layout seed (BP only), data / stochastic-rounding seed of each fine-tuning run, init / SR seeds of the reduced task.
"""
from __future__ import annotations

import math

import kwa_cases as KW
import wps_cases as WC

D, G, M, K, S, R, H = WC.D, WC.G, WC.M, WC.K, WC.S, WC.R, WC.H
W0 = WC.W0
C = 256
Q = R * D // S
GRAPH_SEEDS, BP_LAYOUT_SEEDS = (0, 1, 2), (0, 1)
DEV_SEEDS = (0, 1, 2)
FINAL_SEEDS = (6, 7, 8, 9, 10, 11)  # fresh data / SR seeds of the new cohort (3 / 4 / 5 used by the previous round)
GRAPH_CHECK_SEED = 6
TEST_PROTOCOL = "tkc29000"  # rsn_learn.TEST_PROTOCOLS: batches 29_000..29_007, reserved before any screen
TEST_BATCH_SEEDS = tuple(range(29_000, 29_008))
HISTORICAL_TEST = dict(protocol="kwa19000", batches=list(KW.FINAL_TEST_BATCH_SEEDS),
                       status="examined in the weighted-support round: historical evidence, not a locked test here")
INTERACTION_TEST_PROTOCOL = "tkc69000"
INTERACTION_TEST_SEEDS = tuple(range(69_000, 69_008))
INTERACTION_INIT_SR = ((20, 0), (21, 1), (22, 2))  # (init seed, training / SR seed) pairs
INTERACTION_LR, INTERACTION_LR_READOUT = 1e-2, 1e-2
INTERACTION_STEPS, INTERACTION_LONG = 1000, 4000
NONINFERIORITY_MARGIN = KW.NONINFERIORITY_MARGIN  # 0.003 absolute
PRIMARY = (("TopK/16", "U16@0.25"), ("Cover-TopK/8", "TopK/8"), ("Cover-TopK/16", "TopK/16"))
FAMILY_ALPHA = 0.05

K16_SET = ("TopK/16", "Cover-TopK/16", "BP-row#0", "BP-row#1", "U16@0.25", "A8@0.25", "Mix-TopK#0")
K8_SET = ("TopK/8", "Cover-TopK/8", "BP-row/8#0", "BP-row/8#1", "U8@0.125")
FORMAT = {"TopK/16": "indexed", "Cover-TopK/16": "indexed", "BP-row#0": "indexed", "BP-row#1": "indexed",
          "U16@0.25": "wide", "A8@0.25": "wide", "Mix-TopK#0": "grouped", "TopK/8": "indexed",
          "Cover-TopK/8": "indexed", "BP-row/8#0": "indexed", "BP-row/8#1": "indexed", "U8@0.125": "wide"}
FINAL_METHODS = ("TopK/8", "Cover-TopK/8", "BP-row/8#0", "U8@0.125", "TopK/16", "Cover-TopK/16", "BP-row#0",
                 "U16@0.25", "A8@0.25", "Mix-TopK#0", "dense")
INTERACTION_SPECS = ("TopK/8", "Cover-TopK/8", "BP-row/8#0", "TopK/16", "Cover-TopK/16", "BP-row#0", "dense")
INTERACTION_CONTINUED = ("TopK/8", "Cover-TopK/8", "BP-row/8#0")


def per_output_bound(t: int, k: int, n_rec: int = 763) -> float:
    return min(1.0, n_rec * math.comb(k, t) / math.comb(D, t))


def manifest() -> dict:
    from sparse_pooling.experimental.kgen_layout import KLayout

    L = KLayout(D, H, G, K, C)
    n_rec_out = math.ceil(max(int(c) for c in L.counts_tensor("cpu")) / 16)
    return dict(schema="granule.tkc_manifest/1",
                model=dict(D=D, G=G, K_envelope=K, C_envelope=C, S=S, R=R, Q_total=Q, H=H, W0=W0, M=M,
                           pooling="Mean(ReLU)", numerics="BF16 inputs / parameters / moments, FP32 accumulation / "
                           "main_grad, split dZ, optimizer race fix, stochastic rounding on",
                           optimizer="L1AdamW lr 1e-4, no L1, wd 0"),
                budgets=dict(k8=dict(slots=8 * H, rho=0.125), k16=dict(slots=16 * H, rho=0.25),
                             biases=H, pooling_denominator="original n_g"),
                cover_constraint="per original record: every valid row group keeps exactly k distinct local inputs; "
                                 "every one of the 64 local inputs is kept by >= 1 valid group; infeasible iff m k < 64",
                bounds=dict(per_output_records_max=n_rec_out,
                            per_output_union_k8={f"t{t}": per_output_bound(t, 8, n_rec_out) for t in (2, 3, 4)},
                            per_output_union_k16={f"t{t}": per_output_bound(t, 16, n_rec_out) for t in (2, 3, 4)}),
                seeds=dict(graph=list(GRAPH_SEEDS), bp_layout=list(BP_LAYOUT_SEEDS), development=list(DEV_SEEDS),
                           final=list(FINAL_SEEDS), graph_check=GRAPH_CHECK_SEED,
                           development_validation_batches=[9_000, 9_001, 9_002, 9_003],
                           final_test=dict(protocol=TEST_PROTOCOL, batches=list(TEST_BATCH_SEEDS)),
                           historical_test=HISTORICAL_TEST,
                           interaction=dict(init_sr=[list(p) for p in INTERACTION_INIT_SR],
                                            development_batches=[49_000, 49_001, 49_002, 49_003],
                                            final_test=dict(protocol=INTERACTION_TEST_PROTOCOL,
                                                            batches=list(INTERACTION_TEST_SEEDS)))),
                checkpoints={f"C256_g{s}": WC.ckpt(256, s) for s in GRAPH_SEEDS},
                primary_comparisons=[list(p) for p in PRIMARY],
                decision_rule=dict(margin=NONINFERIORITY_MARGIN, family_alpha=FAMILY_ALPHA, family_size=len(PRIMARY),
                                   upper="mean(delta) + t_{1 - alpha / 3, df 5} sd(delta) / sqrt(6) (one-sided, "
                                         "Bonferroni over the 3 primary comparisons)",
                                   noninferior="upper < margin", improvement="upper < 0",
                                   otherwise="inconclusive (never 'equal' or 'inferior')",
                                   descriptive="pointwise two-sided 95 % paired t intervals, labelled descriptive"),
                spec_grammar=__doc__)
