"""Cases of the K8 / weighted-support / budget-matched-anchor study (results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md):
fixed workload, seeds, bounds and the machine-readable manifest (``manifest()``, no GPU; written before any job).

Support specs are rsn_cases specs: ``BP-row/8#ls``, ``FP-mag/8``, ``U8@0.125``, ``WRS/8:L#ls``, ``TopK/8``,
``WRS/16:L#ls``, ``TopK/16``, ``Mix-BP#ls``, ``Mix-WRS:L#ls``, ``Mix-TopK#ls`` and the parent's ``BP-row#ls`` (K16),
``U16@0.25``, ``A8@0.25``, ``dense``.  Seeds: graph seed (envelope graph + its alpha-0 checkpoint), layout /
candidate seed (#ls), parameter init (the checkpoint's), data seed and stochastic-rounding (training) seed of each
fine-tuning run, init seed of the reduced interaction task.
"""
from __future__ import annotations

import math

import rsn_cases as RC
import wps_cases as WC

D, G, M, K, S, R, H = WC.D, WC.G, WC.M, WC.K, WC.S, WC.R, WC.H
W0 = WC.W0
C = 256
Q = R * D // S
GRAPH_SEEDS, LAYOUT_SEEDS = (0, 1, 2), (0, 1)
DEV_SEEDS, FINAL_SEEDS = (0, 1, 2), (3, 4, 5)
FINAL_TEST_BATCH_SEEDS = tuple(range(19_000, 19_008))
FINAL_INTERACTION_SEEDS = tuple(range(59_000, 59_008))
NONINFERIORITY_MARGIN = 0.003
WRS_L = (1, 4, 16)
MIXED_COUNTS = dict(H64=786_432, H16=6_994_976, H8=4_718_592, slots=200_000_000, biases=12_500_000)
K8_CONTROLS = ("U8@0.125", "FP-mag/8")
REFERENCES_25 = ("BP-row", "U16@0.25", "A8@0.25", "dense")


def per_output_bound(t: int, k: int, n_rec: int = 763) -> float:
    return min(1.0, n_rec * math.comb(k, t) / math.comb(D, t))


def manifest() -> dict:
    from sparse_pooling.experimental.kgen_layout import KLayout

    L = KLayout(D, H, G, K, C)
    n_rec_out = math.ceil(max(int(c) for c in L.counts_tensor("cpu")) / 16)
    return dict(schema="granule.kwa_manifest/1",
                model=dict(D=D, G=G, K_envelope=K, C_envelope=C, S=S, R=R, Q_total=Q, H=H, W0=W0, M=M,
                           pooling="Mean(ReLU)", numerics="BF16 inputs / parameters / moments, FP32 accumulation / "
                           "main_grad, split dZ, optimizer race fix", optimizer="L1AdamW lr 1e-4, no L1, wd 0, SR on"),
                budgets=dict(K8=dict(slots=8 * H, rho=0.125, executed="kgen K8, C16 (16-neuron row groups)"),
                             K25=dict(slots=16 * H, rho=0.25), mixed=MIXED_COUNTS),
                bounds=dict(per_output_records_max=n_rec_out,
                            per_output_union_k8={f"t{t}": per_output_bound(t, 8, n_rec_out) for t in (2, 3, 4)},
                            per_output_union_k16={f"t{t}": per_output_bound(t, 16, n_rec_out) for t in (2, 3, 4)}),
                seeds=dict(graph=list(GRAPH_SEEDS), layout=list(LAYOUT_SEEDS), development=list(DEV_SEEDS),
                           final=list(FINAL_SEEDS), final_test_batches=list(FINAL_TEST_BATCH_SEEDS),
                           development_validation_batches=[9_000, 9_001, 9_002, 9_003],
                           final_interaction_batches=list(FINAL_INTERACTION_SEEDS), input=WC.INPUT_SEED),
                checkpoints={f"C256_g{s}": WC.ckpt(256, s) for s in GRAPH_SEEDS},
                noninferiority=dict(margin=NONINFERIORITY_MARGIN, rule="paired 95 % t interval of (new - control) "
                                    "over the final training seeds: noninferior iff upper < margin; improvement iff "
                                    "upper < 0; otherwise inconclusive (never 'equivalent')"),
                wrs_candidate_counts=list(WRS_L), spec_grammar=RC.__doc__)
