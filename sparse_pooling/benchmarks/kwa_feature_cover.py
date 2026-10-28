"""Per-output feature coverage of local supports on the reduced interaction geometry (H 262,144, C256, graph 0,
init seed 17 = the interaction task's init): distinct local inputs (of the record's 64) enabled in at least one
row group of each output.  CPU only.  Prediction for independent per-group TopK picks: 64 (1 - (1 - k/64)^16)."""
import sys

import torch

import common  # noqa: F401
import wps_cases as WC
from sparse_pooling.experimental import kgen_graph as KG
from sparse_pooling.experimental import local_supports as LS
from sparse_pooling.experimental.kgen_layout import KLayout, init_parameters_by_neuron
from sparse_pooling.supergroups import ConnectivityPlan

H = 262_144
L = KLayout(1024, H, 1024, 64, 256)
counts = L.counts_tensor("cpu")
idx, feat, ubk, _ = KG.generate(1024, counts, 256, 64, L.blocks_per_group, 1, ConnectivityPlan(64, 256, seed=0), "cpu")
bk = KG.unit_to_block_buckets(ubk, counts, 256, 1)
w, _ = init_parameters_by_neuron(L, "cpu", seed=WC.INIT_SEED)
for pol, k in (("BP-row", 8), ("TopK", 8), ("BP-row", 16), ("TopK", 16)):
    s = LS.build(pol, k, L, bk, w=w, graph_seed=0, layout_seed=0, Q=feat.shape[0])
    per_out = s.group_mask().view(1024, -1, 64).any(1).sum(1).float()
    print(f"{pol}/{k}: features covered per output (of 64) mean {float(per_out.mean()):.3f} min {int(per_out.min())} "
          f"| independent-pick prediction {64 * (1 - (1 - k / 64) ** 16):.3f}", flush=True)
sys.exit(0)
