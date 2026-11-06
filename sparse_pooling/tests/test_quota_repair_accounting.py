"""quota_repair synchronization accounting (experimental/local_supports.py; integration review 5.2): the counter is
named for what it counts (explicit stop-condition polls), ``host_syncs`` survives only as a documented deprecated alias
for the historical records, and the record carries the qualification that per-round dynamic-length boolean indexing
also synchronizes on CUDA.  (Balanced-TopK is a research comparison; Cover-TopK/16 never calls quota_repair.)"""
import torch

from sparse_pooling.experimental import local_supports as LS
from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_
from sparse_pooling.experimental.kgen_lowered import Envelope
from sparse_pooling.supergroups import ConnectivityPlan


def test_polling_checks_named_and_alias_documented():
    D, H, G = 256, 8 * 431 + 4, 8
    L = KLayout(D, H, G, 64, 256)
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(L.weight_shape, generator=g) / 8).bfloat16()
    b = torch.zeros(L.bias_shape, dtype=torch.bfloat16)
    zero_padding_(L, w, b)
    env = Envelope.generate(D, H, G, 256, ConnectivityPlan(64, 2, seed=1), w, b, device="cpu")
    s = LS.build("Balanced-TopK", 16, L, env.kg_buckets, w=w, graph_seed=1, device="cpu", Q=env.sg_feat.shape[0])
    rep = s.stats["repair"]
    assert rep["polling_checks"] >= 1 and rep["host_syncs"] == rep["polling_checks"]
    assert "boolean indexing" in rep["synchronization_note"]
    doc = LS.quota_repair.__doc__
    assert "DEPRECATED alias" in doc and "act[live]" in doc and "polling_checks" in doc
    assert "SYNCHRONIZES" in LS.cover_repair.__doc__  # the corrected Cover-TopK claim (research erratum 2.2)
