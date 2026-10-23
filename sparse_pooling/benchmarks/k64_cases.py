"""The K64 large-C / index-free / L1 study: architecture manifest, anchors, fixed budget and seeds
(results/K64_LARGE_C_INDEXFREE_L1.md).

The 36 cases (K64 x C {64..2048} x S {64, 128} x R {64, 128, 256}) and their pre-pruning model expectations
are read from the supplied companion JSON (results/k64_large_c_indexfree_l1/supplied), never re-derived;
``check_manifest`` re-runs the supplied script and asserts that it reproduces the JSON byte for byte.
Anchors of the previous sweep are listed separately and never counted as cases.
"""
from __future__ import annotations

import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results" / "k64_large_c_indexfree_l1"
SUPPLIED = RAW / "supplied"
MANIFEST = SUPPLIED / "20261023-02-GranuleLayer_K64_LargeC_L1_Coverage.json"
SCRIPT = SUPPLIED / "20261023-02-GranuleLayer_K64_LargeC_L1_Coverage.py"

D, G, M = 1024, 1024, 8192
K = 64
WEIGHTS = 800_000_000
H = WEIGHTS // K
GRAPH_SEED, INIT_SEED, INPUT_SEED = 0, 17, 23
LR = 1e-4
SOURCE_COMMIT = "87d89aad18b5abebb2eb81d5083ab01d9f3f7689"

# same-session anchors of the random K/C/S/R sweep (labelled separately; their own init rule)
ANCHORS = {
    "E24": dict(K=64, C=256, S=64, R=256, H=12_500_000, backend="kcsr_factory:kgen", role="previous E24 (K64/C256/S64/R256)"),
    "E22": dict(K=64, C=64, S=128, R=32, H=12_500_000, backend="kcsr_factory:kgen",
                role="previous E22 (K64/C64/S128/R32, off-grid anchor)"),
    "E04": dict(K=8, C=64, S=128, R=64, H=100_000_000, backend="native", role="previous E04 (production K8/C64/S128/R64)"),
    "E08": dict(K=16, C=16, S=128, R=256, H=50_000_000, backend="kcsr_factory:kgen", role="previous E08 (K16/C16/S128/R256)"),
}


def load_manifest():
    return json.loads(MANIFEST.read_text())


def cases():
    return [dict(r) for r in load_manifest()["cases"]]


def case(cid: str) -> dict:
    if cid in ANCHORS:
        return dict(ANCHORS[cid], id=cid)
    for r in cases():
        if r["id"] == cid:
            return r
    raise KeyError(cid)


def case_id(C: int, S: int, R: int) -> str:
    return f"K64_C{C}_S{S}_R{R}"


def per_output_bounds():
    return {r["C"]: r for r in load_manifest()["per_output_upper_bounds_before_pruning"]}


def check_manifest():
    """Re-run the supplied script: its manifest() must equal the supplied JSON."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("k64_supplied", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.self_test()
    m = mod.manifest()
    assert json.dumps(m, indent=2) + "\n" == MANIFEST.read_text(), "supplied JSON differs from the supplied script"
    assert len(m["cases"]) == 36 and m["source_commit"] == SOURCE_COMMIT
    return mod
