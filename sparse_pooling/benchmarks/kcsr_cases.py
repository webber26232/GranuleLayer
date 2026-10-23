"""The random K/C/S/R sweep: case manifest, fixed budget and seeds (RANDOM_KCSR_COVERAGE_PERFORMANCE.md).

The 24 cases are read from the supplied manifest (results/random_kcsr_coverage_performance/supplied),
never re-derived here; ``check_manifest`` re-runs the supplied sampler and asserts the cases agree.
Controls are listed separately and are never counted as randomly selected cases.
"""
from __future__ import annotations

import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results" / "random_kcsr_coverage_performance"
SUPPLIED = RAW / "supplied"
MANIFEST = SUPPLIED / "20261023-01-GranuleLayer_Random_KCSR_Manifest.json"
REPRODUCE = SUPPLIED / "20261023-01-GranuleLayer_Random_KCSR_Reproduce.py"

D, G, M = 1024, 1024, 8192
WEIGHTS = 800_000_000
GRAPH_SEED, INIT_SEED, INPUT_SEED = 0, 17, 23
BIAS_STD = 0.0  # documented bias distribution: zero-initialized (the layer default), identical for every K
LR = 1e-4

# separately labelled controls (not part of the random sample)
CONTROLS = {
    "X01": dict(K=8, C=8, S=64, R=16, role="established K8/C8/S64/R16 low-R control (native)"),
}


def load_manifest():
    return json.loads(MANIFEST.read_text())


def cases():
    """[{id, K, C, S, R, H, ...supplied prediction columns}] in manifest order (E01..E24)."""
    return [dict(r) for r in load_manifest()["primary"]]


def case(cid: str):
    if cid in CONTROLS:
        c = dict(CONTROLS[cid], id=cid)
        c["H"] = WEIGHTS // c["K"]
        return c
    for r in cases():
        if r["id"] == cid:
            return r
    raise KeyError(cid)


def hidden(K: int) -> int:
    assert WEIGHTS % K == 0
    return WEIGHTS // K


def check_manifest():
    """Re-run the supplied sampler; the manifest cases must be its accepted sample."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kcsr_reproduce", REPRODUCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows, attempt = mod.select_cases()
    got = [(r["K"], r["C"], r["S"], r["R"]) for r in cases()]
    assert attempt == 17 and list(map(tuple, rows)) == got, "manifest disagrees with the supplied sampler"
    return mod
