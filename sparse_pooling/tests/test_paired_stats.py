"""Fail-closed keyed seed pairing (benchmarks/paired_stats.py; results/INTERACTION_COVERAGE_K16_DX.md 2.1)."""
import json
import math
import random
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))

import paired_stats as PS  # noqa: E402

SEEDS = [(d, d) for d in range(6, 12)]


def key(d, sup="A", proto="tkc29000", steps=256, g=0, be="dense-masked", sr=None, norm="energy"):
    return PS.ObsKey(g, d, d if sr is None else sr, sup, norm, proto, steps, be)


def cohort(name, vals, **kw):
    c = PS.Cohort(name)
    for d, v in vals.items():
        c.add(key(d, sup=name, **kw), v, f"{name}_{d}")
    return c


def base(name, off=0.0):
    return {d: 0.17 + 0.001 * ((d * 7) % 5) + off for d in range(6, 12)}


def test_complete_pairing_matches_manual_rule():
    a, b = cohort("A", base("A", 0.002)), cohort("B", base("B"))
    r = PS.paired(a, b, SEEDS, family_size=4, alpha=0.05, margin=0.003)
    assert r["status"] == "complete" and r["n"] == 6 and r["df"] == 5
    assert abs(r["t_crit"] - 3.16338145) < 1e-6
    d = [0.002] * 6
    assert abs(r["mean"] - 0.002) < 1e-12 and r["sd"] < 1e-12
    assert r["verdict"] == "noninferior"


def test_different_missing_seeds_on_opposite_sides_never_mispair():
    va, vb = base("A", 0.001), base("B")
    va.pop(7)
    vb.pop(9)
    a, b = cohort("A", va), cohort("B", vb)
    r = PS.paired(a, b, SEEDS, family_size=4, alpha=0.05, margin=0.003)
    assert r["status"] == "incomplete" and r["verdict"] is None and r["upper"] is None
    assert r["inventory"]["a"]["missing"] == [(7, 7)] and r["inventory"]["b"]["missing"] == [(9, 9)]
    # descriptive deltas only on seeds held by both, each joined to ITS OWN partner (zip would shift seeds 8..11)
    assert set(r["deltas"]) == {"6/6", "8/8", "10/10", "11/11"}
    for k, v in r["deltas"].items():
        d = int(k.split("/")[0])
        assert abs(v - (va[d] - vb[d])) < 1e-15


def test_same_missing_seed_is_incomplete():
    va, vb = base("A"), base("B")
    va.pop(8)
    vb.pop(8)
    r = PS.paired(cohort("A", va), cohort("B", vb), SEEDS, family_size=4, alpha=0.05, margin=0.003)
    assert r["status"] == "incomplete" and r["verdict"] is None
    assert r["descriptive"]["n"] == 5


def test_duplicate_key_rejected():
    c = cohort("A", base("A"))
    with pytest.raises(PS.DuplicateKey):
        c.add(key(6, sup="A"), 0.2, "copy_of_6")


def test_wrong_protocol_or_steps_or_backend_rejected():
    for kw in (dict(proto="tkc29000x"), dict(steps=32), dict(be="kgen-dx-mt2"), dict(g=1), dict(norm="none")):
        c = cohort("A", {6: 0.1})
        with pytest.raises(PS.WrongIdentity):
            c.add(key(7, sup="A", **kw), 0.1, "odd")


def test_undeclared_seed_blocks_verdict():
    va = base("A")
    va[12] = 0.5
    r = PS.paired(cohort("A", va), cohort("B", base("B")), SEEDS, family_size=4, alpha=0.05, margin=0.003)
    assert r["status"] == "incomplete" and r["inventory"]["a"]["undeclared"] == [(12, 12)]


def test_mismatched_sr_seed_is_a_different_key():
    c = PS.Cohort("A")
    c.add(key(6, sup="A", sr=6), 0.1)
    c.add(key(6, sup="A", sr=7), 0.1)  # different seed pair: allowed, but undeclared
    assert PS.inventory(c, SEEDS)["undeclared"] == [(6, 7)]


def test_reordered_files_give_identical_result():
    va, vb = base("A", 0.0013), base("B")
    va[9] += 0.0007
    r1 = PS.paired(cohort("A", va), cohort("B", vb), SEEDS, family_size=4, alpha=0.05, margin=0.003)
    items = list(va.items())
    random.Random(3).shuffle(items)
    r2 = PS.paired(cohort("A", dict(items)), cohort("B", dict(reversed(list(vb.items())))), SEEDS,
                   family_size=4, alpha=0.05, margin=0.003)
    assert r1["upper"] == r2["upper"] and r1["deltas"] == r2["deltas"]


def test_family_size_preserved_and_df_from_cohort():
    r = PS.paired(cohort("A", base("A", 0.001)), cohort("B", base("B")), SEEDS[:3], family_size=4, alpha=0.05,
                  margin=0.003)
    # declared three seeds, cohort has six: undeclared -> incomplete (n is never silently changed)
    assert r["status"] == "incomplete"
    a3 = cohort("A", {d: v for d, v in base("A", 0.001).items() if d < 9})
    b3 = cohort("B", {d: v for d, v in base("B").items() if d < 9})
    r = PS.paired(a3, b3, SEEDS[:3], family_size=4, alpha=0.05, margin=0.003)
    from scipy.stats import t

    assert r["df"] == 2 and abs(r["t_crit"] - t.ppf(1 - 0.05 / 4, 2)) < 1e-12


def test_not_estimable_keeps_alpha():
    r = PS.not_estimable("opt", "legacy", 4, 0.05, "no backend qualified")
    assert r["status"] == "not estimable" and r["family_size"] == 4 and r["verdict"] is None


def test_report_withholds_verdict_for_missing_files(tmp_path):
    """tkc_report.final on a copy of the previous round's final records with different files removed on the two
    sides of a primary comparison: no verdict, inventory reported; the complete comparisons are unchanged."""
    src = HERE.parent / "results" / "topk16_coverage_constrained_topk"
    if not (src / "final").is_dir():
        pytest.skip("previous round's raw records not present")
    import tkc_report as TR

    root = tmp_path / "r"
    (root / "final").mkdir(parents=True)
    shutil.copy(src / "final_selection.json", root / "final_selection.json")
    for f in (src / "final").glob("learn_*.json"):
        shutil.copy(f, root / "final" / f.name)
    full = TR.Report(root, tmp_path)
    full.final()
    prim_full = dict((r[0], r) for r in full.csvs["final_primary"][1])
    (root / "final" / "learn_TopK_k16_g0_energy_s256_d7_tkc29000.json").unlink()
    (root / "final" / "learn_U16_r0.25_g0_energy_s256_d9_tkc29000.json").unlink()
    # a reordered, renamed duplicate of another record must be rejected
    rep = TR.Report(root, tmp_path)
    rep.final()
    prim = dict((r[0], r) for r in rep.csvs["final_primary"][1])
    row = prim["TopK/16 - U16@0.25"]
    assert row[-1] == "no verdict" and "INCOMPLETE" in row[1]
    assert prim["Cover-TopK/16 - TopK/16"][-1] == "no verdict"
    assert prim["Cover-TopK/8 - TopK/8"] == prim_full["Cover-TopK/8 - TopK/8"]
    shutil.copy(root / "final" / "learn_TopK_k8_g0_energy_s256_d6_tkc29000.json", root / "final" / "zz_dup.json")
    with pytest.raises(PS.DuplicateKey):
        TR.Report(root, tmp_path).final()
    (root / "final" / "zz_dup.json").unlink()
    j = json.loads((root / "final" / "learn_TopK_k8_g0_energy_s256_d6_tkc29000.json").read_text())
    j["final_test_set"] = "kwa19000"
    (root / "final" / "learn_TopK_k8_g0_energy_s256_d6_tkc29000.json").write_text(json.dumps(j))
    with pytest.raises(PS.WrongIdentity):
        TR.Report(root, tmp_path).final()
