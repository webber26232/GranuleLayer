"""Fail-closed keyed seed pairing against DECLARED experiment / method identities (benchmarks/paired_stats.py;
integration review 5.3).  The research version learned a cohort's identity from its first record: six observations of
A under "old-protocol" and six of B under icdx39000 (seeds 12..17, equal values) came out complete / noninferior."""
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))

import paired_stats as PS  # noqa: E402

SEEDS = [(d, d) for d in range(12, 18)]
EXP = PS.Experiment(graph=0, protocol="icdx39000", steps=256)
MA = PS.Method("A", "energy")
MB = PS.Method("B", "energy")


def key(d, sup="A", proto="icdx39000", steps=256, g=0, be="dense-masked", sr=None, norm="energy"):
    return PS.ObsKey(g, d, d if sr is None else sr, sup, norm, proto, steps, be)


def cohort(name, vals, exp=EXP, method=None, **kw):
    m = method or PS.Method(name, kw.get("norm", "energy"), kw.get("be", "dense-masked"))
    c = PS.Cohort(name, exp, m)
    for d, v in vals.items():
        c.add(key(d, sup=m.support, **kw), v, f"{name}_{d}")
    return c


def base(off=0.0):
    return {d: 0.17 + 0.001 * ((d * 7) % 5) + off for d in range(12, 18)}


def pair(a, b, seeds=SEEDS, exp=EXP, ma=MA, mb=MB, family_size=4):
    return PS.paired(a, b, seeds, experiment=exp, method_a=ma, method_b=mb, family_size=family_size, alpha=0.05,
                     margin=0.003)


def test_complete_pairing_matches_manual_rule():
    r = pair(cohort("A", base(0.002)), cohort("B", base()))
    assert r["status"] == "complete" and r["n"] == 6 and r["df"] == 5
    assert abs(r["t_crit"] - 3.16338145) < 1e-6
    assert abs(r["mean"] - 0.002) < 1e-12 and r["sd"] < 1e-12 and r["verdict"] == "noninferior"
    assert r["experiment"] == dict(graph=0, protocol="icdx39000", steps=256)


def test_review_reproducer_wrong_protocol_cohort_is_rejected():
    """A: six records of 'old-protocol'; B: six of icdx39000; same seeds, equal values."""
    with pytest.raises(PS.WrongIdentity):
        cohort("A", base(), proto="old-protocol")


def test_same_wrong_protocol_on_both_sides_is_rejected():
    with pytest.raises(PS.WrongIdentity):
        cohort("A", base(), proto="old-protocol")
    with pytest.raises(PS.WrongIdentity):
        cohort("B", base(), proto="old-protocol")
    # and cohorts built for another (internally consistent) experiment are refused by the comparison
    old = PS.Experiment(0, "old-protocol", 256)
    a, b = cohort("A", base(), exp=old, proto="old-protocol"), cohort("B", base(), exp=old, proto="old-protocol")
    with pytest.raises(PS.WrongIdentity, match="experiment"):
        pair(a, b)


@pytest.mark.parametrize("kw", [dict(g=1), dict(steps=32), dict(proto="tkc29000"), dict(norm="none"),
                                dict(be="mt4_nw8_wg4")])
def test_wrong_graph_steps_protocol_treatment_backend_rejected(kw):
    c = cohort("A", {12: 0.1})
    with pytest.raises(PS.WrongIdentity):
        c.add(key(13, sup="A", **kw), 0.1, "odd")
    fresh = PS.Cohort("A2", EXP, PS.Method("A2", "energy"))
    with pytest.raises(PS.WrongIdentity):  # even when it is the cohort's FIRST record
        fresh.add(key(12, sup="A2", **kw), 0.1)


def test_valid_cross_method_identities_are_allowed():
    """A comparison may vary support, treatment and backend when it declares them; graph / protocol / steps shared."""
    ml, mo = PS.Method("Cover-TopK/16", "energy", "legacy"), PS.Method("Cover-TopK/16", "energy", "mt4_nw8_wg4")
    a = cohort("opt", base(1e-7), method=mo, be="mt4_nw8_wg4")
    b = cohort("leg", base(), method=ml, be="legacy")
    r = pair(a, b, ma=mo, mb=ml)
    assert r["status"] == "complete" and r["verdict"] == "noninferior"
    dn = PS.Method("dense", "none")
    r = pair(cohort("A", base(0.001)), cohort("dense", base(), method=dn, norm="none"), mb=dn)
    assert r["status"] == "complete"
    with pytest.raises(PS.WrongIdentity, match="method"):  # a cohort of an undeclared method
        pair(a, b, ma=ml, mb=ml)


def test_cohort_requires_a_declaration():
    with pytest.raises(TypeError):
        PS.Cohort("A")  # no identity is inferred from records


def test_different_missing_seeds_on_opposite_sides_never_mispair():
    va, vb = base(0.001), base()
    va.pop(13)
    vb.pop(15)
    r = pair(cohort("A", va), cohort("B", vb))
    assert r["status"] == "incomplete" and r["verdict"] is None and r["upper"] is None
    assert r["inventory"]["a"]["missing"] == [(13, 13)] and r["inventory"]["b"]["missing"] == [(15, 15)]
    assert set(r["deltas"]) == {"12/12", "14/14", "16/16", "17/17"}
    for k, v in r["deltas"].items():
        d = int(k.split("/")[0])
        assert abs(v - (va[d] - vb[d])) < 1e-15


def test_same_missing_seed_is_incomplete():
    va, vb = base(), base()
    va.pop(14)
    vb.pop(14)
    r = pair(cohort("A", va), cohort("B", vb))
    assert r["status"] == "incomplete" and r["verdict"] is None and r["descriptive"]["n"] == 5


def test_duplicate_key_rejected():
    c = cohort("A", base())
    with pytest.raises(PS.DuplicateKey):
        c.add(key(12, sup="A"), 0.2, "copy_of_12")


def test_undeclared_seed_blocks_verdict():
    va = base()
    va[18] = 0.5
    r = pair(cohort("A", va), cohort("B", base()))
    assert r["status"] == "incomplete" and r["inventory"]["a"]["undeclared"] == [(18, 18)]


def test_mismatched_sr_seed_is_a_different_key():
    c = PS.Cohort("A", EXP, MA)
    c.add(key(12, sup="A", sr=12), 0.1)
    c.add(key(12, sup="A", sr=13), 0.1)
    assert PS.inventory(c, SEEDS)["undeclared"] == [(12, 13)]


def test_reordered_files_give_identical_result():
    va, vb = base(0.0013), base()
    va[15] += 0.0007
    r1 = pair(cohort("A", va), cohort("B", vb))
    items = list(va.items())
    random.Random(3).shuffle(items)
    r2 = pair(cohort("A", dict(items)), cohort("B", dict(reversed(list(vb.items())))))
    assert r1["upper"] == r2["upper"] and r1["deltas"] == r2["deltas"]


def test_family_size_preserved_and_df_from_cohort():
    r = pair(cohort("A", base(0.001)), cohort("B", base()), seeds=SEEDS[:3])
    assert r["status"] == "incomplete"  # declared three seeds, cohorts hold six: n is never silently changed
    a3 = cohort("A", {d: v for d, v in base(0.001).items() if d < 15})
    b3 = cohort("B", {d: v for d, v in base().items() if d < 15})
    r = pair(a3, b3, seeds=SEEDS[:3])
    from scipy.stats import t

    assert r["df"] == 2 and abs(r["t_crit"] - t.ppf(1 - 0.05 / 4, 2)) < 1e-12 and r["family_size"] == 4


def test_not_estimable_keeps_alpha():
    r = PS.not_estimable("opt", "legacy", 4, 0.05, "no backend qualified")
    assert r["status"] == "not estimable" and r["family_size"] == 4 and r["verdict"] is None
