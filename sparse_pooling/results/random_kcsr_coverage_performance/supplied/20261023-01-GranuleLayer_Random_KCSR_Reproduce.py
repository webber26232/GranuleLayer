"""Reproduce the stratified K/C/S/R sample and analytical coverage predictions.

Predictions concern an ideal balanced-record local-permutation model, not measured
connectivity from GranuleLayer's packed generator. Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import math
from pathlib import Path
import random

KS = (8, 16, 32, 64)
CS = (8, 16, 32, 64, 128, 256)
SS = (16, 32, 64, 128)
RS = (4, 8, 16, 32, 64, 128, 256)
SAMPLE_SEED = 20261020
D, G, WEIGHTS = 1024, 1024, 800_000_000


def select_cases(seed: int = SAMPLE_SEED):
    """One row per (K,C); choose legal S and R uniformly, then accept diversity constraints."""
    rng = random.Random(seed)
    for attempt in range(1, 100_001):
        rows = [(k, c, rng.choice([s for s in SS if s >= k]), rng.choice(RS)) for k in KS for c in CS]
        sc = collections.Counter(s for k, c, s, r in rows)
        rc = collections.Counter(r for k, c, s, r in rows)
        large_k_wide = sum(k == 64 and s == 128 for k, c, s, r in rows)
        if all(sc[s] >= 3 for s in SS) and all(rc[r] >= 2 for r in RS) and 2 <= large_k_wide <= 4:
            return rows, attempt
    raise RuntimeError("No sample satisfied the declared constraints")


def valid_records(h: int, c: int, g: int = G) -> int:
    """Each pooling output forms C-shared blocks independently; count partial valid blocks."""
    n, rem = divmod(h, g)
    return (g - rem) * ((n + c - 1) // c) + rem * ((n + 1 + c - 1) // c)


def local_hit(s: int, k: int, records: int, order: int) -> float:
    """Conditional tuple hit in independent local permutation passes, prefix records in last pass."""
    if order > k or records == 0:
        return 0.0
    records_per_pass = s // k
    full, tail = divmod(records, records_per_pass)
    a = math.comb(k - 1, order - 1) / math.comb(s - 1, order - 1)
    if full and a == 1.0:
        return 1.0
    log_miss = full * math.log1p(-a) if full else 0.0
    if tail:
        log_miss += math.log1p(-(tail / records_per_pass) * a)
    return -math.expm1(log_miss)


def predict(d: int, k: int, c: int, s: int, r: int, h: int, order: int, g: int = G):
    """Expected layer-wide coverage under explicitly idealized balanced record allocation.

    Per-supergroup record quotas differ by at most one; remainder quotas use the
    round-interleaved order. Full local passes partition S into disjoint K-sets.
    This omits actual packed work-unit coupling, invalid-record holes, and seed realization.
    """
    if not (0 < k <= s <= d and s % k == 0 and d % s == 0 and r >= 1 and h >= g):
        raise ValueError("Invalid graph/model dimensions")
    per_round = d // s
    q = r * per_round
    nrec = valid_records(h, c, g)
    base, rem = divmod(nrec, q)
    pg = math.comb(s - 1, order - 1) / math.comb(d - 1, order - 1) if order <= s else 0.0
    group = -math.expm1(r * math.log1p(-pg)) if pg < 1.0 else 1.0
    hlo = local_hit(s, k, base, order)
    hhi = local_hit(s, k, base + 1, order)
    lm = 0.0
    for rd in range(r):
        high = rem // r + int(rd < rem % r)
        mean_hit = ((per_round - high) * hlo + high * hhi) / per_round
        p = pg * mean_hit
        if p == 1.0:
            model = 1.0
            break
        lm += math.log1p(-p)
    else:
        model = -math.expm1(lm)
    iid_p = math.comb(k, order) / math.comb(d, order) if order <= k else 0.0
    iid = -math.expm1(nrec * math.log1p(-iid_p)) if iid_p < 1.0 else 1.0
    return dict(model=model, group_bound_expectation=group, iid_record_reference=iid,
                valid_records=nrec, records_per_set_mean=nrec / q, Q=q)


def support(k: int, c: int) -> str:
    if k != 8:
        return "requires_fan_in_extension"
    if c in (128, 256):
        return "requires_logical_C_extension_or_validated_lowering"
    return "existing_grouped_eligible" if c in (8, 64) else "existing_generic_eligible"


def rows_for_budget(cases, h_fixed=None):
    out = []
    for i, (k, c, s, r) in enumerate(cases, 1):
        h = h_fixed if h_fixed is not None else WEIGHTS // k
        row = dict(id=f"E{i:02}", K=k, C=c, S=s, R=r, D=D, G=G, H=h, valid_weights=h * k,
                   valid_biases=h, budget="H100M_theory_only" if h_fixed else "W800M_primary",
                   current_support=support(k, c))
        for order, label in ((2, "pair"), (3, "triple"), (4, "quad")):
            p = predict(D, k, c, s, r, h, order)
            row[label + "_model_pct"] = 100 * p["model"]
            row[label + "_group_expectation_pct"] = 100 * p["group_bound_expectation"]
            row[label + "_iid_reference_pct"] = 100 * p["iid_record_reference"]
            row["Q"] = p["Q"]
            row["valid_records"] = p["valid_records"]
            row["records_per_set_mean"] = p["records_per_set_mean"]
            assert 0 <= p["model"] <= p["group_bound_expectation"] + 1e-14 <= 1 + 1e-14
            assert p["model"] <= min(1.0, p["valid_records"] * math.comb(k, order) / math.comb(D, order)) + 1e-13
        assert row["quad_model_pct"] <= row["triple_model_pct"] + 1e-12 <= row["pair_model_pct"] + 1e-12
        out.append(row)
    return out


def self_test():
    cases, attempt = select_cases()
    assert attempt == 17 and len(cases) == 24 and len(set(cases)) == 24
    legal = [(k, c, s, r) for k, c, s, r in itertools.product(KS, CS, SS, RS) if k <= s]
    assert len(legal) == 546 and all(x in legal for x in cases)
    assert len({(k, c) for k, c, s, r in cases}) == 24
    # One global/local pass gives exactly D/K disjoint records: count tuples directly.
    for t in (2, 3, 4):
        v = predict(16, 4, 1, 8, 1, 4, t, g=1)["model"]
        expected = 4 * math.comb(4, t) / math.comb(16, t)
        assert math.isclose(v, expected, rel_tol=1e-13)
    # Full fan-in S: each occupied set contributes its complete tuple set.
    for t in (2, 3, 4):
        p = predict(16, 8, 1, 8, 3, 12, t, g=1)
        assert math.isclose(p["model"], p["group_bound_expectation"], rel_tol=1e-13)
    # Enumerate every local permutation to check a partial pass and two independent passes.
    perms = list(itertools.permutations(range(4)))
    hit_prefix = sum({0, 1} <= set(p[:2]) for p in perms) / len(perms)
    assert math.isclose(local_hit(4, 2, 1, 2), hit_prefix, rel_tol=1e-13)
    miss_twopass = 0
    for p1 in perms:
        for p2 in perms:
            edges = [set(p1[:2]), set(p1[2:]), set(p2[:2]), set(p2[2:])]
            miss_twopass += not any({0, 1} <= e for e in edges)
    exact = 1 - miss_twopass / len(perms)**2
    assert math.isclose(local_hit(4, 2, 4, 2), exact, rel_tol=1e-13)
    # Sharing changes record diversity, not the total edge budget.
    assert predict(D, 8, 8, 128, 256, 100_000_000, 3)["model"] >= predict(D, 8, 256, 128, 256, 100_000_000, 3)["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    self_test()
    args.out.mkdir(parents=True, exist_ok=True)
    cases, attempt = select_cases()
    primary = rows_for_budget(cases)
    alternate = rows_for_budget(cases, h_fixed=100_000_000)
    for name, rows in (("sampled_coverage_W800M.csv", primary), ("sampled_coverage_H100M.csv", alternate)):
        with (args.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    manifest = dict(seed=SAMPLE_SEED, accepted_attempt=attempt, raw_grid_size=672, legal_grid_size=546,
                    constraints="one per (K,C); each S >=3 rows; each R >=2 rows; K64 has 2..4 S128 rows",
                    model="ideal balanced records, independent global/local permutations, prefix tail; no packed-unit effects",
                    measured=False, primary=primary, alternate_H100M=alternate)
    (args.out / "sampled_coverage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("| ID | K | C | S | R | pair % | triple % | quad % | current support |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for v in primary:
        vals = [f"{v[k + '_model_pct']:.4f}" for k in ("pair", "triple", "quad")]
        print(f"| {v['id']} | {v['K']} | {v['C']} | {v['S']} | {v['R']} | {' | '.join(vals)} | {v['current_support']} |")


if __name__ == "__main__":
    main()
