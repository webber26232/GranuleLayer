"""Reproduce the K64 large-C experiment manifest and pre-pruning coverage predictions.

Standard library only. These are ideal balanced-record model expectations, NOT
measured graphs or predictions of weights learned with L1. Run with --self-test.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

D = G = 1024
K = 64
WEIGHTS = 800_000_000
H = WEIGHTS // K
CS = (64, 128, 256, 512, 1024, 2048)
SS = (64, 128)
RS = (64, 128, 256)
SOURCE_COMMIT = "87d89aad18b5abebb2eb81d5083ab01d9f3f7689"


def valid_records(h, c, g=G):
    base, rem = divmod(h, g)
    return (g - rem) * math.ceil(base / c) + rem * math.ceil((base + 1) / c)


def local_hit(s, k, records, order):
    if records == 0 or order > k:
        return 0.0
    per_pass = s // k
    full, tail = divmod(records, per_pass)
    a = math.comb(k - 1, order - 1) / math.comb(s - 1, order - 1)
    if a == 1.0 and full:
        return 1.0
    log_miss = full * math.log1p(-a) if full else 0.0
    if tail:
        log_miss += math.log1p(-(tail / per_pass) * a)
    return -math.expm1(log_miss)


def predict(d, k, c, s, r, h, order, g=G):
    if not (0 < order <= k <= s <= d and s % k == 0 and d % s == 0 and r >= 1 and h >= g):
        raise ValueError("invalid dimensions")
    sets_per_round = d // s
    q = r * sets_per_round
    b = valid_records(h, c, g)
    base, rem = divmod(b, q)
    pg = math.comb(s - 1, order - 1) / math.comb(d - 1, order - 1)
    group = -math.expm1(r * math.log1p(-pg)) if pg < 1 else 1.0
    hlo, hhi = (local_hit(s, k, m, order) for m in (base, base + 1))
    lm = 0.0
    for rd in range(r):
        high = rem // r + int(rd < rem % r)
        hbar = ((sets_per_round - high) * hlo + high * hhi) / sets_per_round
        p = pg * hbar
        if p >= 1:
            result = 1.0
            break
        lm += math.log1p(-p)
    else:
        result = -math.expm1(lm)
    return {"model": result, "group_expectation": group, "valid_records": b,
            "Q": q, "records_per_set_mean": b / q, "all_sets_occupied_in_ideal_allocation": base >= 1}


def manifest():
    cases = []
    for s, c, r in itertools.product(SS, CS, RS):
        row = {"id": f"K64_C{c}_S{s}_R{r}", "K": K, "C": c, "S": s, "R": r,
               "D": D, "G": G, "H": H, "valid_weight_slots": WEIGHTS, "valid_biases": H,
               "local_index_mode_candidate": "all_features_of_set" if s == K else "sampled_K_record"}
        for t, name in ((2, "pair"), (3, "triple"), (4, "quad")):
            p = predict(D, K, c, s, r, H, t)
            row[name + "_model_pct"] = 100 * p["model"]
            row[name + "_group_expectation_pct"] = 100 * p["group_expectation"]
        row.update({key: p[key] for key in ("valid_records", "Q", "records_per_set_mean",
                                           "all_sets_occupied_in_ideal_allocation")})
        cases.append(row)
    per_output = []
    nmin, nmax = H // G, math.ceil(H / G)
    for c in CS:
        bmin, bmax = math.ceil(nmin / c), math.ceil(nmax / c)
        row = {"C": c, "neurons_per_output_min": nmin, "neurons_per_output_max": nmax,
               "records_per_output_min": bmin, "records_per_output_max": bmax,
               "distinct_input_features_per_output_upper_bound": min(D, bmax * K)}
        for t, name in ((1, "feature"), (2, "pair"), (3, "triple"), (4, "quad")):
            row[name + "_per_output_union_bound_pct"] = 100 * min(1.0, bmax * math.comb(K, t) / math.comb(D, t))
        per_output.append(row)
    return {"source_branch": "random-kcsr-coverage-sweep", "source_commit": SOURCE_COMMIT,
            "semantics": "K64, fixed 800M valid weight slots; H12.5M, D=G1024; primary timing M8192",
            "R_definition": "independent partitions of all D features; Q=R*D/S, not local permutation passes",
            "case_count": len(cases), "C_values": list(CS), "S_values": list(SS), "R_values": list(RS),
            "prediction_status": "pre-pruning model expectation, NOT measured connectivity",
            "model": "balanced logical record quotas; independent global and local permutations; per-output C tails",
            "limitations": ["Actual assignment quantum, empty sets, valid local-pass holes and seed realization require measurement.",
                            "All S=K sets are occupied in the ideal full-size grid; verify this on each generated graph.",
                            "L1 strength or density alone does not determine pruned tuple coverage.",
                            "Layer-wide coverage and per-output coverage are different statistics.",
                            "Values rounded to 100% do not certify exact complete coverage."],
            "tuple_universe_counts": {str(t): math.comb(D, t) for t in (2, 3, 4)},
            "cases": cases, "per_output_upper_bounds_before_pruning": per_output,
            "effective_coverage_definition": "tuple fits in the enabled/nonzero input support of at least one VALID neuron; never union different neurons",
            "effective_L1_coverage": None,
            "effective_L1_coverage_status": "requires the measured trained weights / pruning masks"}


def markdown(m):
    lines = ["## Pre-experiment coverage predictions", "",
             "Fixed 800M valid weight slots; K64; H12.5M; D=G1024. Percentages below are MODEL EXPECTATIONS before pruning.",
             "They are neither measured graph statistics nor learned-L1 sparsity predictions. Exact full-precision values are in the companion JSON.", "",
             "### S=K=64: same model expectation for every requested C", "",
             "All sets are occupied in the ideal full-size allocation. Verify occupied sets in the actual generator.", "",
             "| R | Pair % | Triple % | Quad % |", "|---:|---:|---:|---:|"]
    for r in RS:
        v = next(x for x in m["cases"] if x["S"] == 64 and x["C"] == 64 and x["R"] == r)
        lines.append(f"| {r} | {v['pair_model_pct']:.4f} | {v['triple_model_pct']:.4f} | {v['quad_model_pct']:.4f} |")
    lines += ["", "### S128: finite-record coverage depends on C", "",
              "| C | R64 pair / triple / quad % | R128 pair / triple / quad % | R256 pair / triple / quad % |",
              "|---:|---:|---:|---:|"]
    for c in CS:
        cells = []
        for r in RS:
            v = next(x for x in m["cases"] if x["S"] == 128 and x["C"] == c and x["R"] == r)
            cells.append(" / ".join(f"{v[name + '_model_pct']:.4f}" for name in ("pair", "triple", "quad")))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "### Per-output limits, before pruning", "",
              "These are union bounds, not expected or measured coverage. Repeated tuples make real coverage lower.", "",
              "| C | Logical records per output | Distinct input features at most | Pair upper bound % | Triple upper bound % |",
              "|---:|---:|---:|---:|---:|"]
    for x in m["per_output_upper_bounds_before_pruning"]:
        lines.append(f"| {x['C']} | {x['records_per_output_max']} | {x['distinct_input_features_per_output_upper_bound']} | "
                     f"{x['pair_per_output_union_bound_pct']:.4f} | {x['triple_per_output_union_bound_pct']:.4f} |")
    lines += ["", "After L1/pruning, recompute coverage from each neuron's retained support. A density-only estimate is insufficient.", ""]
    return "\n".join(lines)


def self_test():
    m = manifest()
    assert len(m["cases"]) == 36
    assert len({x["id"] for x in m["cases"]}) == 36
    for x in m["cases"]:
        p, t, q = (x[name + "_model_pct"] for name in ("pair", "triple", "quad"))
        assert 0 <= q <= t <= p <= 100
        assert x["all_sets_occupied_in_ideal_allocation"]
        for name in ("pair", "triple", "quad"):
            assert x[name + "_model_pct"] <= x[name + "_group_expectation_pct"] + 1e-10
            if x["S"] == K:
                assert math.isclose(x[name + "_model_pct"], x[name + "_group_expectation_pct"], abs_tol=1e-10)
    assert valid_records(H, 2048) == 6144
    assert valid_records(H, 64) == 195584
    assert math.isclose(100 * predict(D, K, 256, 64, 256, H, 3)["model"], 61.6419, abs_tol=0.0001)
    assert math.isclose(100 * predict(D, K, 256, 128, 256, H, 4)["model"], 31.2895, abs_tol=0.0001)
    assert m["per_output_upper_bounds_before_pruning"][-1]["distinct_input_features_per_output_upper_bound"] == 384
    # Exhaustive local-permutation oracle: one and two prefix records from one pass.
    for order in (2, 3):
        for nrecords in (1, 2):
            target = set(range(order))
            hits = total = 0
            for perm in itertools.permutations(range(6)):
                total += 1
                hits += any(target <= set(perm[j * 3:(j + 1) * 3]) for j in range(nrecords))
            assert math.isclose(hits / total, local_hit(6, 3, nrecords, order), abs_tol=1e-14)
    # An empty-set case demonstrates why K=S alone is not enough for group saturation.
    p = predict(64, 8, 256, 8, 4, 32, 3, g=4)
    assert p["valid_records"] < p["Q"]
    assert p["model"] < p["group_expectation"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="write full-precision JSON manifest")
    ap.add_argument("--markdown", type=Path, help="write human-readable prediction tables")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    m = manifest()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=2) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(m))
    if not args.out and not args.markdown:
        print(markdown(m))
    else:
        print(f"Validated {m['case_count']} cases; predictions are pre-pruning expectations.")


if __name__ == "__main__":
    main()
