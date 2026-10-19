"""Compact table of the deterministic training control (deterministic_training.py JSON).

    python summarize_deterministic.py results/scale_dispatch_fused_backward/deterministic_training.json
"""
import json
import sys


def main():
    d = json.load(open(sys.argv[1]))
    print(f"determinism warnings: {d['determinism_warnings']}")
    for r in d["results"]:
        own = r["ownership"]
        same = [n for n, v in r["variants"].items() if v["category"] == "same"]
        ok = [n for n in same if v_ok(r, n)]
        print(f"{r['case']:26s} {r['opt_mode']:4s} seed {r['sr_seed']:>9d} {r['precision']:5s} {r['steps']:>2d} steps | "
              f"single writer {own['single_writer']} (writers/feature {own['writer_ctas_per_feature_token_max']}, "
              f"chunks/q {own['chunks_per_supergroup_max']}, token tail {own['token_tail']}) | same-order bitwise "
              f"{len(ok)}/{len(same)}" + (f" restore {r['restore_check']['trajectory_bitwise_after_restore']}"
                                           if r.get("restore_check") else ""))
        for n, v in r["variants"].items():
            if v["category"] == "order" or (v["category"] == "same" and not v["bitwise_all"]):
                fd, fs = v["first_diff"], v["final_step"]
                if fd is None:
                    print(f"    {n:28s} [{v['category']}] bitwise")
                    continue
                print(f"    {n:28s} [{v['category']}] first diff step {fd['step']} {fd['tensor']:6s} | final rel L2: "
                      + " ".join(f"{k} {fs[k].get('rel_l2', 0):.1e}" for k in ("y", "loss", "dX", "dW", "W", "proj.W")))


def v_ok(r, n):
    return r["variants"][n]["bitwise_all"]


if __name__ == "__main__":
    main()
