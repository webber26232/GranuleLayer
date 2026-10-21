"""Markdown table of a run_det_regression.sh output directory (compare / p2 JSONs)."""
import glob, json, os, sys
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    n = os.path.basename(f)[:-5]
    if not (n.endswith("_new_compare") or n.endswith("_p2")):
        continue
    j = json.load(open(f)); a = j["args"]
    steps = len(j["trajectory1"]) + len(j["trajectory2"])
    comp = sum(s["compared"] for s in j["trajectory1"] + j["trajectory2"])
    diff = sum(len(s["differing"]) for s in j["trajectory1"] + j["trajectory2"])
    kind = "old vs new" if n.endswith("_new_compare") else "new vs new (generic_cuda)"
    name = n.rsplit("_new_compare", 1)[0].rsplit("_p2", 1)[0]
    rows.append((name, kind, a["case"], a["opt_mode"], a["precision"], a["plan"] or "-", a["steps"], steps, comp, diff,
                 j["status"], j["resolved_plan_equal"], all(all(x) for x in j["restore_fingerprint_equal"]) if j["restore_fingerprint_equal"] and isinstance(j["restore_fingerprint_equal"][0], list) else all(j["restore_fingerprint_equal"]),
                 j["provenance"]["commit"][:7], (j.get("reference_provenance") or {}).get("commit", "")[:7],
                 j["peak_allocated_bytes"] / 2**30))
print("| Case | Kind | det_training case | optimizer | dZ | plan | steps | compared steps | tensors compared | differing | status | same plan | state restored | tested | reference | peak GiB |")
print("|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---:|")
for r in rows:
    print("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |")
print(f"\n{len(rows)} comparisons, {sum(r[10] == 'pass' for r in rows)} pass, {sum(r[9] for r in rows)} differing tensors")
