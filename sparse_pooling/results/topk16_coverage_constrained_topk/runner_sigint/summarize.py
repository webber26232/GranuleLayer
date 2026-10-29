import json, glob, collections
out = {}
for f in sorted(glob.glob("m_*.jsonl")):
    rows = [json.loads(l) for l in open(f) if '"trial"' in l]
    c = collections.Counter((r["sig"], "hang>30s" if r["hung"] else "exited", r["rc"], "survivors" if r["survivors"] else "no survivors") for r in rows)
    lat = [r["latency_s"] for r in rows if not r["hung"]]
    out[f[2:-6]] = dict(trials=len(rows), outcomes={" | ".join(map(str, k)): v for k, v in sorted(c.items())},
                        exit_latency_s=dict(max=max(lat), mean=sum(lat) / len(lat)) if lat else None)
json.dump(out, open("matrix_summary.json", "w"), indent=1)
for k, v in out.items(): print(k, v["outcomes"], v["exit_latency_s"])
