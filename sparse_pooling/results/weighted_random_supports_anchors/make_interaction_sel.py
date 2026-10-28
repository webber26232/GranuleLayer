"""Interaction screen selection from the frozen method selections (selection_rules.json 'interaction')."""
import json, sys
R = sys.argv[1]
w = json.load(open(f"{R}/w8_final_selection.json"))
m = json.load(open(f"{R}/m25_final_selection.json"))
specs = ["BP-row/8#0", w["w_star"], m["mixed"], "A8@0.25", "dense"]
jobs = [dict(spec=s, terms=16, init_seed=17, sr_seed=0, steps=1000, lr=lr, save_at="1000")
        for s in specs for lr in (3e-3, 1e-2)]
jobs += [dict(spec=m["mixed"], terms=16, oracle="anchor", steps=1000, lr=3e-3),
         dict(spec=m["mixed"], terms=16, oracle="anchor", perturb=0.1, steps=1000, lr=3e-3)]
json.dump(dict(rule="selection_rules.json interaction (screen)", specs=specs, jobs=jobs),
          open(f"{R}/interaction_selection.json", "w"), indent=1)
print(len(jobs), "interaction jobs")
