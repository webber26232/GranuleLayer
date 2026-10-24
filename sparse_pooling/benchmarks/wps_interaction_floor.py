"""Exact additive floors of the interaction-diagnostic targets (wps_interaction.py), per order, on its held-out
batches -- the normalized MSE below which no support whose neurons each see inputs of only ONE 16-input block
can go.

For independent N(0, 1) inputs the best additive approximation (over single inputs, equivalently over blocks,
since the targets couple only inputs of different blocks) is the sum of the conditional means of every term:
E[f_k | x_i] = (x_i - 1) Phi((x_i - 1) / s) + s phi((x_i - 1) / s) - mu_k, s^2 = k - 1, divided by sigma_k
(order 1: the term itself, floor 0).  With T > 1 terms per record several terms share inputs, so their
additive parts add coherently while their non-additive parts stay uncorrelated: the floor is LOWER than the
single-term constant (which the first capacity analysis divided by; this script replaces it).  Analysis only:
it rebuilds the targets from the recorded configuration (graph seed, C, H, hash seed, terms), no training.

    python wps_interaction_floor.py --root RAW --out RAW/interaction_floors.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

import common  # noqa: F401
import wps_cases as WC
import wps_interaction as WI


@torch.no_grad()
def floors(hidden: int, C: int, graph_seed: int, terms: int, dev="cuda") -> dict:
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan

    lay = KgxLayer(WC.D, hidden, WC.G, C, ConnectivityPlan(WC.S, WC.R, seed=graph_seed), variant="idf", tile=128,
                   device=dev, seed=WC.INIT_SEED, keep_indices=False)
    ids, valid, order, _ = WI.target_inputs(lay, terms)
    sig = WI.sigmas(dev)
    nrm = torch.distributions.Normal(torch.tensor(0.0, device=dev), torch.tensor(1.0, device=dev))
    G, NB, T, _ = ids.shape
    se = torch.zeros(G, device=dev, dtype=torch.float64)
    var = torch.zeros(G, device=dev, dtype=torch.float64)
    for s in WI.HELD_SEEDS:
        x = WI.xbatch(s, dev)
        t = WI.targets(x, ids, valid, order, sig)
        A = torch.zeros_like(t)
        for j in range(NB):
            for u in range(T):
                xs = x[:, ids[:, j, u]]  # [M, G, 4]
                for k in (1, 2, 3, 4):
                    sel = order == k
                    if k == 1:
                        part = (xs[:, sel, 0] - sig[1][0]) / sig[1][1]
                    else:
                        sk = (k - 1) ** 0.5
                        part = torch.zeros(x.shape[0], int(sel.sum()), device=dev)
                        for i in range(k):
                            v = xs[:, sel, i] - WI.OFFSET
                            part += v * nrm.cdf(v / sk) + sk * torch.exp(nrm.log_prob(v / sk)) - sig[k][0]
                        part = part / sig[k][1]
                    A[:, sel] += part * valid[sel, j][None, :]
        A = A / (valid.sum(1).clamp_min(1) * T).sqrt()[None, :]
        se += ((t - A) ** 2).sum(0).double()
        var += (t ** 2).sum(0).double()
    fl = se / var
    return {f"order{o}": float(fl[order == o].mean()) for o in (1, 2, 3, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from provenance import provenance

    cfgs = {}
    for st in ("interaction", "interaction_capacity"):
        for f in glob.glob(f"{a.root}/{st}/int*.json"):
            d = json.loads(Path(f).read_text())
            cfgs[(d.get("hidden", 262_144), d["C"], d["graph_seed"], d.get("terms", 1))] = st
    out = dict(rule=__doc__.split("\n\n")[1], provenance=provenance(Path(common.__file__).resolve().parent.parent),
               floors=[])
    for (hidden, C, gs, terms), st in sorted(cfgs.items()):
        fl = floors(hidden, C, gs, terms)
        out["floors"].append(dict(stage=st, hidden=hidden, C=C, graph_seed=gs, terms=terms, floor=fl))
        print(st, terms, fl, flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
