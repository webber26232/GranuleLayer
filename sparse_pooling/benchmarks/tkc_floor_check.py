"""Bounded independent Monte Carlo check of the analytic order-1 missing-input floor (rsn_interaction.missing_input_floor;
results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md section 6.1) on the reduced raw-input task.

For each support: build the layer / mask exactly as rsn_interaction does (init seed, graph seed 0, T 16), compute the
analytic floor, then on fresh input batches (seeds 77_000 + i, disjoint from train / held-out / test seeds) the
residual of the BEST predictor given the visible inputs, t - (gamma + sum_{i in U_g} beta x_i), with t from the actual
target code (wps_interaction.targets) -- a wrong beta / gamma would leave extra error, so the check validates the
coefficients against the code as well as the floor.  Reported: analytic vs Monte Carlo order-1 mean nMSE floor with a
batch-level standard error, and the BF16-input target control.

    python tkc_floor_check.py --supports TopK/8,Cover-TopK/8,BP-row/8#0,TopK/16 --init-seed 20 --batches 64 --out X.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import common  # noqa: F401
import rsn_cases as RC
import rsn_interaction as RI
import wps_cases as WC
import wps_interaction as WI

MC_SEED0 = 77_000


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supports", required=True)
    ap.add_argument("--init-seed", type=int, default=20)
    ap.add_argument("--terms", type=int, default=16)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    dev = torch.device("cuda")
    lay = KgxLayer(WC.D, 262_144, WC.G, RC.C, ConnectivityPlan(WC.S, WC.R, seed=0), variant="idf", tile=128,
                   device=dev, seed=a.init_seed, keep_indices=False)
    ids, rvalid, order, loc = WI.target_inputs(lay, a.terms)
    sig = WI.sigmas(dev)
    seeds = [MC_SEED0 + i for i in range(a.batches)]
    clash = set(seeds) & ({WI.TRAIN_SEED + i for i in range(4000)} | set(WI.HELD_SEEDS) | {WI.TRAIN_SEED - 1} |
                          set().union(*(set(v) for v in RI.TEST_PROTOCOLS.values())))
    if clash:
        raise SystemExit(f"Monte Carlo seeds collide: {sorted(clash)[:5]}")
    out = dict(schema="granule.tkc_floor_check/1", init_seed=a.init_seed, terms=a.terms, batches=a.batches,
               tokens_per_batch=WI.M_TRAIN, mc_seeds=[seeds[0], seeds[-1]],
               provenance=provenance(Path(common.__file__).resolve().parent.parent), supports={})
    w0 = lay.weight.detach().clone()
    for spec in a.supports.split(","):
        with torch.no_grad():
            lay.weight.copy_(w0)
        em, sel, srec = RC.support(spec, lay, 0)
        an = RI.missing_input_floor(lay, em, ids, rvalid, order, sig, a.terms)
        # visible set and coefficients, rebuilt exactly as the analytic function does
        L = lay.layout
        G, NB = L.groups, L.blocks_per_group
        valid = L.valid_mask(dev)
        m = torch.ones(L.weight_shape, dtype=torch.bool, device=dev) if em is None else em.bool()
        loc_on = (m & valid[..., None]).any(2)
        b = lay.kg_buckets.to(torch.int64)
        rb = torch.repeat_interleave(torch.arange(len(b), device=dev), b[:, 3])
        kb = torch.arange(len(rb), device=dev) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
        bq = torch.full((G * NB,), -1, dtype=torch.int64, device=dev)
        bq[b[rb, 1] * NB + b[rb, 2] + kb] = b[rb, 0]
        q = bq.view(G, NB)
        seen = torch.zeros((G, L.input_dim), dtype=torch.bool, device=dev)
        gi = torch.arange(G, device=dev)[:, None, None].expand(G, NB, 64)
        on = loc_on & (q >= 0)[..., None]
        seen[gi[on], lay.sg_feat.to(torch.int64)[q.clamp_min(0)][on]] = True
        mu1, sd1 = sig[1][0], sig[1][1]
        n_g = rvalid.sum(1).double()
        beta = torch.zeros((G, L.input_dim), dtype=torch.float64, device=dev)
        gj = torch.arange(G, device=dev)[:, None, None].expand(G, NB, a.terms)
        vt = rvalid[:, :, None].expand(G, NB, a.terms)
        beta.index_put_((gj[vt], ids[..., 0][vt]), (1.0 / (sd1 * (n_g * a.terms).sqrt()))[gj[vt]], accumulate=True)
        gamma = -(n_g * a.terms).sqrt() * mu1 / sd1
        bs = (beta * seen).t()  # [D, G]
        o1 = order == 1
        per_batch, full_res = [], []
        se_tot = torch.zeros(G, dtype=torch.float64, device=dev)
        var_tot = torch.zeros(G, dtype=torch.float64, device=dev)
        for s in seeds:
            x = WI.xbatch(s, dev)
            t = WI.targets(x, ids, rvalid, order, sig).double()
            pred = x.double() @ bs + gamma
            se = ((t - pred) ** 2).sum(0)
            var = (t ** 2).sum(0)
            se_tot += se
            var_tot += var
            per_batch.append(float((se / var)[o1].mean()))
            # dense-coverage sanity: with every input visible the linear predictor is exact up to FP32 targets
            full_res.append(float(((t - (x.double() @ beta.t() + gamma)) ** 2).sum(0)[o1].max()))
        mc = float((se_tot / var_tot)[o1].mean())
        out["supports"][spec] = dict(analytic=an["order1_mean_floor"], monte_carlo=mc,
                                     monte_carlo_batch_se=statistics.stdev(per_batch) / len(per_batch) ** 0.5,
                                     relative_difference=(mc - an["order1_mean_floor"]) / an["order1_mean_floor"]
                                     if an["order1_mean_floor"] > 0 else None,
                                     absolute_difference=mc - an["order1_mean_floor"],
                                     full_visibility_max_residual_sse=max(full_res),
                                     missing_target_inputs_per_output=an["order1_missing_target_inputs_per_output"],
                                     outputs_with_missing=an["order1_outputs_with_missing"],
                                     mean_local_inputs_per_record=an["mean_local_inputs_per_record"],
                                     support_fingerprint=srec.get("support_fingerprint"))
        print(f"{spec}: analytic {an['order1_mean_floor']:.6f} MC {mc:.6f} "
              f"(+- {out['supports'][spec]['monte_carlo_batch_se']:.6f})", flush=True)
    out["bf16_input_target_control"] = RI.bf16_target_control(ids, rvalid, order, sig, dev, seeds[:16])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
