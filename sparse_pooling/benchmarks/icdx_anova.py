"""Functional-ANOVA structural lower bound of the reduced raw-input task for one support and init seed
(results/INTERACTION_COVERAGE_K16_DX.md section 4.3; benchmarks/anova_bound.py).

Scope (declared): a fixed sample of ``--per-order`` outputs per order (seeded permutation, seed ``--sample-seed``,
chosen before any loss is examined), the analysis batches 97_000..97_063 (M 512 each; never used for training,
calibration, development or final tests).  The support is selected from the init weights exactly as the training run
(rsn_cases spec, init seed), so weighted supports are per init.

Outputs: json (bound estimate per order with batch-cluster bootstrap intervals, per-output values, validation checks,
time and device peak) and npz (per batch x sampled output: residual energy sum and target energy sum) -- the report
joins it with the model's analysis-batch errors (rsn_interaction --eval-state --analysis-out) on the SAME outputs and
batches.  Caps: 2 GiB temporary device allocation and 10 minutes; exceeding either is recorded (status "incomplete")
and nothing is published as a bound.

    python icdx_anova.py --support Cover-TopK/16 --init-seed 30 --out X.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401
import anova_bound as AB
import rsn_cases as RC
import wps_cases as WC
import wps_interaction as WI

CAP_BYTES, CAP_SECONDS = 2 << 30, 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", required=True)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, required=True)
    ap.add_argument("--terms", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=262_144)
    ap.add_argument("--per-order", type=int, default=32)
    ap.add_argument("--sample-seed", type=int, default=20261105)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    import rsn_interaction as RI

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    lay = KgxLayer(WC.D, a.hidden, WC.G, RC.C, ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed), variant="idf",
                   tile=128, device=dev, seed=a.init_seed, keep_indices=False)
    L = lay.layout
    ids, rvalid, order, loc = WI.target_inputs(lay, a.terms)
    sig = WI.sigmas(dev)
    em, sel, srec = RC.support(a.support, lay, a.graph_seed)
    valid = L.valid_mask(dev)
    feat = lay.sg_feat.to(torch.int64).cpu()
    b = lay.kg_buckets.to(torch.int64)
    NB = L.blocks_per_group
    rb = torch.repeat_interleave(torch.arange(len(b), device=dev), b[:, 3])
    kb = torch.arange(len(rb), device=dev) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
    bq = torch.full((L.groups * NB,), -1, dtype=torch.int64, device=dev)
    bq[b[rb, 1] * NB + b[rb, 2] + kb] = b[rb, 0]
    bq = bq.view(L.groups, NB).cpu()
    sample = AB.sample_outputs(order, a.per_order, a.sample_seed)
    outs = [g for r in (1, 2, 3, 4) for g in sample[r]]
    em_c = None if em is None else em.cpu()
    valid_c = valid.cpu()
    ids_c, rv_c, ord_c = ids.cpu(), rvalid.cpu(), order.cpu()
    progs, fulls, terms_of, consts = [], [], {}, []
    t_prog = time.perf_counter()
    for g in outs:
        sup = AB.output_supports(em_c, valid_c, feat, bq, g)
        terms = AB.output_terms(ids_c, rv_c, ord_c, sig, g, a.terms)
        terms_of[g] = terms
        progs.append(AB.residual_program(terms, sup.allowed))
        fulls.append(AB.full_program(terms))
        r = int(ord_c[g])
        ef = 0.0 if r == 1 else float(AB.psi(torch.tensor(-1.0, dtype=torch.float64), math.sqrt(r)))
        consts.append(sum(w * (ef - sig[r][0]) for _, _, w in terms))
    prog_s = time.perf_counter() - t_prog
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    seeds = RI.ANALYSIS_SEEDS
    num = torch.zeros((len(seeds), len(outs)), dtype=torch.float64)
    den = torch.zeros((len(seeds), len(outs)), dtype=torch.float64)
    recon = 0.0
    oidx = torch.tensor(outs, device=dev)
    consts_t = torch.tensor(consts, dtype=torch.float64, device=dev)
    status, why = "complete", ""
    for bi, s_ in enumerate(seeds):
        x = WI.xbatch(s_, dev).double()
        t = RI.targets_fp64(x, ids, rvalid, order, sig)[:, oidx]
        R = AB.evaluate_many(progs, x)
        num[bi] = (R ** 2).sum(0).cpu()
        den[bi] = (t ** 2).sum(0).cpu()
        if bi < 2:  # reconstruction of the actual target from all components (+ its constant)
            F = AB.evaluate_many(fulls, x) + consts_t[None, :]
            recon = max(recon, float((F - t).abs().max()))
        if time.perf_counter() - t0 > CAP_SECONDS:
            status, why = "incomplete", f"time cap {CAP_SECONDS} s reached after {bi + 1} batches"
            num, den = num[:bi + 1], den[:bi + 1]
            break
    peak_extra = torch.cuda.max_memory_allocated() - base_alloc
    if peak_extra > CAP_BYTES:
        status, why = "incomplete", f"temporary device allocation {peak_extra / 2 ** 30:.2f} GiB > 2 GiB"
    res = dict(schema="granule.icdx_anova/1", support=a.support, graph_seed=a.graph_seed, init_seed=a.init_seed,
               terms=a.terms, hidden=a.hidden, support_record=dict(mask_sha256=srec.get("mask_sha256"),
                                                                   support_fingerprint=srec.get("support_fingerprint")),
               scope=dict(outputs_per_order=a.per_order, sample_seed=a.sample_seed, outputs=sample,
                          analysis_batches=[seeds[0], seeds[-1]], tokens_per_batch=WI.M_TRAIN,
                          note="a declared output sample, chosen by seed before any loss was examined; NOT the "
                               "all-output locked-test metric"),
               definition="L_g = E[(sum over nonempty disallowed A of H[g, A](x_A))^2] / E[t_g^2]: a mathematical lower "
                          "bound of the nMSE of any function of the available supports; estimated by Monte Carlo on "
                          "the analysis tokens (per output ratio of sums, averaged over outputs), batch-cluster "
                          "bootstrap; BF16 input quantization excluded; never clipped",
               program_terms=dict(mean=float(np.mean([len(p) for p in progs])), max=max(len(p) for p in progs)),
               validation=dict(reconstruction_max_abs=recon, reconstruction_tolerance=1e-9,
                               reconstruction_ok=recon < 1e-9),
               status=status, status_reason=why, provenance=provenance(Path(common.__file__).resolve().parent.parent))
    order_s = ord_c[outs]
    per_order = {}
    for r in (1, 2, 3, 4):
        cols = torch.nonzero(order_s == r).squeeze(1)
        per_order[f"order{r}"] = AB.cluster_bootstrap(num[:, cols], den[:, cols], seed=r)
        per_order[f"order{r}"]["per_output"] = (num[:, cols].sum(0) / den[:, cols].sum(0)).tolist()
        per_order[f"order{r}"]["outputs_with_positive_bound"] = int(((num[:, cols].sum(0)) > 0).sum())
    res["bound"] = per_order
    if a.support == "dense" or em is None:
        res["validation"]["dense_zero_residual"] = bool(all(len(p) == 0 for p in progs))
    # order-1 agreement with the analytic missing-input floor (rsn_interaction.missing_input_floor) on these outputs
    mf = RI.missing_input_floor(lay, em, ids, rvalid, order, sig, a.terms)
    o1_all = torch.nonzero(ord_c == 1).squeeze(1).tolist()
    pf = dict(zip(o1_all, mf["per_output_floor"]))
    an = [pf[g] for g in sample[1]]
    mc = per_order["order1"]["per_output"]
    res["validation"]["order1_vs_missing_input_floor"] = dict(
        analytic_mean=float(np.mean(an)), monte_carlo_mean=float(np.mean(mc)),
        max_abs_difference=float(np.max(np.abs(np.array(an) - np.array(mc)))),
        note="analytic population floor vs the Monte Carlo bound estimate on the analysis tokens (same outputs)")
    res["seconds"] = dict(total=time.perf_counter() - t0, programs=prog_s)
    res["device_peak_extra_bytes"] = int(peak_extra)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(Path(a.out).with_suffix("")) + ".npz", num=num.numpy(), den=den.numpy(),
                        outputs=np.array(outs), order=order_s.numpy(), seeds=np.array(seeds[:num.shape[0]]))
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"{a.support} i{a.init_seed}: " + " ".join(f"o{r} {per_order[f'order{r}']['estimate']:.5f}" for r in (1, 2, 3, 4))
          + f" | recon {recon:.1e} | {res['seconds']['total']:.0f}s {status}", flush=True)


if __name__ == "__main__":
    main()
