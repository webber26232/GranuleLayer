"""Interaction diagnostic with optimization / representation controls (results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md
section 6; a synthetic diagnostic, NOT an application-accuracy claim).

The reduced workload, targets, data, loss and readout of the parent (benchmarks/wps_interaction.py): D = G = 1024,
K64 envelope, C256, S64 / R256, H = 262,144 (one original record per output), raw inputs x ~ N(0, 1) without a
trainable projection, per-output affine readout yhat_g = a_g y_g + c_g, single-index ReLU terms of order
k = 1 + (g mod 4), T terms per record (hash seed 777), train batches seeds 50_000 + i (M = 512), held-out seeds
49_000..49_003.  Identical for every support; the exact additive floors (wps_interaction_floor.floors, the
actual target configuration) are recorded with every run.

Per run: ``--support`` (rsn_cases spec) selected from the init weights; ``--init-seed`` (by-neuron init,
zero bias; the parent used 17); ``--norm`` one-time row normalization of the retained values
(experimental/row_normalize, energy relative to the init row); ``--oracle anchor|match`` (rsn_oracle: the
teacher-mapped construction with the ANALYTIC readout, never recalibrated) and ``--perturb eps`` (declared
perturbation of the oracle, analytic readout kept); ``--sr-seed`` offsets the optimizer's stochastic-rounding
counter (training seed); ``--steps`` with evaluations every ``--eval-every`` updates (the 1,000-update prefix
of a 4,000-update run is the parent's protocol: constant learning rates, no schedule); ``--save-at`` stores the
complete state (layer values / main_grad / moments / step / rounding counter, readout and its AdamW state) for
a continuation with ``--resume``.  Ordinary initializations calibrate the readout on one batch exactly as the
parent (centered, scale 0.1 / std).

Reported: normalized held-out MSE per order (estimates on 2,048 held-out tokens), explained interaction
1 - nMSE / floor_k (exact additive floor of THIS configuration), full curves, the initial / pruned /
readout-calibrated losses, term-support eligibility (neurons whose enabled support contains all k inputs of
a term), oracle statistics and, for the oracle, the quantization floor: nMSE of the oracle against targets of the
FP32 inputs vs of the BF16-rounded inputs the layer actually reads.

Weighted-support study (results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md section 7.3): ``--final-test`` evaluates the
reserved final interaction-test batches (seeds 59_000..59_007, never used for selection) once after training;
the support record carries the authoritative total Q and the occupied / empty set counts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import common  # noqa: F401
import rsn_cases as RC
import wps_cases as WC
import wps_interaction as WI

FINAL_SEEDS = tuple(range(59_000, 59_008))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", default="dense")
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--terms", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=262_144)
    ap.add_argument("--init-seed", type=int, default=WC.INIT_SEED)
    ap.add_argument("--norm", default="none", choices=["none", "fanin", "energy"])
    ap.add_argument("--oracle", default="", choices=["", "anchor", "match"])
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--perturb-seed", type=int, default=99)
    ap.add_argument("--sr-seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-readout", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-at", default="", help="comma list of steps at which to store the complete state")
    ap.add_argument("--state-dir", default="")
    ap.add_argument("--resume", default="", help="state file of a previous run (continuation, same protocol)")
    ap.add_argument("--floors", default="", help="json cache of exact additive floors")
    ap.add_argument("--final-test", action="store_true", help="evaluate the reserved final test once after training")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.experimental import row_normalize as RN
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    import rsn_oracle as RO
    import wps_interaction_floor as WF

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    lay = KgxLayer(WC.D, a.hidden, WC.G, RC.C, ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed), variant="idf",
                   tile=128, device=dev, seed=a.init_seed, keep_indices=False)
    lay.enable_main_grad()
    L = lay.layout
    ids, rvalid, order, loc = WI.target_inputs(lay, a.terms)
    sig = WI.sigmas(dev)
    key = f"H{a.hidden}_C{RC.C}_g{a.graph_seed}_T{a.terms}"
    floors = {}
    if a.floors and Path(a.floors).exists():
        floors = json.loads(Path(a.floors).read_text())
    if key not in floors:
        floors[key] = WF.floors(a.hidden, RC.C, a.graph_seed, a.terms)
        if a.floors:
            Path(a.floors).write_text(json.dumps(floors, indent=1))
    floor = floors[key]
    res = dict(schema="granule.rsn_interaction/1", support=a.support, graph_seed=a.graph_seed, terms=a.terms,
               hidden=a.hidden, C=RC.C, init_seed=a.init_seed, norm=a.norm, oracle=a.oracle or None,
               perturb=a.perturb, perturb_seed=a.perturb_seed if a.perturb else None, sr_seed=a.sr_seed,
               lr_layer=a.lr, lr_readout=a.lr_readout, steps=a.steps, M_train=WI.M_TRAIN, hash_seed=WI.HASH_SEED,
               data=dict(train="seeds 50_000 + i", heldout="seeds 49_000..49_003", fixed_teacher=True),
               target_constants={k: dict(mu=v[0], sigma=v[1], single_term_floor=v[2]) for k, v in sig.items()},
               additive_floor=floor, floor_rule="wps_interaction_floor.floors (exact for the actual targets)",
               provenance=provenance(Path(common.__file__).resolve().parent.parent),
               occupied_sets=int(torch.unique(lay.kg_buckets[:, 0]).numel()))
    w_init = lay.weight.detach().clone()
    em, sel, srec = RC.support(a.support, lay, a.graph_seed)
    res["support_record"] = srec
    valid = L.valid_mask(dev)
    with torch.no_grad():
        if em is not None:
            lay.weight.mul_(em.to(lay.weight.dtype))
    # term-support eligibility (first term of each record, and all terms): neurons seeing all k inputs
    m = (torch.ones(L.weight_shape, dtype=torch.bool, device=dev) if em is None else em.bool())
    elig = {}
    for o in (1, 2, 3, 4):
        sel_o = order == o
        need = torch.arange(4, device=dev) < o
        cnt = torch.zeros(int(sel_o.sum()), loc.shape[2], device=dev)
        for u in range(loc.shape[2]):
            seen = m[sel_o].gather(3, loc[sel_o][:, :, None, u, :].expand(-1, -1, L.block, 4))
            ok = (seen | ~need).all(3) & valid[sel_o]
            cnt[:, u] = ok.sum((1, 2)).float()
        elig[f"order{o}"] = dict(mean_neurons_per_term=float(cnt.mean()), terms_with_no_neuron=int((cnt == 0).sum()),
                                 terms=int(cnt.numel()))
    res["term_support_eligibility"] = elig
    a_g = torch.ones(WC.G, device=dev, requires_grad=True)
    c_g = torch.zeros(WC.G, device=dev, requires_grad=True)
    if a.oracle:
        w, b, sc, off, ost = RO.build(lay, em, sel if hasattr(sel, "anchor") else None, loc, order, rvalid, sig,
                                      a.terms, a.oracle)
        if a.perturb:
            w, b = RO.perturb(w, b, em, a.perturb, a.perturb_seed)
        with torch.no_grad():
            lay.weight.copy_(w)
            lay.bias.copy_(b)
            a_g.copy_(sc)
            c_g.copy_(off)
        res["oracle_stats"] = ost
        del w, b
    elif a.norm != "none":
        with torch.no_grad():
            wn, bn, nrec = RN.apply(w_init, lay.bias.detach(), em if em is not None else
                                    torch.ones(L.weight_shape, dtype=torch.uint8, device=dev), valid, a.norm)
            lay.weight.copy_(wn)
            lay.bias.copy_(bn)
        res["normalization"] = nrec
        del wn, bn
    del w_init
    opt = L1.L1AdamW([lay.weight, lay.bias], lr=a.lr)
    opt._seed = a.sr_seed * 1_000_003
    if em is not None:
        L1.apply_mask_(opt, lay.weight, em)
    opt2 = torch.optim.AdamW([a_g, c_g], lr=a.lr_readout, weight_decay=0.0)

    def predict(x):
        return a_g * lay(x.bfloat16()).float() + c_g

    @torch.no_grad()
    def evaluate(xform=None, seeds=WI.HELD_SEEDS):
        se = torch.zeros(WC.G, device=dev, dtype=torch.float64)
        var = torch.zeros(WC.G, device=dev, dtype=torch.float64)
        for s in seeds:
            x = WI.xbatch(s, dev)
            t = WI.targets(x if xform is None else xform(x), ids, rvalid, order, sig)
            se += ((predict(x) - t) ** 2).sum(0).double()
            var += (t ** 2).sum(0).double()
        nm = se / var
        r = {f"order{o}": float(nm[order == o].mean()) for o in (1, 2, 3, 4)}
        r.update({f"explained_interaction_order{o}": 1.0 - r[f"order{o}"] / floor[f"order{o}"] for o in (2, 3, 4)})
        return r

    start = 0
    if a.resume:
        st = torch.load(a.resume, map_location=dev, weights_only=False)
        with torch.no_grad():
            lay.weight.copy_(st["weight"])
            lay.bias.copy_(st["bias"])
            lay.weight.main_grad.copy_(st["main_grad_w"])
            lay.bias.main_grad.copy_(st["main_grad_b"])
            a_g.copy_(st["a_g"])
            c_g.copy_(st["c_g"])
        opt.state[lay.weight].update({k: v.clone() for k, v in st["opt_w"].items()})
        opt.state[lay.bias].update({k: v.clone() for k, v in st["opt_b"].items()})
        opt._seed = st["sr_counter"]
        opt2.load_state_dict(st["opt2"])
        start = st["step"]
        if em is not None:
            opt.set_mask(lay.weight, em)
        res["resumed_from"] = dict(path=a.resume, step=start, sha256=st.get("sha256"))
        hist = st["history"]
    else:
        if not a.oracle:
            with torch.no_grad():  # the parent's calibration rule (one batch: centered, scale 0.1 / std)
                y = lay(WI.xbatch(WI.TRAIN_SEED - 1, dev).bfloat16()).float()
                sd = y.std(0).clamp_min(1e-12)
                a_g.copy_(0.1 / sd)
                c_g.copy_(-0.1 * y.mean(0) / sd)
        hist = [dict(step=0, **evaluate())]
        if a.oracle:
            res["oracle_quantization"] = dict(fp32_targets=hist[0], bf16_input_targets=evaluate(
                lambda x: x.bfloat16().float()))
    save_at = {int(s) for s in a.save_at.split(",") if s}
    for i in range(start, a.steps):
        x = WI.xbatch(WI.TRAIN_SEED + i, dev)
        loss = ((predict(x) - WI.targets(x, ids, rvalid, order, sig)) ** 2).mean()
        loss.backward()
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()
        if not torch.isfinite(loss):
            res["nonfinite_at"] = i + 1
            break
        if (i + 1) % a.eval_every == 0 or i + 1 == a.steps:
            hist.append(dict(step=i + 1, train=float(loss), **evaluate()))
            print(f"[{a.support} T{a.terms} {a.norm} {a.oracle or 'ordinary'}] {i + 1}: " + " ".join(
                f"{k[-6:]} {v:.4f}" for k, v in hist[-1].items() if k.startswith("expl")), flush=True)
        if i + 1 in save_at and a.state_dir:
            sd_ = Path(a.state_dir)
            sd_.mkdir(parents=True, exist_ok=True)
            p = sd_ / (Path(a.out).stem + f"_step{i + 1}.pt")
            torch.save(dict(step=i + 1, weight=lay.weight.detach(), bias=lay.bias.detach(),
                            main_grad_w=lay.weight.main_grad, main_grad_b=lay.bias.main_grad,
                            opt_w={k: v for k, v in opt.state[lay.weight].items()},
                            opt_b={k: v for k, v in opt.state[lay.bias].items()}, sr_counter=opt._seed,
                            opt2=opt2.state_dict(), a_g=a_g.detach(), c_g=c_g.detach(), history=list(hist),
                            args=vars(a)), p)
            import hashlib

            res.setdefault("saved_states", []).append(dict(step=i + 1, path=str(p),
                                                           sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    res["history"] = hist
    res["final"] = hist[-1]
    if a.final_test:
        if set(FINAL_SEEDS) & ({WI.TRAIN_SEED + i for i in range(a.steps)} | set(WI.HELD_SEEDS) | {WI.TRAIN_SEED - 1}):
            raise SystemExit("final interaction-test seeds collide with train / held-out seeds")
        res["final_test"] = dict(seeds=list(FINAL_SEEDS), **evaluate(seeds=FINAL_SEEDS))
    res["mask_preserved"] = (None if em is None else
                             bool(((lay.weight.detach() != 0) <= em.bool()).all()))
    res["wall_seconds"] = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
