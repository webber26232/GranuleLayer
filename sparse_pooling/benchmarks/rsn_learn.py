"""Frozen-support fine-tuning with one-time row normalization and signal diagnostics (random-support study,
results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md section 5; the synthetic task of k64_learn.py, M_train = 512; NOT
a downstream-accuracy claim).

Common to every treatment (the parent protocol of wps_learn.py): the actual alpha = 0 checkpoint (sha256
verified) with its proj / head, full H / W0, the same teacher and held-out batches (seeds 9_000..9_003), loss,
dense masked execution of the frozen support (rsn_cases spec, selected from the checkpoint's pre-pruning values),
no L1, FRESH optimizer state (L1AdamW lr 1e-4 layer, AdamW lr 1e-3 proj / head, stochastic rounding).
Per run: ``--norm`` none / fanin / energy (experimental/row_normalize: applied ONCE to the retained values before
fine-tuning, energies of the same pre-pruning canonical row; ``--scale-bias`` the W-and-bias control);
``--data-seed`` (train batch i uses seed 10_000 + 128 + i + 1_000_000 data_seed; 0 = the parent's batches) and
``--sr-seed`` (offset of the optimizer's stochastic-rounding counter); ``--steps`` (32 = the parent; 256 extended).

Diagnostics on a FIXED canonical neuron sample (65,536 valid neurons, seed 4242, all valid rows -- weak rows are
never excluded; connected-only distributions reported beside them) before normalization, after normalization and
after training:

* ``probe``: FIXED layer-input probes (the checkpoint's proj on the 2,048 held-out tokens, computed once before
  any update): isolates weight / support changes;
* ``e2e``: each model's CURRENT proj on the same tokens (includes projection adaptation);
* structural / numerical bias-only counts, fan-in histogram, weight norms; input-driven variance Var_t(z - b)
  absolute and relative to the SAME neuron's original dense row on the probes; ReLU activation / switching
  fractions; per-neuron gradient norms (one held-out batch); output variance of the layer (Var_t y_g);
* a bounded activation-diversity diagnostic: 2,048 neurons of the sample (seed 4243) x 2,048 probe tokens,
  amplitude-normalized ReLU outputs (each neuron standardized; constant neurons -> zero columns, counted),
  participation ratio (sum lambda)^2 / sum lambda^2 of the neuron correlation matrix and mean |off-diagonal
  correlation| (rank <= 2,048: a diagnostic, not a capacity measure).

The historical absolute statistic Var > 0.01 is kept only for comparison with the parent; crossing it after
rescaling is not counted as an improvement by itself.
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
from k64_learn import LR_LAYER, LR_SMALL, REG, WARMUP, Teacher, batch, build, forward_loss, heldout
from wps_learn import DIAG_TOKENS_SEEDS, N_SAMPLE_NEURONS, SAMPLE_NEURON_SEED, neuron_sample, support_monitor

PR_NEURONS, PR_SEED = 2048, 4243


def _q(v):
    v = v.double().cpu()
    if v.numel() == 0:
        return None
    qs = torch.quantile(v, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], dtype=torch.float64))
    return dict(mean=float(v.mean()), p1=float(qs[0]), p10=float(qs[1]), median=float(qs[2]), p90=float(qs[3]),
                p99=float(qs[4]), max=float(v.max()), n=int(v.numel()))


@torch.no_grad()
def signal(lay, mask, pick, ids, xs, var_dense, pr_idx, grad_rows=None) -> dict:
    """Per-neuron signal statistics of the sampled rows on token inputs xs [T, D] (module docstring)."""
    dev = xs.device
    w = lay.weight.detach().reshape(-1, 64)[pick].float()
    m = None if mask is None else mask.reshape(-1, 64)[pick].bool()
    if m is not None:
        w = w * m
    bias = lay.bias.detach().reshape(-1)[pick].float()
    deg = torch.full((len(pick),), 64, device=dev) if m is None else m.sum(1)
    nz = (w != 0).sum(1)
    var = torch.empty(len(pick), device=dev)
    act = torch.empty(len(pick), device=dev)
    for s0 in range(0, len(pick), 4096):
        u = torch.einsum("tnk,nk->tn", xs[:, ids[s0:s0 + 4096]], w[s0:s0 + 4096])
        var[s0:s0 + 4096] = u.var(0)
        act[s0:s0 + 4096] = ((u + bias[s0:s0 + 4096]) > 0).float().mean(0)
    con = nz > 0
    rel = var / var_dense.clamp_min(1e-30)
    switching = (act > 0) & (act < 1)
    out = dict(sample=len(pick), tokens=xs.shape[0],
               classes=dict(structurally_disconnected=int((deg == 0).sum()),
                            connected_all_zero=int(((deg > 0) & (nz == 0)).sum()),
                            never_active=int((con & (act == 0)).sum()), always_active=int((con & (act == 1)).sum()),
                            switching=int((con & switching).sum())),
               switching_fraction_all=float(switching.float().mean()),
               fan_in_hist=torch.bincount(nz, minlength=65).tolist(),
               weight_norm=_q(w.norm(dim=1)), weight_norm_connected=_q(w.norm(dim=1)[con]),
               input_variance=_q(var), input_variance_connected=_q(var[con]),
               relative_variance_vs_dense_row=_q(rel), relative_variance_vs_dense_row_connected=_q(rel[con]),
               activation_fraction=_q(act), activation_fraction_connected=_q(act[con]),
               variance_threshold_counts={f"{t:g}": int((var > t).sum()) for t in (1e-6, 1e-4, 1e-2)},
               relative_variance_threshold_counts={f"{t:g}": int((rel > t).sum()) for t in (0.01, 0.1, 0.5)})
    if grad_rows is not None:
        gn = grad_rows.norm(dim=1)
        out["grad_norm"], out["grad_norm_connected"] = _q(gn), _q(gn[con])
    # activation diversity on the bounded sub-sample
    wp, bp, ip = w[pr_idx], bias[pr_idx], ids[pr_idx]
    z = torch.einsum("tnk,nk->tn", xs[:, ip], wp) + bp
    a = torch.relu(z).double()
    sd = a.std(0)
    const = sd == 0
    zs = torch.where(const[None, :], torch.zeros_like(a), (a - a.mean(0)) / sd.clamp_min(1e-300))
    corr = zs.t() @ zs / (a.shape[0] - 1)
    ev = torch.linalg.eigvalsh(corr).clamp_min(0)
    off = corr[~torch.eye(corr.shape[0], dtype=torch.bool, device=dev)]
    nc = int((~const).sum())
    out["diversity"] = dict(neurons=int(len(pr_idx)), tokens=int(a.shape[0]), constant_neurons=int(const.sum()),
                            participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()) if float(ev.sum()) > 0 else 0.0,
                            rank_limit=min(int(len(pr_idx)), int(a.shape[0])),
                            mean_abs_offdiag_corr_nonconstant=float(
                                corr[~const][:, ~const][~torch.eye(nc, dtype=torch.bool, device=dev)].abs().mean())
                            if nc > 1 else None,
                            mean_abs_offdiag_corr_all=float(off.abs().mean()))
    return out


@torch.no_grad()
def layer_output_variance(lay, xs) -> dict:
    y = torch.cat([lay(xs[s:s + 512].bfloat16()).float() for s in range(0, xs.shape[0], 512)])
    v = y.var(0)
    return dict(mean=float(v.mean()), median=float(v.median()), min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--support", default="dense")
    ap.add_argument("--norm", default="none", choices=["none", "fanin", "energy"])
    ap.add_argument("--scale-bias", action="store_true")
    ap.add_argument("--s-max", type=float, default=8.0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--sr-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.experimental import row_normalize as RN
    from provenance import hardware_state, provenance

    dev = torch.device("cuda")
    t_start = time.perf_counter()
    ck = WC.ckpt(a.C, a.graph_seed)
    lay, proj, head = build(a.C, 256, dev, a.graph_seed)
    rec, w, b, _ = SP.load_checkpoint(ck)
    if SP.mask_fingerprint(w.view(torch.int16)) != rec["weight_sha256"]:
        raise SystemExit("checkpoint weight sha256 differs from its record")
    with torch.no_grad():
        lay.weight.copy_(w)
        lay.bias.copy_(b)
    del w, b
    small = torch.load(ck.replace(".pt", "_small.pt"), map_location=dev)
    proj.load_state_dict(small["proj"])
    head.load_state_dict(small["head"])
    teacher = Teacher(dev)
    valid = SP.valid_slots(lay)
    pick, ids = neuron_sample(lay)
    gpr = torch.Generator(device="cpu").manual_seed(PR_SEED)
    pr_idx = torch.randperm(len(pick), generator=gpr)[:PR_NEURONS].sort().values.to(dev)
    with torch.no_grad():
        probes = torch.cat([proj(batch(s, dev)).bfloat16() for s in DIAG_TOKENS_SEEDS]).float()  # fixed [2048, D]
        wd = lay.weight.detach().reshape(-1, 64)[pick].float()
        var_dense = torch.cat([torch.einsum("tnk,nk->tn", probes[:, ids[s:s + 4096]], wd[s:s + 4096]).var(0)
                               for s in range(0, len(pick), 4096)])
        del wd
    res = dict(schema="granule.rsn_learn/1", C=a.C, graph_seed=a.graph_seed, support=a.support, norm=a.norm,
               scale_bias=a.scale_bias, s_max=a.s_max, data_seed=a.data_seed, sr_seed=a.sr_seed, steps=a.steps,
               checkpoint=dict(path=ck, weight_sha256=rec["weight_sha256"], bias_sha256=rec["bias_sha256"],
                               tag=rec.get("meta", {}).get("tag")),
               protocol=dict(lr_layer=LR_LAYER, lr_small=LR_SMALL, fresh_optimizer_state=True, l1="none",
                             train_batches=f"seeds 10_000 + {WARMUP + REG} + i + 1_000_000 * data_seed",
                             heldout="seeds 9_000..9_003", stochastic_rounding=True,
                             neuron_sample=dict(n=N_SAMPLE_NEURONS, seed=SAMPLE_NEURON_SEED),
                             diversity_sample=dict(n=PR_NEURONS, seed=PR_SEED),
                             probes="checkpoint proj on held-out tokens (2,048), fixed before any update",
                             execution="dense masked (the compact formats' function)"),
               provenance=provenance(Path(common.__file__).resolve().parent.parent), hardware_state=hardware_state())
    res["heldout_checkpoint"] = heldout(lay, proj, head, teacher, dev)
    res["dense_row_reference"] = dict(input_variance=_q(var_dense))
    t0 = time.perf_counter()
    em, sel, srec = RC.support(a.support, lay, a.graph_seed)
    res["support_record"] = srec
    res["support_select_seconds"] = time.perf_counter() - t0
    w_pre = lay.weight.detach().clone() if a.norm != "none" else None
    if em is not None:
        with torch.no_grad():
            lay.weight.mul_(em.to(lay.weight.dtype))
    ms = sel if hasattr(sel, "mandatory") else None
    res["heldout_pruned"] = heldout(lay, proj, head, teacher, dev)
    res["signal_pruned"] = dict(probe=signal(lay, em, pick, ids, probes, var_dense, pr_idx),
                                output_variance=layer_output_variance(lay, probes))
    if a.norm != "none":
        t0 = time.perf_counter()
        with torch.no_grad():
            wn, bn, nrec = RN.apply(w_pre, lay.bias.detach(), em if em is not None else
                                    torch.ones(lay.weight.shape, dtype=torch.uint8, device=dev), valid, a.norm,
                                    scale_bias=a.scale_bias, s_max=a.s_max)
            lay.weight.copy_(wn)
            lay.bias.copy_(bn)
        del wn, bn, w_pre
        nrec["seconds"] = time.perf_counter() - t0
        res["normalization"] = nrec
        res["heldout_normalized"] = heldout(lay, proj, head, teacher, dev)
    opt = L1.L1AdamW([lay.weight, lay.bias], lr=LR_LAYER)
    opt._seed = a.sr_seed * 1_000_003
    if em is not None:
        L1.apply_mask_(opt, lay.weight, em)
    opt2 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=LR_SMALL, weight_decay=0.0)

    def grad_rows():  # main_grad of the sampled neurons on one held-out batch (no update)
        lay.zero_main_grad()
        x0 = batch(DIAG_TOKENS_SEEDS[0], dev)
        forward_loss(lay, proj, head, x0, teacher(x0)).backward()
        for p in list(proj.parameters()) + list(head.parameters()):
            p.grad = None
        g = lay.weight.main_grad.reshape(-1, 64)[pick].clone()
        if em is not None:
            g *= em.reshape(-1, 64)[pick]
        lay.zero_main_grad()
        return g

    def e2e_tokens():
        with torch.no_grad():
            return torch.cat([proj(batch(s, dev)).bfloat16() for s in DIAG_TOKENS_SEEDS]).float()

    res["signal_start"] = dict(probe=signal(lay, em, pick, ids, probes, var_dense, pr_idx, grad_rows()),
                               output_variance=layer_output_variance(lay, probes))
    res["support_start"] = support_monitor(lay, em, ms, valid)
    curve, evals = [], []
    step0 = WARMUP + REG
    small_params = list(proj.parameters()) + list(head.parameters())
    for i in range(a.steps):
        x0 = batch(10_000 + step0 + i + 1_000_000 * a.data_seed, dev)
        loss = forward_loss(lay, proj, head, x0, teacher(x0))
        loss.backward()
        mg = lay.weight.main_grad
        gl = float(mg.norm()) if em is None else float((mg * em).norm())
        gs = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in small_params)))
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()
        curve.append(dict(step=i + 1, loss=float(loss), grad_norm_layer=gl, grad_norm_small=gs))
        if (i + 1) % a.eval_every == 0 or i + 1 == a.steps:
            evals.append(dict(step=i + 1, heldout=heldout(lay, proj, head, teacher, dev)))
            print(f"[{a.support} {a.norm}{'+b' if a.scale_bias else ''} d{a.data_seed}] step {i + 1}: "
                  f"train {float(loss):.6f} held-out {evals[-1]['heldout']:.6f}", flush=True)
    res["train_curve"], res["evals"] = curve, evals
    res["heldout_final"] = evals[-1]["heldout"]
    gr = grad_rows()
    res["signal_after"] = dict(probe=signal(lay, em, pick, ids, probes, var_dense, pr_idx, gr),
                               e2e=signal(lay, em, pick, ids, e2e_tokens(), var_dense, pr_idx),
                               output_variance=layer_output_variance(lay, probes),
                               output_variance_e2e=layer_output_variance(lay, e2e_tokens()))
    res["support_after"] = support_monitor(lay, em, ms, valid)
    if em is not None:
        res["mask_preserved"] = bool(((lay.weight.detach() != 0) <= em.bool()).all())
    res["wall_seconds"] = time.perf_counter() - t_start
    res["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"{a.support} {a.norm}: held-out {res['heldout_checkpoint']:.6f} -> pruned {res['heldout_pruned']:.6f} -> "
          f"final {res['heldout_final']:.6f}", flush=True)


if __name__ == "__main__":
    main()
