"""Fixed-support fine-tuning and effective-width diagnostics of the width-preserving study (synthetic
diagnostic task of k64_learn.py, M_train = 512; NOT a downstream-accuracy claim).

Controlled comparison (assignment section 7), identical for every treatment:

* the actual alpha = 0 checkpoint of the K64 study (layer after 128 updates; its proj / head state), full
  W0 / H, the same teacher, batches (train seeds 10_000 + 128 + i, held-out seeds 9_000..9_003), loss;
* the support of (policy, rho, score, seed) selected from the SAME pre-pruning weights as for timing
  (element-mask sha256 recorded), applied as the dense masked layer (mathematically the compact formats'
  function: tests/test_kgx_wide.py), mandatory support never pruned, NO L1 anywhere;
* 32 fine-tuning updates (k64_learn FINETUNE) with FRESH optimizer state for every parameter (L1AdamW lr 1e-4
  for the layer, AdamW lr 1e-3 for proj / head), stochastic rounding as in the parent; "dense" = the
  unpruned model continued for the same updates.

Recorded: held-out loss at the checkpoint, after pruning, every 8 updates and after fine-tuning; training
losses; gradient norms (layer within the support, proj / head); numerical support over time (nonzero
fan-in, numerically bias-only rows, protected-support violations: a P / A row whose protected tile became
all-zero, an H row whose backbone weights are both zero -- disclosed, never patched with epsilon values);
on a FIXED uniform sample of valid neurons (seed 4242, 65,536 neurons) and 2,048 held-out tokens, before and
after fine-tuning: input-driven preactivation variance Var_t(z - b), weight norm, ReLU activation fraction,
gradient norm, and the classes structurally-disconnected / connected-but-all-zero / connected-never-active /
connected-always-active / input-dependent-and-switching.  ``--save-support`` stores the exact nonzero support
(+ the fine-tuned W_eff) after fine-tuning for coverage (outside git).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import common  # noqa: F401
import wps_cases as WC
from k64_learn import FINETUNE, LR_LAYER, LR_SMALL, REG, WARMUP, Teacher, batch, build, forward_loss, heldout

N_SAMPLE_NEURONS = 65_536
SAMPLE_NEURON_SEED = 4242
DIAG_TOKENS_SEEDS = (9_000, 9_001, 9_002, 9_003)


@torch.no_grad()
def neuron_sample(lay, n=N_SAMPLE_NEURONS, seed=SAMPLE_NEURON_SEED):
    """(slots [n], global input ids [n, 64]) of a fixed uniform sample of valid neurons (canonical order)."""
    L = lay.layout
    dev = lay.weight.device
    valid = L.valid_mask(dev).reshape(-1)
    vs = torch.nonzero(valid).squeeze(1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    pick = vs[torch.randperm(vs.numel(), generator=g)[:n].to(dev)].sort().values
    b = lay.kg_buckets.to(torch.int64)
    NB, C = L.blocks_per_group, L.block
    rb = torch.repeat_interleave(torch.arange(len(b), device=dev), b[:, 3])
    kb = torch.arange(len(rb), device=dev) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
    bq = torch.full((L.groups * NB,), -1, dtype=torch.int64, device=dev)
    bq[b[rb, 1] * NB + b[rb, 2] + kb] = b[rb, 0]
    q = bq[pick // C]
    return pick, lay.sg_feat.to(torch.int64)[q]


@torch.no_grad()
def diagnose(lay, proj, mask, pick, ids, teacher, head, dev, grad_rows=None) -> dict:
    """Per-neuron input sensitivity on held-out tokens (see the module docstring)."""
    xs = torch.cat([proj(batch(s, dev)).bfloat16() for s in DIAG_TOKENS_SEEDS]).float()  # [2048, D]
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
        u = torch.einsum("tnk,nk->tn", xs[:, ids[s0:s0 + 4096]], w[s0:s0 + 4096])  # input-driven part [T, n]
        var[s0:s0 + 4096] = u.var(0)
        act[s0:s0 + 4096] = ((u + bias[s0:s0 + 4096]) > 0).float().mean(0)
    wn = w.norm(dim=1)
    gn = None if grad_rows is None else grad_rows.norm(dim=1)
    cls = dict(structurally_disconnected=int((deg == 0).sum()),
               connected_all_zero=int(((deg > 0) & (nz == 0)).sum()),
               never_active=int(((nz > 0) & (act == 0)).sum()), always_active=int(((nz > 0) & (act == 1)).sum()),
               switching=int(((nz > 0) & (act > 0) & (act < 1)).sum()))

    def q(v):
        v = v.double().cpu()
        qs = torch.quantile(v, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], dtype=torch.float64))
        return dict(mean=float(v.mean()), p1=float(qs[0]), p10=float(qs[1]), median=float(qs[2]), p90=float(qs[3]),
                    p99=float(qs[4]), max=float(v.max()))

    con = nz > 0
    out = dict(sample=len(pick), tokens=xs.shape[0], classes=cls,
               input_variance=q(var), input_variance_connected=q(var[con]) if bool(con.any()) else None,
               weight_norm=q(wn), activation_fraction=q(act), activation_fraction_connected=q(act[con])
               if bool(con.any()) else None,
               variance_threshold_counts={f"{t:g}": int((var > t).sum()) for t in (1e-6, 1e-4, 1e-2)})
    if gn is not None:
        out["grad_norm"] = q(gn)
        out["grad_norm_connected"] = q(gn[con]) if bool(con.any()) else None
    return out


@torch.no_grad()
def support_monitor(lay, mask, ms, valid) -> dict:
    """Numerical support: nonzero fan-in, numerically bias-only rows, protected-support violations."""
    w = lay.weight.detach()
    wv = w.reshape(-1, 64)
    ev = None if mask is None else mask.reshape(-1, 64)
    nz = torch.empty(wv.shape[0], dtype=torch.int16, device=w.device)
    for s0 in range(0, wv.shape[0], 1 << 21):  # chunked: a whole-tensor bool sum materialises 8 B per element
        z = wv[s0:s0 + (1 << 21)] != 0
        if ev is not None:
            z &= ev[s0:s0 + (1 << 21)].bool()
        nz[s0:s0 + (1 << 21)] = z.sum(1, dtype=torch.int16)
    nz = nz[valid.reshape(-1)].to(torch.int64)
    out = dict(nonzero_slots=int(nz.sum()), mean_nonzero_fan_in=float(nz.double().mean()),
               numerically_bias_only=int((nz == 0).sum()))
    if ms is not None:
        n16 = ms.nv.numel()
        vv = valid.reshape(n16, 16)
        prot = ms.mandatory  # [n16, nkt]
        if ms.policy.protect or ms.policy.anchors:
            viol = 0
            for u0 in range(0, n16, 1 << 16):
                tz = (w.reshape(n16, 16, ms.policy.nkt, ms.tb)[u0:u0 + (1 << 16)] != 0).any(3)
                pr = prot[u0:u0 + (1 << 16)]
                viol += int(((vv[u0:u0 + (1 << 16)] & ~(tz & pr[:, None, :]).any(2)) & pr.any(1)[:, None]).sum())
            out["protected_tile_violations"] = viol
        if ms.bb is not None:
            ar = torch.arange(n16, device=w.device)
            wr = w.reshape(n16, 16, 64)
            b0 = wr[ar, :, ms.bb[:, 0]] != 0
            b1 = wr[ar, :, ms.bb[:, 1]] != 0
            out["backbone_violations"] = int((vv & ~(b0 | b1)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--policy", default="dense")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--score", default="magnitude")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=FINETUNE)
    ap.add_argument("--save-support", default="", help="directory (outside git) for the fine-tuned support")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.experimental import width_masks as WM
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
    res = dict(C=a.C, graph_seed=a.graph_seed, policy=a.policy, rho_target=a.rho, score=a.score, seed=a.seed,
               checkpoint=dict(path=ck, weight_sha256=rec["weight_sha256"], bias_sha256=rec["bias_sha256"],
                               tag=rec.get("meta", {}).get("tag")),
               protocol=dict(steps=a.steps, lr_layer=LR_LAYER, lr_small=LR_SMALL, fresh_optimizer_state=True,
                             l1="none", train_batches=f"seeds 10_000 + {WARMUP + REG} + i", heldout="seeds 9_000..9_003",
                             stochastic_rounding=True, neuron_sample=dict(n=N_SAMPLE_NEURONS, seed=SAMPLE_NEURON_SEED),
                             note="dense masked execution of the support (same function as the compact formats)"),
               provenance=provenance(Path(common.__file__).resolve().parent.parent), hardware_state=hardware_state())
    res["heldout_checkpoint"] = heldout(lay, proj, head, teacher, dev)
    mask, ms = None, None
    if a.policy != "dense":
        ms = WM.build(a.policy, a.rho, lay.weight.detach(), lay.layout, score=a.score, seed=a.seed)
        mask = ms.element_mask(lay.layout)
        res["mask_stats"] = ms.stats
        res["mask_sha256"] = SP.mask_fingerprint(mask)
    opt = L1.L1AdamW([lay.weight, lay.bias], lr=LR_LAYER)
    if mask is not None:
        L1.apply_mask_(opt, lay.weight, mask)
    opt2 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=LR_SMALL, weight_decay=0.0)
    res["heldout_pruned"] = heldout(lay, proj, head, teacher, dev)

    def grad_rows():  # main_grad of the sampled neurons on one held-out batch (no update)
        lay.zero_main_grad()
        x0 = batch(DIAG_TOKENS_SEEDS[0], dev)
        forward_loss(lay, proj, head, x0, teacher(x0)).backward()
        for p in list(proj.parameters()) + list(head.parameters()):
            p.grad = None
        g = lay.weight.main_grad.reshape(-1, 64)[pick].clone()
        if mask is not None:
            g *= mask.reshape(-1, 64)[pick]
        lay.zero_main_grad()
        return g

    res["diagnostics_before"] = diagnose(lay, proj, mask, pick, ids, teacher, head, dev, grad_rows())
    res["support_before"] = support_monitor(lay, mask, ms, valid)
    curve, evals = [], []
    step0 = WARMUP + REG
    small_params = list(proj.parameters()) + list(head.parameters())
    for i in range(a.steps):
        x0 = batch(10_000 + step0 + i, dev)
        loss = forward_loss(lay, proj, head, x0, teacher(x0))
        loss.backward()
        gl = float(lay.weight.main_grad.norm()) if mask is None else float((lay.weight.main_grad * mask).norm())
        gs = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in small_params)))
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()
        curve.append(dict(step=i + 1, loss=float(loss), grad_norm_layer=gl, grad_norm_small=gs))
        if (i + 1) % 8 == 0:
            evals.append(dict(step=i + 1, heldout=heldout(lay, proj, head, teacher, dev),
                              support=support_monitor(lay, mask, ms, valid)))
            print(f"[{a.policy} {a.rho:g}] step {i + 1}: train {float(loss):.6f} held-out {evals[-1]['heldout']:.6f} "
                  f"bias-only(num) {evals[-1]['support']['numerically_bias_only']}", flush=True)
    res["train_curve"], res["evals"] = curve, evals
    res["heldout_finetuned"] = heldout(lay, proj, head, teacher, dev)
    res["diagnostics_after"] = diagnose(lay, proj, mask, pick, ids, teacher, head, dev, grad_rows())
    res["support_after"] = support_monitor(lay, mask, ms, valid)
    if mask is not None:
        res["mask_preserved"] = bool(((lay.weight.detach() != 0) <= mask.bool()).all())
    if a.save_support:
        d = Path(a.save_support)
        d.mkdir(parents=True, exist_ok=True)
        nzm = ((lay.weight.detach() != 0) & (mask.bool() if mask is not None else True)).to(torch.uint8)
        tag = f"{WC.case_id(a.policy, a.rho, a.C, a.score, a.graph_seed)}_finetuned"
        p = d / f"{tag}.pt"
        torch.save(dict(rec=dict(kind=f"{a.policy} rho {a.rho:g} ({a.score}) exact nonzero support after "
                                      f"{a.steps} fine-tuning updates", mask_sha256=SP.mask_fingerprint(nzm),
                                 weight_sha256=SP.mask_fingerprint(lay.weight.detach().view(torch.int16))),
                        mask_bits=SP._pack_bits(nzm).cpu(), weight=lay.weight.detach().cpu()), p)
        res["saved_support"] = dict(path=str(p), file_sha256=SP._file_sha(p))
    res["wall_seconds"] = time.perf_counter() - t_start
    res["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"{a.policy} {a.rho:g} {a.score}: held-out {res['heldout_checkpoint']:.6f} -> pruned "
          f"{res['heldout_pruned']:.6f} -> fine-tuned {res['heldout_finetuned']:.6f}", flush=True)


if __name__ == "__main__":
    main()
