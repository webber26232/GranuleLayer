"""Pre-training gate at the primary shape, on the real tasks: direct vs accumulated gradients before the optimizer, one matched update,
several updates, and resume against a floating-point baseline.  The tolerances are those of tests/test_lrf_followup.py.

    python lrf_gate.py --root R --out gate.json
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import learned_reduction_common as C
import lrf_train as TR
import torch

TOL = dict(sparse=1e-4, dense=1e-4, input=2e-3, loss=1e-5, one_update_fp32=1e-4, one_update_bf16=4e-3)


def relerr(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def ns(a, model, task, m_phys, accum, updates, seed=0):
    return argparse.Namespace(root=a.root, H=a.H, s=2048, model=model, task=task, seed=seed, m_phys=m_phys, accum=accum, updates=updates, eval_every=updates, grad_every=0,
                              full_every=0, milestones=[], lr_w=1e-3, lr_small=1e-3, p_mult=1.0, lr_multiplier=1.0, verbose=0, stage="gate")


def grads(t):
    m = t.model
    d = dict(W=m.sparse.weight.main_grad.clone(), b=m.sparse.bias.main_grad.clone(), head=t.head.weight.grad.clone(), proj=t.proj.weight.grad.clone())
    if m.reducer is not None:
        d["P"] = m.reducer.P.main_grad.clone()
    for k in ("c", "V", "d"):
        p = getattr(m, k)
        if p is not None:
            d[k] = p.grad.clone()
    return d


def tol_of(k):
    return TOL["input"] if k == "proj" else TOL["sparse"] if k in ("W", "b", "P") else TOL["dense"]


def direct_vs_accum(a, model, task):
    x, y = TR.Trainer(ns(a, model, task, 8192, 1, 4)), TR.Trainer(ns(a, model, task, 512, 16, 4))
    if x.model.gamma.item() != y.model.gamma.item():
        raise AssertionError("gamma differs between execution modes")
    lx, ly = x.backward_calls(), y.backward_calls()
    gx, gy = grads(x), grads(y)
    rec = dict(model=model, task=task, gamma=x.model.gamma.item(), loss_direct=lx, loss_accum=ly, loss_relerr=abs(lx - ly) / abs(lx),
               grad_relerr={k: relerr(gy[k], gx[k]) for k in gx})
    del gx, gy
    x.optimizer_step(), y.optimizer_step()
    x.update_count += 1
    y.update_count += 1
    rec["one_update_relerr"] = {k: relerr(p2.detach().float(), p1.detach().float()) for (k, p1), (_, p2) in zip(x.model.named_parameters(), y.model.named_parameters())}
    rec["one_update_tol"] = {k: TOL["one_update_fp32"] if p.dtype == torch.float32 else TOL["one_update_bf16"] for k, p in x.model.named_parameters()}
    for _ in range(3):
        lx, ly = x.effective_update(), y.effective_update()
    x.check_position(), y.check_position()
    rec.update(after4=dict(loss_direct=lx, loss_accum=ly, val_direct=x.validation(), val_accum=y.validation(), updates=x.update_count, samples=x.samples_seen,
                           blocks=x.next_block, fused_calls=[x.fused_calls, y.fused_calls], fused_seed=[x.fused._seed, y.fused._seed]))
    rec["ok"] = (rec["loss_relerr"] < TOL["loss"] and all(v < tol_of(k) for k, v in rec["grad_relerr"].items())
                 and all(v < rec["one_update_tol"][k] for k, v in rec["one_update_relerr"].items()) and x.samples_seen == y.samples_seen == 4 * 8192
                 and x.fused_calls == y.fused_calls == 4)
    print(rec, flush=True)
    return rec


def resume(a, model, task):
    def run(updates, ckpt=None, save=None):
        t = TR.Trainer(ns(a, model, task, 512, 1, updates))
        if ckpt:
            t.load_full(ckpt)
        pos = t.next_block
        t.train()
        if save:
            t.save_full(save)
        return t, pos

    U = 32
    p, _ = run(U)
    q, _ = run(U)
    base = {k: relerr(q.parameter_state()[k].float(), p.parameter_state()[k].float()) for k in p.parameter_state()}
    with tempfile.TemporaryDirectory(dir=a.tmp) as d:
        run(U // 2, save=Path(d))
        r, pos = run(U, ckpt=sorted(Path(d).glob("full_u*.pt"))[-1])
    diff = {k: relerr(r.parameter_state()[k].float(), p.parameter_state()[k].float()) for k in p.parameter_state()}
    rec = dict(model=model, task=task, updates=U, resumed_at_block=pos, final_block=r.next_block, baseline_relerr=base, resume_relerr=diff, val_continuous=p.curve[-1]["val"],
               val_repeat=q.curve[-1]["val"], val_resumed=r.curve[-1]["val"], ok=pos == U // 2 and r.next_block == U and max(diff.values()) <= max(10 * max(base.values()), 1e-2))
    print(rec, flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--H", type=int, default=10_485_760)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.tmp).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    out = dict(schema="granule.lrf_gate/1", H=a.H, tolerances=TOL, accumulation=[direct_vs_accum(a, m, t) for m in ("A0", "A2", "A3mean") for t in ("compose", "interact")],
               resume=[resume(a, m, "compose") for m in ("A0", "A2", "A3mean")], seconds=None, env=C.env_record())
    out["seconds"] = time.perf_counter() - t0
    out["ok"] = all(r["ok"] for r in out["accumulation"] + out["resume"])
    C.atomic_json(a.out, out)
    print("gate ok" if out["ok"] else "GATE FAILED", out["seconds"], flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
