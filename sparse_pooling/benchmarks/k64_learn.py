"""L1 learning diagnostic of the K64 study (labelled workload M_train = 512; NOT a performance measurement and
NOT a downstream-accuracy claim).

Composed deterministic synthetic regression:

    x0 [M, 256] ~ N(0, 1)  ->  proj = Linear(256, 1024)  ->  BF16  ->  K64 layer (H = 12.5M, W0 = 800M valid slots,
    logical C, S = 64, R = 256, index-free kgx)  ->  y [M, 1024]  ->  head = Linear(1024, 64)  ->  MSE vs teacher

Teacher (fixed, seed 31): t(x0) = tanh(A2 relu(A1 x0 + c1)), A1 [512, 256], A2 [64, 512]; train batch i draws x0
with seed 10_000 + i; four fixed held-out batches use seeds 9_000..9_003; loss = mean over batch and outputs.
proj / head: torch AdamW lr 1e-3 (recorded); layer: L1AdamW lr 1e-4 (the main lr), BF16 moments, stochastic
rounding, weight_decay 0.  Layer init: per-valid-neuron draws N(0, 1/64), seed 17 (identical valid draws for
every C), zero bias; proj / head init seed 29.

Schedule: 32 unregularized warm-up updates; calibration (4 fixed batches, seeds 8_000..8_003: averaged FP32 task
gradient; g_ref = median |g| over a fixed sample of 1M valid weight positions, seed 7; fallback: mean of the
nonzero |g| if the median is 0); then 96 regularized updates with mu ramped linearly over the first 16 to
mu = alpha g_ref (fixed afterwards); checkpoint; then 32 fixed-mask fine-tuning updates (mask = the exact
nonzero support of the checkpoint; mu = 0; inactive weights / moments frozen at zero).

Objective of the regularized phase: L = L_task + mu sum_valid |w| (lambda_mean = mu W0, W0 = 800M fixed).
Treatments: "prox" (diagonal-metric proximal AdamW; data-gradient moments) and "coupled" (mu sign(w) added to
main_grad before AdamW); alpha = 0 is the dense baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import common  # noqa: F401
import k64_cases as KC

D0, G2, M_TRAIN = 256, 64, 512
WARMUP, REG, RAMP, FINETUNE = 32, 96, 16, 32
LR_LAYER, LR_SMALL = 1e-4, 1e-3
EVAL_EVERY = 8
TEACHER_SEED, SMALL_SEED, CAL_SEED, SAMPLE_SEED = 31, 29, 8_000, 7
N_SAMPLE = 1_000_000


class Teacher:
    def __init__(self, device):
        g = torch.Generator(device="cpu").manual_seed(TEACHER_SEED)
        self.A1 = (torch.randn(512, D0, generator=g) / math.sqrt(D0)).to(device)
        self.c1 = (0.1 * torch.randn(512, generator=g)).to(device)
        self.A2 = (torch.randn(G2, 512, generator=g) * math.sqrt(2.0 / 512)).to(device)

    def __call__(self, x0):
        return torch.tanh(torch.relu(x0 @ self.A1.t() + self.c1) @ self.A2.t())


def batch(seed: int, device, M=M_TRAIN):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(M, D0, generator=g).to(device)


def build(C: int, R: int, device):
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan

    lay = KgxLayer(KC.D, KC.H, KC.G, C, ConnectivityPlan(64, R, seed=KC.GRAPH_SEED), variant="idf", tile=128,
                   device=device, seed=KC.INIT_SEED, keep_indices=False)
    lay.enable_main_grad()
    torch.manual_seed(SMALL_SEED)
    proj = torch.nn.Linear(D0, KC.D, device=device)
    head = torch.nn.Linear(KC.G, G2, device=device)
    return lay, proj, head


def forward_loss(lay, proj, head, x0, tgt):
    y = lay(proj(x0).bfloat16())
    return ((head(y.float()) - tgt) ** 2).mean()


@torch.no_grad()
def heldout(lay, proj, head, teacher, device):
    tot = 0.0
    for s in range(9_000, 9_004):
        x0 = batch(s, device)
        tot += float(forward_loss(lay, proj, head, x0, teacher(x0)))
    return tot / 4


def grad_norm(params):
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--treatment", choices=["prox", "coupled"], default="prox")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--reg-steps", type=int, default=REG)
    ap.add_argument("--ckpt-dir", required=True, help="checkpoints (outside git)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.kernels import cuda_ops
    from provenance import hardware_state, provenance

    dev = torch.device("cuda")
    t_start = time.perf_counter()
    lay, proj, head = build(a.C, a.R, dev)
    teacher = Teacher(dev)
    valid = SP.valid_slots(lay)
    W0 = int(valid.sum()) * 64
    assert W0 == KC.WEIGHTS
    opt = L1.L1AdamW([{"params": [lay.weight], "l1": a.treatment, "mu": 0.0}, {"params": [lay.bias], "l1": "none"}],
                     lr=LR_LAYER, moments="bf16")
    small = list(proj.parameters()) + list(head.parameters())
    opt2 = torch.optim.AdamW(small, lr=LR_SMALL, weight_decay=0.0)
    log = dict(config=dict(C=a.C, S=64, R=a.R, K=64, H=KC.H, W0=W0, M_train=M_TRAIN, D0=D0, G2=G2, treatment=a.treatment,
                           alpha=a.alpha, warmup=WARMUP, reg_steps=a.reg_steps, ramp=RAMP, finetune=FINETUNE,
                           lr_layer=LR_LAYER, lr_small=LR_SMALL, weight_decay=0.0, moments="bf16",
                           stochastic_rounding=True, init="granule.init_by_neuron/1 seed 17, bias 0",
                           teacher_seed=TEACHER_SEED, small_seed=SMALL_SEED, graph_seed=KC.GRAPH_SEED,
                           variant="idf", tile=128),
               steps=[], evals=[], provenance=provenance(Path(common.__file__).resolve().parent.parent),
               hardware_state=hardware_state())
    ev = {}

    def timed_step(i, mu, phase):
        opt.param_groups[0]["mu"] = mu
        x0 = batch(10_000 + i, dev)
        tgt = teacher(x0)
        events = []
        cuda_ops.PHASE_EVENTS = events
        try:
            cuda_ops._mark("step.begin")
            loss = forward_loss(lay, proj, head, x0, tgt)
            cuda_ops._mark("fwd.end")
            loss.backward()
            cuda_ops._mark("bwd.end")
            gn_layer = float(lay.weight.main_grad.norm())
            gn_small = grad_norm(small)
            opt2.step()
            opt2.zero_grad(set_to_none=True)
            cuda_ops._mark("small.end")
            opt.step()
            cuda_ops._mark("opt.end")
        finally:
            cuda_ops.PHASE_EVENTS = None
        torch.cuda.synchronize()
        m = dict(events)
        rec = dict(step=i, phase=phase, mu=mu, loss=float(loss), grad_norm_layer=gn_layer, grad_norm_small=gn_small,
                   fwd_ms=m["step.begin"].elapsed_time(m["fwd.end"]), bwd_ms=m["fwd.end"].elapsed_time(m["bwd.end"]),
                   opt_ms=m["small.end"].elapsed_time(m["opt.end"]), step_ms=m["step.begin"].elapsed_time(m["opt.end"]))
        log["steps"].append(rec)
        return rec

    def evaluate(i, phase, mu):
        st = L1.weight_stats(lay.weight.detach(), valid)
        rec = dict(step=i, phase=phase, heldout=heldout(lay, proj, head, teacher, dev), l1_mean_abs=st["mean_abs"],
                   l1_term=mu * st["l1"], density_nonzero=st["density_nonzero"], exact_zeros=st["exact_zeros"],
                   near_zero_fraction=st["near_zero_fraction"], all_zero_neurons=st["all_zero_neurons"],
                   active_fan_in_mean=st["active_fan_in_mean"])
        log["evals"].append(rec)
        print(f"[{phase} {i:4d}] heldout {rec['heldout']:.6f} density {rec['density_nonzero']:.4f} "
              f"all-zero neurons {rec['all_zero_neurons']} train {log['steps'][-1]['loss'] if log['steps'] else 0:.6f}",
              flush=True)
        return st

    step = 0
    evaluate(0, "init", 0.0)
    for _ in range(WARMUP):
        timed_step(step, 0.0, "warmup")
        step += 1
    evaluate(step, "warmup", 0.0)
    # calibration (no parameter change): averaged task gradient of 4 fixed batches
    lay.zero_main_grad()
    for s in range(CAL_SEED, CAL_SEED + 4):
        x0 = batch(s, dev)
        forward_loss(lay, proj, head, x0, teacher(x0)).backward()
    for p in small:
        p.grad = None
    g = lay.weight.main_grad / 4
    pos = L1.sample_positions(valid, N_SAMPLE, SAMPLE_SEED)
    cal = L1.calibrate_mu(g, pos)
    lay.zero_main_grad()
    mu = a.alpha * cal["g_ref"]
    log["calibration"] = dict(cal, alpha=a.alpha, mu=mu, lambda_mean=L1.lambda_mean(mu, W0),
                              batches=[CAL_SEED + i for i in range(4)], sample_seed=SAMPLE_SEED, sample=N_SAMPLE,
                              at_step=step)
    print(f"calibration: g_ref {cal['g_ref']:.4e} (median, fallback {cal['fallback']}) alpha {a.alpha} mu {mu:.4e} "
          f"lambda_mean {L1.lambda_mean(mu, W0):.4e}", flush=True)
    for r in range(a.reg_steps):
        m_r = mu * min(1.0, (r + 1) / RAMP)
        timed_step(step, m_r, f"{a.treatment}")
        step += 1
        if (r + 1) % EVAL_EVERY == 0:
            evaluate(step, a.treatment, m_r)
    st = L1.weight_stats(lay.weight.detach(), valid)
    log["checkpoint_stats"] = st
    mask = SP.learned_mask(lay.weight.detach(), valid)
    tag = f"C{a.C}_R{a.R}_{a.treatment}_a{a.alpha:g}" + ("" if a.reg_steps == REG else f"_reg{a.reg_steps}")
    ck = Path(a.ckpt_dir) / f"{tag}_step{step}.pt"
    log["checkpoint"] = SP.save_checkpoint(ck, lay, mask, dict(tag=tag, step=step, phase="after regularized updates",
                                                               config=log["config"], calibration=log["calibration"]))
    log["checkpoint"]["path"] = str(ck)
    torch.save(dict(proj=proj.state_dict(), head=head.state_dict()), Path(a.ckpt_dir) / f"{tag}_step{step}_small.pt")
    # fixed-mask fine-tuning (the nonzero support of the checkpoint; mu = 0)
    opt.param_groups[0]["l1"] = "none"
    L1.apply_mask_(opt, lay.weight, mask)
    for _ in range(FINETUNE):
        timed_step(step, 0.0, "finetune_fixed_mask")
        step += 1
        if (step - WARMUP - a.reg_steps) % EVAL_EVERY == 0:
            evaluate(step, "finetune_fixed_mask", 0.0)
    st2 = L1.weight_stats(lay.weight.detach(), valid)
    log["finetune_stats"] = st2
    log["finetune_mask_preserved"] = bool(((lay.weight.detach() != 0) <= mask.bool()).all())
    log["wall_seconds"] = time.perf_counter() - t_start
    log["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(log, indent=1, default=str))
    print(f"done {tag}: density {st['density_nonzero']:.4f} -> fine-tuned held-out {log['evals'][-1]['heldout']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
