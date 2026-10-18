"""Nonlinear-loss and multi-step training comparisons of grouped dX scatter variants.

Three questions are kept separate (results/C8_DX_SCATTER_OPTIMIZATION.md, section 4):
  1. fixed-dY kernel comparison (tests/test_dx_scatter.py),
  2. nonlinear-loss gradients: dY is computed from the actual (atomic, nondeterministic)
     grouped forward output, so forward summation-order differences reach dW / db / dX,
  3. multi-step training: updates, BF16 moment / parameter rounding (deterministic or
     stochastic) can amplify small differences.

Models (production dtypes: BF16 GranuleLayer parameters / input, FP32 main_grad, FusedAdamW):
  "mse"     y = layer(x); loss = mean((y - target)^2) with a nonzero target
  "square"  loss = mean(y^2) (the timing-harness loss)
  "compose" x0 -> Linear(D0, D) (trainable, FP32) -> BF16 -> layer -> Linear(G, G2) -> tanh
            -> mean((h - target)^2): dX of the layer drives the projection's update
Every trajectory starts from the same seeds (identical initial state) and uses the same
pre-generated, changing batches.

PREDECLARED CHECKS (fixed before any candidate was run; numerics_training.py applies them):
  C1  every trajectory stays finite.
  C2  "same"-order variants (identical per-warp arithmetic; only global FP32 atomic order can
      differ, as between two baseline runs): at every step, the candidate-vs-baseline
      divergence of each tracked quantity <= 2 x the largest baseline-vs-baseline divergence
      among REPEATS baseline runs (+ 1e-12 absolute).
  C3  "mma_accumulate" variants (feature_owner: different FP32 summation order): the same
      comparison with factor 10; a variant needing more stays experimental (it is opt-in in
      any case).
  Amendments after the first smoke run (before any recorded result; they apply to every
  variant alike): (a) equal numbers of pairs: REPEATS runs of every variant, candidate run i
  paired with baseline run j for i < j (as many pairs as baseline-vs-baseline); (b) the
  scalar FP32 loss gets a floor of one FP32 ulp (2^-23 relative): with 3 runs its envelope
  can be exactly 0 while another run differs by one ulp.  (c) a null control: "baseline_null" (an independent set of
  REPEATS baseline runs) is checked exactly like a candidate, so the false-positive rate of C2
  itself is measured in every configuration (with 3 runs the smoke test failed C2 also where
  dX cannot influence training at all).
Tracked per step: loss, layer output y, layer dW / db norms, dX norm (compose), parameter
and moment state (relative L2 differences between runs).
"""
from __future__ import annotations

import torch

from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.supergroups import ConnectivityPlan

CHECK_FACTOR = {"same": 2.0, "mma_accumulate": 10.0}
ABS_FLOOR = 1e-12
LOSS_FLOOR = 2.0 ** -23  # one FP32 ulp, relative (amendment (b))


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def batches(cfg, steps, seed=1234):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    D0 = cfg.get("D0", cfg["D"])
    out = []
    for _ in range(steps):
        x = torch.randn(cfg["B"], D0 if cfg["model"] == "compose" else cfg["D"], device="cuda", generator=g)
        nt = cfg.get("G2", cfg["G"]) if cfg["model"] == "compose" else cfg["G"]
        tgt = 0.5 * torch.randn(cfg["B"], nt, device="cuda", generator=g)
        out.append((x, tgt))
    return out


def build(cfg, plan, opt_mode):
    lay = SparsePoolingLayer(cfg["D"], cfg["H"], cfg["G"], shared_neurons=cfg["C"], pooling=cfg.get("pooling", "mean"),
                             device="cuda", seed=cfg.get("init_seed", 0), bias_std=0.2, plan=plan,
                             connectivity=ConnectivityPlan(cfg["S"], cfg["R"], seed=cfg.get("graph_seed", 0)))
    lay.enable_main_grad()
    if opt_mode == "fp32":
        opt = FusedAdamW([lay.weight, lay.bias], lr=cfg.get("lr", 1e-3), moments="fp32", master_weights=True)
    else:
        opt = FusedAdamW([lay.weight, lay.bias], lr=cfg.get("lr", 1e-3), moments="bf16",
                         stochastic_rounding=(opt_mode == "sr"))
    extra = None
    if cfg["model"] == "compose":
        torch.manual_seed(cfg.get("init_seed", 0) + 17)
        proj = torch.nn.Linear(cfg.get("D0", cfg["D"]), cfg["D"], device="cuda")
        head = torch.nn.Linear(cfg["G"], cfg.get("G2", cfg["G"]), device="cuda")
        extra = (proj, head, torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=1e-3))
    return lay, opt, extra


class Run:
    """One training trajectory, advanced one step at a time (runs are kept in lockstep so
    that only the current step's tensors exist)."""

    def __init__(self, cfg, plan_kw, opt_mode):
        self.cfg = cfg
        plan = PhysicalPlan(grad_precision=cfg.get("precision", "split"), **plan_kw)
        self.lay, self.opt, self.extra = build(cfg, plan, opt_mode)
        self.cur = None
        self.info = None

    def step(self, x0, tgt):
        lay, extra = self.lay, self.extra
        if extra is not None:
            proj, head, opt2 = extra
            xin = proj(x0).bfloat16()
            xin.retain_grad()
            y = lay(xin)
            h = torch.tanh(head(y.float()))
            loss = ((h - tgt) ** 2).mean()
        else:  # dX is computed (as in the timing harness) but does not influence training
            xin = x0.bfloat16().requires_grad_(True)
            y = lay(xin)
            loss = ((y.float() - tgt) ** 2).mean() if self.cfg["model"] == "mse" else y.float().square().mean()
        loss.backward()
        rec = {"loss": float(loss), "y": y.detach().float(), "dW": lay.weight.main_grad.clone(),
               "db": lay.bias.main_grad.clone(), "dX": xin.grad.float()}
        if extra is not None:
            extra[2].step()
            extra[2].zero_grad(set_to_none=True)
        self.opt.step()
        st = self.opt.state[lay.weight]
        rec.update(W=lay.weight.detach(), b=lay.bias.detach(), m=st["m"], v=st["v"])
        if extra is not None:
            rec["P"] = extra[0].weight.detach()
        self.info = {k: lay.last_info.get(k) for k in ("forward_kernel", "dx_kernel", "dw_kernel")}
        self.info["scatter"] = (lay.last_info.get("grouped_dx") or {}).get("scatter")
        self.cur = rec
        return rec


TRACKED = ("loss", "y", "dW", "db", "dX", "W", "b", "m", "v", "P")


def diff(ra, rb):
    d = {}
    for k in TRACKED:
        if k not in ra:
            continue
        d[k] = abs(ra[k] - rb[k]) / (abs(rb[k]) + 1e-30) if k == "loss" else rel(ra[k], rb[k])
    return d


def finite(rec):
    return rec["loss"] == rec["loss"] and all(torch.isfinite(rec[k]).all().item()
                                              for k in TRACKED if k in rec and k != "loss")


def lockstep(cfg, variants, opt_mode, data, repeats):
    """variants: {name: (plan kwargs, order category | None)}; "baseline" must be present.
    Returns per-step divergence series: baseline-vs-baseline pairs (envelope) and every
    other variant's runs vs every baseline run, plus losses and finiteness."""
    runs = {name: [Run(cfg, kw, opt_mode) for _ in range(repeats if order is not None or name == "baseline" else 1)]
            for name, (kw, order) in variants.items()}
    series = {name: [] for name in variants}
    losses = {name: [] for name in variants}
    ok = {name: True for name in variants}
    kernels = {}
    for x0, tgt in data:
        for name, rs in runs.items():
            for r in rs:
                r.step(x0, tgt)
                ok[name] &= finite(r.cur)
            kernels[name] = rs[0].info
            losses[name].append(rs[0].cur["loss"])
        base = runs["baseline"]
        for name, rs in runs.items():
            pairs = ([(base[i], base[j]) for i in range(len(base)) for j in range(i + 1, len(base))]
                     if name == "baseline" else
                     [(rs[i], base[j]) for i in range(len(rs)) for j in range(i + 1, len(base))] or
                     [(rs[0], b) for b in base])  # single-run context variants: vs every baseline run
            step = {}
            for a, b in pairs:
                for k, v in diff(a.cur, b.cur).items():
                    step[k] = max(step.get(k, 0.0), v)
            series[name].append(step)
    return {"series": series, "losses": losses, "finite": ok, "kernels": kernels}


def check(env, cand, order):
    """Predeclared check C2 / C3 on per-step max divergences."""
    f = CHECK_FACTOR[order]
    worst, fails = 0.0, []
    for s, (e, c) in enumerate(zip(env, cand)):
        for k in e:
            lim = f * e[k] + (LOSS_FLOOR if k == "loss" else ABS_FLOOR)
            worst = max(worst, c[k] / lim)
            if c[k] > lim:
                fails.append((s, k, c[k], e[k]))
    return {"factor": f, "pass": not fails, "worst_ratio_to_limit": worst, "failures": fails[:20]}
