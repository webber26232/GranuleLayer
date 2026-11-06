"""Quality runs of the learned-reduction study (one job = one (H, reducer, task, seed, mode, LR); fresh process, atomic
JSON, resumable sessions).  Synthetic diagnostics, not downstream-accuracy claims.

Modes: "e2e": W, b, P, c, V, d, proj, head train, input support frozen.  "frozen": W, b, support and proj fixed; only the
readout (P, c, V, d, head) trains: recoverable information and readout capacity; A0 then trains only the common head.

Test losses are evaluated only by jobs with --test 1, after the validation-based choices are frozen.  Per-step time is the
GPU-synchronized time of forward + backward + optimizers (data generation and evaluation excluded).

    python learned_reduction_train.py job --root R --H 2621440 --model A4 --task compose --seed 0 --steps 2048 --out f.json
    python learned_reduction_train.py session --manifest m.json --root R --out DIR
    python learned_reduction_train.py replicate --checkpoint g0.pt --out f.json      # historical Cover-TopK/16 protocol
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import learned_reduction_common as C
import learned_reduction_tasks as T
import torch


def gnorm(t):
    return float(t.norm()) if t is not None else 0.0


class Run:
    def __init__(self, a):
        from sparse_pooling.experimental import learned_reduction as LR
        from sparse_pooling.optim import FusedAdamW

        self.a, self.LR = a, LR
        dev = self.dev = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False  # fixed across every comparison
        self.task = T.TASKS[a.task](dev)
        t0 = time.perf_counter()
        base = C.get_base(a.root, a.H, a.seed, init_seed=C.INIT_SEED + a.seed)
        cfgs = LR.core_configs(a.H, L=a.L, s=a.s)
        self.cfg = cfgs[a.model]
        frozen = a.mode == "frozen"
        self.model = LR.ReductionModel(base, self.cfg, output_dim=C.G, seed=a.seed, train_sparse=not frozen)
        self.frozen = frozen
        torch.manual_seed(29 + a.seed)
        self.proj = torch.nn.Linear(self.task.D0, C.D, device=dev)
        self.head = torch.nn.Linear(C.G, T.OUT, device=dev)
        if self.task.identity_proj:
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(C.D, device=dev))
                self.proj.bias.zero_()
        if frozen:
            self.proj.requires_grad_(False)
        cal = self.batch(0)[0]
        if self.cfg.dense and self.cfg.dense_init != "identity":
            with torch.no_grad():
                self.model.gamma.fill_(LR.calibrate_gamma(self.model, self.proj(cal).to(torch.bfloat16)))
        self.setup_s = time.perf_counter() - t0
        fused_groups = []
        if self.model.sparse_parameters():
            fused_groups.append(dict(params=self.model.sparse_parameters(), lr=a.lr_w))
        if self.model.reducer_parameters():
            fused_groups.append(dict(params=self.model.reducer_parameters(), lr=a.lr_w * a.p_mult))
        self.fused = FusedAdamW(fused_groups, lr=a.lr_w, moments="bf16", stochastic_rounding=True) if fused_groups else None
        if self.fused is not None:
            self.fused._seed = a.seed * 1_000_003
        small = self.model.dense_parameters() + list(self.head.parameters()) + (
            [] if frozen else list(self.proj.parameters()))
        self.small = torch.optim.AdamW(small, lr=a.lr_small, weight_decay=0.0, fused=True)

    def batch(self, i):
        a = self.a
        s = T.TRAIN_BASE + (i % T.POOL) + 1_000_000 * a.seed
        x0 = T.gaussian(s, T.M_TRAIN, self.task.D0, self.dev)
        return x0, self.task(x0)

    def loss_of(self, x0, tgt):
        y = self.model(self.proj(x0).to(torch.bfloat16))
        return (self.head(y.float()) - tgt).square().mean()

    @torch.no_grad()
    def heldout(self, seeds, shift=None):
        tot = 0.0
        for s in seeds:
            x0 = T.gaussian(s, T.M_TRAIN, self.task.D0, self.dev)
            if shift:
                x0 = T.shifted(x0, shift)
            tot += float(self.loss_of(x0, self.task(x0)))
        return tot / len(seeds)

    @torch.no_grad()
    def train_eval(self):
        tot = 0.0
        for i in range(T.TRAIN_EVAL_BATCHES):
            x0, tgt = self.batch(i * (T.POOL // T.TRAIN_EVAL_BATCHES))
            tot += float(self.loss_of(x0, tgt))
        return tot / T.TRAIN_EVAL_BATCHES

    @torch.no_grad()
    def hidden_sample(self):
        """Pre / post statistics of a fixed sample of 2,048 C16 records (M 512 x 32,768 neurons, not the layer)."""
        sp = self.model.sparse
        idx = sp.indices.reshape(-1, 16)
        n = idx.shape[0]
        g = torch.Generator(device="cpu").manual_seed(4242)
        pick = torch.randperm(n, generator=g)[:min(2048, n)].to(self.dev)
        x = self.proj(T.gaussian(9_000, T.M_TRAIN, self.task.D0, self.dev)).to(torch.bfloat16).float()
        W = sp.weight.detach().reshape(-1, 16, 16)[pick].float()
        b = sp.bias.detach().reshape(-1, 16)[pick].float()
        z = torch.einsum("mrk,rnk->mrn", x[:, idx[pick].long()], W) + b[None]
        U = self.model.features(self.proj(T.gaussian(9_000, T.M_TRAIN, self.task.D0, self.dev)).to(torch.bfloat16))
        r = dict(z_mean=float(z.mean()), z_rms=float(z.square().mean().sqrt()), relu_zero_fraction=float((z <= 0).float().mean()),
                 h_mean=float(z.clamp_min(0).mean()), h_rms=float(z.clamp_min(0).square().mean().sqrt()),
                 U=self.LR.feature_stats(U), U_token_std_mean=float(U.std(0).mean()), gamma=float(self.model.gamma))
        if self.model.reducer is not None:
            P = self.model.reducer.P.detach().float()
            r.update(P_rms=float(P.square().mean().sqrt()), P_max=float(P.abs().max()))
        return r

    def grad_stats(self):
        m = self.model
        sp = m.sparse
        s = dict(W=gnorm(getattr(sp.weight, "main_grad", None)), b=gnorm(getattr(sp.bias, "main_grad", None)))
        s["W_ratio"] = s["W"] / max(float(sp.weight.detach().float().norm()), 1e-30)
        if m.reducer is not None:
            s["P"] = gnorm(getattr(m.reducer.P, "main_grad", None))
            s["P_ratio"] = s["P"] / max(float(m.reducer.P.detach().float().norm()), 1e-30)
        if m.V is not None:
            s["V"] = gnorm(m.V.grad)
            s["V_ratio"] = s["V"] / max(float(m.V.detach().norm()), 1e-30)
        s["head"] = gnorm(self.head.weight.grad)
        return s

    def train(self):
        a = self.a
        rec = dict(schema="granule.lsw_train/1", H=a.H, L=a.L, s=a.s, model=a.model, task=a.task, seed=a.seed, mode=a.mode,
                   steps=a.steps, lr_w=a.lr_w, lr_small=a.lr_small, p_mult=a.p_mult, batch=T.M_TRAIN, pool=T.POOL,
                   config=self.cfg.describe(), weights=self.model.weight_counts(), head_weights=T.OUT * C.G + T.OUT,
                   proj_weights=self.task.D0 * C.D + C.D, fingerprints=self.model.fingerprints(),
                   setup_seconds=self.setup_s, protocol=dict(
                       val=T.VAL_SEEDS, test=T.LSW_TEST if a.test else None, pool=T.POOL, tf32=False,
                       gelu="exact" if self.cfg.psi == "gelu" else None, moments="bf16", stochastic_rounding=True,
                       optimizers="FusedAdamW W / b / P; torch AdamW fused proj / head / c / V / d", weight_decay=0.0),
                   hidden_start=self.hidden_sample())
        curve = [dict(step=0, gpu_s=0.0, val=self.heldout(T.VAL_SEEDS), train=self.train_eval())]
        running, n, gpu_s = 0.0, 0, 0.0
        t_wall = time.perf_counter()
        finite = True
        gs = []
        for i in range(a.steps):
            x0, tgt = self.batch(i)
            torch.cuda.synchronize()
            t = time.perf_counter()
            loss = self.loss_of(x0, tgt)
            loss.backward()
            if (i + 1) % a.eval_every == 0 and (i + 1) in a.grad_steps:
                gs.append(dict(step=i + 1, **self.grad_stats()))
            if self.fused is not None:
                self.fused.step()
            self.small.step()
            self.small.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            gpu_s += time.perf_counter() - t
            lv = float(loss)
            finite &= math.isfinite(lv)
            running += lv
            n += 1
            if not finite:
                break
            if (i + 1) % a.eval_every == 0 or i + 1 == a.steps:
                curve.append(dict(step=i + 1, gpu_s=gpu_s, val=self.heldout(T.VAL_SEEDS), train=self.train_eval(),
                                  running=running / n))
                running, n = 0.0, 0
                if a.verbose:
                    print(curve[-1], flush=True)
        rec.update(curve=curve, finite=finite, gpu_seconds=gpu_s, wall_seconds=time.perf_counter() - t_wall,
                   ms_per_step=1e3 * gpu_s / max(a.steps, 1), grad_stats=gs, hidden_end=self.hidden_sample() if finite else None,
                   final_val=curve[-1]["val"], final_train=curve[-1]["train"], gap_val_train=curve[-1]["val"] - curve[-1]["train"],
                   best_val=min(c["val"] for c in curve), peak_alloc_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   support_after=self.model.fingerprints(), env=C.env_record())
        if a.test and finite:
            rec["test"] = self.heldout(T.LSW_TEST)
            rec["shifts"] = {k: self.heldout(T.LSW_TEST, k) for k in T.SHIFTS}
            rec["test_target_var"] = float(torch.cat([self.task(T.gaussian(s, T.M_TRAIN, self.task.D0, self.dev))
                                                      for s in T.LSW_TEST]).var())
        if self.model.reducer is not None:
            rec["kernel"] = self.model.reducer.execution_record()
        rec["dx_variant_record"] = {k: v for k, v in self.model.sparse.last_info.items() if k in ("kgen_dx", "reduction_backward")}
        return rec


def job(a):
    a.grad_steps = {a.eval_every, a.steps - a.steps % a.eval_every} if a.eval_every else set()
    r = Run(a).train()
    C.atomic_json(a.out, r)
    print(f"H{a.H} {a.model} {a.task} s{a.seed} {a.mode} lr {a.lr_w}/{a.lr_small}/m{a.p_mult} steps {a.steps}: val {r['final_val']:.5f} "
          f"train {r['final_train']:.5f} test {r.get('test')} ms/step {r['ms_per_step']:.1f} finite {r['finite']}", flush=True)


def session(a):
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    defaults = man.get("defaults", {})
    fails = out / "failures.jsonl"
    for k, j in enumerate(man["jobs"]):
        j = {**defaults, **j}
        name = "_".join(f"{x}{j[x]}" for x in sorted(j))
        f = out / f"{name}.json"
        if f.exists():
            continue
        cmd = [sys.executable, __file__, "job", "--root", a.root, "--out", str(f)]
        for x, v in j.items():
            cmd += [f"--{x.replace('_', '-')}", str(v)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + r.stderr)
        print(f"{k + 1}/{len(man['jobs'])} rc {r.returncode} {time.perf_counter() - t0:.0f}s {r.stdout.strip()[-200:]}"
              + (f" ERR {r.stderr.strip()[-300:]}" if r.returncode else ""), flush=True)
        if r.returncode:
            with open(fails, "a") as h:
                h.write(json.dumps(dict(job=j, rc=r.returncode, err=r.stderr[-2000:])) + "\n")


def replicate(a):
    """Stage-0 provenance check: Cover-TopK/16 from the historical alpha-0 K64 checkpoint with its proj / head, the
    k64_learn teacher, fresh optimizers, 256 updates (rsn_learn protocol); compare with the historical records."""
    from sparse_pooling import CoverTopK16Layer
    from sparse_pooling.optim import FusedAdamW

    from sparse_pooling.supergroups import ConnectivityPlan

    dev = torch.device("cuda")
    # from_envelope, not from_checkpoint: the latter's mask check (cover_topk16.py, boolean index over 805 M elements)
    # materialises about 25.6 GiB of host indices and systemd-oomd kills the job on this 31 GiB machine
    blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    rec0 = blob["rec"]
    sp = rec0["spec"]
    conn = ConnectivityPlan(sp["S"], sp["R"], sp.get("generation", "balanced_permutation"), sp["seed"])
    lay = CoverTopK16Layer.from_envelope(blob["weight"].to(dev), blob["bias"].to(dev), input_dim=1024, hidden_dim=int(rec0["H"]),
                                         output_dim=1024, connectivity=conn, envelope_C=int(rec0["C"]), normalization="energy",
                                         device=dev, dx_variant="auto")
    del blob
    lay.enable_main_grad()
    small = torch.load(a.checkpoint.replace(".pt", "_small.pt"), map_location=dev)
    proj = torch.nn.Linear(256, 1024, device=dev)
    head = torch.nn.Linear(1024, 64, device=dev)
    proj.load_state_dict(small["proj"])
    head.load_state_dict(small["head"])
    teacher = T.Compose(dev)
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4, moments="bf16", stochastic_rounding=True)
    opt._seed = a.seed * 1_000_003
    opt2 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.0)

    def loss_of(x0):
        return (head(lay(proj(x0).to(torch.bfloat16)).float()) - teacher(x0)).square().mean()

    @torch.no_grad()
    def held(seeds):
        return sum(float(loss_of(T.gaussian(s, 512, 256, dev))) for s in seeds) / len(seeds)

    val0 = held(T.VAL_SEEDS)
    for i in range(a.steps):
        loss = loss_of(T.gaussian(10_000 + 128 + i + 1_000_000 * a.seed, 512, 256, dev))
        loss.backward()
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()
    rec = dict(schema="granule.lsw_replicate/1", checkpoint=Path(a.checkpoint).name, seed=a.seed, steps=a.steps, val_start=val0,
               val_final=held(T.VAL_SEEDS), test_icdx39000=held(tuple(range(39_000, 39_008))),
               support=lay.metadata["support_fingerprint"],
               support_expected="e4465520b9cca51b91cf19cbfc6ba9771f7ec988149b548425574d5eeb643a81 (graph 0 validation record)",
               executed_dx=lay.last_info["kgen_dx"]["variant"], env=C.env_record(),
               historical="Cover-TopK/16 development 0.1713 +- 0.0042 (validation), research confirmation about 0.1697 (test, "
                          "icdx39000), dense-masked execution of the same support on another code head")
    C.atomic_json(a.out, rec)
    print({k: rec[k] for k in ("val_start", "val_final", "test_icdx39000")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--root", required=True)
    j.add_argument("--H", type=int, required=True)
    j.add_argument("--L", type=int, default=10_240)
    j.add_argument("--s", type=int, default=2_048)
    j.add_argument("--model", required=True)
    j.add_argument("--task", required=True, choices=sorted(T.TASKS))
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--mode", default="e2e", choices=["e2e", "frozen"])
    j.add_argument("--steps", type=int, default=2048)
    j.add_argument("--eval-every", type=int, default=64)
    j.add_argument("--lr-w", type=float, default=1e-4)
    j.add_argument("--lr-small", type=float, default=1e-3)
    j.add_argument("--p-mult", type=float, default=1.0)
    j.add_argument("--test", type=int, default=0)
    j.add_argument("--verbose", type=int, default=0)
    j.add_argument("--out", required=True)
    s = sub.add_parser("session")
    s.add_argument("--manifest", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("replicate")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--seed", type=int, default=12)
    r.add_argument("--steps", type=int, default=256)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(job=job, session=session, replicate=replicate)[a.cmd](a)


if __name__ == "__main__":
    main()
