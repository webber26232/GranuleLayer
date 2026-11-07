"""Unified training loop of the A3mean follow-up (one job = one (H, model, task, seed, M_phys, accum_steps, LR); fresh process,
atomic JSON, resumable).  One code path runs all execution modes; synthetic diagnostics, not downstream-accuracy claims.

    M_phys          input rows of one forward / backward call
    accum_steps     forward / backward calls per optimizer update
    M_eff           M_phys * accum_steps: rows contributing to one optimizer update
    samples_seen    M_eff * optimizer_updates: training input vectors consumed (no calibration, no evaluation)

An effective update: main_grad and ordinary gradients are zero on entry; each call backpropagates (mean loss of its rows) /
accum_steps; each optimizer steps exactly once at the end, which is where FusedAdamW consumes and zeroes main_grad.

train_s: GPU-synchronized forward + backward + optimizers.  wall_s: everything since the job started (setup excluded), with
data, evaluation and checkpoint time also listed separately.  Test blocks are never touched here: `evaluate` reads milestone
checkpoints and refuses to run without the unlock file written after all choices are frozen.

    python lrf_train.py job --root R --ckpt-root C --H 10485760 --model A3mean --task compose --seed 0 --m-phys 8192 --accum 1 \\
        --updates 1024 --eval-every 64 --lr-w 3e-4 --lr-small 3e-4 --p-mult 0.3 --milestones 256,512,1024 --out f.json
    python lrf_train.py session --manifest m.json --root R --ckpt-root C --out DIR
    python lrf_train.py evaluate --record f.json --root R --unlock U --out g.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import learned_reduction_common as C
import learned_reduction_tasks as T
import lrf_stream as S
import torch

SCHEMA = "granule.lrf_train/1"
EVAL_SCHEMA = "granule.lrf_test/1"
NEAR_ZERO_FRACTION = 0.1  # |P| below this fraction of the mean-initial value 1 / L counts as near zero (declared)
CHECK_BLOCKS = 8  # already consumed blocks in each of the two training-loss diagnostics


def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def norm(t) -> float:
    return float(t.float().norm()) if t is not None else 0.0


class Trainer:
    """``base`` / ``task`` / ``output_dim`` may be injected (tests use small shapes); production jobs load the packed base."""

    def __init__(self, a, base=None, task=None, output_dim=C.G, on_block=None):
        from sparse_pooling.experimental import learned_reduction as LR
        from sparse_pooling.optim import FusedAdamW

        self.a, self.LR = a, LR
        dev = self.dev = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False  # fixed across every comparison
        self.task = task if task is not None else T.TASKS[a.task](dev)
        t0 = time.perf_counter()
        if base is None:
            base = C.get_base(a.root, a.H, a.seed, init_seed=C.INIT_SEED + a.seed)
        self.cfg = LR.core_configs(a.H, s=a.s)[a.model]
        self.model = LR.ReductionModel(base, self.cfg, output_dim=output_dim, seed=a.seed)
        del base
        D = self.model.input_dim
        torch.manual_seed(29 + a.seed)
        self.proj = torch.nn.Linear(self.task.D0, D, device=dev)
        self.head = torch.nn.Linear(output_dim if self.model.V is not None else self.model.J, T.OUT, device=dev)  # A0: y is the group mean
        if self.task.identity_proj:
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(D, device=dev))
                self.proj.bias.zero_()
        self.on_block = on_block  # test hook: observes every (split, block id) read from the stream
        cal = self._read("calibration", a.seed, 0).to(dev)
        if self.cfg.dense and self.cfg.dense_init != "identity":
            with torch.no_grad():  # frozen from here on, for every M_phys, and never recalibrated after a resume
                self.model.gamma.fill_(LR.calibrate_gamma(self.model, self.proj(cal).to(torch.bfloat16)))
        self.setup_s = time.perf_counter() - t0
        groups = [dict(params=self.model.sparse_parameters(), lr=a.lr_w)]
        if self.model.reducer_parameters():
            groups.append(dict(params=self.model.reducer_parameters(), lr=a.lr_w * a.p_mult))
        self.fused = FusedAdamW(groups, lr=a.lr_w, moments="bf16", stochastic_rounding=True)
        self.fused._seed = a.seed * 1_000_003
        small = self.model.dense_parameters() + list(self.head.parameters()) + list(self.proj.parameters())
        self.small = torch.optim.AdamW(small, lr=a.lr_small, weight_decay=0.0, fused=True)
        self.update_count, self.next_block = 0, 0
        self.train_s = self.data_s = self.eval_s = self.ckpt_s = self.wall_prior_s = 0.0
        self.curve, self.gstats, self.pstats = [], [], []
        self.milestones = {}
        self.fused_calls = self.small_calls = 0
        self.fp = self.model.fingerprints()
        self.P0 = self.model.reducer.P.detach().cpu() if self.model.reducer is not None else None  # diagnostics: host memory
        self.val_cache = None
        self.peak_train = 0

    # ------------------------------------------------------------------ data
    def _read(self, split, data_seed, block_id):
        if self.on_block is not None:
            self.on_block(split, block_id)
        return S.block(split, data_seed, block_id, self.task.D0)

    def next_microbatch(self):
        """The next M_phys rows: consecutive canonical blocks from the sample position."""
        n = self.a.m_phys // S.BLOCK
        x0 = torch.cat([self._read("train", self.a.seed, self.next_block + i) for i in range(n)]).to(self.dev)
        self.next_block += n
        return x0, self.task(x0)

    @property
    def samples_seen(self):
        return self.update_count * self.a.m_phys * self.a.accum

    # ------------------------------------------------------------------ training
    def loss_of(self, x0, tgt):
        y = self.model(self.proj(x0).to(torch.bfloat16))
        return (self.head(y.float()) - tgt).square().mean()

    def backward_calls(self, batches=None):
        """All forward / backward calls of one effective update; returns its mean loss (no optimizer)."""
        a = self.a
        total = 0.0
        for k in range(a.accum):
            t = time.perf_counter()
            x0, tgt = batches[k] if batches is not None else self.next_microbatch()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.data_s += t1 - t
            loss = self.loss_of(x0, tgt) / a.accum
            loss.backward()
            torch.cuda.synchronize()
            self.train_s += time.perf_counter() - t1
            total += float(loss)
            del loss, x0, tgt
        return total

    def optimizer_step(self):
        t = time.perf_counter()
        self.fused.step()
        self.fused_calls += 1
        self.small.step()
        self.small_calls += 1
        self.small.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        self.train_s += time.perf_counter() - t

    def effective_update(self, capture_grads=False):
        lv = self.backward_calls()
        if capture_grads:
            self.gstats.append(dict(update=self.update_count + 1, **self.grad_stats()))
        self.optimizer_step()
        self.update_count += 1
        return lv

    # ------------------------------------------------------------------ evaluation (M = 512 blocks, no parameter changes)
    @torch.no_grad()
    def _loss_on(self, x0):
        x0 = x0.to(self.dev)
        return float(self.loss_of(x0, self.task(x0)))

    def validation(self):
        if self.val_cache is None:
            self.val_cache = [self._read("validation", S.HELDOUT_DATA_SEED, i) for i in S.validation_ids()]
        return sum(self._loss_on(x) for x in self.val_cache) / len(self.val_cache)

    def seen_diagnostics(self):
        """recent_seen: the last CHECK_BLOCKS blocks already consumed.  fixed_seen: the first CHECK_BLOCKS blocks of the stream,
        once all of them have been consumed.  Neither ever reads a block beyond the current sample position."""
        n = min(CHECK_BLOCKS, self.next_block)
        if n == 0:
            return dict(recent_seen=None, fixed_seen=None)
        first = self.next_block - n
        recent = sum(self._loss_on(self._read("train", self.a.seed, first + i)) for i in range(n)) / n
        fixed = None
        if self.next_block >= CHECK_BLOCKS:
            fixed = sum(self._loss_on(self._read("train", self.a.seed, i)) for i in range(CHECK_BLOCKS)) / CHECK_BLOCKS
        return dict(recent_seen=recent, fixed_seen=fixed)

    def grad_stats(self):
        m = self.model
        sp = m.sparse
        s = dict(W=norm(sp.weight.main_grad), b=norm(sp.bias.main_grad))
        s["W_ratio"] = s["W"] / max(norm(sp.weight.detach()), 1e-30)
        if m.reducer is not None:
            s["P"] = norm(m.reducer.P.main_grad)
            s["P_ratio"] = s["P"] / max(norm(m.reducer.P.detach()), 1e-30)
        if m.V is not None:
            s["V"] = norm(m.V.grad)
            s["V_ratio"] = s["V"] / max(norm(m.V.detach()), 1e-30)
        s["head"] = norm(self.head.weight.grad)
        s["proj"] = norm(self.proj.weight.grad)
        return s

    @torch.no_grad()
    def p_stats(self):
        """Distribution of the learned first-stage weights and their movement from the mean initialization."""
        if self.model.reducer is None:
            return None
        red = self.model.reducer
        mask = red.routing.member_mask(self.dev)
        P = red.P.detach().float()[mask]
        P0 = self.P0.to(self.dev).float()[mask]
        ref = 1.0 / red.spec.window
        step = max(1, P.numel() // 2_000_000)
        q = torch.quantile(P[::step], torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=self.dev))
        return dict(update=self.update_count, valid=P.numel(), mean=float(P.mean()), std=float(P.std()), rms=float(P.square().mean().sqrt()),
                    max_abs=float(P.abs().max()), positive=float((P > 0).float().mean()), negative=float((P < 0).float().mean()),
                    near_zero=float((P.abs() < NEAR_ZERO_FRACTION * ref).float().mean()), near_zero_threshold=NEAR_ZERO_FRACTION * ref,
                    q01=float(q[0]), q10=float(q[1]), q50=float(q[2]), q90=float(q[3]), q99=float(q[4]),
                    rel_movement=norm(P - P0) / max(norm(P0), 1e-30), mean_abs_movement=float((P - P0).abs().mean()), init_value=ref)

    def eval_point(self, running=None, t_start=None, with_p=False):
        t = time.perf_counter()
        d = dict(update=self.update_count, samples=self.samples_seen, train_s=self.train_s, val=self.validation(), running=running,
                 **self.seen_diagnostics())
        ps = self.p_stats() if with_p else None
        if ps is not None:
            self.pstats.append(ps)
        self.eval_s += time.perf_counter() - t
        d["wall_s"] = self.wall_prior_s + (time.perf_counter() - t_start if t_start is not None else 0.0)
        self.curve.append(d)
        if self.a.verbose:
            print(d, flush=True)
        return d

    # ------------------------------------------------------------------ checkpoints
    def parameter_state(self):
        d = {f"model.{k}": v.detach().cpu() for k, v in self.model.named_parameters()}
        d.update({f"proj.{k}": v.detach().cpu() for k, v in self.proj.named_parameters()})
        d.update({f"head.{k}": v.detach().cpu() for k, v in self.head.named_parameters()})
        return d

    @torch.no_grad()
    def load_parameter_state(self, st):
        for name, mod in (("model", self.model), ("proj", self.proj), ("head", self.head)):
            for k, v in mod.named_parameters():
                v.copy_(st[f"{name}.{k}"].to(v.device, v.dtype))

    def identity(self):
        a = self.a
        return dict(H=a.H, s=a.s, model=a.model, task=a.task, seed=a.seed, m_phys=a.m_phys, accum=a.accum, lr_w=a.lr_w, lr_small=a.lr_small,
                    p_mult=a.p_mult, fingerprints=self.fp, study=S.STUDY)

    def save_milestone(self, directory: Path):
        t = time.perf_counter()
        p = directory / f"model_u{self.update_count}.pt"
        atomic_save(dict(kind="model", update=self.update_count, samples=self.samples_seen, identity=self.identity(),
                         gamma=float(self.model.gamma), params=self.parameter_state()), p)
        self.milestones[self.update_count] = dict(path=str(p), sha256=C.sha256_file(p), bytes=p.stat().st_size, samples=self.samples_seen,
                                                  val=self.curve[-1]["val"] if self.curve and self.curve[-1]["update"] == self.update_count else None,
                                                  train_s=self.train_s)
        self.ckpt_s += time.perf_counter() - t

    def full_state(self):
        return dict(kind="full", identity=self.identity(), update=self.update_count, next_block=self.next_block, gamma=float(self.model.gamma),
                    params=self.parameter_state(), fused=self.fused.state_dict(), fused_seed=self.fused._seed, small=self.small.state_dict(),
                    rng=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                    P0=self.P0,
                    log=dict(train_s=self.train_s, data_s=self.data_s, eval_s=self.eval_s, ckpt_s=self.ckpt_s, curve=self.curve, gstats=self.gstats,
                             pstats=self.pstats, milestones=self.milestones, fused_calls=self.fused_calls, small_calls=self.small_calls))

    def save_full(self, directory: Path, keep=2):
        t = time.perf_counter()
        atomic_save(self.full_state(), directory / f"full_u{self.update_count:07d}.pt")
        for old in sorted(directory.glob("full_u*.pt"))[:-keep]:
            old.unlink()
        self.ckpt_s += time.perf_counter() - t

    def load_full(self, path, strict_lr=True):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        want, got = self.identity(), blob["identity"]
        if not strict_lr:
            for k in ("lr_w", "lr_small", "p_mult"):
                want.pop(k), got.pop(k)
        if want != got:
            raise ValueError(f"checkpoint identity {got} != job {want}")
        self.load_parameter_state(blob["params"])
        self.model.gamma.fill_(blob["gamma"])  # the training value, never recalibrated
        self.fused.load_state_dict(blob["fused"])
        self.fused._seed = blob["fused_seed"]
        self.small.load_state_dict(blob["small"])
        torch.set_rng_state(blob["rng"])
        torch.cuda.set_rng_state_all(blob["rng_cuda"])
        self.update_count, self.next_block = blob["update"], blob["next_block"]
        self.P0 = blob["P0"]
        log = blob["log"]
        self.train_s, self.data_s, self.eval_s, self.ckpt_s = log["train_s"], log["data_s"], log["eval_s"], log["ckpt_s"]
        self.curve, self.gstats, self.pstats, self.milestones = log["curve"], log["gstats"], log["pstats"], log["milestones"]
        self.fused_calls, self.small_calls = log["fused_calls"], log["small_calls"]
        self.wall_prior_s = self.curve[-1]["wall_s"] if self.curve else 0.0
        self.check_position()

    def check_position(self):
        n = self.a.m_phys // S.BLOCK * self.a.accum
        if self.next_block != self.update_count * n:
            raise RuntimeError(f"stream position {self.next_block} != updates {self.update_count} x {n} blocks")
        for g in self.fused.param_groups:
            for p in g["params"]:
                st = self.fused.state.get(p)
                if st and st["step"] != self.update_count:
                    raise RuntimeError(f"FusedAdamW step {st['step']} != updates {self.update_count}")

    # ------------------------------------------------------------------ run
    def train(self, ckpt_dir=None, stop_after=None):
        a = self.a
        d = Path(ckpt_dir) if ckpt_dir else None
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
        milestones = set(a.milestones)
        t_start = time.perf_counter()
        if not self.curve:
            self.eval_point(t_start=t_start, with_p=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        running, n, finite = 0.0, 0, True
        while self.update_count < a.updates:
            capture = a.grad_every and (self.update_count + 1) % a.grad_every == 0
            lv = self.effective_update(capture_grads=bool(capture))
            finite &= math.isfinite(lv)
            running += lv
            n += 1
            if not finite:
                break
            u = self.update_count
            do_eval = u % a.eval_every == 0 or u == a.updates
            do_ms = d is not None and u in milestones
            do_full = d is not None and a.full_every and u % a.full_every == 0
            if do_eval or do_ms or do_full:
                self.peak_train = max(self.peak_train, torch.cuda.max_memory_allocated())
            if do_eval or do_ms:
                self.eval_point(running=running / n, t_start=t_start, with_p=do_ms or u == a.updates)
                running, n = 0.0, 0
            if do_ms:
                self.save_milestone(d)
            if do_full:
                self.save_full(d)
            if do_eval or do_ms or do_full:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            if stop_after is not None and u >= stop_after:
                break
        self.peak_train = max(self.peak_train, torch.cuda.max_memory_allocated())
        return finite, time.perf_counter() - t_start + self.wall_prior_s

    def record(self, finite, wall):
        a = self.a
        rec = dict(schema=SCHEMA, study=S.STUDY, stage=a.stage, H=a.H, L=self.cfg.L, s=a.s, model=a.model, task=a.task, seed=a.seed, regime="fresh",
                   M_phys=a.m_phys, accum_steps=a.accum, M_eff=a.m_phys * a.accum, optimizer_updates=self.update_count,
                   updates_planned=a.updates, samples_seen=self.samples_seen, unique_samples_seen=self.samples_seen,
                   blocks_consumed=self.next_block, lr_w=a.lr_w, lr_small=a.lr_small, p_mult=a.p_mult, lr_policy="constant",
                   lr_multiplier=a.lr_multiplier, config=self.cfg.describe(), weights=self.model.weight_counts(),
                   head_weights=T.OUT * self.head.in_features + T.OUT, proj_weights=self.proj.weight.numel() + self.proj.bias.numel(),
                   gamma=float(self.model.gamma), fingerprints=self.fp, support_after=self.model.fingerprints(), setup_seconds=self.setup_s,
                   finite=finite, train_s=self.train_s, data_s=self.data_s, eval_s=self.eval_s, ckpt_s=self.ckpt_s, wall_s=wall,
                   ms_per_update=1e3 * self.train_s / max(self.update_count, 1), curve=self.curve, grad_stats=self.gstats, p_stats=self.pstats,
                   milestones=self.milestones, score_val=score_of(self.curve, finite), final_val=self.curve[-1]["val"],
                   peak_train_window_bytes=self.peak_train, stream=dict(first_block=S.checksum(S.block("train", a.seed, 0, self.task.D0)),
                                                                        last_block=S.checksum(S.block("train", a.seed, max(self.next_block - 1, 0), self.task.D0))),
                   fused_calls=self.fused_calls, small_calls=self.small_calls, fused_final_seed=self.fused._seed,
                   numerics=dict(tf32=False, moments="bf16", stochastic_rounding=True, weight_decay=0.0,
                                 optimizers="FusedAdamW W / b / P; torch AdamW fused proj / head / c / V / d"),
                   task_note="proj trainable, initialized to the identity" if self.task.identity_proj else "proj trainable", env=C.env_record(),
                   init_from=getattr(a, "init_from", None))
        if self.model.reducer is not None:
            rec["kernel"] = {k: v for k, v in self.model.reducer.execution_record().items() if k != "work_lists"}
        rec["dx_variant_record"] = {k: v for k, v in self.model.sparse.last_info.items() if k in ("kgen_dx", "reduction_backward")}
        rec["dx_request"] = self.model.sparse.dx_variant
        return rec


def score_of(curve, finite, last=4):
    """Selection score: the mean validation loss over the last four scheduled evaluations (declared), infinite if not finite."""
    vals = [c["val"] for c in curve[-last:]]
    if not finite or len(vals) < last or not all(math.isfinite(v) for v in vals):
        return float("inf")
    return sum(vals) / len(vals)


def job_args(ap):
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt-root", default="")
    ap.add_argument("--stage", default="adhoc")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--s", type=int, default=2_048)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=sorted(T.TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m-phys", type=int, default=512)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--updates", type=int, default=2048)
    ap.add_argument("--eval-every", type=int, default=128)
    ap.add_argument("--grad-every", type=int, default=0)
    ap.add_argument("--full-every", type=int, default=0)
    ap.add_argument("--keep-final-full", type=int, default=0)
    ap.add_argument("--milestones", default="", type=lambda v: [int(x) for x in str(v).split(",") if x])
    ap.add_argument("--lr-w", type=float, default=1e-4)
    ap.add_argument("--lr-small", type=float, default=1e-3)
    ap.add_argument("--p-mult", type=float, default=1.0)
    ap.add_argument("--lr-multiplier", type=float, default=1.0)
    ap.add_argument("--init-from", default="", help="full checkpoint to continue from: an extension of a finished run")
    ap.add_argument("--verbose", type=int, default=0)


def job_name(a_dict) -> str:
    keys = ("H", "model", "task", "seed", "m_phys", "accum", "lr_w", "lr_small", "p_mult")
    return "_".join(f"{k}{a_dict[k]}" for k in keys)


def job(a):
    a.init_from = a.init_from or None
    if a.m_phys % S.BLOCK:
        raise ValueError(f"M_phys {a.m_phys} is not a whole number of {S.BLOCK}-row blocks")
    t = Trainer(a)
    name = job_name(vars(a))
    d = Path(a.ckpt_root) / a.stage / name if a.ckpt_root else None
    resumes = []
    existing = sorted(d.glob("full_u*.pt")) if d and d.exists() else []
    if existing:
        t.load_full(existing[-1])
        resumes.append(dict(path=str(existing[-1]), update=t.update_count, next_block=t.next_block))
    elif a.init_from:
        t.load_full(a.init_from)
        resumes.append(dict(path=a.init_from, update=t.update_count, next_block=t.next_block, kind="extension"))
    finite, wall = t.train(ckpt_dir=d)
    if d is not None and a.keep_final_full and finite:
        t.save_full(d, keep=1)
    elif d is not None:
        for f in d.glob("full_u*.pt"):
            f.unlink()
    r = t.record(finite, wall)
    r["resumes"] = resumes
    C.atomic_json(a.out, r)
    print(f"{a.stage} H{a.H} {a.model} {a.task} s{a.seed} M{a.m_phys}x{a.accum} lr {a.lr_w}/{a.lr_small}/m{a.p_mult}: u {t.update_count} val {r['final_val']:.5f} "
          f"score {r['score_val']:.5f} ms/update {r['ms_per_update']:.1f} wall {wall:.0f}s finite {finite}", flush=True)


def session(a):
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    defaults = man.get("defaults", {})
    fails = out / "failures.jsonl"
    for k, j in enumerate(man["jobs"]):
        j = {**defaults, **j}
        name = job_name(j)
        f = out / f"{name}.json"
        if f.exists():
            continue
        cmd = [sys.executable, __file__, "job", "--root", a.root, "--ckpt-root", a.ckpt_root, "--out", str(f)]
        for x, v in j.items():
            cmd += [f"--{x.replace('_', '-')}", ",".join(map(str, v)) if isinstance(v, list) else str(v)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + r.stderr)
        print(f"{k + 1}/{len(man['jobs'])} rc {r.returncode} {time.perf_counter() - t0:.0f}s {r.stdout.strip()[-200:]}"
              + (f" ERR {r.stderr.strip()[-300:]}" if r.returncode else ""), flush=True)
        if r.returncode:
            with open(fails, "a") as h:
                h.write(json.dumps(dict(job=j, rc=r.returncode, err=r.stderr[-2000:])) + "\n")


def evaluate(a):
    """Locked test split, on every retained milestone of one finished run.  Needs the unlock file."""
    if not Path(a.unlock).exists():
        raise SystemExit(f"test split locked: {a.unlock} does not exist")
    rec = json.loads(Path(a.record).read_text())
    ns = argparse.Namespace(root=a.root, H=rec["H"], s=rec["s"], model=rec["model"], task=rec["task"], seed=rec["seed"], m_phys=rec["M_phys"], accum=rec["accum_steps"],
                            lr_w=rec["lr_w"], lr_small=rec["lr_small"], p_mult=rec["p_mult"], verbose=0)
    t = Trainer(ns)
    if t.fp != rec["fingerprints"]:
        raise RuntimeError("support identity differs from the training record")
    blocks = [S.block("test", S.HELDOUT_DATA_SEED, i, t.task.D0) for i in S.test_ids()]
    out = {}
    for u, m in sorted(rec["milestones"].items(), key=lambda kv: int(kv[0])):
        if a.verify_sha and C.sha256_file(m["path"]) != m["sha256"]:
            raise RuntimeError(f"checksum of {m['path']}")
        blob = torch.load(m["path"], map_location="cpu", weights_only=False)
        t.load_parameter_state(blob["params"])
        t.model.gamma.fill_(blob["gamma"])
        out[str(u)] = dict(update=int(u), samples=m["samples"], train_s=m["train_s"], val=m["val"], test=sum(t._loss_on(x) for x in blocks) / len(blocks))
    with torch.no_grad():
        tv = float(torch.cat([t.task(x.to(t.dev)) for x in blocks]).var())
    C.atomic_json(a.out, dict(schema=EVAL_SCHEMA, record=Path(a.record).name, stage=rec["stage"], model=rec["model"], task=rec["task"], seed=rec["seed"], H=rec["H"],
                              M_phys=rec["M_phys"], accum_steps=rec["accum_steps"], M_eff=rec["M_eff"], lr_w=rec["lr_w"], lr_small=rec["lr_small"], p_mult=rec["p_mult"],
                              milestones=out, test_target_var=tv, test_block_ids=list(S.test_ids()), unlock=json.loads(Path(a.unlock).read_text()), env=C.env_record()))
    print(rec["stage"], rec["model"], rec["task"], rec["seed"], {k: round(v["test"], 5) for k, v in out.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    job_args(j)
    j.add_argument("--out", required=True)
    s = sub.add_parser("session")
    s.add_argument("--manifest", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--ckpt-root", required=True)
    s.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--record", required=True)
    e.add_argument("--root", required=True)
    e.add_argument("--unlock", required=True)
    e.add_argument("--verify-sha", type=int, default=1)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(job=job, session=session, evaluate=evaluate)[a.cmd](a)


if __name__ == "__main__":
    main()
