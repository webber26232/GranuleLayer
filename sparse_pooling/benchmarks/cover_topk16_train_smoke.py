"""Compact-backend training smoke check of the named Cover-TopK/16 layer: legacy dX vs dx_variant="auto" (which resolves
to mt4_nw8_wg4 on the RTX 3090 for the validated model at M 512), 32 updates, seeds 0 / 1 / 2, development data
(docs/validation/cover_topk16_rtx3090_dx.md section 4).  INTEGRATION / TRAJECTORY validation, not an accuracy
confirmation: architecture selection and the six-seed locked-test cohorts are research evidence (f4c538a).

Model (all synthetic, seeded; no research campaign code): z [512, 1024] ~ N(0, 1) per (data seed, step); target
t = tanh(z A) B (fixed teacher, seed 777); y = head(layer(proj(z) as BF16)); MSE.  proj / head: FP32 Linear(1024,
1024), AdamW lr 1e-3, wd 0.  Layer: the full 200M-slot model built ONCE from the user-supplied graph-0 K64 envelope
checkpoint with energy normalization, saved packed, and RELOADED for every run (identical packed start state), FP32
main_grad, FusedAdamW lr 1e-4, BF16 moments, stochastic rounding with counter offset sr_seed * 1_000_003, split dZ.
Validation: data seeds 9000 .. 9003 after the last update.

Recorded per run: start-state fingerprints (packed weight / bias, main_grad, moments empty, rounding counter, proj /
head), per-step loss and layer gradient norms (FP32 main_grad before the optimizer step), the executed kgen dX
decision and library hash, final validation loss.  Pairing of legacy and auto runs: benchmarks/paired_stats with the
declared experiment (graph 0, protocol dev-val-9000-9003, 32 updates) and methods (Cover-TopK/16@<support>, energy,
legacy | mt4_nw8_wg4).

    python cover_topk16_train_smoke.py --checkpoint /path/to/C256_R256_prox_a0_step128.pt --out smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import common  # noqa: F401  (sys.path)
import torch

STEPS, M, LR_LAYER, LR_SMALL = 32, 512, 1e-4, 1e-3
VAL_SEEDS = (9000, 9001, 9002, 9003)
SEEDS = (0, 1, 2)


def sha(*ts) -> str:
    h = hashlib.sha256()
    for t in ts:
        t = t.detach().contiguous()
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def batch(seed: int, D: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(M, D, device="cuda", generator=g)


def teacher(D: int, G: int):
    g = torch.Generator(device="cuda").manual_seed(777)
    A = torch.randn(D, 256, device="cuda", generator=g) / math.sqrt(D)
    B = torch.randn(256, G, device="cuda", generator=g) / math.sqrt(256)
    return lambda z: torch.tanh(z @ A * 2.0) @ B


def run(packed: Path, request: str, seed: int, T) -> dict:
    from sparse_pooling import CoverTopK16Layer
    from sparse_pooling.optim import FusedAdamW

    lay = CoverTopK16Layer.load_packed(packed, device="cuda", dx_variant=request)
    D, G = lay.layout.input_dim, lay.layout.groups
    lay.enable_main_grad()
    lay.zero_main_grad()
    torch.manual_seed(4242)  # proj / head init: identical for every run
    proj = torch.nn.Linear(D, D, device="cuda")
    head = torch.nn.Linear(G, G, device="cuda")
    opt = FusedAdamW([lay.weight, lay.bias], lr=LR_LAYER, moments="bf16", stochastic_rounding=True)
    opt._seed = seed * 1_000_003
    opt2 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=LR_SMALL, weight_decay=0.0)
    start = dict(weight=sha(lay.weight), bias=sha(lay.bias), main_grad=sha(lay.weight.main_grad, lay.bias.main_grad),
                 moments_empty=len(opt.state) == 0, sr_counter=opt._seed,
                 proj_head=sha(*proj.parameters(), *head.parameters()))
    losses, gnorm, finite = [], [], True
    for i in range(STEPS):
        z = batch(10_000 + i + 1_000_000 * seed, D)
        y = head(lay(proj(z).to(torch.bfloat16)).float())
        loss = (y - T(z)).square().mean()
        loss.backward()
        gw = float(lay.weight.main_grad.norm())
        gb = float(lay.bias.main_grad.norm())
        finite &= math.isfinite(float(loss)) and math.isfinite(gw) and math.isfinite(gb)
        losses.append(float(loss))
        gnorm.append([gw, gb])
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()
    with torch.no_grad():
        val = sum(float((head(lay(proj(batch(s, D)).to(torch.bfloat16)).float()) - T(batch(s, D))).square().mean())
                  for s in VAL_SEEDS) / len(VAL_SEEDS)
    ex = lay.last_info["kgen_dx"]
    return dict(request=request, data_seed=seed, sr_seed=seed, steps=STEPS, batch=M, start_state=start, losses=losses,
                grad_norms=gnorm, all_finite=finite and math.isfinite(val), validation_loss=val,
                executed=dict(variant=ex["variant"], entry=ex["entry"], library_hash=ex["library_hash"],
                              shared_bytes=ex["shared_bytes"], threads=ex["threads"], tokens=ex["tokens"]),
                support=lay.metadata["support_fingerprint"], normalization=lay.normalization["treatment"],
                padded_slots_zero=bool((lay.weight.detach()[~lay.layout.valid_mask(lay.weight.device)] == 0).all()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import paired_stats as PS
    from provenance import provenance
    from sparse_pooling import CoverTopK16Layer

    lay = CoverTopK16Layer.from_checkpoint(a.checkpoint, normalization="energy", device="cuda")
    D, G = lay.layout.input_dim, lay.layout.groups
    sup = lay.metadata["support_fingerprint"]
    T = teacher(D, G)
    runs = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        packed = Path(td) / "cover.pt"
        lay.save_packed(packed)
        del lay
        torch.cuda.empty_cache()
        for seed in SEEDS:
            for request in ("legacy", "auto"):  # interleaved
                r = run(packed, request, seed, T)
                runs.append(r)
                print(f"seed {seed} {request}: executed {r['executed']['variant']['name']} final loss "
                      f"{r['losses'][-1]:.6f} val {r['validation_loss']:.6f}", flush=True)
                torch.cuda.empty_cache()
    exp = PS.Experiment(graph=0, protocol="dev-val-9000-9003", steps=STEPS)
    meth = {v: PS.Method(f"Cover-TopK/16@{sup[:16]}", "energy", v) for v in ("legacy", "mt4_nw8_wg4")}
    coh = {v: PS.Cohort(v, exp, meth[v]) for v in meth}
    for r in runs:
        v = r["executed"]["variant"]["name"]
        coh[v].add(PS.ObsKey(0, r["data_seed"], r["sr_seed"], f"Cover-TopK/16@{r['support'][:16]}", "energy",
                             "dev-val-9000-9003", r["steps"], v), r["validation_loss"], f"{r['request']}_s{r['data_seed']}")
    pr = PS.paired(coh["mt4_nw8_wg4"], coh["legacy"], [(s, s) for s in SEEDS], experiment=exp,
                   method_a=meth["mt4_nw8_wg4"], method_b=meth["legacy"], family_size=1, alpha=0.05, margin=0.003)
    per_seed = []
    for s in SEEDS:
        L = next(r for r in runs if r["data_seed"] == s and r["request"] == "legacy")
        A = next(r for r in runs if r["data_seed"] == s and r["request"] == "auto")
        per_seed.append(dict(seed=s, identical_start=L["start_state"] == A["start_state"],
                             max_abs_loss_diff=max(abs(x - y) for x, y in zip(L["losses"], A["losses"])),
                             final_loss_diff=A["losses"][-1] - L["losses"][-1],
                             validation_diff=A["validation_loss"] - L["validation_loss"],
                             max_rel_grad_norm_diff=max(abs(x[0] - y[0]) / max(y[0], 1e-30)
                                                        for x, y in zip(A["grad_norms"], L["grad_norms"])),
                             executed=(L["executed"]["variant"]["name"], A["executed"]["variant"]["name"]),
                             all_finite=L["all_finite"] and A["all_finite"]))
    out = dict(schema="granule.cover_topk16_train_smoke/1", label="integration / trajectory validation (32 updates, "
               "3 seeds); not a six-seed accuracy confirmation", checkpoint=Path(a.checkpoint).name, runs=runs,
               per_seed=per_seed, paired_validation=pr,
               provenance=provenance(Path(common.__file__).resolve().parent.parent),
               pass_=all(p["identical_start"] and p["all_finite"] and p["executed"] == ("legacy", "mt4_nw8_wg4")
                         for p in per_seed))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(dict(per_seed=per_seed, pass_=out["pass_"], paired=dict(status=pr["status"],
                                                                             descriptive=pr.get("descriptive"))), indent=1))


if __name__ == "__main__":
    main()
