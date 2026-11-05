"""Compatibility smoke of the existing fan-in-8 paths touched by the shared build / dispatch / profile edits of the
Cover-TopK/16 integration (docs/validation/cover_topk16_rtx3090_dx.md section 6).

    python cover_topk16_compat_smoke.py compare --baseline <checkout of main> --out compat.json

``compare`` runs ``case`` in a fresh process for this checkout and for the baseline checkout (GRANULE_IMPL_ROOT, same
tool), alternating, and checks: identical production library identity (same compiled code), identical resolved plans,
outputs / parameter gradients within FP32 tolerance and BF16 input gradients within one BF16 ulp (FP32 atomics are
not bitwise), and a short step
timing per path (informational; same session, not an acceptance measurement).
Cases: SparsePoolingLayer D 1024 / G 256, H 1M, input supergroups S 64 / R 16, C 8 and C 64, automatic plan (grouped /
hybrid as resolved), forced generic_cuda profile, and an explicit generic backend.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

CASES = [dict(name="C8_auto", C=8, plan=dict()), dict(name="C64_auto", C=64, plan=dict()),
         dict(name="C8_generic_profile", C=8, plan=dict(tuning_profile="generic_cuda")),
         dict(name="C64_generic_backend", C=64, plan=dict(cuda_input_supergroup_backend="generic"))]


def case():
    import common  # noqa: F401  (GRANULE_IMPL_ROOT)
    import torch
    from sparse_pooling import PhysicalPlan, SparsePoolingLayer
    from sparse_pooling.optim import FusedAdamW
    from sparse_pooling.supergroups import ConnectivityPlan

    D, H, G, B = 1024, 1_000_000, 256, 1024
    out = {}
    for c in CASES:
        torch.manual_seed(0)
        lay = SparsePoolingLayer(D, H, G, shared_neurons=c["C"], pooling="mean", device="cuda",
                                 connectivity=ConnectivityPlan(64, 16, seed=1), plan=PhysicalPlan(**c["plan"]))
        lay.enable_main_grad()
        g = torch.Generator(device="cuda").manual_seed(5)
        x = torch.randn(B, D, device="cuda", generator=g).to(torch.bfloat16).requires_grad_(True)
        y = lay(x)
        y.float().square().mean().backward()
        ex = lay.explain(B)
        rec = dict(y=y.detach().float().cpu(), dx=x.grad.float().cpu(), dw=lay.weight.main_grad.cpu(),
                   backend=str(ex.get("backend")), plan=json.loads(json.dumps(ex, default=str)))
        opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4)
        ts = []
        x.grad = None
        for i in range(10):
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            lay(x).float().square().mean().backward()
            opt.step()
            x.grad = None
            e1.record()
            torch.cuda.synchronize()
            if i >= 2:
                ts.append(e0.elapsed_time(e1))
        rec["step_ms_median"] = statistics.median(ts)
        out[c["name"]] = rec
        del lay, opt, x, y
        torch.cuda.empty_cache()
    from sparse_pooling.backends.cuda.runtime import load_bundle
    from sparse_pooling.backends.cuda.build import resolve_target
    from sparse_pooling.backends.capabilities import query

    target, _ = resolve_target(query(torch.device("cuda")))
    ident = load_bundle("production", target).ident
    torch.save(dict(cases=out, production=dict(library_hash=ident["library_hash"], unit_hashes=ident["unit_hashes"],
                                               flags=ident["flags"], kind=ident["kind"])), os.environ["COMPAT_OUT"])


def compare(a):
    import torch

    me = str(Path(__file__).resolve())
    work = Path(a.out).with_suffix(".work")
    work.mkdir(parents=True, exist_ok=True)
    runs = {}
    for label, root in (("pr", str(Path(__file__).resolve().parents[2])), ("baseline", str(Path(a.baseline).resolve()))):
        f = work / f"{label}.pt"
        env = dict(os.environ, GRANULE_IMPL_ROOT=root, COMPAT_OUT=str(f), PYTHONPATH=str(Path(me).parent))
        r = subprocess.run([sys.executable, me, "case"], env=env, capture_output=True, text=True, cwd=str(Path(me).parent))
        if r.returncode:
            raise SystemExit(f"{label} failed:\n{r.stdout}\n{r.stderr}")
        runs[label] = torch.load(f, weights_only=False)

    def rel(x, y):
        return float((x.double() - y.double()).abs().max() / (y.double().abs().max() + 1e-300))

    res = dict(schema="granule.cover_topk16_compat_smoke/1", production_identical=runs["pr"]["production"] ==
               runs["baseline"]["production"], production_library_hash=runs["pr"]["production"]["library_hash"], cases={})
    ok = res["production_identical"]
    for c in CASES:
        n = c["name"]
        p, b = runs["pr"]["cases"][n], runs["baseline"]["cases"][n]
        keys = ("forward", "backward", "profile", "dispatch", "plan")
        same_plan = {k: p["plan"].get(k) == b["plan"].get(k) for k in keys if k in p["plan"] or k in b["plan"]}
        prof_pr, prof_b = dict(p["plan"].get("profile") or {}), dict(b["plan"].get("profile") or {})
        prof_pr.pop("version", None), prof_b.pop("version", None)
        if "profile" in same_plan:
            same_plan["profile"] = prof_pr == prof_b  # (the RTX 3090 profile's version moved; its C8 / C64 data did not)
        d = dict(plan_equal=all(same_plan.values()), plan_fields=same_plan, y_rel=rel(p["y"], b["y"]), dx_rel=rel(p["dx"], b["dx"]),
                 dw_rel=rel(p["dw"], b["dw"]), step_ms=dict(pr=p["step_ms_median"], baseline=b["step_ms_median"]),
                 backend=p["backend"])
        # x.grad is BF16: two runs whose FP32 atomic sums differ in order may round to adjacent BF16 values (one ulp,
        # at most 2^-7 relative at the largest element); dW is compared in FP32
        d["pass"] = d["plan_equal"] and d["y_rel"] < 1e-5 and d["dx_rel"] <= 2.0 ** -7 and d["dw_rel"] < 1e-5
        ok &= d["pass"]
        res["cases"][n] = d
        print(n, d, flush=True)
    res["pass"] = ok
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print("PASS" if ok else "FAIL")
    for f in work.glob("*.pt"):
        f.unlink()
    work.rmdir()
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("case")
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "case":
        case()
    else:
        compare(a)


if __name__ == "__main__":
    main()
