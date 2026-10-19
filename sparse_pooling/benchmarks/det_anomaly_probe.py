"""Probe of the deterministic-control anomaly (D = 1024, G = 8, H = 100k, M = 64, S = 16, R = 1):
two runs, identical up to the dX scatter, compared tensor by tensor around every kernel of a
step (before / after backward, after each optimizer) to find where the first difference
appears.  Also runnable under compute-sanitizer (--steps small)."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

import common  # noqa: F401,E402
from deterministic_training import CASES, Run, batches, bits, det_plan  # noqa: E402


def snap(r):
    st = r.opt.state.get(r.lay.weight, {})
    out = {"W": r.lay.weight.detach().clone(), "b": r.lay.bias.detach().clone(),
           "mg": r.lay.weight.main_grad.clone(), "mgb": r.lay.bias.main_grad.clone()}
    for k in ("m", "v"):
        if k in st:
            out[k] = st[k].clone()
    return out


def diff(a, b):
    return [k for k in a if not torch.equal(bits(a[k]), bits(b[k]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="D1024_G8_H100k_M64_S16")
    ap.add_argument("--cand", default="vectorized_hoisted")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--T", type=int, default=64)
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)
    case = {c[0]: c for c in CASES}[args.case]
    ra = Run(case, det_plan(args.T, 1, args.ref), "det")
    rb = Run(case, det_plan(args.T, 1, args.cand), "det")
    data = batches(case, args.steps)
    for s, (x0, tgt) in enumerate(data):
        for tag, fn in (("pre", None),):
            d = diff(snap(ra), snap(rb))
            if d:
                print(f"step {s} {tag}: differ {d}")
        outs = []
        for r in (ra, rb):
            xin = r.proj(x0).bfloat16()
            xin.retain_grad()
            y = r.lay(xin)
            h = torch.tanh(r.head(y.float()))
            loss = ((h - tgt) ** 2).mean()
            loss.backward()
            torch.cuda.synchronize()
            outs.append((y.detach().clone(), xin.grad.clone(), snap(r)))
        (ya, gxa, sa), (yb, gxb, sb) = outs
        d = diff(sa, sb)
        print(f"step {s} after backward: y eq {torch.equal(bits(ya), bits(yb))} dX eq {torch.equal(bits(gxa), bits(gxb))}"
              f" state differ {d}", flush=True)
        for r in (ra, rb):
            r.opt2.step()
            r.opt2.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        d = diff(snap(ra), snap(rb))
        if d:
            print(f"step {s} after torch AdamW: differ {d}")
        pre = (snap(ra), snap(rb))
        for r in (ra, rb):
            r.opt.step()
        torch.cuda.synchronize()
        d = diff(snap(ra), snap(rb))
        if d:
            print(f"step {s} after FusedAdamW: differ {d}; inputs differed: {diff(*pre)}")
            # replay the FusedAdamW step from the saved inputs on both runs' tensors
            break


if __name__ == "__main__":
    main()


def lockstep_probe(case_name="D1024_G8_H100k_M64_S16", steps=16):
    """All variants in lockstep (as deterministic_training.lockstep); after every Run.step the
    layer state (W, b, main_grad, m, v) of EVERY run is fingerprinted, so a change of a run's
    state while another run executes (corruption) is detected and attributed."""
    from deterministic_training import lockstep, variants_for  # noqa: F401
    from train_state import tensor_fingerprint
    case = {c[0]: c for c in CASES}[case_name]
    variants = variants_for(case[5])
    runs = {n: Run(case, kw, "det") for n, (kw, _) in variants.items()}
    data = batches(case, steps)

    def fp(r):
        return {k: tensor_fingerprint(v) for k, v in snap(r).items()}

    last = {}
    for s, (x0, tgt) in enumerate(data):
        for n, r in runs.items():
            # every other run must be unchanged since its own last step
            for m_, f in last.items():
                if m_ != n and fp(runs[m_]) != f:
                    bad = [k for k in f if fp(runs[m_])[k] != f[k]]
                    print(f"step {s}: state of {m_} changed ({bad}) before {n} ran", flush=True)
                    last[m_] = fp(runs[m_])
            r.step(x0, tgt)
            torch.cuda.synchronize()
            last[n] = fp(r)
        print(f"step {s} done", flush=True)
