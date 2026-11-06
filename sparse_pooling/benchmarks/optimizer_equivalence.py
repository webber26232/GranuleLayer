"""Bitwise equivalence of optim.FusedAdamW and the research branch's L1AdamW(l1="none", no mask) -- the optimizer the
research compact-backend training used (interaction-coverage-k16-dx f4c538a, experimental/l1.py) -- before this
integration substitutes FusedAdamW (docs/validation/cover_topk16_rtx3090_dx.md section 4).

    python optimizer_equivalence.py --l1 <research checkout>/sparse_pooling/experimental/l1.py --out eq.json

Same BF16 parameters, FP32 main_grad, BF16 / FP32 moments, stochastic rounding on / off, weight decay 0 / 0.01, five
steps with fresh gradients each step, two parameters per optimizer (the rounding counter advances per parameter):
parameters, both moments, step counts, the rounding counter and the fused main_grad zeroing must be bitwise equal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import common  # noqa: F401  (sys.path)
import torch


def load_l1(path):
    spec = importlib.util.spec_from_file_location("research_l1", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.optim import FusedAdamW

    L1 = load_l1(a.l1)
    src = Path(a.l1).read_bytes()
    cases = []
    for moments in ("bf16", "fp32"):
        for sr in (True, False):
            for wd in (0.0, 0.01):
                g = torch.Generator(device="cuda").manual_seed(7)
                shapes = [(1000, 16, 16), (3001,)]
                p0 = [(torch.randn(s, device="cuda", generator=g) * 0.1).bfloat16() for s in shapes]
                grads = [[torch.randn(s, device="cuda", generator=g) * 1e-2 for s in shapes] for _ in range(5)]
                sides = {}
                for name in ("fused", "l1"):
                    ps = [torch.nn.Parameter(p.clone()) for p in p0]
                    for p in ps:
                        p.main_grad = torch.zeros(p.shape, device="cuda", dtype=torch.float32)
                    opt = (FusedAdamW(ps, lr=1e-3, moments=moments, stochastic_rounding=sr, weight_decay=wd)
                           if name == "fused" else
                           L1.L1AdamW(ps, lr=1e-3, moments=moments, stochastic_rounding=sr, weight_decay=wd, l1="none"))
                    opt._seed = 3 * 1_000_003
                    zeroed = True
                    for gs in grads:
                        for p, gg in zip(ps, gs):
                            p.main_grad.copy_(gg)
                        opt.step()
                        zeroed &= all(float(p.main_grad.abs().max()) == 0.0 for p in ps)
                    sides[name] = dict(params=[p.detach().clone() for p in ps],
                                       m=[opt.state[p]["m"].clone() for p in ps], v=[opt.state[p]["v"].clone() for p in ps],
                                       step=[opt.state[p]["step"] for p in ps], seed=opt._seed, zeroed=zeroed)
                f, l = sides["fused"], sides["l1"]
                eq = dict(params=all(torch.equal(x, y) for x, y in zip(f["params"], l["params"])),
                          m=all(torch.equal(x, y) for x, y in zip(f["m"], l["m"])),
                          v=all(torch.equal(x, y) for x, y in zip(f["v"], l["v"])),
                          step=f["step"] == l["step"], rounding_counter=f["seed"] == l["seed"],
                          main_grad_zeroed=f["zeroed"] and l["zeroed"])
                cases.append(dict(moments=moments, stochastic_rounding=sr, weight_decay=wd, steps=5, equal=eq,
                                  all_equal=all(eq.values())))
                print(cases[-1], flush=True)
    out = dict(schema="granule.optimizer_equivalence/1", research_l1_sha256=hashlib.sha256(src).hexdigest(),
               cases=cases, all_equal=all(c["all_equal"] for c in cases), device=torch.cuda.get_device_name(0))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("ALL EQUAL" if out["all_equal"] else "DIFFERENT")


if __name__ == "__main__":
    main()
