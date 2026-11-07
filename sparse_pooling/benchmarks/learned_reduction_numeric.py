"""Measured errors of the lrd kernels against the FP64 same-operand oracle, over the test geometries (the tests assert the
declared tolerances; this records the values).  Worst case over the geometries, per token count and quantity.

    python learned_reduction_numeric.py --out numeric_errors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1] / "tests")]
import test_learned_reduction as TL  # noqa: E402
from sparse_pooling.experimental.learned_reduction import ReducerConfig, ReductionModel  # noqa: E402


def errors(a, b):
    a, b = a.double(), b.double()
    return dict(rel_l2=float((a - b).norm() / b.norm().clamp_min(1e-30)), max_abs=float((a - b).abs().max()),
                max_normalized=float((a - b).abs().max() / b.abs().max().clamp_min(1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for H, G, L, s, ch, desc in TL.GEOMS:
        for M in (8, 40, 136):
            m = ReductionModel(TL.build_base(H, G), ReducerConfig("t", "learned", L=L, s=s, channels=ch), output_dim=32)
            red = m.reducer
            x = torch.randn(M, 256, device="cuda", dtype=torch.float32).requires_grad_(True)
            U = red(x)
            gy = torch.randn_like(U)
            Uo, dxo, dwo, dbo, dpo = TL.oracle_grads(m, x, gy)
            m.zero_main_grad()
            torch.autograd.backward(U, gy)
            rows.append(dict(H=H, G=G, L=L, s=s, channels=ch, M=M, geometry=desc, U=errors(U.detach(), Uo), dx=errors(x.grad, dxo),
                             dW=errors(m.sparse.weight.main_grad, dwo), db=errors(m.sparse.bias.main_grad, dbo),
                             dP=errors(red.P.main_grad, dpo)))
    q = ("U", "dx", "dW", "db", "dP")
    worst = {k: {e: max(r[k][e] for r in rows) for e in ("rel_l2", "max_abs", "max_normalized")} for k in q}
    rec = dict(schema="granule.lsw_numeric/1", tolerances=dict(forward_rel_l2=TL.TOL_FWD_SPLIT, gradient_rel_l2=TL.TOL_GRAD,
                                                               max_normalized=TL.TOL_MAX_NORM), worst=worst, cases=rows)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(worst, indent=1))


if __name__ == "__main__":
    main()
