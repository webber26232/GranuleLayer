"""Uninterrupted vs resumed short run of the reduced task (results/INTERACTION_COVERAGE_K16_DX.md section 4.2): the same
protocol run 0 -> 40 updates in one process, and 0 -> 20 (complete state saved) + resumed 20 -> 40 in a second process
(identity-checked resume).  Compared: the saved 40-update states (values, main_grad, moments, rounding counter, step,
readout) and the evaluation histories.  The dense-masked layer reduces with FP32 atomics, so bitwise equality is not
required; the differences are reported, with a tolerance gate.  Also checks that a state of another support / init is
REFUSED by the identity check.

    python icdx_resume_check.py --out X.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def run(args, log):
    with open(log, "w") as fh:
        return subprocess.run([sys.executable, str(HERE / "rsn_interaction.py")] + args, cwd=HERE, stdout=fh,
                              stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", default="Cover-TopK/16")
    ap.add_argument("--init-seed", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.out).with_suffix("")
    d.mkdir(parents=True, exist_ok=True)
    base = ["--support", a.support, "--terms", "16", "--init-seed", str(a.init_seed), "--sr-seed", "0", "--lr", "0.01",
            "--lr-readout", "0.01", "--eval-every", "10", "--state-dir", str(d)]
    rc1 = run(base + ["--steps", "40", "--save-at", "20,40", "--out", str(d / "full.json")], d / "full.log")
    rc2 = run(base + ["--steps", "40", "--save-at", "40", "--resume", str(d / "full_step20.pt"), "--out",
                      str(d / "resumed.json")], d / "resumed.log")
    res = dict(schema="granule.icdx_resume_check/1", support=a.support, init_seed=a.init_seed, exit=[rc1, rc2])
    if rc1 == 0 and rc2 == 0:
        A = torch.load(d / "full_step40.pt", map_location="cpu", weights_only=False)
        B = torch.load(d / "resumed_step40.pt", map_location="cpu", weights_only=False)
        diff = {}
        for k in ("weight", "bias", "main_grad_w", "main_grad_b", "a_g", "c_g"):
            x, y = A[k].float(), B[k].float()
            diff[k] = dict(max_abs=float((x - y).abs().max()), equal=bool(torch.equal(A[k], B[k])))
        for k, v in A["opt_w"].items():
            if torch.is_tensor(v) and v.numel() > 1:
                diff[f"opt_w.{k}"] = dict(max_abs=float((v.float() - B["opt_w"][k].float()).abs().max()),
                                          equal=bool(torch.equal(v, B["opt_w"][k])))
        res["state_differences"] = diff
        res["sr_counter_equal"] = A["sr_counter"] == B["sr_counter"]
        res["step_equal"] = A["step"] == B["step"] == 40
        ja, jb = json.loads((d / "full.json").read_text()), json.loads((d / "resumed.json").read_text())
        ha = {h["step"]: h for h in ja["history"]}
        hb = {h["step"]: h for h in jb["history"]}
        res["history_max_abs_difference"] = max(abs(ha[s][k] - hb[s][k]) for s in ha for k in ha[s]
                                                if k.startswith("order") and s in hb)
        res["identity_checked_on_resume"] = jb.get("state_identity_checked")
        res["pass"] = bool(res["sr_counter_equal"] and res["step_equal"] and res["identity_checked_on_resume"] and
                           res["history_max_abs_difference"] < 1e-3)
    # a mismatching state must be refused
    rc3 = run(["--support", "BP-row#0", "--terms", "16", "--init-seed", str(a.init_seed), "--steps", "40", "--lr", "0.01",
               "--resume", str(d / "full_step20.pt"), "--out", str(d / "mismatch.json")], d / "mismatch.log")
    log = (d / "mismatch.log").read_text()
    res["mismatched_state_refused"] = rc3 != 0 and "identity mismatch" in log
    res["pass"] = bool(res.get("pass") and res["mismatched_state_refused"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "state_differences"}))


if __name__ == "__main__":
    main()
