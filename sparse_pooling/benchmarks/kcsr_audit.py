"""Support audit of the 24 sampled K/C/S/R cases against an implementation (GRANULE_IMPL_ROOT).

For every case: does the public constructor path accept (fan_in = K, shared_neurons = C, S, R)?  The
exact exception text is recorded for rejected cases.  For accepted cases, the exact canonical bucket
statistics of the graph (supergroups.bucket_table: the compact metadata only, no index records) feed
the resolver of the real device and the automatically selected profile, giving the per-phase plan that
``auto`` will choose before anything runs (the timing runs verify it against last_info).

    GRANULE_IMPL_ROOT=../GranuleLayer-kcsr-base python kcsr_audit.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import kcsr_cases as KC
from common import gpu_info  # noqa: F401  (puts the implementation under test on sys.path)
import sparse_pooling
from sparse_pooling.config import LevelSpec, PhysicalPlan
from sparse_pooling.layout import SparseLayout
from sparse_pooling.supergroups import ConnectivityPlan, bucket_statistics, bucket_table


def impl_commit():
    d = str(Path(sparse_pooling.__file__).resolve().parent)
    sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return d, sha, bool(dirty)


def audit_case(c, caps, sel, B=KC.M):
    out = dict(id=c["id"], K=c["K"], C=c["C"], S=c["S"], R=c["R"], H=c["H"])
    try:
        L = SparseLayout(KC.D, c["H"], KC.G, c["K"], c["C"], True, packed=True)
    except Exception as e:  # noqa: BLE001
        out.update(layout_ok=False, status="unsupported", reason=f"SparseLayout: {type(e).__name__}: {e}")
        return out
    try:
        plan = ConnectivityPlan(c["S"], c["R"], seed=KC.GRAPH_SEED)
        Q = plan.num_supergroups(KC.D)
    except Exception as e:  # noqa: BLE001
        out.update(layout_ok=True, status="unsupported", reason=f"ConnectivityPlan: {type(e).__name__}: {e}")
        return out
    t0 = time.perf_counter()
    feat, buckets = bucket_table(L, plan)
    bs = bucket_statistics(L, buckets, Q, c["S"], plan)
    out.update(layout_ok=True, status="existing-supported", Q=Q, bucket_stats=bs,
               bucket_table_seconds=time.perf_counter() - t0, unit_blocks=L.unit_blocks,
               blocks_per_group=L.blocks_per_group, padding_fraction=L.padding_fraction(),
               param_bytes=L.param_bytes())
    # resolver decision with the real device record (host-only)
    from sparse_pooling.backends.dispatch import Resolver
    from sparse_pooling.kernels import ops

    lv = LevelSpec(type="mean")  # Mean(ReLU)
    sg = SimpleNamespace(S=c["S"], bucket_stats=bs, mean_bucket_units=lambda: bs["mean_units_per_nonempty_bucket"] or 0.0)
    rt = SimpleNamespace(layout=L, level=lv, plan=PhysicalPlan(grad_precision="split"), sg=sg, det=False, last_info={},
                         units=ops.Units(1, L.blocks_per_group, None, L.counts_tensor("cpu").float()),
                         idx=SimpleNamespace(device="cuda:0"))
    r = Resolver(caps, sel)
    out["resolved_forward"] = r.resolve(rt, B, "fwd").as_dict()
    out["resolved_backward"] = r.resolve(rt, B, "bwd").as_dict()
    out["predicted_kernels"] = dict(fwd=out["resolved_forward"].get("kernel"),
                                    dx=out["resolved_backward"]["dx"].get("kernel"),
                                    dw=out["resolved_backward"]["dw"].get("kernel"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    KC.check_manifest()
    import torch
    from sparse_pooling.backends.capabilities import query
    from sparse_pooling.backends.profiles import select

    caps = query(torch.device("cuda", 0))
    sel = select(caps)
    d, sha, dirty = impl_commit()
    rows = [audit_case(c, caps, sel) for c in KC.cases()]
    rows.append(audit_case(KC.case("X01"), caps, sel))
    res = dict(implementation=d, commit=sha, dirty=dirty, device=caps.summary(), profile=sel.summary(),
               cases=rows, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    for r in rows:
        print(f"{r['id']} K={r['K']:>2} C={r['C']:>3} S={r['S']:>3} R={r['R']:>3}: {r['status']:<19}"
              f"{r.get('predicted_kernels', '') or r.get('reason', '')}")


if __name__ == "__main__":
    main()
