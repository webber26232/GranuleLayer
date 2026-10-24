"""Checkpoint provenance of the width-preserving study (audit item 3, results/WIDTH_PRESERVING_SPARSITY.md 1.2).

The study REUSES the actual alpha = 0 checkpoints of the K64 study (no regeneration from seeds: fast FP32
atomic reductions do not make byte-identical reconstruction possible).  For every referenced checkpoint this
records: the path (outside git), size, file sha256 recomputed now and compared with its sidecar record, the
sidecar's weight / bias / mask sha256 and model spec, the parent learning record that produced it (commit,
source-dirty flags, source tree sha256, kgx library hash, hardware), and the retrieval / reconstruction
description.  Run without a GPU:

    python wps_checkpoints.py --out RAW/checkpoints/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import wps_cases as WC

PARENT_RAW = Path(__file__).resolve().parent.parent / "results" / "k64_large_c_indexfree_l1"
PARENT_COMMIT = "982760d84ba9f149fb2c55fae882859003354be0"


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def learn_record(C: int, seed: int):
    """The K64 study's learning record that wrote the checkpoint (stage learn / confirm_sparse)."""
    cands = [PARENT_RAW / "learn" / f"learn_C{C}_prox_a0.json"] if seed == 0 else \
        [PARENT_RAW / "confirm_sparse" / f"learn_C{C}_prox_a0_g{seed}.json"]
    for c in cands:
        if c.exists():
            return c, json.loads(c.read_text())
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for C, seeds in ((256, WC.GRAPH_SEEDS), (2048, WC.GRAPH_SEEDS), (64, (0,))):
        for s in seeds:
            p = Path(WC.ckpt(C, s))
            side = json.loads(Path(str(p) + ".json").read_text())
            small = Path(str(p).replace(".pt", "_small.pt"))
            fs = file_sha(p)
            lp, lr = learn_record(C, s)
            prov = (lr or {}).get("provenance", {})
            ck = (lr or {}).get("checkpoint", {})
            rows.append(dict(
                id=f"C{C}_g{s}", path=str(p), bytes=p.stat().st_size, file_sha256_now=fs,
                file_sha256_recorded=side.get("file_sha256"), file_sha256_match=fs == side.get("file_sha256"),
                weight_sha256=side["weight_sha256"], bias_sha256=side["bias_sha256"], mask_sha256=side["mask_sha256"],
                spec=side["spec"], C=side["C"], H=side["H"], tag=side["meta"]["tag"], step=side["meta"]["step"],
                small_state=dict(path=str(small), bytes=small.stat().st_size, file_sha256=file_sha(small)),
                learn_record=str(lp) if lp else None,
                learn_record_checkpoint_weight_sha256=ck.get("weight_sha256"),
                learn_record_match=ck.get("weight_sha256") == side["weight_sha256"] if ck else None,
                parent_code=dict(commit=prov.get("commit"), branch=prov.get("branch"),
                                 source_dirty=prov.get("source_dirty"), generated_dirty=prov.get("generated_dirty"),
                                 source_tree_sha256=prov.get("source_tree_sha256")),
                parent_hardware=(lr or {}).get("hardware_state"),
                config=side["meta"].get("config")))
    out = dict(schema="granule.wps_checkpoints/1", parent_study_commit=PARENT_COMMIT,
               retrieval=("local files outside git under " + WC.CKPT_DIR + " (kept; never overwritten). Each is the "
                          "dense layer after 128 updates of k64_learn.py (alpha 0: 32 warm-up + 96 'regularized' "
                          "updates with mu = 0) at the recorded parent commit; *_small.pt holds the proj / head state."),
               reconstruction=("python k64_learn.py --C <C> --treatment prox --alpha 0 [--graph-seed <s>] --ckpt-dir DIR "
                               "--out X.json from a frozen checkout of the parent commit; the process uses fast FP32 "
                               "atomic reductions and stochastic rounding, so a regenerated checkpoint is EXPECTED to "
                               "differ in its bytes; exact regeneration has NOT been verified (the regen stage of this "
                               "study records one attempt and its hashes)."),
               checkpoints=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    for r in rows:
        print(r["id"], r["file_sha256_match"], r["learn_record_match"], r["parent_code"]["commit"], flush=True)


if __name__ == "__main__":
    main()
