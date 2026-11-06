"""Full-size identity / budget / coverage / round-trip check of the named Cover-TopK/16 layer built from a user-supplied
K64 envelope checkpoint (docs/validation/cover_topk16_rtx3090_dx.md section 3).

    python cover_topk16_identity.py --checkpoint /path/to/C256_R256_prox_a0_step128.pt --normalization energy \
        --expect-support e4465520... --expect-canonical 00922491... --expect-mask b8c5d17f... \
        --expect-weight-sha 29efaf12... --out identity_g0.json

Checks: exact 16 sorted distinct inputs per valid neuron (index / value pairing), all 64 candidates of every original
record kept by >= 1 row group (union coverage), 16 H valid slots and H biases, padded slots zero, original per-output
denominators, the frozen support / canonical values / mask / packed tensors against the research references (when
given), memory by unique storage, one-time construction costs, and a save_packed / load_packed round trip that
restores the identical layer without selecting again (the envelope values are dropped before loading).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import common  # noqa: F401  (sys.path)
import torch


def sha_tensors(*ts) -> str:
    """Byte hash of raw tensors (dtype / shape included): the research replay records' start-state hash."""
    h = hashlib.sha256()
    for t in ts:
        t = t.detach().contiguous()
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.view(torch.uint8).cpu().numpy().tobytes() if t.element_size() == 1 else
                 t.reshape(-1).view(torch.int16 if t.element_size() == 2 else torch.int32).cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def structure(lay) -> dict:
    """Degree / sortedness / union coverage / budget / padding of a CoverTopK16Layer."""
    L = lay.layout
    env = lay.env.long()  # [G nb16, 16] local positions (0..63) per 16-neuron record
    counts = lay.counts.long()
    G, nb16 = L.groups, L.blocks_per_group
    nvb = torch.clamp(counts[:, None] - 16 * torch.arange(nb16, device=env.device)[None, :], 0, 16)  # valid neurons
    live = (nvb > 0).reshape(-1)
    sorted_distinct = bool((env[live][:, 1:] > env[live][:, :-1]).all())
    idx = lay.indices.long().reshape(-1, 16)
    idx_sorted = bool((idx[live][:, 1:] > idx[live][:, :-1]).all())
    # original records: envelope block j = (16-block) // (C_env / 16)
    gpb = lay.envelope_KC[1] // 16
    rec = (torch.arange(G, device=env.device)[:, None] * ((nb16 + gpb - 1) // gpb) +
           torch.arange(nb16, device=env.device)[None, :] // gpb).reshape(-1)
    hit = torch.zeros((int(rec.max()) + 1, 64), dtype=torch.bool, device=env.device)
    hit[rec[live][:, None].expand(-1, 16), env[live]] = True
    rec_live = torch.unique(rec[live])
    covered = hit[rec_live].all(1)
    w = lay.weight.detach()
    valid = lay.layout.valid_mask(w.device)
    pad_zero = bool((w[~valid] == 0).all()) and bool((lay.bias.detach()[~valid] == 0).all())
    return dict(valid_neurons=int(counts.sum()), valid_slots=int(16 * counts.sum()), biases=int(counts.sum()),
                records_16=int(live.sum()), original_records=int(rec_live.numel()),
                original_records_all_64_covered=int(covered.sum()), sorted_distinct_local=sorted_distinct,
                sorted_global_indices=idx_sorted, padded_slots_zero=pad_zero,
                denominators=dict(min=int(counts.min()), max=int(counts.max()), sum=int(counts.sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--normalization", default="energy", choices=["none", "energy"])
    ap.add_argument("--expect-support", default="")
    ap.add_argument("--expect-canonical", default="")
    ap.add_argument("--expect-mask", default="")
    ap.add_argument("--expect-weight-sha", default="")
    ap.add_argument("--expect-bias-sha", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from provenance import provenance
    from sparse_pooling import CoverTopK16Layer
    from sparse_pooling.experimental import canonical as Cn

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    lay = CoverTopK16Layer.from_checkpoint(a.checkpoint, normalization=a.normalization, device="cuda")
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    out = dict(schema="granule.cover_topk16_identity/1", checkpoint=dict(name=Path(a.checkpoint).name),
               normalization=a.normalization, build_seconds_total=build_s,
               construction_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               provenance=provenance(Path(common.__file__).resolve().parent.parent))
    m = lay.metadata
    out["metadata"] = {k: v for k, v in m.items() if k != "support_stats"}
    out["support_stats_keys"] = sorted(m["support_stats"])
    out["structure"] = structure(lay)
    ident = dict(support=m["support_fingerprint"], canonical_values=Cn.digest(lay)["sha"],
                 canonical_mask=Cn.mask_digest(lay)["sha"], packed_weight=sha_tensors(lay.weight),
                 packed_bias=sha_tensors(lay.bias), immutable=lay.immutable_fingerprints())
    out["identity"] = ident
    exp = dict(support=a.expect_support, canonical_values=a.expect_canonical, canonical_mask=a.expect_mask,
               packed_weight=a.expect_weight_sha, packed_bias=a.expect_bias_sha)
    out["expected"] = {k: dict(expected=v, got=ident[k], equal=ident[k] == v) for k, v in exp.items() if v}
    lay.enable_main_grad()
    out["memory_unique_storage_before_runtime"] = lay.memory_report()
    lay._runtime()
    out["memory_unique_storage"] = lay.memory_report()
    # round trip: save, drop everything, load without the envelope / checkpoint / selection
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        p = Path(td) / "cover.pt"
        t1 = time.perf_counter()
        lay.save_packed(p)
        save_s = time.perf_counter() - t1
        size = p.stat().st_size
        del lay
        torch.cuda.empty_cache()
        import sparse_pooling.experimental.local_supports as LS

        calls = []
        orig = LS.build
        LS.build = lambda *x, **k: calls.append(1) or orig(*x, **k)
        t1 = time.perf_counter()
        lay2 = CoverTopK16Layer.load_packed(p, device="cuda")
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t1
        LS.build = orig
    rt = dict(save_seconds=save_s, load_seconds=load_s, file_bytes=size, support_selection_calls_on_load=len(calls),
              support=lay2.metadata["support_fingerprint"] == ident["support"],
              canonical_values=Cn.digest(lay2)["sha"] == ident["canonical_values"],
              canonical_mask=Cn.mask_digest(lay2)["sha"] == ident["canonical_mask"],
              packed_weight=sha_tensors(lay2.weight) == ident["packed_weight"],
              packed_bias=sha_tensors(lay2.bias) == ident["packed_bias"],
              normalization_recorded=lay2.normalization.get("treatment"))
    rt["pass"] = all(v for k, v in rt.items() if k in ("support", "canonical_values", "canonical_mask", "packed_weight",
                                                        "packed_bias")) and not calls
    out["round_trip"] = rt
    s = out["structure"]
    out["pass"] = bool(rt["pass"] and all(e["equal"] for e in out["expected"].values()) and s["sorted_distinct_local"]
                       and s["sorted_global_indices"] and s["padded_slots_zero"]
                       and s["original_records_all_64_covered"] == s["original_records"]
                       and s["valid_slots"] == 16 * s["biases"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(dict(pass_=out["pass"], expected=out["expected"], structure=s, round_trip=rt), indent=1,
                     default=str))


if __name__ == "__main__":
    main()
