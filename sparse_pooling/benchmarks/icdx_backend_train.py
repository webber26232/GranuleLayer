"""Explicit COMPACT-BACKEND training replay (results/INTERACTION_COVERAGE_K16_DX.md section 6.2, rows 7 / 8): the
synthetic fine-tuning task of rsn_learn.py executed on the actual lowered kgen layer instead of the dense-masked
reference execution (architecture-quality rows stay on rsn_learn.py; NOT a downstream-accuracy claim).

Construction is the same as rsn_learn: the actual alpha-0 checkpoint (sha256 verified) with its proj / head, the
support selected from its pre-pruning values (rsn_cases spec), the values masked, the one-time normalization
(``--norm``; row_normalize, cap 8) applied to the CANONICAL dense-masked values -- then the layer is LOWERED ONCE
(indexed K16 / C16: kgx_onetile.lower_local; mixed: kgx_grouped.lower_mixed) and the canonical identity (values and
masks) of the lowered layer is checked against the dense-masked one before the dense layer is freed.  Training: the
compact parameters themselves (L1AdamW lr 1e-4, fresh moments, no L1, wd 0, stochastic rounding with counter offset
sr_seed * 1_000_003; AdamW lr 1e-3 on proj / head), the same data / validation / locked-test batches as rsn_learn.

``--dx-variant`` selects the K16 dX launch (experimental/kgen_dx_variants.py; "legacy" = the original launch).  Two
runs differing only in it start from IDENTICAL packed values, biases, moments (fresh), main_grad (zero), rounding
counter and proj / head: the record carries a start-state fingerprint and the report requires equality.  Dense vs
compact parameter layouts index stochastic rounding differently, so a compact run is not the same random-rounding
trajectory as its dense-masked counterpart (secondary lowering / trajectory check only).

Recorded: the executed backend (forward / dX / dW kernels and the resolved dX variant from last_info), library
identity (kgen build info), start / final canonical digests (the canonical adapter maps the compact layout back to
envelope rows), padded-slot zero checks, curves, held-out, the locked test (once, after training), and the final state
(packed weight / bias / main_grad / moments / step / rounding counter / proj / head / their AdamW state) with sha256.

    python icdx_backend_train.py --support Cover-TopK/16 --norm energy --data-seed 12 --sr-seed 12 --steps 256 \
        --final-test --test-protocol icdx39000 --dx-variant legacy --save-state DIR --out X.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import common  # noqa: F401
import rsn_cases as RC
import wps_cases as WC
from k64_learn import LR_LAYER, LR_SMALL, REG, WARMUP, Teacher, batch, build, forward_loss, heldout
from rsn_learn import TEST_PROTOCOLS, check_seed_collisions, final_test

SCHEMA = "granule.icdx_backend_train/1"


def sha_tensors(*ts) -> str:
    h = hashlib.sha256()
    for t in ts:
        if t is None:
            h.update(b"none")
            continue
        t = t.detach().contiguous()
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.view(torch.uint8).cpu().numpy().tobytes() if t.element_size() == 1 else
                 t.reshape(-1).view(torch.int16 if t.element_size() == 2 else torch.int32).cpu().numpy().tobytes())
    return h.hexdigest()


def state_fingerprint(lo, opt, proj, head, opt2) -> dict:
    st_w, st_b = opt.state.get(lo.weight, {}), opt.state.get(lo.bias, {})
    mom = {k: sha_tensors(v) for k, v in list(st_w.items()) + [(f"bias_{k}", v) for k, v in st_b.items()]
           if torch.is_tensor(v)}
    small = sha_tensors(*[p for p in list(proj.parameters()) + list(head.parameters())])
    return dict(weight=sha_tensors(lo.weight), bias=sha_tensors(lo.bias),
                main_grad=sha_tensors(getattr(lo.weight, "main_grad", None), getattr(lo.bias, "main_grad", None)),
                moments=mom, moments_empty=not any(torch.is_tensor(v) for v in st_w.values()),
                sr_counter=int(getattr(opt, "_seed", 0)), proj_head=small,
                opt_small_empty=len(opt2.state) == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--support", required=True)
    ap.add_argument("--norm", default="none", choices=["none", "fanin", "energy"])
    ap.add_argument("--s-max", type=float, default=8.0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--sr-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=32)
    ap.add_argument("--dx-variant", default="legacy")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--test-protocol", default="icdx39000", choices=sorted(TEST_PROTOCOLS))
    ap.add_argument("--save-state", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.config import PhysicalPlan
    from sparse_pooling.experimental import canonical as Cn
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.experimental import row_normalize as RN
    from sparse_pooling.experimental.kgx_grouped import lower_mixed
    from sparse_pooling.experimental.kgx_onetile import lower_local
    from provenance import hardware_state, provenance

    dev = torch.device("cuda")
    t_start = time.perf_counter()
    ck = WC.ckpt(256, a.graph_seed)
    lay, proj, head = build(256, 256, dev, a.graph_seed)
    rec, w, b, _ = SP.load_checkpoint(ck)
    if SP.mask_fingerprint(w.view(torch.int16)) != rec["weight_sha256"]:
        raise SystemExit("checkpoint weight sha256 differs from its record")
    with torch.no_grad():
        lay.weight.copy_(w)
        lay.bias.copy_(b)
    del w, b
    small = torch.load(ck.replace(".pt", "_small.pt"), map_location=dev)
    proj.load_state_dict(small["proj"])
    head.load_state_dict(small["head"])
    teacher = Teacher(dev)
    valid = SP.valid_slots(lay)
    res = dict(schema=SCHEMA, graph_seed=a.graph_seed, support=a.support, norm=a.norm, s_max=a.s_max,
               data_seed=a.data_seed, sr_seed=a.sr_seed, steps=a.steps,
               checkpoint=dict(path=ck, weight_sha256=rec["weight_sha256"], bias_sha256=rec["bias_sha256"]),
               protocol=dict(lr_layer=LR_LAYER, lr_small=LR_SMALL, fresh_optimizer_state=True, l1="none",
                             train_batches=f"seeds 10_000 + {WARMUP + REG} + i + 1_000_000 * data_seed",
                             heldout="seeds 9_000..9_003", stochastic_rounding=True,
                             execution="COMPACT backend: lowered once after construction / normalization"),
               provenance=provenance(Path(common.__file__).resolve().parent.parent), hardware_state=hardware_state())
    if a.final_test:
        res["final_test_protocol"] = dict(check_seed_collisions(steps=max(256, a.steps), protocol=a.test_protocol),
                                          evaluated="once, after training", batch=512)
    t0 = time.perf_counter()
    em, sel, srec = RC.support(a.support, lay, a.graph_seed)
    if em is None or sel is None:
        raise SystemExit("the compact backend replay needs a local or mixed support")
    res["support_record"] = srec
    w_pre = lay.weight.detach().clone() if a.norm != "none" else None
    with torch.no_grad():
        lay.weight.mul_(em.to(lay.weight.dtype))
    if a.norm != "none":
        with torch.no_grad():
            wn, bn, nrec = RN.apply(w_pre, lay.bias.detach(), em, valid, a.norm, s_max=a.s_max)
            lay.weight.copy_(wn)
            lay.bias.copy_(bn)
        del wn, bn
        res["normalization"] = nrec
    del w_pre
    res["heldout_dense_masked_start"] = heldout(lay, proj, head, teacher, dev)
    lay.mask = em
    canon_dense = Cn.digest(lay)["sha"]
    mask_dense = Cn.mask_digest(lay)["sha"]
    plan = PhysicalPlan(grad_precision="split")
    if hasattr(sel, "kc"):
        lo = lower_mixed(lay, sel, plan)
        fmt = "grouped"
    else:
        lo = lower_local(lay, sel, plan)
        fmt = "indexed"
    lo.dx_variant = a.dx_variant
    canon_lo, mask_lo = Cn.digest(lo)["sha"], Cn.mask_digest(lo)["sha"]
    if canon_lo != canon_dense or mask_lo != mask_dense:
        raise SystemExit("lowered layer differs from the dense-masked canonical values / mask")
    res["lowering"] = dict(format=fmt, seconds=time.perf_counter() - t0, canonical_values_sha=canon_lo,
                           canonical_mask_sha=mask_lo, identical_to_dense_masked=True)
    del lay, em
    torch.cuda.empty_cache()
    lo.enable_main_grad()
    lo.zero_main_grad()
    res["heldout_compact_start"] = heldout(lo, proj, head, teacher, dev)
    opt = L1.L1AdamW([lo.weight, lo.bias], lr=LR_LAYER)
    opt._seed = a.sr_seed * 1_000_003
    opt2 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=LR_SMALL, weight_decay=0.0)
    res["start_state"] = state_fingerprint(lo, opt, proj, head, opt2)
    live = lo.weight.detach() != 0
    curve, evals = [], []
    step0 = WARMUP + REG
    for i in range(a.steps):
        x0 = batch(10_000 + step0 + i + 1_000_000 * a.data_seed, dev)
        loss = forward_loss(lo, proj, head, x0, teacher(x0))
        loss.backward()
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()  # L1AdamW zeroes main_grad (fused), as in rsn_learn
        curve.append(dict(step=i + 1, loss=float(loss)))
        if (i + 1) % a.eval_every == 0 or i + 1 == a.steps:
            evals.append(dict(step=i + 1, heldout=heldout(lo, proj, head, teacher, dev)))
            print(f"[{a.support} {a.norm} {a.dx_variant} d{a.data_seed}] step {i + 1}: train {float(loss):.6f} "
                  f"held-out {evals[-1]['heldout']:.6f}", flush=True)
    info = dict(lo.last_info)
    be = lo._runtime().backend if fmt == "indexed" else lo.backend
    res["backend"] = dict(format=fmt, variant=a.dx_variant,
                          resolved=(lo.dx_variant_resolved().describe() if fmt == "indexed" else
                                    {f"K{pk.K}": lo.dx_variant_of(pk).describe() for pk in lo.packs}),
                          forward=info.get("forward_backend"), dx=info.get("dx_kernel"), dw=info.get("dw_kernel"),
                          kgen_dx=info.get("kgen_dx") or info.get("dx_variants"), library=be.build_info(),
                          grad_precision=info.get("grad_precision"))
    res["train_curve"], res["evals"] = curve, evals
    res["heldout_final"] = evals[-1]["heldout"]
    res["padded_slots_zero"] = bool((lo.weight.detach()[~live] == 0).all()) if hasattr(lo, "packs") else None
    res["canonical_final_values_sha"] = Cn.digest(lo)["sha"]
    res["canonical_mask_after_sha"] = Cn.mask_digest(lo)["sha"]
    res["mask_preserved"] = res["canonical_mask_after_sha"] == mask_dense
    if a.final_test:
        res["final_test"] = final_test(lo, proj, head, teacher, dev, a.test_protocol)
        res["final_test_set"] = a.test_protocol
    if a.save_state:
        sd_ = Path(a.save_state)
        sd_.mkdir(parents=True, exist_ok=True)
        st_w, st_b = opt.state.get(lo.weight, {}), opt.state.get(lo.bias, {})
        state = dict(schema=SCHEMA + "_state", support=a.support, norm=a.norm, data_seed=a.data_seed, sr_seed=a.sr_seed,
                     steps=a.steps, dx_variant=a.dx_variant, format=fmt, support_fingerprint=srec.get(
                         "support_fingerprint"), weight=lo.weight.detach().clone(), bias=lo.bias.detach().clone(),
                     main_grad=[lo.weight.main_grad.clone(), lo.bias.main_grad.clone()],
                     moments={k: v.clone() for k, v in st_w.items() if torch.is_tensor(v)},
                     bias_moments={k: v.clone() for k, v in st_b.items() if torch.is_tensor(v)},
                     step_state={k: v for k, v in st_w.items() if not torch.is_tensor(v)}, sr_counter=opt._seed,
                     proj=proj.state_dict(), head=head.state_dict(), opt_small=opt2.state_dict(),
                     note="compact packed layout of the lowered layer (canonical adapter: experimental/canonical.py)")
        f = sd_ / (Path(a.out).stem + "_final.pt")
        torch.save(state, f)
        del state
        hh = hashlib.sha256()
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                hh.update(blk)
        res["saved_state"] = dict(path=str(f), sha256=hh.hexdigest(), bytes=f.stat().st_size)
    res["wall_seconds"] = time.perf_counter() - t_start
    res["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"{a.support} {a.norm} {a.dx_variant}: held-out start {res['heldout_compact_start']:.6f} -> final "
          f"{res['heldout_final']:.6f}" + (f" test {res['final_test']:.6f}" if a.final_test else ""), flush=True)


if __name__ == "__main__":
    main()
