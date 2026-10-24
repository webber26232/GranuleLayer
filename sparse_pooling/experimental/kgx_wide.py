"""EXPERIMENTAL width-preserving frozen-support layers of the K = 64 S = K index-free layout
(results/WIDTH_PRESERVING_SPARSITY.md; kernels kernels/cuda/experimental/kgx/kgx_wide.cuh).

``KgxWideLayer(dense, maskset)`` stores a support selected by ``width_masks.build`` compactly:

* tiles of width b = 16 or 8 per 16-neuron row group: values ``weight`` [n_tiles, 16, b] (the trainable
  parameter), ``tmeta`` [G NB C / 16] int32 = (first compact tile << NKT) | NKT-bit tile mask;
* optionally (H2 / H2A) a protected two-input backbone per row group: ``bbk`` [G NB C / 16] int32 local
  coordinates a0 | a1 << 8 and ``bb`` [G NB C, 2] BF16 weights per neuron slot (trainable; padding 0).
  Extra tiles overlapping a backbone coordinate keep a STRUCTURAL ZERO there (``tvmask`` = 0: the
  optimizer keeps value, moments and update at zero), so every enabled edge has exactly one trainable
  copy; W_eff (``dense_params``) is the single canonical weight.

Every valid neuron keeps its bias, its activation and the pooling denominator n_g; neurons without any
enabled input (only possible without a backbone) are bias-only rows handled exactly outside the kernels
(as KgxTileLayer).  Variants: "tile8" (b = 8), "tilew16" (b = 16 on the new kernels; cross-check of the
measured "tile" kernels), "hyb16" (backbone + b = 16).  Never selected by SparsePoolingLayer.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .kgx_layer import KgxLayer, _mark
from .kgx_sparse import KgxTileLayer, _fp_tensor, valid_slots
from .width_masks import MaskSet

WIDE_VARIANTS = {"tilew16": (16, False), "tile8": (8, False), "hyb16": (16, True)}


def variant_for(tb: int, backbone: bool) -> str:
    for k, v in WIDE_VARIANTS.items():
        if v == (tb, backbone):
            return k
    raise ValueError(f"no wide kgx variant for tile width {tb} with backbone {backbone}")


class KgwFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, bbw, layer: "KgxWideLayer"):
        rt = layer._runtime()
        L = layer.layout
        B = x.shape[0]
        layer._check_call(B)
        xc = x.to(torch.bfloat16).contiguous()
        _mark("fwd.begin")
        xt = xc.t().contiguous()
        _mark("fwd.layout")
        ysum = torch.zeros((L.groups, B), device=x.device, dtype=torch.float32)
        _mark("fwd.zero")
        wa = layer._wargs(rt, xt, w.contiguous(), b.contiguous(), bbw, B)
        a = wa.a
        a.items, a.chunks, a.ysum = rt.sg.items.data_ptr(), rt.sg.chunks.data_ptr(), ysum.data_ptr()
        tok = layer._tokens("fwd", rt)
        n_tiles = math.ceil(B / tok)
        rt.backend.launch_wide("fwd", wa, 0, rt.sg.chunks.shape[0], n_tiles, x.device)
        _mark("fwd.kernel")
        if rt.sg.bias_only is not None:  # rows without any enabled input: z = bias, exactly
            slot, gg = rt.sg.bias_only
            const = torch.zeros(L.groups, device=x.device, dtype=torch.float32).index_add_(
                0, gg, torch.relu(b.reshape(-1)[slot].float()))
            _mark("fwd.bias_only")
            pre = ((ysum + const[:, None]) / layer.counts_f[:, None]).t().contiguous()
        else:
            pre = (ysum / layer.counts_f[:, None]).t().contiguous()
        _mark("fwd.finalize")
        ctx.layer = layer
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(xc, w, b, bbw)
        layer.last_info.update(forward_backend="kgx", forward_kernel=f"kgx_{layer.kvariant}",
                               kgx_fwd=dict(ctas=rt.sg.chunks.shape[0] * n_tiles, token_tiles=n_tiles,
                                            chunks=rt.sg.chunks.shape[0], items=rt.sg.items.shape[0], tokens=tok,
                                            launches=1))
        return pre

    @staticmethod
    def backward(ctx, dy):
        layer: KgxWideLayer = ctx.layer
        xc, w, b, bbw = ctx.saved_tensors
        rt = layer._runtime()
        B, D = xc.shape
        _mark("bwd.begin")
        c1T = (dy.float() / layer.counts_f[None, :]).t().contiguous()
        _mark("bwd.coef")
        xt = xc.t().contiguous()
        _mark("bwd.layout")
        dxt = torch.zeros((D, B), device=xc.device, dtype=torch.float32)
        _mark("bwd.dx_init")
        ps = [w, b] + ([bbw] if layer.backbone else [])
        use_main = all(getattr(p, "main_grad", None) is not None for p in ps)
        if use_main:
            dw, db = w.main_grad, b.main_grad
            dbb = bbw.main_grad if layer.backbone else None
        else:
            dw = torch.zeros(w.shape, device=w.device, dtype=torch.float32)
            db = torch.zeros(b.shape, device=b.device, dtype=torch.float32)
            dbb = torch.zeros(bbw.shape, device=b.device, dtype=torch.float32) if layer.backbone else None
        wa = layer._wargs(rt, xt, w.contiguous(), b.contiguous(), bbw, B, dbb)
        a = wa.a
        a.c1T, a.dXT, a.dW, a.dB = c1T.data_ptr(), dxt.data_ptr(), dw.data_ptr(), db.data_ptr()
        a.items, a.chunks, a.dx_group_warps = rt.sg.items.data_ptr(), rt.sg.chunks.data_ptr(), rt.dx_group_warps
        split = int(layer.plan.grad_precision == "split")
        tok = layer._tokens("dx", rt)
        n_tiles = math.ceil(B / tok)
        rt.backend.launch_wide("dx", wa, split, rt.sg.chunks.shape[0], n_tiles, xc.device)
        _mark("bwd.dx")
        a.tasks, a.chunks = rt.sg.tasks.data_ptr(), rt.sg.dw_chunks.data_ptr()
        rt.backend.launch_wide("dw", wa, split, rt.sg.dw_chunks.shape[0], 0, xc.device)
        _mark("bwd.dw")
        if rt.sg.bias_only is not None:
            slot, gg = rt.sg.bias_only
            csum = c1T.sum(1)
            bo = b.detach().reshape(-1)[slot].float()
            db.view(-1).index_add_(0, slot, csum[gg] * (bo > 0).float())
            _mark("bwd.bias_only")
        layer.last_info.update(backward_backend="kgx", dx_kernel=f"kgx_{layer.kvariant}",
                               dw_kernel=f"kgx_{layer.kvariant}", grad_precision=layer.grad_scheme(),
                               grad_precision_requested=layer.plan.grad_precision,
                               kgx_dx=dict(ctas=rt.sg.chunks.shape[0] * n_tiles, tokens=tok,
                                           group_warps=rt.dx_group_warps, reduction=layer._dx_reduction(rt),
                                           launches=1),
                               kgx_dw=dict(ctas=rt.sg.dw_chunks.shape[0], tasks=rt.sg.tasks.shape[0], launches=1))
        dx = dxt.t().to(ctx.x_dtype, memory_format=torch.contiguous_format)
        if use_main:
            return dx, None, None, None, None
        return dx, dw.to(w.dtype), db.to(b.dtype), (dbb.to(bbw.dtype) if layer.backbone else None), None


class KgxWideLayer(KgxTileLayer):
    """Frozen-support b16 / b8 tile layer, optionally with a protected two-input backbone (module docstring)."""

    kvariant_family = tuple(WIDE_VARIANTS)

    def __init__(self, dense: KgxLayer, ms: MaskSet, variant: str = ""):
        nn.Module.__init__(self)
        self._copy_graph(dense)
        L = dense.layout
        n16 = L.groups * L.blocks_per_group * L.block // 16
        tb, nkt = ms.tb, ms.policy.nkt
        tm = ms.tiles
        if tm.shape != (n16, nkt):
            raise ValueError("tile mask shape")
        self.tb, self.nkt = tb, nkt
        self.backbone = ms.bb is not None
        bits = (tm.to(torch.int32) << torch.arange(nkt, device=tm.device, dtype=torch.int32)).sum(1)
        cnt = tm.sum(1).to(torch.int64)
        first = torch.cumsum(cnt, 0) - cnt
        if int(cnt.sum()) >= (1 << (31 - nkt)):
            raise ValueError(f"too many tiles ({int(cnt.sum())}) for the {31 - nkt}-bit compact tile index")
        self.register_buffer("tmeta", ((first << nkt) | bits).to(torch.int32).contiguous())
        self.register_buffer("tile_mask_bool", tm.contiguous())
        self.register_buffer("anchor_bool", ms.anchor.contiguous())
        self.register_buffer("mandatory_bool", ms.mandatory.contiguous())
        wv = dense.weight.detach().view(n16, 16, nkt, tb).permute(0, 2, 1, 3)  # [n16, kt, neuron, k]
        vals = wv[tm].contiguous()  # [n_tiles, 16, tb]
        self.mask_stats = dict(ms.stats)
        if self.backbone:
            bbk = (ms.bb[:, 0] | (ms.bb[:, 1] << 8)).to(torch.int32)
            self.register_buffer("bbk", bbk.contiguous())
            self.register_buffer("bb_coords", ms.bb.to(torch.int16).contiguous())
            wd = dense.weight.detach().view(n16, 16, 64)
            ar = torch.arange(n16, device=wd.device)
            bbv = torch.stack([wd[ar, :, ms.bb[:, j]] for j in range(2)], -1)  # [n16, 16, 2]
            valid = valid_slots(dense).reshape(n16, 16, 1)
            self.bb = nn.Parameter((bbv * valid).reshape(-1, 2).contiguous())
            # structural zeros of the extra tiles at the backbone coordinates (one trainable copy per edge)
            tv = torch.ones((n16, nkt, 16, tb), dtype=torch.uint8, device=tm.device)
            for j in range(2):
                tv[ar, ms.bb[:, j] // tb, :, ms.bb[:, j] % tb] = 0
            tvm = tv[tm].contiguous()
            vals = vals * tvm.to(vals.dtype)
            self.register_buffer("tvmask", tvm if bool((tvm == 0).any()) else None)
        else:
            self.register_buffer("bbk", None)
            self.register_buffer("bb_coords", None)
            self.bb = None
            self.register_buffer("tvmask", None)
        self.weight = nn.Parameter(vals)
        self.bias = nn.Parameter(dense.bias.detach().clone())
        self.select_variant(variant or variant_for(tb, self.backbone), dense.tile)

    def select_variant(self, variant, tile=None):
        if variant not in WIDE_VARIANTS or WIDE_VARIANTS[variant] != (self.tb, self.backbone):
            raise ValueError(f"KgxWideLayer (tile width {self.tb}, backbone {self.backbone}) cannot run {variant!r}")
        self.kvariant = variant
        if tile is not None:
            self.tile = tile
        self.last_info["execution"] = None
        return self

    def group_keep(self):
        if self.backbone:
            return None  # every valid row group has its backbone: all run
        return self.tile_mask_bool.any(1)

    def run_items(self) -> bool:
        return True if self.backbone else super().run_items()

    def _wargs(self, rt, xt, w, b, bbw, B, dbb=None):
        from .kgx_runtime import KwArgs

        a = self._args(rt, xt, w, b, B)
        return KwArgs(a, self.bbk.data_ptr() if self.backbone else 0, bbw.data_ptr() if self.backbone else 0,
                      dbb.data_ptr() if dbb is not None else 0, self.tb, int(self.backbone))

    def enable_main_grad(self):
        for p in self.canon_params().values():
            if getattr(p, "main_grad", None) is None:
                p.main_grad = torch.zeros(p.shape, device=p.device, dtype=torch.float32)

    def zero_main_grad(self):
        for p in self.canon_params().values():
            if getattr(p, "main_grad", None) is not None:
                p.main_grad.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        bbw = self.bb if self.backbone else self.bias  # (placeholder tensor without a backbone; unused)
        y = KgwFunction.apply(x.reshape(-1, x.shape[-1]), self.weight, self.bias, bbw, self)
        return y.reshape(*lead, y.shape[-1])

    def grad_scheme(self) -> str:
        if self.backbone:
            return f"mixed: tiles {self.plan.grad_precision} BF16 MMA dZ, backbone fp32_scalar dz"
        return self.plan.grad_precision

    def numerics(self) -> str:
        p = "dZ as BF16 hi + lo pair" if self.plan.grad_precision == "split" else "dZ as one BF16"
        s = (f"BF16 x / W / bias operands; {self.tb}-input tile products on tensor cores (FP32 accumulation"
             + (", m16n8k8 per 8-input tile" if self.tb == 8 else "") + ")")
        if self.backbone:
            s += ("; two backbone inputs per neuron: BF16 x BF16 products FP32-exact, FP32 FMA into the SAME "
                  "accumulator before one ReLU")
        s += f"; backward recomputes z; tile dX / dW products: {p}"
        if self.backbone:
            s += "; backbone dX / dv products: dz in FP32 on CUDA cores (mixed scalar / MMA scheme)"
        return s + "; FP32 atomics for cross-CTA sums"

    def _dx_reduction(self, rt) -> str:
        s = "register accumulation over records, warp-private slab copies (ordered sum), FP32 global atomics"
        return s + ("; backbone dA added into the warp's copy by its owning lane" if self.backbone else "")

    # ---- dense / canonical views
    def _tile_dense(self, src_w, n16):
        L = self.layout
        w = torch.zeros((n16, self.nkt, 16, self.tb), dtype=src_w.dtype, device=src_w.device)
        w[self.tile_mask_bool] = src_w
        return w.permute(0, 2, 1, 3).reshape(n16, 16, 64)

    def dense_params(self):
        L = self.layout
        n16 = L.groups * L.blocks_per_group * L.block // 16
        w = self._tile_dense(self.weight.detach(), n16)
        if self.backbone:
            ar = torch.arange(n16, device=w.device)
            bbv = self.bb.detach().view(n16, 16, 2)
            c = self.bb_coords.long()
            for j in range(2):
                w[ar, :, c[:, j]] = bbv[:, :, j]
        return w.reshape(L.weight_shape).contiguous(), self.bias.detach()

    def element_mask(self):
        L = self.layout
        n16 = L.groups * L.blocks_per_group * L.block // 16
        e = self.tile_mask_bool[:, None, :, None].expand(n16, 16, self.nkt, self.tb).reshape(n16, 16, 64).clone()
        if self.backbone:
            ar = torch.arange(n16, device=e.device)
            c = self.bb_coords.long()
            for j in range(2):
                e[ar, :, c[:, j]] = True
        valid = valid_slots(self).reshape(n16, 16, 1)
        return (e & valid).to(torch.uint8).view(L.weight_shape)

    def canon_params(self):
        d = {"weight": self.weight, "bias": self.bias}
        if self.backbone:
            d["bb"] = self.bb
        return d

    def canon_rows(self, src, g, n):
        from .canonical import dense_slots

        s = dense_slots(self, g, n)
        u, r = s // 16, s % 16
        tm = self.tile_mask_bool[u]
        first = (self.tmeta[u].to(torch.int64) >> self.nkt)
        rank = torch.cumsum(tm.to(torch.int64), 1) - tm.to(torch.int64)
        W = src["weight"]
        rows = torch.zeros((len(s), 64), dtype=W.dtype, device=W.device)
        for kt in range(self.nkt):
            sel = tm[:, kt]
            rows[sel, self.tb * kt:self.tb * kt + self.tb] = W[(first + rank[:, kt])[sel], r[sel]]
        if self.backbone:
            c = self.bb_coords[u].long()
            bv = src["bb"].reshape(-1, 2)[s].to(W.dtype)
            ar = torch.arange(len(s), device=W.device)
            for j in range(2):
                rows[ar, c[:, j]] = bv[:, j]
        return rows, src["bias"].reshape(-1)[s]

    def canon_mask_rows(self, g, n):
        from .canonical import dense_slots

        s = dense_slots(self, g, n)
        u = s // 16
        m = self.tile_mask_bool[u].repeat_interleave(self.tb, 1)
        if self.backbone:
            c = self.bb_coords[u].long()
            ar = torch.arange(len(s), device=m.device)
            m = m.clone()
            for j in range(2):
                m[ar, c[:, j]] = True
        return m

    # ---- bookkeeping
    def metadata_bytes(self):
        d = dict(tile_meta=self.tmeta.numel() * 4, compact_values=self.weight.numel() * 2,
                 active_tiles=int(self.weight.shape[0]), tile_width=self.tb)
        if self.backbone:
            d.update(backbone_coords=self.bbk.numel() * 4, backbone_values=self.bb.numel() * 2,
                     tile_value_mask=0 if self.tvmask is None else self.tvmask.numel())
        return d

    def extra_persistent_bytes(self):
        d = dict(sparse_metadata=self.tmeta.numel() * 4 + (self.bbk.numel() * 4 if self.backbone else 0) +
                 (0 if self.tvmask is None else self.tvmask.numel()),
                 sparse_format=f"{self.kvariant} (16 x {self.tb} tiles" + (" + 2-input backbone)" if self.backbone
                                                                         else ")"),
                 compact_values=self.weight.numel() * 2 + (self.bb.numel() * 2 if self.backbone else 0),
                 active_tiles=int(self.weight.shape[0]))
        if self.backbone:
            d["backbone_values"] = self.bb.numel() * 2
            d["backbone_main_grad"] = self.bb.numel() * 4
        return d

    def immutable_fingerprints(self):
        d = dict(tmeta=list(_fp_tensor(self.tmeta)), tile_mask=list(_fp_tensor(self.tile_mask_bool)),
                 anchor=list(_fp_tensor(self.anchor_bool)))
        if self.backbone:
            d["bbk"] = list(_fp_tensor(self.bbk))
        if self.tvmask is not None:
            d["tvmask"] = list(_fp_tensor(self.tvmask))
        return d

    def executed_work(self, B: int, split: bool = True) -> dict:
        """Executed tensor-core products and CUDA-core FMAs (multiply-adds) of one training step at batch B,
        from the kernels' loop structure (padding / structural zeros included)."""
        return executed_work(self, B, split)


@torch.no_grad()
def executed_work(layer, B: int, split: bool = True) -> dict:
    """Executed work of a dense KgxLayer (idf / tile families) or KgxWideLayer at batch B (per step):

    * fwd: per row group with an active tile (or a backbone), every 8-neuron n-tile holding a valid neuron
      runs its active tiles: 8 x b products per token and tile (MMA), + 2 backbone FMAs per row;
    * dX: the z recomputation (as fwd) + dA: (16 or 8 neurons) x b per active tile (x 2 split), + backbone
      recompute / dA (2 + 2 per row);
    * dW: z^T 16 x b per active tile + dW 16 x b (x 2 split) per token; backbone 2 + 2 per row (16 rows).
    Valid-slot products (the enabled edges) are reported alongside as rho-normalized densities."""
    L = layer.layout
    nv = L.valid_mask(layer.counts.device).reshape(-1, 16).sum(1).to(torch.int64)
    if hasattr(layer, "tile_mask_bool"):
        tm = layer.tile_mask_bool
        tb = getattr(layer, "tb", 16)
    else:
        tm = torch.ones((nv.numel(), 4), dtype=torch.bool, device=nv.device)
        tb = 16
    bbk = getattr(layer, "backbone", False)
    tiles = tm.sum(1)
    run = (nv > 0) & ((tiles > 0) | bbk)
    rows_f = torch.where(run, (nv + 7) // 8 * 8, torch.zeros_like(nv))
    two = torch.where(run, torch.where(nv > 8, 16, 8), torch.zeros_like(nv))
    rows_w = torch.where(run, torch.full_like(nv, 16), torch.zeros_like(nv))
    sp = 2 if split else 1
    fwd = int((rows_f * tiles).sum()) * tb
    dxz = fwd
    dxa = int((two * tiles).sum()) * tb * sp
    dwz = int((rows_w * tiles).sum()) * tb
    dww = dwz * sp
    sc = dict(fwd=0, dx=0, dw=0)
    if bbk:
        sc = dict(fwd=2 * int(rows_f.sum()), dx=4 * int(rows_f.sum()), dw=4 * int(rows_w.sum()))
    W0 = 64 * int(nv.sum())
    mma = dict(fwd=fwd * B, dx=(dxz + dxa) * B, dw=(dwz + dww) * B)
    return dict(batch=B, split=split, tile_width=tb, mma_products=mma, scalar_fma={k: v * B for k, v in sc.items()},
                mma_products_per_token=dict(fwd=fwd, dx_recompute=dxz, dx_da=dxa, dw_recompute=dwz, dw=dww),
                fwd_executed_density=(fwd + sc["fwd"]) / W0, row_groups_run=int(run.sum()),
                note="products = multiply-adds; dense reference: fwd density ~1 (+ partial n-tile padding)")


# ------------------------------------------------------------------ benchmark factory (k64_factory:wide)
FORMATS = ("dense", "dense_masked", "wide", "tile", "lowered")


def build_policy_layer(args, conn, plan):
    """Layer of one width-preserving policy built from a saved pre-pruning checkpoint (outside every timed
    region).  args.factory: checkpoint, format (FORMATS), policy (width_masks.POLICIES; omitted for "dense",
    the unpruned control), density, score ("magnitude" | "random"), seed, tile (physical neuron tile).

    The checkpoint's graph spec and weight / bias sha256 are verified; every format of the same (checkpoint,
    policy, density, score, seed) holds the same W_eff, mask and neuron identity (canonical hashes)."""
    import json
    import time

    from . import kgx_sparse as SP
    from . import width_masks as WM
    from .kgx_onetile import lower_one_tile

    f = args.factory
    t0 = time.perf_counter()
    rec, w, b, _ = SP.load_checkpoint(f["checkpoint"])
    ws, bs = SP.mask_fingerprint(w.view(torch.int16)), SP.mask_fingerprint(b.view(torch.int16))
    if ws != rec["weight_sha256"] or bs != rec["bias_sha256"]:
        raise SystemExit(f"checkpoint {f['checkpoint']}: weight / bias sha256 differ from its record")
    lay = KgxLayer(args.input_dim, args.hidden, args.groups, args.C, conn, variant="idf", tile=int(f.get("tile", 128)),
                   device="cuda", plan=plan, weights=(w, b), keep_indices=False)
    del w, b
    if json.loads(lay.spec.to_json()) != rec["spec"]:
        raise SystemExit(f"checkpoint graph spec {rec['spec']} != generated {lay.spec}")
    load_s = time.perf_counter() - t0
    fmt = f.get("format", "wide")
    if fmt not in FORMATS:
        raise SystemExit(f"format {fmt!r} (one of {FORMATS})")
    ck = dict(path=f["checkpoint"], weight_sha256=rec["weight_sha256"], bias_sha256=rec["bias_sha256"],
              file_sha256=rec.get("file_sha256"), tag=rec.get("meta", {}).get("tag"), verified=True,
              load_seconds=load_s)
    if fmt == "dense" or f.get("policy") in (None, "dense"):
        lay.mask = None
        lay.graph_stats.update(checkpoint=ck, policy="dense", format="dense", rho_mask=1.0)
        return lay
    L = lay.layout
    t0 = time.perf_counter()
    ms = WM.build(f["policy"], float(f["density"]), lay.weight.detach(), L, score=f.get("score", "magnitude"),
                  seed=int(f.get("seed", 0)))
    em = ms.element_mask(L)
    torch.cuda.synchronize()
    sel_s = time.perf_counter() - t0
    mask_sha = SP.mask_fingerprint(em)
    t0 = time.perf_counter()
    with torch.no_grad():
        lay.weight.data.mul_(em.to(lay.weight.dtype))
    if fmt == "dense_masked":
        out = lay
        out.mask = em
    elif fmt == "wide":
        out = KgxWideLayer(lay, ms)
    elif fmt == "tile":
        if ms.tb != 16 or ms.bb is not None:
            raise SystemExit("format 'tile' (the measured 16 x 16 kernels) needs a b16 policy without backbone")
        out = SP.KgxTileLayer(lay, ms.tiles)
    else:
        out = lower_one_tile(lay, ms, plan)
    torch.cuda.synchronize()
    conv_s = time.perf_counter() - t0
    out.mask_set_stats = ms.stats
    out.graph_stats.update(checkpoint=ck, policy=ms.stats["policy"], format=fmt, mask_stats=ms.stats,
                           rho_mask=ms.stats["rho_mask"], mask_sha256=mask_sha, select_seconds=sel_s,
                           conversion_seconds=conv_s)
    if fmt != "dense_masked":
        del lay, em
    torch.cuda.empty_cache()
    return out
