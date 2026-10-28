"""EXPERIMENTAL grouped execution of a mixed fan-in local support (experimental/mixed_supports.py: K64 anchors,
K16 and K8 row groups inside the C256 K64 envelope; weighted-support study,
results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md section 5.2) on the EXISTING kgen kernels (bundle "kgen", K64 / K16 /
K8 at C = 16; no new CUDA code).

Packing: every fan-in type T keeps the original row groups of class T in canonical (ascending u = (g, j, c // 16))
order; inside output g they are the 16-neuron kgen blocks 0, 1, ... of a type layout ``KLayout(D, H_T, G, T, 16)``
whose per-output counts must EQUAL the actual per-output type populations (asserted; the primary allocation fits the
balanced counts: 768 K64, 4,608 K8 and 6,831 / 6,832 K16 neurons per output).  Only the last block of an output may
be partial (the original partial tail group); anything else is rejected (``Unsupported``) -- a neuron is never
padded, moved to another output or duplicated.  The record of a block is feat[q_u][env_u] (its original set q, its
sorted local subset), its weights / bias those of the original row group.  Per type: buckets = maximal runs of
consecutive blocks of one output in one set (a duplicate (q, g) bucket is rejected), KgExec work lists.

Function (the original pooling, not three KgenFunctions):

    y[b, g] = sum over ALL original valid neurons n of g of ReLU(z[b, g, n]) / n_original[g]

* forward: X^T is transposed ONCE; the three forward launches atomically add into ONE FP32 ysum [G, B]
  (the kgen kernels accumulate); ysum is divided once by the ORIGINAL counts.  The type-local counts are passed
  to the kernels only for physical validity (the kernels never divide by them);
* backward: one coefficient c1 = dy / n_original for every type; ONE FP32 dX^T buffer zeroed once, all three dX
  launches accumulate into it, and it is cast to the input dtype once at the end (no per-type BF16 rounding);
  each type's dW / db go to its own slices of the flat FP32 main_grad.

Parameters: ONE flat BF16 ``weight`` (the K64, K16, K8 packs back to back: [G NB_T 16 T] each) and ONE flat
``bias`` ([G NB_T 16] each), so the optimizer and the benchmark's state machinery see two tensors as for every other
format; padded slots stay exactly zero (no gradient reaches them; weight decay 0).  Canonical identity
(experimental/canonical.py): ``canon_rows`` scatters every original neuron's T values back into its 64-input
envelope row (``env`` maps, uint8; identity only, never read by a kernel).
"""
from __future__ import annotations

import json
import math

import torch
from torch import nn

from . import kgen_graph as KG
from .kgen_layout import KLayout
from .mixed_supports import CLASSES, MixedSupport, Unsupported


def _mark(name):
    from ..kernels import cuda_ops

    cuda_ops._mark(name)


class TypePack:
    """One fan-in type: geometry, records, buckets, offsets into the flat parameters."""

    def __init__(self, K, layout, counts, indices, buckets, env, u_of_block, w_off, b_off):
        self.K, self.layout, self.counts = K, layout, counts
        self.indices, self.buckets, self.env, self.u_of_block = indices, buckets, env, u_of_block
        self.w_off, self.b_off = w_off, b_off
        self.w_len = math.prod(layout.weight_shape)
        self.b_len = math.prod(layout.bias_shape)
        self.sg = None
        self.cfg = None
        self.dx_group_warps = None

    def tensors(self):
        return (self.indices, self.buckets, self.env, self.u_of_block, self.counts)


@torch.no_grad()
def pack_type(dense, sup: MixedSupport, K: int, w_off: int, b_off: int) -> TypePack:
    L = dense.layout
    dev = dense.weight.device
    G, NB = L.groups, L.blocks_per_group
    gpb = L.block // 16
    n16 = G * NB * gpb
    kc, nv = sup.kc.to(dev), sup.nv.to(dev)
    u = torch.nonzero((kc == K) & (nv > 0)).squeeze(1)  # ascending u: canonical order
    g = u // (NB * gpb)
    cnt_g = torch.zeros(G, dtype=torch.int64, device=dev).index_add_(0, g, nv[u])
    HT = int(cnt_g.sum())
    TL = KLayout(L.input_dim, HT, G, K, 16)
    want = TL.counts_tensor(dev).to(torch.int64)
    if not torch.equal(cnt_g, want):
        bad = int((cnt_g != want).sum())
        raise Unsupported(f"type K{K}: per-output populations do not fit the balanced KLayout counts ({bad} outputs "
                          f"differ; min {int(cnt_g.min())}, max {int(cnt_g.max())}); no neuron is moved or padded")
    ng = torch.bincount(g, minlength=G)
    first = torch.cumsum(ng, 0) - ng
    j = torch.arange(len(u), device=dev) - first[g]  # block index inside the output
    last = j == ng[g] - 1
    if bool((nv[u][~last] != 16).any()):
        raise Unsupported(f"type K{K}: a partial row group is not the last block of its output")
    NBT = TL.blocks_per_group
    fb = g * NBT + j
    u_of_block = torch.full((G * NBT,), -1, dtype=torch.int64, device=dev)
    u_of_block[fb] = u
    # records: feat[q_u][env_u]; padded blocks copy the output's first record (never computed)
    from .local_supports import block_sets

    bq = block_sets(L, dense.kg_buckets, dev)
    q = bq[u // gpb]
    gm = sup.gm.to(dev)[u]
    env = torch.nonzero(gm)[:, 1].view(len(u), K)
    feat = dense.sg_feat.to(torch.int64)
    rec = feat[q[:, None], env]
    idx = torch.empty((G * NBT, K), dtype=torch.int64, device=dev)
    first_rec = rec[first.clamp_max(len(u) - 1)]  # [G, K] (every output has >= 1 block of every non-empty type)
    idx[:] = first_rec.repeat_interleave(NBT, 0)
    idx[fb] = rec
    envb = torch.zeros((G * NBT, K), dtype=torch.int64, device=dev)
    envb[:] = env[first.clamp_max(len(u) - 1)].repeat_interleave(NBT, 0)
    envb[fb] = env
    # buckets: runs of consecutive blocks of one output in one set
    brk = torch.ones(len(u), dtype=torch.bool, device=dev)
    brk[1:] = (g[1:] != g[:-1]) | (q[1:] != q[:-1])
    starts = torch.nonzero(brk).squeeze(1)
    lens = torch.diff(torch.cat([starts, starts.new_tensor([len(u)])]))
    bk = torch.stack([q[starts], g[starts], j[starts], lens], 1)
    key = bk[:, 0] * G + bk[:, 1]
    if torch.unique(key).numel() != len(bk):
        raise Unsupported(f"type K{K}: an output has two non-adjacent runs in one set (duplicate (q, g) bucket)")
    bk = bk[torch.argsort(key)].to(torch.int32).contiguous()
    return TypePack(K, TL, want.to(torch.int32).contiguous(), idx.to(torch.int16).contiguous(), bk,
                    envb.to(torch.uint8).contiguous(), u_of_block, w_off, b_off)


class GroupedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, layer: "GroupedMixedLayer"):
        layer._runtime()
        L = layer.layout
        B = x.shape[0]
        layer._check_call(B)
        xc = x.to(torch.bfloat16).contiguous()
        wc, bc = w.contiguous(), b.contiguous()
        _mark("fwd.begin")
        xt = xc.t().contiguous()
        _mark("fwd.layout")
        ysum = torch.zeros((L.groups, B), device=x.device, dtype=torch.float32)
        _mark("fwd.zero")
        for pk in layer.packs:
            a = layer._args(pk, xt, wc, bc, B)
            a.items, a.chunks, a.ysum = pk.sg.items.data_ptr(), pk.sg.chunks.data_ptr(), ysum.data_ptr()
            n_tiles = math.ceil(B / (16 * pk.cfg[0]))
            layer.backend.launch(pk.K, "fwd", a, layer.level.act_id, 0, pk.sg.chunks.shape[0], n_tiles, x.device)
            _mark(f"fwd.t{pk.K}")
        _mark("fwd.kernel")
        pre = (ysum / layer.counts_f[:, None]).t().contiguous()  # ORIGINAL counts, once
        _mark("fwd.finalize")
        ctx.layer = layer
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(xc, w, b)
        layer.last_info.update(forward_backend="kgen-grouped", forward_kernel="kgen_grouped",
                               grouped_types=[pk.K for pk in layer.packs], launches_fwd=len(layer.packs))
        return pre

    @staticmethod
    def backward(ctx, dy):
        layer: GroupedMixedLayer = ctx.layer
        xc, w, b = ctx.saved_tensors
        B, D = xc.shape
        _mark("bwd.begin")
        c1T = (dy.float() / layer.counts_f[None, :]).t().contiguous()  # one coefficient for every type
        _mark("bwd.coef")
        xt = xc.t().contiguous()
        _mark("bwd.layout")
        dxt = torch.zeros((D, B), device=xc.device, dtype=torch.float32)  # shared FP32 accumulator
        _mark("bwd.dx_init")
        mg_w, mg_b = getattr(w, "main_grad", None), getattr(b, "main_grad", None)
        use_main = mg_w is not None and mg_b is not None
        dw = mg_w if use_main else torch.zeros(w.shape, device=w.device, dtype=torch.float32)
        db = mg_b if use_main else torch.zeros(b.shape, device=b.device, dtype=torch.float32)
        split = int(layer.plan.grad_precision == "split")
        wc, bc = w.contiguous(), b.contiguous()
        args = []
        for pk in layer.packs:
            a = layer._args(pk, xt, wc, bc, B)
            a.c1T, a.dXT = c1T.data_ptr(), dxt.data_ptr()
            a.dW = dw.data_ptr() + 4 * pk.w_off
            a.dB = db.data_ptr() + 4 * pk.b_off
            a.items, a.chunks, a.dx_group_warps = pk.sg.items.data_ptr(), pk.sg.chunks.data_ptr(), pk.dx_group_warps
            n_tiles = math.ceil(B / (16 * pk.cfg[2]))
            layer.backend.launch(pk.K, "dx", a, layer.level.act_id, split, pk.sg.chunks.shape[0], n_tiles, xc.device)
            _mark(f"bwd.dx.t{pk.K}")
            args.append(a)
        _mark("bwd.dx")
        for pk, a in zip(layer.packs, args):
            a.tasks, a.chunks = pk.sg.tasks.data_ptr(), pk.sg.dw_chunks.data_ptr()
            layer.backend.launch(pk.K, "dw", a, layer.level.act_id, split, pk.sg.dw_chunks.shape[0], 0, xc.device)
            _mark(f"bwd.dw.t{pk.K}")
        _mark("bwd.dw")
        layer.last_info.update(backward_backend="kgen-grouped", dx_kernel="kgen_grouped", dw_kernel="kgen_grouped",
                               grad_precision=layer.plan.grad_precision, launches_bwd=2 * len(layer.packs),
                               dx_accumulation="one FP32 dX^T for all types, one cast")
        dx = dxt.t().to(ctx.x_dtype, memory_format=torch.contiguous_format)  # the only dX cast
        gw = None if use_main else dw.to(w.dtype)
        gb = None if use_main else db.to(b.dtype)
        return dx, gw, gb, None


class GroupedMixedLayer(nn.Module):
    """Grouped K64 / K16 / K8 execution of a MixedSupport (module docstring)."""

    def __init__(self, dense, sup: MixedSupport, plan=None):
        super().__init__()
        from .kgx_sparse import valid_slots  # noqa: F401  (import check: same package state)

        L = dense.layout
        dev = dense.weight.device
        self.level, self.layout, self.connectivity, self.spec = dense.level, L, dense.connectivity, dense.spec
        self.plan = plan or dense.plan
        self.counts, self.counts_f = dense.counts, dense.counts_f
        self.init_rule, self.init_seed = getattr(dense, "init_rule", None), getattr(dense, "init_seed", None)
        self.graph_stats = dict(dense.graph_stats)
        self.register_buffer("sg_feat", dense.sg_feat)
        self.register_buffer("kg_buckets", dense.kg_buckets)
        self.register_buffer("kg_spec", dense.kg_spec)
        self.packs = []
        w_off = b_off = 0
        for K in CLASSES:
            if not bool(((sup.kc.to(dev) == K) & (sup.nv.to(dev) > 0)).any()):
                continue  # an entirely empty type is skipped
            pk = pack_type(dense, sup, K, w_off, b_off)
            self.packs.append(pk)
            w_off += pk.w_len
            b_off += pk.b_len
        wf = torch.zeros(w_off, dtype=torch.bfloat16, device=dev)
        bf = torch.zeros(b_off, dtype=torch.bfloat16, device=dev)
        wd = dense.weight.detach().reshape(-1, 16, 64)
        bd = dense.bias.detach().reshape(-1, 16)
        valid = L.valid_mask(dev).reshape(-1, 16)
        for pk in self.packs:
            ub = pk.u_of_block
            live = ub >= 0
            uu = ub[live]
            env = pk.env[live].long()
            v = torch.zeros((ub.numel(), 16, pk.K), dtype=torch.bfloat16, device=dev)
            v[live] = torch.gather(wd[uu], 2, env[:, None, :].expand(-1, 16, pk.K)) * valid[uu][:, :, None]
            bb = torch.zeros((ub.numel(), 16), dtype=torch.bfloat16, device=dev)
            bb[live] = bd[uu] * valid[uu]
            wf[pk.w_off:pk.w_off + pk.w_len] = v.reshape(-1)
            bf[pk.b_off:pk.b_off + pk.b_len] = bb.reshape(-1)
            for i, t in enumerate(pk.tensors()):
                self.register_buffer(f"t{pk.K}_{i}", t)
        self.weight = nn.Parameter(wf)
        self.bias = nn.Parameter(bf)
        # canonical map original neuron slot -> (pack, block) (identity only)
        n16 = L.groups * L.blocks_per_group * L.block // 16
        self.register_buffer("u_pack", torch.full((n16,), -1, dtype=torch.int8, device=dev))
        self.register_buffer("u_block", torch.full((n16,), -1, dtype=torch.int32, device=dev))
        for i, pk in enumerate(self.packs):
            live = pk.u_of_block >= 0
            self.u_pack[pk.u_of_block[live]] = i
            self.u_block[pk.u_of_block[live]] = torch.nonzero(live).squeeze(1).to(torch.int32)
        if bool((self.u_pack[sup.nv.to(dev) > 0] < 0).any()):
            raise AssertionError("a valid original row group is in no type pack")
        self.support_stats = dict(sup.stats)
        self.graph_stats.update(grouped=dict(types=[pk.K for pk in self.packs],
                                             neurons={f"K{pk.K}": int(pk.counts.sum()) for pk in self.packs},
                                             blocks_per_output={f"K{pk.K}": pk.layout.blocks_per_group for pk in self.packs},
                                             buckets={f"K{pk.K}": int(len(pk.buckets)) for pk in self.packs}))
        self._envelope_table = dense.kcsr_table() if hasattr(dense, "kcsr_table") else None
        self.backend = None
        self._rt_key = None
        self.last_info = {}

    @property
    def sg_buckets(self):  # benchmark compatibility (envelope buckets)
        return self.kg_buckets

    # ---- runtime
    def _runtime(self):
        key = (self.weight.device, self.packs[0].indices.data_ptr())
        if self._rt_key == key:
            return
        from ..backends.capabilities import query
        from ..backends.dispatch import kgen_dx_group_warps
        from .kgen_runtime import kgen_backend

        dev = self.weight.device
        self.backend = kgen_backend(dev)
        self.caps = query(dev)
        st = {}
        for pk in self.packs:
            pk.cfg = self.backend.cfgs[pk.K]
            pk.sg = KG.KgExec(pk.layout.blocks_per_group, 16, pk.K, pk.counts, self.sg_feat, pk.buckets, pk.cfg)
            pk.dx_group_warps = kgen_dx_group_warps(self.caps, self.layout.input_dim, self.sg_feat.shape[1], pk.cfg)
            st[f"K{pk.K}"] = dict(pk.sg.stats, dx_group_warps=pk.dx_group_warps)
        self._rt_key = key
        self.last_info["kgen_work_lists"] = st

    @property
    def _rt(self):  # benchmark compatibility (work-list bytes of all types)
        class _R:
            pass

        r = _R()
        r.sg = _AllSg(self.packs)
        return r

    def _check_call(self, B: int):
        from ..backends.dispatch import kgen_reason
        from ..config import resolve_reduction

        mode, why, _ = resolve_reduction(self.plan)
        for pk in self.packs:
            r = kgen_reason(self.caps, pk.K, 16, self.sg_feat.shape[1], self.layout.input_dim, B, self.level, self.plan,
                            mode == "deterministic", pk.cfg)
            if r:
                raise ValueError(f"grouped kgen layer cannot run this call (K{pk.K}): {r}")
        self.last_info["reduction"] = {"mode": "fast", "reason": why}
        self.last_info.setdefault("execution", dict(batch=B, grad_precision=self.plan.grad_precision,
                                                    types=[pk.K for pk in self.packs]))

    def _args(self, pk: TypePack, xt, w, b, B):
        from .kgen_runtime import KgArgs

        L = self.layout
        return KgArgs(xt.data_ptr(), xt.stride(0), pk.indices.data_ptr(), w.data_ptr() + 2 * pk.w_off,
                      b.data_ptr() + 2 * pk.b_off, pk.counts.data_ptr(), self.sg_feat.data_ptr(), 0, 0, 0, 0, 0, 0,
                      0, 0, B, L.groups, L.input_dim, self.sg_feat.shape[1], pk.layout.blocks_per_group, 16, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        y = GroupedFunction.apply(x.reshape(-1, x.shape[-1]), self.weight, self.bias, self)
        return y.reshape(*lead, y.shape[-1])

    def enable_main_grad(self):
        for p in (self.weight, self.bias):
            if getattr(p, "main_grad", None) is None:
                p.main_grad = torch.zeros(p.shape, device=p.device, dtype=torch.float32)

    def zero_main_grad(self):
        for p in (self.weight, self.bias):
            if getattr(p, "main_grad", None) is not None:
                p.main_grad.zero_()

    def select_variant(self, *a, **k):
        pass

    # ---- canonical identity (experimental/canonical.py)
    def canon_params(self):
        return {"weight": self.weight, "bias": self.bias}

    def _locate(self, g, n):
        L = self.layout
        s = g * (L.blocks_per_group * L.block) + n
        u, r = s // 16, s % 16
        return u, r, self.u_pack[u].long(), self.u_block[u].long()

    def canon_rows(self, src, g, n):
        u, r, p, blk = self._locate(g, n)
        W, Bv = src["weight"], src["bias"]
        rows = torch.zeros((len(u), 64), dtype=W.dtype, device=W.device)
        bias = torch.zeros(len(u), dtype=Bv.dtype, device=Bv.device)
        for i, pk in enumerate(self.packs):
            sel = p == i
            if not bool(sel.any()):
                continue
            bb, rr = blk[sel], r[sel]
            wv = W[pk.w_off:pk.w_off + pk.w_len].view(-1, 16, pk.K)
            rows[sel] = rows[sel].scatter(1, pk.env[bb].long(), wv[bb, rr])
            bias[sel] = Bv[pk.b_off:pk.b_off + pk.b_len].view(-1, 16)[bb, rr]
        return rows, bias

    def canon_mask_rows(self, g, n):
        u, r, p, blk = self._locate(g, n)
        m = torch.zeros((len(u), 64), dtype=torch.bool, device=u.device)
        for i, pk in enumerate(self.packs):
            sel = p == i
            if bool(sel.any()):
                m[sel] = m[sel].scatter(1, pk.env[blk[sel]].long(), True)
        return m

    # ---- graph identity / description (benchmark compatibility)
    def kcsr_table(self):
        return self._envelope_table

    def kcsr_validate(self):
        L = self.layout
        out = {}
        for pk in self.packs:
            out[f"K{pk.K}"] = KG.validate(L.input_dim, pk.counts, 16, pk.K, pk.indices.view(L.groups, -1, pk.K),
                                          self.sg_feat, pk.buckets, self.spec.R)
        return out

    def kcsr_describe(self):
        self._runtime()
        return dict(backend="kgen-grouped", status="experimental", envelope_K=64, envelope_C=self.layout.block,
                    types={f"K{pk.K}": dict(executed_K=pk.K, executed_C=16, neurons=int(pk.counts.sum()),
                                            NB=pk.layout.blocks_per_group, tiling=self.backend.cfg_dict(pk.K),
                                            dx_group_warps=pk.dx_group_warps, work_lists=dict(pk.sg.stats),
                                            index_bytes=pk.indices.numel() * 2)
                           for pk in self.packs},
                    spec=json.loads(self.spec.to_json()), build=self.backend.build_info(),
                    numerics="per type: the kgen numerics (BF16 operands, FP32 accumulation, split dZ); one shared FP32 "
                             "ysum / dX^T across types, original pooling counts, one final dX cast")

    def immutable_fingerprints(self):
        from .kgx_sparse import _fp_tensor

        out = dict(feat=list(_fp_tensor(self.sg_feat)), u_pack=list(_fp_tensor(self.u_pack)),
                   u_block=list(_fp_tensor(self.u_block)))
        for pk in self.packs:
            out[f"K{pk.K}"] = [list(_fp_tensor(t)) for t in pk.tensors()]
        return out

    def extra_persistent_bytes(self) -> dict:
        d = {}
        for pk in self.packs:
            d[f"index_records_int16_K{pk.K}"] = pk.indices.numel() * 2
            d[f"buckets_K{pk.K}"] = pk.buckets.numel() * 4
            d[f"type_counts_K{pk.K}"] = pk.counts.numel() * 4
            d[f"canonical_env_map_uint8_K{pk.K}"] = pk.env.numel()
            d[f"block_to_group_map_int64_K{pk.K}"] = pk.u_of_block.numel() * 8
            if pk.sg is not None:
                d[f"work_lists_K{pk.K}"] = pk.sg.runtime_bytes()
        d["canonical_group_maps"] = self.u_pack.numel() + self.u_block.numel() * 4
        d["padded_weight_slots"] = int(self.weight.numel()) - sum(int(pk.counts.sum()) * pk.K for pk in self.packs)
        return d

    def executed_work(self, B: int, split: bool = True) -> dict:
        tot = dict(fwd=0, dx=0, dw=0)
        per = {}
        for pk in self.packs:
            cnt = pk.counts.to(torch.int64)
            j = torch.arange(pk.layout.blocks_per_group, device=cnt.device)[None, :]
            nvb = torch.clamp(cnt[:, None] - 16 * j, 0, 16)
            rows8 = int(((nvb + 7) // 8 * 8).sum())
            nb = int(((cnt + 15) // 16).sum())
            sp = 2 if split else 1
            f = rows8 * pk.K * B
            e = dict(fwd=f, dx=f * (1 + sp), dw=16 * pk.K * nb * (1 + sp) * B)
            per[f"K{pk.K}"] = e
            for k in tot:
                tot[k] += e[k]
        return dict(batch=B, split=split, mma_products=tot, per_type=per,
                    fwd_executed_density=tot["fwd"] / B / (64 * self.layout.hidden_dim),
                    note="estimate from the kgen block structure of every type (rows rounded to 8 per block)")


class _AllSg:
    def __init__(self, packs):
        self.packs = packs
        self.stats = {f"K{p.K}": dict(p.sg.stats) for p in packs if p.sg is not None}

    def runtime_bytes(self):
        return sum(p.sg.runtime_bytes() for p in self.packs if p.sg is not None)


@torch.no_grad()
def lower_mixed(dense, sup: MixedSupport, plan=None) -> GroupedMixedLayer:
    lay = GroupedMixedLayer(dense, sup, plan)
    lay.graph_stats.update(lowered_from_policy=sup.policy, local_support=sup.stats.get("support_fingerprint"),
                           nominal_envelope=dict(K=64, C=dense.layout.block),
                           executed=dict(K=[pk.K for pk in lay.packs], C=16, sharing="16 neurons per record"))
    return lay
