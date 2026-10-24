"""EXPERIMENTAL frozen-mask sparse representations of the K = 64 index-free layout (S = K = 64).

Masks (uint8 / bool over the dense [G, NB, C, 64] slots; padded slots always inactive):

* ``learned_mask``     the exact nonzero support of a checkpoint (L1 / proximal training);
* ``magnitude_mask``   global magnitude pruning over valid weights to an exact kept count (an additional
                       pruning policy, NOT the L1 treatment); ties of the BF16 threshold value are broken by
                       flat index (deterministic);
* ``random_mask``      density-matched control: each valid slot kept independently with probability rho
                       (seeded; actual density recorded);
* ``tile_mask``        structured: complete 16-neuron x 16-input tiles ranked by their L1 magnitude (exact
                       kept tile count); an all-zero 16-neuron group keeps its neurons, biases and db;
* ``random_tile_mask`` density-matched random tiles (control of the tile geometry).

Formats (all keep every valid neuron, its bias, its activation and the pooling denominator n_g):

* dense masked   KgxLayer ("idf"), inactive weights 0, L1AdamW mask (the reference of every sparse format);
* ``KgxTileLayer``  compact tiles [n_tiles, 16, 16] (the trainable parameter) + tmeta [G NB C / 16] int32
                 = (first compact tile << 4) | k16-tile mask; inactive tiles are never loaded / multiplied;
* ``KgxCsrLayer``   unstructured: vals [nnz] bf16 (the trainable parameter), kix [nnz] uint8 (local input),
                 rowptr [G NB C + 1] int32; CUDA-core kernels that touch only stored entries.

The compact values ARE the optimizer's parameter: an update never leaves a packed copy stale; only a mask
change needs re-compression (``refresh``: timed separately).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .kgx_layer import KgxLayer

CKPT_SCHEMA = "granule.k64_checkpoint/1"


# ------------------------------------------------------------------------------ masks
def valid_slots(layer) -> torch.Tensor:
    """[G, NB, C] bool valid neurons."""
    return layer.layout.valid_mask(layer.sg_feat.device)


@torch.no_grad()
def learned_mask(w: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return ((w != 0) & valid[..., None]).to(torch.uint8)


@torch.no_grad()
def magnitude_mask(w: torch.Tensor, valid: torch.Tensor, density: float) -> torch.Tensor:
    """Keep exactly round(density * W_valid) valid weights of largest |w| (BF16 magnitude histogram)."""
    vb = valid[..., None].expand_as(w)
    W0 = int(valid.sum()) * w.shape[-1]
    keep_n = int(round(density * W0))
    mag = (w.view(torch.int16) & 0x7FFF).to(torch.int32)  # BF16 |w| bit pattern: monotone in |w|
    mag = torch.where(vb, mag, torch.full_like(mag, -1))
    hist = torch.bincount((mag + 1).reshape(-1), minlength=0x8001)  # bin 0: invalid slots
    hist = hist[1:].flip(0)  # descending magnitude
    cum = torch.cumsum(hist, 0)
    i = int(torch.searchsorted(cum, torch.tensor(keep_n, device=cum.device)))  # bin where the count is reached
    thr = 0x7FFF - i
    above = mag > thr
    need = keep_n - int(above.sum())
    tie = torch.nonzero((mag == thr).reshape(-1)).squeeze(1)[:max(need, 0)]
    m = above.reshape(-1).clone()
    m[tie] = True
    out = m.view_as(w).to(torch.uint8)
    del mag
    return out


@torch.no_grad()
def random_mask(valid: torch.Tensor, density: float, seed: int, K: int = 64) -> torch.Tensor:
    g = torch.Generator(device=valid.device).manual_seed(seed)
    out = torch.empty(valid.shape + (K,), dtype=torch.uint8, device=valid.device)
    flat = out.view(valid.shape[0], -1)
    for g0 in range(valid.shape[0]):
        flat[g0] = (torch.rand(flat.shape[1], generator=g, device=valid.device) < density).to(torch.uint8)
    out &= valid[..., None].to(torch.uint8)
    return out


@torch.no_grad()
def tile_scores(w: torch.Tensor) -> torch.Tensor:
    """[G NB C / 16, 4] L1 magnitude of every (16-neuron group, k16 tile)."""
    n16 = w.numel() // (16 * 64)
    return w.view(n16, 16, 4, 16).float().abs().sum((1, 3))


@torch.no_grad()
def tile_mask(w: torch.Tensor, valid: torch.Tensor, density: float) -> torch.Tensor:
    """[G NB C / 16, 4] bool: the round(density * T) tiles of largest L1 magnitude among tiles that hold a
    valid neuron (ties: flat order)."""
    sc = tile_scores(w)
    has_valid = valid.reshape(-1, 16).any(1)
    sc = torch.where(has_valid[:, None], sc, torch.full_like(sc, -1.0))
    T = int(has_valid.sum()) * 4
    keep = int(round(density * T))
    order = torch.argsort(sc.reshape(-1), descending=True, stable=True)[:keep]
    m = torch.zeros(sc.numel(), dtype=torch.bool, device=w.device)
    m[order] = True
    return m.view_as(sc) & has_valid[:, None]


@torch.no_grad()
def random_tile_mask(valid: torch.Tensor, density: float, seed: int) -> torch.Tensor:
    """Density-matched random control of ``tile_mask``: round(density * T) tiles drawn uniformly (seeded) among
    the tiles that hold a valid neuron."""
    has_valid = valid.reshape(-1, 16).any(1)
    cand = torch.nonzero(has_valid[:, None].expand(-1, 4).reshape(-1)).squeeze(1)
    keep = int(round(density * cand.numel()))
    g = torch.Generator(device="cpu").manual_seed(seed)
    pick = cand[torch.randperm(cand.numel(), generator=g)[:keep].to(cand.device)]
    m = torch.zeros(has_valid.numel() * 4, dtype=torch.bool, device=valid.device)
    m[pick] = True
    return m.view(-1, 4)


def tiles_to_elements(tm: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Element mask [G, NB, C, 64] uint8 of a tile mask (valid neurons only)."""
    n16 = tm.shape[0]
    e = tm[:, None, :, None].expand(n16, 16, 4, 16).reshape(valid.shape + (64,))
    return (e & valid[..., None]).to(torch.uint8)


def mask_fingerprint(m: torch.Tensor) -> str:
    h = hashlib.sha256()
    t = m.contiguous().view(-1)
    for i in range(0, t.numel(), 1 << 28):
        h.update(t[i:i + (1 << 28)].cpu().numpy().tobytes())
    return h.hexdigest()


def _fp_tensor(t: torch.Tensor):
    from importlib import import_module

    try:
        tf = import_module("train_state").tensor_fingerprint
    except ImportError:  # outside benchmarks/
        b = t.detach().reshape(-1)
        return (int(b.view(torch.uint8).sum(dtype=torch.int64)),)
    return tf(t)


# ------------------------------------------------------------------------------ layers
class _SparseBase(KgxLayer):
    kvariant_family = ("tile", "csr")

    def _init_sparse(self, dense: KgxLayer):
        self.__dict__["_dense_shapes"] = (tuple(dense.weight.shape), tuple(dense.bias.shape))

    def bench_variants(self):
        return [self.kvariant]

    def enable_main_grad(self):
        for p in (self.weight, self.bias):
            if getattr(p, "main_grad", None) is None:
                p.main_grad = torch.zeros(p.shape, device=p.device, dtype=torch.float32)

    def index_bytes(self) -> dict:
        d = super().index_bytes()
        d.update(self.metadata_bytes())
        return d


class KgxTileLayer(_SparseBase):
    """Tile-sparse frozen-mask layer: weight = compact active tiles [n_tiles, 16, 16]."""

    def __init__(self, dense: KgxLayer, tmask: torch.Tensor):
        nn.Module.__init__(self)
        self._copy_graph(dense)
        L = dense.layout
        n16 = L.groups * L.blocks_per_group * L.block // 16
        if tmask.shape != (n16, 4):
            raise ValueError("tile mask shape")
        tm = tmask.to(torch.int32)
        bits = tm[:, 0] | (tm[:, 1] << 1) | (tm[:, 2] << 2) | (tm[:, 3] << 3)
        cnt = tm.sum(1)
        first = torch.cumsum(cnt, 0) - cnt
        if int(cnt.sum()) >= (1 << 27):
            raise ValueError("too many tiles for the 27-bit tile index")
        self.register_buffer("tmeta", ((first << 4) | bits).to(torch.int32).contiguous())
        self.register_buffer("tile_mask_bool", tmask.contiguous())
        w = dense.weight.detach().view(n16, 16, 4, 16).permute(0, 2, 1, 3)  # [n16, kt, neuron, k]
        self.weight = nn.Parameter(w[tmask].contiguous())  # [n_tiles, 16, 16]
        self.bias = nn.Parameter(dense.bias.detach().clone())
        self.select_variant("tile", dense.tile)

    def _copy_graph(self, dense: KgxLayer):
        for k in ("level", "layout", "plan", "connectivity", "spec", "index_free_ok", "counts", "counts_f",
                  "init_rule", "init_seed", "graph_stats"):
            setattr(self, k, getattr(dense, k))
        self.graph_stats = dict(dense.graph_stats)
        self.register_buffer("indices", None)
        self.register_buffer("sg_feat", dense.sg_feat)
        self.register_buffer("kg_buckets", dense.kg_buckets)
        self.register_buffer("kg_spec", dense.kg_spec)
        self.tile = dense.tile
        self._rt = None
        self._rt_key = None
        self.last_info = {}
        self.kvariant = None
        if not self.index_free_ok:
            raise ValueError("sparse kgx formats need S = K = 64")

    def select_variant(self, variant, tile=None):
        if variant != "tile":
            raise ValueError("KgxTileLayer runs variant 'tile' only")
        self.kvariant = "tile"
        if tile is not None:
            self.tile = tile
        self.last_info["execution"] = None
        return self

    def _wptrs(self, w):
        return w.data_ptr(), self.tmeta.data_ptr(), 0, 0

    def group_keep(self):
        return self.tile_mask_bool.any(1)

    RUN_ITEMS_BELOW = 0.5  # kept-group fraction below which items are runs of kept groups (else: standard items)

    def run_items(self) -> bool:
        return float(self.group_keep().double().mean()) < self.RUN_ITEMS_BELOW

    def dense_params(self):
        L = self.layout
        n16 = L.groups * L.blocks_per_group * L.block // 16
        w = torch.zeros((n16, 4, 16, 16), dtype=self.weight.dtype, device=self.weight.device)
        w[self.tile_mask_bool] = self.weight.detach()
        return w.permute(0, 2, 1, 3).reshape(L.weight_shape).contiguous(), self.bias.detach()

    def element_mask(self):
        return tiles_to_elements(self.tile_mask_bool, valid_slots(self))

    def canon_rows(self, src, g, n):
        from .canonical import dense_slots

        s = dense_slots(self, g, n)
        u, r = s // 16, s % 16
        tm = self.tile_mask_bool[u]
        first = (self.tmeta[u] >> 4).to(torch.int64)
        rank = torch.cumsum(tm.to(torch.int64), 1) - tm.to(torch.int64)
        W = src["weight"]
        rows = torch.zeros((len(s), 64), dtype=W.dtype, device=W.device)
        for kt in range(4):
            sel = tm[:, kt]
            rows[sel, 16 * kt:16 * kt + 16] = W[(first + rank[:, kt])[sel], r[sel]]
        return rows, src["bias"].reshape(-1)[s]

    def canon_mask_rows(self, g, n):
        from .canonical import dense_slots

        s = dense_slots(self, g, n)
        return self.tile_mask_bool[s // 16].repeat_interleave(16, 1)

    def metadata_bytes(self):
        return dict(tile_meta=self.tmeta.numel() * 4, compact_values=self.weight.numel() * 2,
                    active_tiles=int(self.weight.shape[0]))

    def extra_persistent_bytes(self):
        return dict(sparse_metadata=self.tmeta.numel() * 4, sparse_format="tile16x16",
                    compact_values=self.weight.numel() * 2, active_tiles=int(self.weight.shape[0]))

    def immutable_fingerprints(self):
        return dict(tmeta=list(_fp_tensor(self.tmeta)), tile_mask=list(_fp_tensor(self.tile_mask_bool)))


class KgxCsrLayer(KgxTileLayer):
    """Unstructured frozen-mask layer: weight = vals [nnz] in (neuron slot, input) order."""

    def __init__(self, dense: KgxLayer, emask: torch.Tensor, chunk_slots: int = 1 << 22):
        nn.Module.__init__(self)
        self._copy_graph(dense)
        L = dense.layout
        slots = L.groups * L.blocks_per_group * L.block
        m = (emask.view(slots, 64) != 0) & valid_slots(dense).reshape(slots, 1)
        cnt = m.sum(1, dtype=torch.int32)
        rowptr = torch.zeros(slots + 1, dtype=torch.int64, device=m.device)
        rowptr[1:] = torch.cumsum(cnt, 0)
        nnz = int(rowptr[-1])
        if nnz >= (1 << 31):
            raise ValueError("nnz >= 2^31")
        w = dense.weight.detach().view(slots, 64)
        vals = torch.empty(nnz, dtype=w.dtype, device=w.device)
        kix = torch.empty(nnz, dtype=torch.uint8, device=w.device)
        ar = torch.arange(64, device=w.device, dtype=torch.uint8)
        for s0 in range(0, slots, chunk_slots):
            s1 = min(slots, s0 + chunk_slots)
            mc = m[s0:s1]
            e0, e1 = int(rowptr[s0]), int(rowptr[s1])
            vals[e0:e1] = w[s0:s1][mc]
            kix[e0:e1] = ar.expand(s1 - s0, 64)[mc]
        self.register_buffer("rowptr", rowptr.to(torch.int32).contiguous())
        self.register_buffer("kix", kix.contiguous())
        self.weight = nn.Parameter(vals)
        self.bias = nn.Parameter(dense.bias.detach().clone())
        self.select_variant("csr", dense.tile)

    def select_variant(self, variant, tile=None):
        if variant != "csr":
            raise ValueError("KgxCsrLayer runs variant 'csr' only")
        self.kvariant = "csr"
        if tile is not None:
            self.tile = tile
        self.last_info["execution"] = None
        return self

    def _wptrs(self, w):
        return w.data_ptr(), 0, self.rowptr.data_ptr(), self.kix.data_ptr()

    # The CSR kernels (kgx_csr.cuh) multiply on CUDA cores with dz in FP32: no BF16 / split rounding of dz
    # exists in them, so plan.grad_precision does not select anything.  A "bf16" request (coarser dz) is
    # rejected rather than silently exceeded; the default "split" request is accepted and the EXECUTED
    # scheme ("fp32_scalar") is what records and numerics report (audit correction of the K64 study, whose
    # inherited description said split BF16).
    def grad_scheme(self) -> str:
        return "fp32_scalar"

    def _check_call(self, B: int):
        if self.plan.grad_precision == "bf16":
            raise ValueError("kgx csr computes dz in FP32 (CUDA cores): a BF16-dZ request (grad_precision='bf16') "
                             "cannot be honoured")
        return super()._check_call(B)

    def numerics(self) -> str:
        return ("BF16 x / W / bias operands; CUDA-core FP32 products of the stored entries only, FP32 accumulation "
                "of z before one ReLU (4-way interleaved partial sums); backward recomputes z; dz in FP32 (no BF16 "
                "or split rounding) in the dX / dW products (FP32 accumulate); FP32 atomics for cross-CTA sums")

    def group_keep(self):
        rp = self.rowptr.to(torch.int64)
        return (rp[16::16] - rp[:-1:16]) > 0  # any entry among the group's 16 neuron slots

    def run_items(self) -> bool:
        return True  # the CSR kernels do not skip dropped groups themselves

    def dx_windows(self, rt):
        """(wins [n, 4] int32, dX chunks [m, 3]) cached per runtime (static: built from the frozen mask)."""
        if getattr(self, "_win_key", None) is not rt:
            t0 = time.perf_counter()
            self._wins = csr_windows(rt.sg, self.rowptr, self.layout.block, self.counts)
            self._win_key = rt
            self.last_info["csr_dx_windows"] = dict(windows=int(self._wins[0].shape[0]),
                                                    bytes=int(self._wins[0].numel() * 4 + self._wins[1].numel() * 4),
                                                    build_seconds=time.perf_counter() - t0)
        return self._wins

    def dense_params(self, chunk_slots: int = 1 << 22):
        L = self.layout
        slots = L.groups * L.blocks_per_group * L.block
        w = torch.zeros((slots, 64), dtype=self.weight.dtype, device=self.weight.device)
        rp = self.rowptr.to(torch.int64)
        for s0 in range(0, slots, chunk_slots):
            s1 = min(slots, s0 + chunk_slots)
            e0, e1 = int(rp[s0]), int(rp[s1])
            row = torch.repeat_interleave(torch.arange(s0, s1, device=w.device), (rp[s0 + 1:s1 + 1] - rp[s0:s1]))
            w[row, self.kix[e0:e1].long()] = self.weight.detach()[e0:e1]
        return w.view(L.weight_shape), self.bias.detach()

    def element_mask(self, chunk_slots: int = 1 << 22):
        L = self.layout
        slots = L.groups * L.blocks_per_group * L.block
        m = torch.zeros((slots, 64), dtype=torch.uint8, device=self.weight.device)
        rp = self.rowptr.to(torch.int64)
        for s0 in range(0, slots, chunk_slots):
            s1 = min(slots, s0 + chunk_slots)
            e0, e1 = int(rp[s0]), int(rp[s1])
            row = torch.repeat_interleave(torch.arange(s0, s1, device=m.device), (rp[s0 + 1:s1 + 1] - rp[s0:s1]))
            m[row, self.kix[e0:e1].long()] = 1
        return m.view(L.weight_shape)

    def _csr_entries(self, s):
        rp = self.rowptr.to(torch.int64)
        e0, cnt = rp[s], rp[s + 1] - rp[s]
        r = torch.repeat_interleave(torch.arange(len(s), device=s.device), cnt)
        e = torch.arange(len(r), device=s.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt) + e0[r]
        return r, e

    def canon_rows(self, src, g, n):
        from .canonical import dense_slots

        s = dense_slots(self, g, n)
        r, e = self._csr_entries(s)
        W = src["weight"]
        rows = torch.zeros((len(s), 64), dtype=W.dtype, device=W.device)
        rows[r, self.kix[e].long()] = W[e]
        return rows, src["bias"].reshape(-1)[s]

    def canon_mask_rows(self, g, n):
        from .canonical import dense_slots

        s = dense_slots(self, g, n)
        r, e = self._csr_entries(s)
        m = torch.zeros((len(s), 64), dtype=torch.bool, device=s.device)
        m[r, self.kix[e].long()] = True
        return m

    def metadata_bytes(self):
        return dict(rowptr=self.rowptr.numel() * 4, kix=self.kix.numel(), compact_values=self.weight.numel() * 2,
                    nnz=int(self.weight.numel()))

    def extra_persistent_bytes(self):
        return dict(sparse_metadata=self.rowptr.numel() * 4 + self.kix.numel(), sparse_format="csr",
                    compact_values=self.weight.numel() * 2, nnz=int(self.weight.numel()))

    def immutable_fingerprints(self):
        return dict(rowptr=list(_fp_tensor(self.rowptr)), kix=list(_fp_tensor(self.kix)))


KC_WE = 256  # entries per CSR neuron window (kgx_csr.cuh)


@torch.no_grad()
def csr_windows(sg, rowptr: torch.Tensor, C: int, counts: torch.Tensor):
    """Static dX neuron windows of a CSR layer: each item's neuron range cut greedily into windows of <= 31
    neurons and <= KC_WE entries (a neuron has <= 64), in item order; dX chunks = the items' chunks mapped to
    window ranges.  Returns (wins [n, 4] = (g, first slot, neurons | entries << 8, first entry), chunks)."""
    dev = rowptr.device
    it = sg.items.to(torch.int64)
    if it.shape[0] and int(it[:, 2].max()) != 1:
        raise ValueError("CSR windows expect single-block items")
    g, fb, cr = it[:, 0], it[:, 1], it[:, 3]
    c0, c1 = cr & 0xFFFF, cr >> 16
    rp = rowptr.to(torch.int64)
    # run items (KxExec group_keep path): c1 is already clipped to the block's valid neurons
    s_beg = fb * C + c0
    s_end = fb * C + c1
    s = s_beg.clone()
    active = s < s_end
    order = 0
    rec = []
    while bool(active.any()):
        ia = torch.nonzero(active).squeeze(1)
        sa = s[ia]
        cap = torch.clamp(s_end[ia] - sa, max=31)
        lim = torch.searchsorted(rp, rp[sa] + KC_WE, right=True) - 1 - sa  # neurons whose entries fit
        m = torch.clamp(torch.minimum(lim, cap), min=1)
        rec.append((ia, sa, m, torch.full_like(ia, order)))
        s[ia] = sa + m
        active = s < s_end
        order += 1
    if not rec:  # no kept group (all neurons bias-only)
        z = torch.zeros((0, 4), dtype=torch.int32, device=dev)
        return z, torch.zeros((0, 3), dtype=torch.int32, device=dev)
    ia = torch.cat([r[0] for r in rec])
    sa = torch.cat([r[1] for r in rec])
    m = torch.cat([r[2] for r in rec])
    od = torch.cat([r[3] for r in rec])
    key = ia * (order + 1) + od
    srt = torch.argsort(key)
    ia, sa, m = ia[srt], sa[srt], m[srt]
    e0 = rp[sa]
    ne = rp[sa + m] - e0
    wins = torch.stack([g[ia], sa, m | (ne << 8), e0], 1).to(torch.int32).contiguous()
    nwin = torch.bincount(ia, minlength=it.shape[0])
    woff = torch.zeros(it.shape[0] + 1, dtype=torch.int64, device=dev)
    woff[1:] = torch.cumsum(nwin, 0)
    ch = sg.chunks.to(torch.int64)
    wch = torch.stack([ch[:, 0], woff[ch[:, 1]], woff[ch[:, 2]]], 1).to(torch.int32).contiguous()
    return wins, wch


# ------------------------------------------------------------------------------ dense masked reference
def dense_masked(dense: KgxLayer, emask: torch.Tensor) -> KgxLayer:
    """Zero the inactive weights of a dense idf layer in place (its optimizer gets the mask: l1.apply_mask_)."""
    with torch.no_grad():
        dense.weight.data.mul_(emask.to(dense.weight.dtype))
    dense.mask = emask
    return dense


# ------------------------------------------------------------------------------ checkpoints
def save_checkpoint(path, layer: KgxLayer, mask: Optional[torch.Tensor], meta: dict) -> dict:
    """Dense weights / bias (+ uint8 mask packed to bits) and metadata; returns sha256 of the file and of the
    parameters / mask.  Checkpoints live outside git (large)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, b = layer.dense_params()
    rec = dict(schema=CKPT_SCHEMA, spec=json.loads(layer.spec.to_json()), C=layer.layout.block,
               H=layer.layout.hidden_dim, meta=meta, weight_sha256=mask_fingerprint(w.view(torch.int16)),
               bias_sha256=mask_fingerprint(b.view(torch.int16)),
               mask_sha256=None if mask is None else mask_fingerprint(mask))
    blob = dict(rec=rec, weight=w.cpu(), bias=b.cpu(),
                mask_bits=None if mask is None else _pack_bits(mask).cpu())
    t0 = time.perf_counter()
    torch.save(blob, path)
    rec["file_sha256"] = _file_sha(path)
    rec["save_seconds"] = time.perf_counter() - t0
    Path(str(path) + ".json").write_text(json.dumps(rec, indent=1))
    return rec


def load_checkpoint(path, device="cuda"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = blob["mask_bits"]
    w = blob["weight"].to(device)
    mask = None if m is None else _unpack_bits(m.to(device), w.numel()).view(w.shape)
    return blob["rec"], w, blob["bias"].to(device), mask


def _pack_bits(m: torch.Tensor) -> torch.Tensor:
    f = m.reshape(-1).to(torch.uint8)
    pad = (-f.numel()) % 8
    if pad:
        f = torch.cat([f, torch.zeros(pad, dtype=torch.uint8, device=f.device)])
    f = f.view(-1, 8)
    sh = torch.arange(8, device=f.device, dtype=torch.uint8)
    return (f << sh).sum(1, dtype=torch.int32).to(torch.uint8)


def _unpack_bits(b: torch.Tensor, n: int) -> torch.Tensor:
    sh = torch.arange(8, device=b.device, dtype=torch.uint8)
    return ((b[:, None] >> sh) & 1).reshape(-1)[:n].to(torch.uint8)


def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def build_from_checkpoint(args, conn, plan):
    """Benchmark factory (k64_factory:sparse): args.factory = dict(checkpoint, format = dense_masked | tile |
    csr, mask = checkpoint | magnitude | random | tile, density, mask_seed, tile)."""
    f = args.factory
    rec, w, b, mask = load_checkpoint(f["checkpoint"])
    lay = KgxLayer(args.input_dim, args.hidden, args.groups, args.C, conn, variant="idf", tile=int(f.get("tile", 128)),
                   device="cuda", plan=plan, weights=(w, b), keep_indices=False)
    if json.loads(lay.spec.to_json()) != rec["spec"]:
        raise SystemExit(f"checkpoint graph spec {rec['spec']} != generated {lay.spec}")
    valid = valid_slots(lay)
    kind = f.get("mask", "checkpoint")
    t0 = time.perf_counter()
    tm = None
    if kind == "checkpoint_dense":  # the checkpoint's values, dense (no mask): dense / dense + L1 controls
        lay.mask = None
        lay.graph_stats.update(checkpoint=rec, mask_kind=kind, format="dense", mask_density=None)
        return lay
    if kind == "checkpoint":
        if mask is None:
            mask = learned_mask(lay.weight.detach(), valid)
    elif kind == "learned":
        mask = learned_mask(lay.weight.detach(), valid)
    elif kind == "magnitude":
        mask = magnitude_mask(lay.weight.detach(), valid, float(f["density"]))
    elif kind == "random":
        mask = random_mask(valid, float(f["density"]), int(f.get("mask_seed", 0)))
    elif kind == "tile":
        tm = tile_mask(lay.weight.detach(), valid, float(f["density"]))
        mask = tiles_to_elements(tm, valid)
    elif kind == "random_tile":
        tm = random_tile_mask(valid, float(f["density"]), int(f.get("mask_seed", 0)))
        mask = tiles_to_elements(tm, valid)
    else:
        raise SystemExit(f"mask {kind!r}")
    mask_s = time.perf_counter() - t0
    fmt = f.get("format", "dense_masked")
    t0 = time.perf_counter()
    with torch.no_grad():
        lay.weight.data.mul_(mask.to(lay.weight.dtype))
    if fmt == "dense_masked":
        out = lay
        out.mask = mask
    elif fmt == "tile":
        if tm is None:
            raise SystemExit("format tile needs mask = tile (a tile mask)")
        out = KgxTileLayer(lay, tm)
        out.mask = None
    elif fmt == "csr":
        out = KgxCsrLayer(lay, mask)
        out.mask = None
    else:
        raise SystemExit(f"format {fmt!r}")
    torch.cuda.synchronize()
    conv_s = time.perf_counter() - t0
    W0 = int(valid.sum()) * 64
    nnz = int(mask.sum(dtype=torch.int64))
    out.graph_stats.update(checkpoint=rec, mask_kind=kind, mask_density_target=f.get("density"),
                           mask_density=nnz / W0, mask_nnz=nnz, mask_sha256=mask_fingerprint(mask),
                           mask_seconds=mask_s, conversion_seconds=conv_s, format=fmt)
    if fmt != "dense_masked":
        del lay
    torch.cuda.empty_cache()
    return out
