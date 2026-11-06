"""Representation-independent identity of K = 64 envelope models (width-preserving sparsity study,
results/WIDTH_PRESERVING_SPARSITY.md section 1).

Every K = 64 representation of this study (dense ``KgxLayer`` incl. its ``lowered`` / ``kgen`` routes, dense
masked, ``KgxTileLayer`` 16 x 16, ``KgxCsrLayer``, ``KgxWideLayer`` 16 x 8 / hybrid backbone + tiles, and the
K16 / K8 lowered execution controls) describes the same mathematical object:

    for every valid neuron (output g, neuron n < n_g):  W_eff[g, n, k]  (k = local input of the neuron's
    64-input envelope record, ascending global feature id),  bias[g, n].

The canonical form streams the valid neurons in (g, n) order (padding never appears), reconstructs every
inactive / absent entry as +0, maps -0 to +0 (signed-zero normalization of the raw bits), and hashes the
weights and the biases with two independent sha256 streams (chunk-size independent):

    params_sha = sha256(header || sha256(W_eff rows) || sha256(bias))

Each representation implements ``canon_params()`` ({name: Parameter}) and ``canon_rows(src, g, n)`` mapping
per-parameter tensors (values, main_grad, optimizer moments) to canonical rows, and ``canon_mask_rows(g, n)``
(enabled support).  So the same function hashes parameters, FP32 main_grad and both Adam moments.  Graph and
mask identity are separate hashes (``graph_fingerprint`` of kcsr_graph; ``mask_sha`` here).  ``storage_sha``
hashes the raw stored tensors (representation dependent).
"""
from __future__ import annotations

import hashlib
import json

import torch

SCHEMA = "granule.canonical_k64/1"
CHUNK_NEURONS = 1 << 20


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def normalized_bits(t: torch.Tensor) -> torch.Tensor:
    """Raw bits with -0 mapped to +0 (sign bit alone)."""
    b = _bits(t)
    neg0 = torch.tensor(torch.iinfo(b.dtype).min, dtype=b.dtype, device=b.device)
    return torch.where(b == neg0, torch.zeros_like(b), b)


def neurons(counts: torch.Tensor, g0: int, g1: int):
    """(g, n) of the valid neurons of outputs [g0, g1) in canonical order."""
    c = counts[g0:g1].to(torch.int64)
    g = torch.repeat_interleave(torch.arange(g0, g1, device=c.device), c)
    n = torch.arange(len(g), device=c.device) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c)
    return g, n


def _chunks(counts: torch.Tensor, chunk: int = CHUNK_NEURONS):
    G = counts.numel()
    cnt = counts.to("cpu", torch.int64).tolist()
    g0 = 0
    while g0 < G:
        g1, tot = g0, 0
        while g1 < G and (tot == 0 or tot + cnt[g1] <= chunk):
            tot += cnt[g1]
            g1 += 1
        yield g0, g1
        g0 = g1


def sources(layer, kind: str, opt=None) -> dict:
    """{param name: tensor} of a representation: kind "param" (values), "main_grad", or an optimizer state key
    ("m" / "v": missing state = zeros)."""
    ps = layer.canon_params()
    if kind == "param":
        return {k: p.detach() for k, p in ps.items()}
    if kind == "main_grad":
        return {k: (p.main_grad if getattr(p, "main_grad", None) is not None else torch.zeros_like(p, dtype=torch.float32))
                for k, p in ps.items()}
    out = {}
    for k, p in ps.items():
        st = opt.state.get(p, {}) if opt is not None else {}
        out[k] = st[kind] if kind in st else torch.zeros_like(p)
    return out


@torch.no_grad()
def digest(layer, kind: str = "param", opt=None, chunk: int = CHUNK_NEURONS) -> dict:
    """{sha, weight_sha, bias_sha, dtype, neurons} of the canonical form of ``kind`` (see module docstring)."""
    src = sources(layer, kind, opt)
    hw, hb = hashlib.sha256(), hashlib.sha256()
    counts = layer.counts
    tot = 0
    dt = None
    for g0, g1 in _chunks(counts, chunk):
        g, n = neurons(counts, g0, g1)
        w, b = layer.canon_rows(src, g, n)
        if dt is None:
            dt = str(w.dtype)
        hw.update(normalized_bits(w).cpu().numpy().tobytes())
        hb.update(normalized_bits(b).cpu().numpy().tobytes())
        tot += len(g)
    head = json.dumps(dict(schema=SCHEMA, kind=kind if kind in ("param", "main_grad") else f"opt.{kind}", dtype=dt,
                           neurons=tot, G=int(counts.numel()), K=64), sort_keys=True).encode()
    h = hashlib.sha256(head)
    h.update(hw.digest())
    h.update(hb.digest())
    return dict(sha=h.hexdigest(), weight_sha=hw.hexdigest(), bias_sha=hb.hexdigest(), dtype=dt, neurons=tot,
                schema=SCHEMA)


@torch.no_grad()
def mask_digest(layer, chunk: int = CHUNK_NEURONS) -> dict:
    """sha256 of the enabled support (64-bit local masks, canonical order) + enabled slot count."""
    h = hashlib.sha256(json.dumps(dict(schema=SCHEMA, kind="mask")).encode())
    sh = torch.arange(64, device=layer.counts.device)
    enabled = 0
    for g0, g1 in _chunks(layer.counts, chunk):
        g, n = neurons(layer.counts, g0, g1)
        m = layer.canon_mask_rows(g, n).to(torch.int64)
        enabled += int(m.sum())
        h.update((m << sh).sum(1).cpu().numpy().tobytes())
    return dict(sha=h.hexdigest(), enabled_slots=enabled)


@torch.no_grad()
def storage_digest(layer) -> str:
    """sha256 of the raw stored parameter tensors (name order; representation dependent)."""
    h = hashlib.sha256()
    for k, p in sorted(layer.canon_params().items()):
        t = _bits(p.detach().reshape(-1))
        h.update(k.encode())
        h.update(str(tuple(p.shape)).encode())
        for i in range(0, t.numel(), 1 << 27):
            h.update(t[i:i + (1 << 27)].cpu().numpy().tobytes())
    return h.hexdigest()


def identity(layer, opt=None, state: bool = True) -> dict:
    """Canonical identity record of a layer (and its optimizer state): params / mask / storage (+ main_grad and
    both moments when ``state``)."""
    out = dict(schema=SCHEMA, params=digest(layer, "param"), mask=mask_digest(layer), storage_sha=storage_digest(layer))
    if state:
        out["main_grad"] = digest(layer, "main_grad")
        if opt is not None:
            out["opt_m"] = digest(layer, "m", opt)
            out["opt_v"] = digest(layer, "v", opt)
    return out


# ------------------------------------------------------------------ helpers for the dense K64 layout
def dense_slots(layer, g, n):
    L = layer.layout
    return g * (L.blocks_per_group * L.block) + n


def dense_rows(layer, src, g, n):
    s = dense_slots(layer, g, n)
    return src["weight"].reshape(-1, 64)[s], src["bias"].reshape(-1)[s]


def dense_mask_rows(layer, g, n):
    m = getattr(layer, "mask", None)
    s = dense_slots(layer, g, n)
    if m is None:
        return torch.ones((len(s), 64), dtype=torch.bool, device=s.device)
    return m.reshape(-1, 64)[s] != 0
