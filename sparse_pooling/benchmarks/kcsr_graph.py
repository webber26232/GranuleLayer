"""Logical record tables and graph fingerprints of the K/C/S/R sweep (backend independent).

A *logical record table* lists every logical index record with at least one valid neuron, in
(output g, logical block j) order:

    records [N, K] int16  sorted distinct global input ids (the shared index record)
    rec_q   [N]    int32  supergroup of the record (-1: unrestricted graph)
    rec_g   [N]    int32  pooling output
    rec_j   [N]    int32  logical block within the output (neurons j*C .. j*C + C - 1)
    rec_n   [N]    int32  valid neurons sharing the record (C, or the partial tail of the output)
    feat    [Q, S] int16  the input supergroups

Padded neurons never appear; a logical C128 / C256 block lowered to several physical C64 blocks is
ONE row (the physical copies add no index diversity).  The fingerprint hashes this table plus
(D, G, H, K, C): two backends that run the same mathematical graph produce the same fingerprint.
"""
from __future__ import annotations

import hashlib
import json

import torch

FINGERPRINT_SCHEMA = "granule.kcsr_graph/1"


@torch.no_grad()
def native_table(layer, logical_C: int | None = None):
    """Record table of a native SparsePoolingLayer (Mode B, optional supergroup graph).
    ``logical_C`` > layout C: the layer is a physical lowering of logical C-blocks (consecutive
    physical blocks of one logical block carry the same record; checked)."""
    from sparse_pooling.supergroups import SupergroupGraph, block_supergroup

    L = layer.layout
    idx = L.unpack_indices(layer.indices) if L.packed else layer.indices
    G, NB, Cp = L.groups, L.blocks_per_group, L.block
    Cl = logical_C or Cp
    assert Cl % Cp == 0
    f = Cl // Cp
    counts = L.counts_tensor(idx.device).to(torch.int64)
    if layer.connectivity is not None:
        bq = block_supergroup(L, SupergroupGraph(layer.connectivity, layer.sg_feat, layer.sg_buckets))
        feat = layer.sg_feat.to("cpu")
    else:
        bq = torch.full((G, NB), -1, dtype=torch.int32, device=idx.device)
        feat = torch.empty((0, 0), dtype=torch.int16)
    NBl = -(-NB // f)
    pad = NBl * f - NB
    idxp = torch.nn.functional.pad(idx.view(G, NB, 8), (0, 0, 0, pad)) if pad else idx.view(G, NB, 8)
    bqp = torch.nn.functional.pad(bq, (0, pad), value=-1) if pad else bq
    rec = idxp.view(G, NBl, f, 8)
    q = bqp.view(G, NBl, f)
    j = torch.arange(NBl, device=idx.device, dtype=torch.int64)
    vn = torch.clamp(counts[:, None] - j[None, :] * Cl, min=0, max=Cl)  # [G, NBl]
    valid = vn > 0
    # physical copies of a logical block that hold valid neurons must carry the same record / supergroup
    pvn = torch.clamp(counts[:, None] - (torch.arange(NBl * f, device=idx.device) * Cp)[None, :], 0, Cp).view(G, NBl, f)
    live = pvn > 0
    same = ((rec == rec[:, :, :1]).all(-1) | ~live).all(-1) & ((q == q[:, :, :1]) | ~live).all(-1)
    if not bool(same[valid].all()):
        raise AssertionError("physical copies of a logical block disagree (not a lowering of logical C)")
    g_ = torch.arange(G, device=idx.device)[:, None].expand(G, NBl)
    return dict(records=rec[:, :, 0][valid].to("cpu").contiguous(), rec_q=q[:, :, 0][valid].to(torch.int32).cpu(),
                rec_g=g_[valid].to(torch.int32).cpu(), rec_j=j[None].expand(G, NBl)[valid].to(torch.int32).cpu(),
                rec_n=vn[valid].to(torch.int32).cpu(), feat=feat.contiguous(),
                meta=dict(D=L.input_dim, G=G, H=L.hidden_dim, K=8, C=Cl, physical_C=Cp))


@torch.no_grad()
def logical_params(layer):
    """(weights [G, NB_l, C, K], bias [G, NB_l, C]) of a layer in logical canonical order (native, lowered or kgen)."""
    if hasattr(layer, "kcsr_logical_C"):  # physical lowering
        from sparse_pooling.experimental.lowering import unlower

        _, w, b, _ = unlower(layer)
        return w, b
    L = layer.layout
    if hasattr(L, "packed"):  # production layout (padded NB): restrict to the logical blocks
        _, w, b = L.to_canonical(layer.indices, layer.weight.detach(), layer.bias.detach())
        nb = -(-L.max_group // L.block)
        return w[:, :nb], b[:, :nb]
    if hasattr(layer, "dense_params"):  # compact K64 formats: dense logical values (was: the stored compact tensor)
        return layer.dense_params()
    return layer.weight.detach(), layer.bias.detach()


PARAMS_SHA_SCHEMA = {"canonical": "granule.canonical_k64/1 (experimental/canonical.py: valid neurons in (g, n) "
                                  "order, 64-input envelope, inactive = +0, -0 -> +0)",
                     "logical": "sha256 of the logical [G, NB, C, K] weight / bias tensors incl. padding"}


def params_sha_schema(layer) -> str:
    return "canonical" if hasattr(layer, "canon_rows") else "logical"


@torch.no_grad()
def params_sha(layer) -> str:
    """sha256 identifying the mathematical parameters.  K64 envelope layers (``canon_rows``): the canonical
    streamed hash of experimental/canonical.py (equal for the same model in every representation: dense,
    dense masked, tile16 / tile8 / hybrid, CSR, lowered controls).  Others: the logical canonical tensors."""
    if hasattr(layer, "canon_rows"):
        from sparse_pooling.experimental import canonical

        return canonical.digest(layer, "param")["sha"]
    w, b = logical_params(layer)
    h = hashlib.sha256()
    for t in (w, b):
        t = t.contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.int16).cpu().numpy().tobytes())
    return h.hexdigest()


def fingerprint(tab) -> str:
    m = tab["meta"]
    h = hashlib.sha256()
    h.update(json.dumps(dict(schema=FINGERPRINT_SCHEMA, D=m["D"], G=m["G"], H=m["H"], K=m["K"], C=m["C"]),
                        sort_keys=True).encode())
    for k in ("records", "rec_q", "rec_g", "rec_j", "rec_n", "feat"):
        t = tab[k].contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def summary(tab) -> dict:
    """Counts of a record table (no coverage)."""
    n = tab["rec_n"].to(torch.int64)
    m = tab["meta"]
    return dict(logical_records=int(n.numel()), valid_neurons=int(n.sum()), H=m["H"], K=m["K"], C=m["C"],
                full_records=int((n == m["C"]).sum()), partial_records=int((n < m["C"]).sum()),
                records_per_output_min=int(torch.bincount(tab["rec_g"].long(), minlength=m["G"]).min()),
                records_per_output_max=int(torch.bincount(tab["rec_g"].long(), minlength=m["G"]).max()))
