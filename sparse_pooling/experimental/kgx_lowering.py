"""EXPERIMENTAL logical-C lowering of the K = 64 study: a logical C = 512 / 1024 / 2048 graph executed as
physical C = 256 subblocks on the existing kgen K = 64 kernels (the validated reference representation).

Logical block j of output g (C_l neurons sharing one 64-record, independent weights / bias) becomes the
f = C_l / 256 physical blocks f j .. f j + f - 1, each carrying a COPY of the record.  In canonical storage
this is a reshape: logical weights [G, NB_l, C_l, 64] ARE physical weights [G, f NB_l, 256, 64] (same memory
order), so neuron identity (slot j C_l + c = (f j + h) 256 + c'), output membership, bias, padding,
main_grad and optimizer state map one to one; a partial logical tail keeps its valid physical subblocks
only (the last bucket of each output is trimmed).  Physical buckets are the logical ones scaled by f, so
the f copies of a record always stay in its supergroup.

``KgenRoute`` runs the kgen kernels on these physical views of a KgxLayer's own tensors (variants
"kgen" for C <= 256 and "lowered" for C > 256), so the reference and the direct / index-free kgx kernels
are timed on exactly the same parameters.  The duplicated physical records are ``index_duplication = f``
in memory; coverage is always measured on the logical records.
"""
from __future__ import annotations

import torch

from . import kgen_graph as KG
from .kgen_layout import KLayout

PHYSICAL_C = 256


@torch.no_grad()
def physical_graph(L: KLayout, idx_l: torch.Tensor, blk_l: torch.Tensor, Cp: int = PHYSICAL_C):
    """(physical blocks per output NB_p = f NB_l, idx [G, NB_p, 64], physical block buckets, f) of logical
    layout L (C_l % Cp == 0).  NB_p = f NB_l (not ceil(max n_g / Cp)), so the physical weights are a VIEW of
    the logical ones; physical blocks beyond an output's valid neurons are padding (in no bucket)."""
    Cl = L.block
    if Cl % Cp:
        raise ValueError(f"logical C = {Cl} is not a multiple of the physical C = {Cp}")
    f = Cl // Cp
    G, NBl, K = L.groups, L.blocks_per_group, L.fan_in
    ip = idx_l.reshape(G, NBl, 1, K).expand(G, NBl, f, K).reshape(G, f * NBl, K)
    b = blk_l.to(torch.int64)
    nvp = (L.counts_tensor(b.device).to(torch.int64) + Cp - 1) // Cp
    first = b[:, 2] * f
    end = torch.minimum((b[:, 2] + b[:, 3]) * f, nvp[b[:, 1]])
    pb = torch.stack([b[:, 0], b[:, 1], first, end - first], 1)
    pb = pb[pb[:, 3] > 0].to(torch.int32).contiguous()
    return f * NBl, ip.contiguous(), pb, f


class KgenRoute:
    """The kgen K = 64 kernels on (physical views of) a KgxLayer's graph and parameters."""

    def __init__(self, layer):
        from ..backends.capabilities import query
        from ..backends.dispatch import kgen_dx_group_warps
        from .kgen_runtime import kgen_backend

        L = layer.layout
        dev = layer.sg_feat.device
        idx_l = layer.logical_indices()
        if L.block <= 256:
            self.NB, self.C, self.idx, self.bk, self.f = L.blocks_per_group, L.block, idx_l.contiguous(), \
                layer.kg_buckets, 1
        else:
            self.NB, self.idx, self.bk, self.f = physical_graph(L, idx_l, layer.kg_buckets)
            self.C = PHYSICAL_C
        self.G, self.D = L.groups, L.input_dim
        self.backend = kgen_backend(dev)
        self.caps = query(dev)
        self.cfg = self.backend.cfgs[64]
        self.sg = KG.KgExec(self.NB, self.C, 64, layer.counts, layer.sg_feat, self.bk, self.cfg)
        self.dx_group_warps = kgen_dx_group_warps(self.caps, L.input_dim, layer.sg_feat.shape[1], self.cfg)
        self.stats = dict(self.sg.stats, physical_C=self.C, physical_subblocks=self.f, index_duplication=self.f,
                          physical_index_bytes=self.idx.numel() * 2, physical_buckets=int(self.bk.shape[0]))

    def args(self, layer, xt, w, b, B):
        from .kgen_runtime import KgArgs

        return KgArgs(xt.data_ptr(), xt.stride(0), self.idx.data_ptr(), w.data_ptr(), b.data_ptr(),
                      layer.counts.data_ptr(), layer.sg_feat.data_ptr(), 0, 0, 0, 0, 0, 0, 0, 0,
                      B, self.G, self.D, layer.sg_feat.shape[1], self.NB, self.C, 1, 0)

    def reason(self, layer, B) -> str:
        from ..backends.dispatch import kgen_reason

        return kgen_reason(self.caps, 64, self.C, layer.sg_feat.shape[1], self.D, B, layer.level,
                           layer.plan, False, self.cfg)

    def tokens(self, kind: str) -> int:
        return 16 * (self.cfg[0] if kind == "fwd" else self.cfg[2])


@torch.no_grad()
def logical_from_physical(L: KLayout, w_p: torch.Tensor, b_p: torch.Tensor):
    """Round trip: logical (weights [G, NB_l, C_l, 64], bias [G, NB_l, C_l]) of physical C = 256 parameters."""
    G, NBl, Cl, K = L.weight_shape
    f = Cl // w_p.shape[2]
    return (w_p[:, :f * NBl].reshape(G, NBl, Cl, K).contiguous(), b_p[:, :f * NBl].reshape(G, NBl, Cl).contiguous())


def describe(route: KgenRoute) -> dict:
    return dict(route="kgen K = 64 kernels" + (f" via physical lowering (C = {route.C} subblocks)"
                                               if route.f > 1 else ""), **route.stats)


def lowering_bytes(L: KLayout) -> dict:
    f = max(1, L.block // PHYSICAL_C)
    nb = L.blocks_per_group
    return dict(logical_record_bytes=L.groups * nb * 64 * 2, physical_record_bytes=L.groups * nb * f * 64 * 2,
                index_duplication=f, physical_blocks_per_output=nb * f, padding_fraction=L.padding_fraction(),
                physical_C=min(L.block, PHYSICAL_C), subblocks=f)
