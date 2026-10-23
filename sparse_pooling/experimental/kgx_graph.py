"""EXPERIMENTAL work lists of the K = 64 kgx kernels and the index-free (S = K) graph checks.

The logical graph is the one of ``kgen_graph.generate`` (unchanged generator, quantum = 1 logical record):
kgx only changes how the records are executed.

Work items (kgx.cuh): ``items [n, 4] = (g, first flat block, blocks, c0 | c1 << 16)``.  A logical block of
C <= tile neurons is executed whole (up to ``max_item_blocks`` consecutive blocks of one bucket per item,
as kgen); a block of C > tile neurons is split along the neuron dimension into ceil(n_valid / tile)
items of ``tile`` neurons (the physical neuron tile), so a large logical group is processed by several
warps / CTAs while its record (and its supergroup) is unchanged.  Chunks are cut at about
``chunk_neurons`` neurons inside one supergroup; dW tasks are 16-neuron groups (independent of the tile).

Index-free execution (S = K): ``generate`` cuts each local permutation of a set into S / K = 1 record, so
every valid record of supergroup q is feat[q] itself (sorted); ``check_index_free`` verifies this on the
actual graph before the record indices are dropped.
"""
from __future__ import annotations

import time
from typing import Dict

import torch

from ..supergroups import chunk_table


@torch.no_grad()
def check_index_free(idx: torch.Tensor, feat: torch.Tensor, blk_buckets: torch.Tensor, counts: torch.Tensor,
                     C: int) -> Dict:
    """Raise unless every valid logical record equals its supergroup's (sorted) feature set; returns counts
    (records, occupied / empty supergroups)."""
    G, NB, K = idx.shape
    Q, S = feat.shape
    if S != K:
        raise ValueError(f"index-free execution needs S = K (S = {S}, K = {K})")
    b = blk_buckets.to(idx.device).to(torch.int64)
    rb = torch.repeat_interleave(torch.arange(len(b), device=idx.device), b[:, 3])
    kb = torch.arange(len(rb), device=idx.device) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
    fb = b[rb, 1] * NB + b[rb, 2] + kb
    rec = idx.view(-1, K)[fb].to(torch.int64)
    want = feat.to(idx.device).to(torch.int64)[b[rb, 0]]
    if not torch.equal(rec, want):
        bad = int((rec != want).any(1).sum())
        raise AssertionError(f"{bad} records differ from their supergroup's feature set: not an S = K graph")
    occ = torch.zeros(Q, dtype=torch.bool, device=idx.device)
    occ[b[:, 0]] = True
    return dict(records=int(len(fb)), occupied_supergroups=int(occ.sum()), empty_supergroups=int((~occ).sum()), Q=Q)


class KxExec:
    """Work lists of a kgx layer: items / chunks (fwd, dX), tasks / dw_chunks (dW).  Built once per layer and
    device (host work outside the timed steps; ``stats['build_seconds']``)."""

    def __init__(self, NB: int, C: int, counts: torch.Tensor, feat: torch.Tensor, blk_buckets: torch.Tensor, tile: int,
                 dw_per_cta: int, chunk_neurons: int = 1 << 14, max_item_blocks: int = 64,
                 group_keep: torch.Tensor = None, run_items: bool = True):
        """``group_keep`` [G NB C / 16] bool (sparse formats): only the kept 16-neuron groups enter the items and
        dW tasks (runs of consecutive kept groups of a block form the items, <= ``tile`` neurons each); the
        valid neurons of the other groups are bias-only neurons (``self.bias_only``), handled exactly outside
        the kernels.  ``run_items=False`` keeps the standard items (whole blocks / neuron tiles) and only filters
        the dW tasks: the kernels then skip the dropped groups themselves (tile format: mask 0)."""
        t0 = time.perf_counter()
        dev = feat.device
        self.feat = feat.contiguous()
        self.Q, self.S = feat.shape
        self.tile = tile
        b = blk_buckets.to("cpu", torch.int64)
        q, g, j0, nb = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cnt = counts.to("cpu", torch.int64)
        self.bias_only = None
        if group_keep is not None:
            keep = group_keep.to("cpu")
            NG = C // 16
            rb = torch.repeat_interleave(torch.arange(len(b)), nb)
            kb = torch.arange(len(rb)) - torch.repeat_interleave(torch.cumsum(nb, 0) - nb, nb)
            bj, bg, bq = j0[rb] + kb, g[rb], q[rb]
            fbk = bg * NB + bj
            nval = torch.clamp(cnt[bg] - bj * C, max=C)
            # groups of the blocks in bucket (supergroup) order
            gi = fbk[:, None] * NG + torch.arange(NG)[None, :]  # [n_blocks, NG] global group ids
            gvalid = torch.arange(NG)[None, :] * 16 < nval[:, None]
            gk = keep[gi] & gvalid
            gkf = gk.reshape(-1)
            pos = torch.arange(NG).repeat(len(fbk))
            prev = torch.cat([torch.zeros(1, dtype=torch.bool), gkf[:-1]])
            start = gkf & ((pos == 0) | ~prev)
            run = torch.cumsum(start.to(torch.int64), 0) - 1
            seg = torch.where(gkf, run, torch.full_like(run, -1))
            # split runs into pieces of <= tile neurons
            idx_k = torch.nonzero(gkf).squeeze(1)
            rid = seg[idx_k]
            rstart = torch.zeros(int(rid.max()) + 1 if rid.numel() else 0, dtype=torch.int64)
            if rid.numel():
                rstart.scatter_reduce_(0, rid, idx_k, reduce="amin", include_self=False)
            piece = (idx_k - rstart[rid]) // max(1, tile // 16)
            key = rid * (NG + 1) + piece
            ukey, inv = torch.unique_consecutive(key, return_inverse=True)
            n_items = len(ukey)
            first = torch.full((n_items,), 1 << 62, dtype=torch.int64).scatter_reduce_(0, inv, idx_k, reduce="amin")
            last = torch.full((n_items,), -1, dtype=torch.int64).scatter_reduce_(0, inv, idx_k, reduce="amax")
            blk = first // NG
            ic0 = (first % NG) * 16
            ic1 = torch.minimum((last % NG + 1) * 16, nval[blk])
            ig, iq = bg[blk], bq[blk]
            if run_items:
                self.items = torch.stack([ig, fbk[blk], torch.ones_like(ig), ic0 | (ic1 << 16)], 1).to(torch.int32)
                neur = ic1 - ic0
                self.item_q = iq.to(torch.int32)
                self.chunks = chunk_table(iq, neur, self.Q, chunk_neurons)
            else:
                neur = self._standard_items(b, NB, C, cnt, tile, chunk_neurons, max_item_blocks)
            tk = idx_k
            tfb, tgrp = fbk[tk // NG], tk % NG
            self.tasks = torch.stack([tfb, tgrp], 1).to(torch.int32)
            tq = bq[tk // NG]
            # bias-only valid neurons: valid neurons of the dropped groups
            drop = (~gk) & gvalid
            dblk, dgrp = torch.nonzero(drop, as_tuple=True)
            nn_ = torch.clamp(nval[dblk] - dgrp * 16, max=16)
            rr = torch.repeat_interleave(torch.arange(len(dblk)), nn_)
            kk = torch.arange(len(rr)) - torch.repeat_interleave(torch.cumsum(nn_, 0) - nn_, nn_)
            slot = fbk[dblk][rr] * C + dgrp[rr] * 16 + kk
            self.bias_only = (slot.to(torch.int64).to(dev), bg[dblk][rr].to(torch.int64).to(dev))
            self._finish(tq, dw_per_cta, dev, C, neur, dict(groups_kept=int(gk.sum()), groups_valid=int(gvalid.sum()),
                                                             bias_only_neurons=int(len(slot)),
                                                             item_mode="runs of kept groups" if run_items else
                                                             "standard items, kernels skip dropped groups"))
            self.stats["build_seconds"] = time.perf_counter() - t0
            return
        neur = self._standard_items(b, NB, C, cnt, tile, chunk_neurons, max_item_blocks)
        # dW tasks: (flat block, 16-neuron group) of every valid neuron group, supergroup order
        rb = torch.repeat_interleave(torch.arange(len(b)), nb)
        kb = torch.arange(len(rb)) - torch.repeat_interleave(torch.cumsum(nb, 0) - nb, nb)
        bj = j0[rb] + kb
        fb = g[rb] * NB + bj
        nval = torch.clamp(cnt[g[rb]] - bj * C, max=C)
        ncg = (nval + 15) // 16
        rt = torch.repeat_interleave(torch.arange(len(fb)), ncg)
        kt = torch.arange(len(rt)) - torch.repeat_interleave(torch.cumsum(ncg, 0) - ncg, ncg)
        self.tasks = torch.stack([fb[rt], kt], 1).to(torch.int32)
        tq = q[rb][rt]
        self._finish(tq, dw_per_cta, dev, C, neur, dict(item_mode="standard"))
        self.stats["build_seconds"] = time.perf_counter() - t0

    def _standard_items(self, b, NB, C, cnt, tile, chunk_neurons, max_item_blocks):
        q, g, j0, nb = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        if C <= tile:
            pieces = (nb + max_item_blocks - 1) // max_item_blocks
            rep = torch.repeat_interleave(torch.arange(len(b)), pieces)
            k = torch.arange(len(rep)) - torch.repeat_interleave(torch.cumsum(pieces, 0) - pieces, pieces)
            iq, ig = q[rep], g[rep]
            ij = j0[rep] + k * max_item_blocks
            inb = torch.clamp(nb[rep] - k * max_item_blocks, max=max_item_blocks)
            ic0 = torch.zeros_like(ij)
            ic1 = torch.full_like(ij, C)
            # neurons of the item (for chunk balancing)
            last = ij + inb - 1
            neur = (inb - 1) * C + torch.clamp(cnt[ig] - last * C, max=C)
        else:
            rb = torch.repeat_interleave(torch.arange(len(b)), nb)
            kb = torch.arange(len(rb)) - torch.repeat_interleave(torch.cumsum(nb, 0) - nb, nb)
            bj = j0[rb] + kb
            bg = g[rb]
            nval = torch.clamp(cnt[bg] - bj * C, max=C)
            nt = (nval + tile - 1) // tile
            rt = torch.repeat_interleave(torch.arange(len(rb)), nt)
            kt = torch.arange(len(rt)) - torch.repeat_interleave(torch.cumsum(nt, 0) - nt, nt)
            iq, ig, ij = q[rb][rt], bg[rt], bj[rt]
            inb = torch.ones_like(ij)
            ic0 = kt * tile
            ic1 = torch.clamp(ic0 + tile, max=C)
            neur = torch.minimum(ic1, nval[rt]) - ic0
        self.items = torch.stack([ig, ig * NB + ij, inb, ic0 | (ic1 << 16)], 1).to(torch.int32)
        self.item_q = iq.to(torch.int32)
        self.chunks = chunk_table(iq, neur, self.Q, chunk_neurons)
        return neur

    def _finish(self, tq, dw_per_cta, dev, C, neur, extra):
        q_off = torch.searchsorted(tq.contiguous(), torch.arange(self.Q + 1))
        n = q_off[1:] - q_off[:-1]
        nch = (n + dw_per_cta - 1) // dw_per_cta
        qs = torch.repeat_interleave(torch.arange(self.Q), nch)
        kk = torch.arange(len(qs)) - torch.repeat_interleave(torch.cumsum(nch, 0) - nch, nch)
        lo = q_off[qs] + kk * dw_per_cta
        hi = torch.minimum(lo + dw_per_cta, q_off[qs + 1])
        self.dw_chunks = torch.stack([qs, lo, hi], 1).to(torch.int32)
        ipc = (self.chunks[:, 2] - self.chunks[:, 1]).to(torch.float64)
        self.stats = dict(items=len(self.items), chunks=len(self.chunks), tasks=len(self.tasks),
                          dw_chunks=len(self.dw_chunks), physical_tile=self.tile, logical_C=C,
                          items_split_along_neurons=bool(C > self.tile),
                          items_per_cta=dict(mean=float(ipc.mean()) if len(ipc) else 0.0,
                                             min=int(ipc.min()) if len(ipc) else 0, max=int(ipc.max()) if len(ipc) else 0),
                          neurons_per_item=dict(mean=float(neur.double().mean()) if len(neur) else 0.0,
                                                min=int(neur.min()) if len(neur) else 0,
                                                max=int(neur.max()) if len(neur) else 0), **extra)
        self.items, self.chunks = self.items.to(dev).contiguous(), self.chunks.to(dev).contiguous()
        self.tasks, self.dw_chunks = self.tasks.to(dev).contiguous(), self.dw_chunks.to(dev).contiguous()
        self.stats["bytes"] = self.runtime_bytes() + (0 if self.bias_only is None else
                                                      sum(t.numel() * t.element_size() for t in self.bias_only))

    def runtime_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.items, self.chunks, self.tasks, self.dw_chunks))
