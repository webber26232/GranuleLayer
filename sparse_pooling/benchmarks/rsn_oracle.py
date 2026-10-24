"""Teacher-mapped oracle of the interaction diagnostic (results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md section 6.1).

Target of output g (wps_interaction.targets; n_rec valid records, T terms of order k per record):

    t_g = sum_{records r, terms u} (ReLU(sum_{i < k} x[a_{r,u,i}] - 1) - mu_k) / (sqrt(n_rec T) sigma_k)

Oracle: map every term (r, u) of an order >= 2 output to ONE distinct eligible neuron of record r -- weights 1 on
the term's k local inputs, 0 elsewhere, bias -1 -- and set every other neuron of the layer to zero weights and
zero bias (ReLU(0) = 0: no contribution).  With the unchanged Mean pooling denominator n_g, the per-output affine
readout

    readout_scale  = n_g / (sqrt(n_rec T) sigma_k),     readout_offset = -sqrt(n_rec T) mu_k / sigma_k

gives yhat_g = t_g exactly (up to arithmetic and input quantization).  Order-1 outputs are a separate linear
control: a non-negative mean of ReLUs cannot represent the signed linear term by this construction, so their
neurons stay zero and they are reported as not applicable.

Eligibility: ``mode="anchor"`` maps terms only onto the full-support anchor neurons (the 16 neurons of the
record's anchor group of an A / H2A mask; the first row group for the dense model) -- the representability
diagnostic for the anchors; ``mode="match"`` uses any valid neuron of the record whose ENABLED support contains
all k term inputs, assigned by maximum bipartite matching (augmenting paths): a failure is a statement about
this construction only, never a general impossibility proof.  Identical terms (same input set) are counted once
as distinct hyperplanes; each copy still needs its own neuron in this construction.

The oracle deliberately violates the broad-nonzero-support operating point (all but <= T neurons per record are
zero); it is a representability diagnostic, not a useful-width / coverage result.
"""
from __future__ import annotations

import math

import torch

SCHEMA = "granule.rsn_oracle/1"


def _match(elig: list) -> list:
    """Maximum bipartite matching: elig[u] = candidate neurons of term u; returns neuron per term or -1."""
    owner = {}

    def aug(u, seen):
        for n in elig[u]:
            if n in seen:
                continue
            seen.add(n)
            if n not in owner or aug(owner[n], seen):
                owner[n] = u
                return True
        return False

    for u in range(len(elig)):
        aug(u, set())
    out = [-1] * len(elig)
    for n, u in owner.items():
        out[u] = n
    return out


@torch.no_grad()
def build(lay, mask, ms, loc, order, rvalid, sig, terms: int, mode: str = "anchor"):
    """(w [G, NB, C, 64] bf16, b [G, NB, C] bf16, readout scale [G], offset [G], record).

    ``loc`` [G, NB, T, 4] local positions of every term (wps_interaction.target_inputs), ``order`` [G],
    ``rvalid`` [G, NB] valid records, ``sig`` {k: (mu, sigma, floor)}, ``mask`` the enabled support (None: dense),
    ``ms`` the MaskSet (anchors) or None."""
    L = lay.layout
    dev = lay.weight.device
    Gn, NB, Cb = L.groups, L.blocks_per_group, L.block
    valid = L.valid_mask(dev)
    w = torch.zeros(L.weight_shape, dtype=torch.bfloat16, device=dev)
    b = torch.zeros(L.bias_shape, dtype=torch.bfloat16, device=dev)
    em = (torch.ones(L.weight_shape, dtype=torch.bool, device=dev) if mask is None else mask.bool())
    if mode == "anchor":
        if ms is not None and getattr(ms, "anchor", None) is not None and bool(ms.anchor.any()):
            anc = ms.anchor.view(Gn, NB, Cb // 16)[..., None].expand(Gn, NB, Cb // 16, 16).reshape(Gn, NB, Cb)
        elif mask is None:
            anc = torch.zeros((Gn, NB, Cb), dtype=torch.bool, device=dev)
            anc[:, :, :16] = True
        else:
            raise ValueError("anchor oracle: the support has no full-support anchors (use mode='match')")
        cand_mask = anc & valid
    elif mode == "match":
        cand_mask = valid
    else:
        raise ValueError(mode)
    T = loc.shape[2]
    assert T == terms
    stats = dict(terms_total=0, terms_matched=0, terms_unmatched=0, distinct_terms=0, records=0,
                 records_fully_matched=0, eligible_neurons_per_term=[], order_counts={})
    em_cpu = em.cpu()
    cand_cpu = cand_mask.cpu()
    loc_cpu, order_cpu, rv_cpu = loc.cpu(), order.cpu(), rvalid.cpu()
    for g in range(Gn):
        k = int(order_cpu[g])
        if k < 2:
            continue
        for j in range(NB):
            if not bool(rv_cpu[g, j]):
                continue
            stats["records"] += 1
            pos = loc_cpu[g, j, :, :k]  # [T, k]
            cand = torch.nonzero(cand_cpu[g, j]).squeeze(1)
            sees = em_cpu[g, j][cand][:, pos]  # [n_cand, T, k]
            ok = sees.all(2)  # [n_cand, T]
            elig = [cand[ok[:, u]].tolist() for u in range(T)]
            stats["eligible_neurons_per_term"].extend(len(e) for e in elig)
            asg = _match(elig)
            stats["terms_total"] += T
            stats["distinct_terms"] += len({tuple(sorted(p.tolist())) for p in pos})
            m = sum(a >= 0 for a in asg)
            stats["terms_matched"] += m
            stats["terms_unmatched"] += T - m
            stats["records_fully_matched"] += int(m == T)
            for u, n in enumerate(asg):
                if n < 0:
                    continue
                w[g, j, n, pos[u]] = 1.0
                b[g, j, n] = -1.0
    n_g = L.counts_tensor(dev).to(torch.float64)
    nrec = rvalid.sum(1).to(torch.float64)
    mu = torch.tensor([sig[int(o)][0] for o in order.tolist()], dtype=torch.float64, device=dev)
    sd = torch.tensor([sig[int(o)][1] for o in order.tolist()], dtype=torch.float64, device=dev)
    rt = (nrec * terms).sqrt()
    scale = n_g / (rt * sd)
    offset = -rt * mu / sd
    lin = order == 1
    scale[lin], offset[lin] = 0.0, 0.0  # order 1: not applicable (prediction 0)
    if mask is not None and bool((w.bool() & ~em).any()):
        raise AssertionError("oracle weight outside the enabled support")
    e = torch.tensor(stats.pop("eligible_neurons_per_term"), dtype=torch.float64)
    stats["eligible_neurons_per_term"] = (dict(min=float(e.min()), median=float(e.median()), max=float(e.max()),
                                               zero=int((e == 0).sum())) if e.numel() else None)
    stats.update(schema=SCHEMA, mode=mode, nonzero_neurons=int((w != 0).any(3).sum()),
                 readout="scale n_g / (sqrt(n_rec T) sigma_k), offset -sqrt(n_rec T) mu_k / sigma_k; order 1: 0")
    return w, b, scale.float(), offset.float(), stats


@torch.no_grad()
def perturb(w, b, mask, eps: float, seed: int, std: float = 1.0 / 8.0):
    """Declared small perturbation: every ENABLED weight += eps * N(0, std^2) (std = 1/sqrt(64), the init scale),
    bias unchanged; returns new (w, b)."""
    g = torch.Generator(device=w.device).manual_seed(seed)
    noise = torch.randn(w.shape, generator=g, device=w.device) * (eps * std)
    if mask is not None:
        noise = noise * mask
    return (w.float() + noise).to(w.dtype), b.clone()
