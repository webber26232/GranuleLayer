"""One-time row normalization of a frozen sparse support before fine-tuning (random-support study,
results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md section 5).

For independent equal-variance inputs Var(z_i - b_i) = sigma_x^2 sum_k w_ik^2, so pruning a row to k_i of its 64
envelope inputs lowers its input-driven variance without proving the row useless.  Treatments (the mask, the bias
count and the optimizer budget are unchanged; the scale multiplies EVERY retained weight of the row exactly once
-- backbone and tile coordinates of a hybrid support are one dense masked row, so an overlap is never scaled
twice; structural zeros stay zero; disconnected rows (k_i = 0) stay all-zero):

* ``none``        the retained checkpoint values;
* ``fanin``       s_i = sqrt(64 / k_i);
* ``energy``      s_i = min(s_max, sqrt(E_dense_i / max(E_kept_i, eps_i))), E over the SAME pre-pruning canonical
                  row (all 64 envelope weights) and its retained part, FP32 sums of the BF16 values;
                  eps_i = eps_rel E_dense_i (relative to the row's own original scale; a connected row whose
                  retained weights are all zero gets s_max and stays zero -- no epsilon-valued edges are added).

``scale_bias``: also b_i <- s_i b_i (the W-and-bias control: positive joint scaling keeps the preactivation sign
and every ReLU decision but scales the neuron's output amplitude).  The primary treatments scale W only.

The scaled values are rounded to BF16 once (the stored parameters): the record reports the rounding (relative
error of the stored product vs the FP32 product), the clipping frequency (s_i = s_max) and the restored energy
ratio E_new / E_dense.  No runtime scale exists afterwards (zero per-step cost); nothing here uses data.
"""
from __future__ import annotations

import torch

SCHEMA = "granule.row_normalize/1"
TREATMENTS = ("none", "fanin", "energy")
S_MAX, EPS_REL = 8.0, 1e-6


def _q(v: torch.Tensor) -> dict:
    v = v.double().cpu()
    if v.numel() == 0:
        return dict(n=0)
    sub = v[::-(-v.numel() // (1 << 24))]  # (torch.quantile limit: an evenly strided subsample above 16M)
    qs = torch.quantile(sub, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], dtype=torch.float64))
    return dict(n=int(v.numel()), mean=float(v.mean()), min=float(v.min()), p1=float(qs[0]), p10=float(qs[1]),
                median=float(qs[2]), p90=float(qs[3]), p99=float(qs[4]), max=float(v.max()))


@torch.no_grad()
def row_scales(w_pre: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor, treatment: str, s_max: float = S_MAX,
               eps_rel: float = EPS_REL, chunk: int = 1 << 21):
    """(s [rows] FP32, k [rows] int64, E_dense, E_kept) for every neuron row of [..., 64] tensors (padded rows:
    s = 0)."""
    if treatment not in TREATMENTS:
        raise ValueError(f"normalization {treatment!r} (one of {TREATMENTS})")
    wv, mv = w_pre.reshape(-1, 64), mask.reshape(-1, 64)
    vv = valid.reshape(-1)
    n = wv.shape[0]
    dev = w_pre.device
    s = torch.zeros(n, dtype=torch.float32, device=dev)
    k = torch.zeros(n, dtype=torch.int64, device=dev)
    ed = torch.zeros(n, dtype=torch.float32, device=dev)
    ek = torch.zeros(n, dtype=torch.float32, device=dev)
    for i in range(0, n, chunk):
        w = wv[i:i + chunk].float()
        m = mv[i:i + chunk] != 0
        e = (w * w).sum(1)
        ekp = (w * w * m).sum(1)
        kk = m.sum(1)
        ok = vv[i:i + chunk] & (kk > 0)
        if treatment == "none":
            si = torch.ones_like(e)
        elif treatment == "fanin":
            si = torch.sqrt(64.0 / kk.clamp_min(1).float())
        else:
            si = torch.clamp(torch.sqrt(e / torch.maximum(ekp, eps_rel * e)), max=s_max)
            si = torch.where(e > 0, si, torch.ones_like(si))  # an all-zero dense row: nothing to restore
        s[i:i + chunk] = torch.where(ok, si, torch.zeros_like(si))
        k[i:i + chunk], ed[i:i + chunk], ek[i:i + chunk] = kk, e, ekp
    return s, k, ed, ek


@torch.no_grad()
def apply(w_pre: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor, treatment: str,
          scale_bias: bool = False, s_max: float = S_MAX, eps_rel: float = EPS_REL, chunk: int = 1 << 21):
    """(w_new BF16 = round(s_i (w_pre * mask)), b_new, record).  ``w_pre`` [G, NB, C, 64] the PRE-pruning values
    (energies use all 64 envelope weights), ``mask`` the frozen support, ``valid`` [G, NB, C]."""
    s, k, ed, ek = row_scales(w_pre, mask, valid, treatment, s_max, eps_rel, chunk)
    out = torch.empty_like(w_pre)
    ov, wv, mv = out.view(-1, 64), w_pre.reshape(-1, 64), mask.reshape(-1, 64)
    rerr_max, changed, nonzero, en = 0.0, 0, 0, torch.zeros_like(ed)
    for i in range(0, wv.shape[0], chunk):
        prod = wv[i:i + chunk].float() * (mv[i:i + chunk] != 0) * s[i:i + chunk, None]
        st = prod.to(w_pre.dtype)
        ov[i:i + chunk] = st
        nz = prod != 0
        rel = ((st.float() - prod).abs() / prod.abs().clamp_min(1e-38))[nz]
        if rel.numel():
            rerr_max = max(rerr_max, float(rel.max()))
        changed += int(((st.float() != prod) & nz).sum())
        nonzero += int(nz.sum())
        flushed = int(((st == 0) & nz).sum())
        if flushed:
            raise AssertionError(f"{flushed} scaled weights rounded to zero (support would shrink)")
        en[i:i + chunk] = (st.float() ** 2).sum(1)
    b_new = b.clone()
    if scale_bias:
        bv = b_new.view(-1)
        b_new.view(-1).copy_((bv.float() * s).to(b.dtype))
    vv = valid.reshape(-1)
    con = vv & (k > 0)
    ratio = en[con] / ed[con].clamp_min(1e-38)
    rec = dict(schema=SCHEMA, treatment=treatment, scale_bias=scale_bias, s_max=s_max if treatment == "energy" else None,
               eps_rel=eps_rel if treatment == "energy" else None, rows=int(vv.sum()), connected_rows=int(con.sum()),
               disconnected_rows=int((vv & (k == 0)).sum()),
               connected_rows_all_zero=int((con & (ek == 0)).sum()),
               scale=_q(s[con]), clipped=int((con & (s >= s_max)).sum()) if treatment == "energy" else 0,
               clipped_fraction=(float((con & (s >= s_max)).sum()) / max(1, int(con.sum())))
               if treatment == "energy" else 0.0,
               energy_ratio_before=_q(ek[con] / ed[con].clamp_min(1e-38)), energy_ratio_after=_q(ratio),
               bf16_rounding=dict(values=nonzero, values_changed_by_rounding=changed, max_relative_error=rerr_max,
                                  note="stored = BF16(round-to-nearest(s_i * w)); the FP32 product is exact for "
                                       "power-of-two s_i"),
               rule=__doc__.split("\n\n")[1].strip())
    return out, b_new, rec
