"""Are the dX and dW pre-activations / ReLU decisions of the grouped C = 8 backward
interchangeable?  (Track F prerequisite, results/SCALE_DISPATCH_FUSED_BACKWARD.md section 6.)

kernels/cuda/diag_proj_c8.cu recomputes, for (unit, 16-token tile) work items, the projection
of every block with the exact instruction sequences of both kernels (dX: z = X W + b with X as
the MMA A operand; dW: z^T = W^T X^T + b with W as the A operand, rows 8-15 zero) and records
z bits and the decisions of the dX mask epilogue, the dW fast epilogue and the dW mask
epilogue (plain and MASKED / ragged).  Regimes: random; signed zeros; subnormal products /
bias; exact cancellation (z = +-0 by construction); near-zero z (bias = -round(x.w)); huge
values (inf / NaN z).  An FP64 reference sign is recorded for context (decisions are NOT
expected to match it near zero: both kernels use the tensor-core FP32 sum).

    python proj_mask_diag.py --out results/scale_dispatch_fused_backward/proj_mask_diag
"""
from __future__ import annotations

import argparse
import ctypes

import torch

import common  # noqa: F401
from common import save
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.kernels.cuda_ops import _ptr
from sparse_pooling.supergroups import ConnectivityPlan

BF = torch.bfloat16


def layer(D, H, G, S, seed=0):
    lay = SparsePoolingLayer(D, H, G, shared_neurons=8, pooling="mean", device="cuda", seed=seed, bias_std=0.2,
                             plan=PhysicalPlan(cuda_input_supergroup_backend="grouped"),
                             connectivity=ConnectivityPlan(S, 4, seed=seed))
    return lay


def run_diag(lay, x, items):
    """items: [n, 3] int32 (flat unit, token tile, n_g) -> dict of tensors [n, 4, 16, 8]."""
    rt = lay._runtime()
    be = rt.backend
    B = x.shape[0]
    xop = x.t().contiguous()
    a = be._sg_args(rt, xop, lay.weight, lay.bias, B)
    n = items.shape[0]
    zdx = torch.empty((n, 4, 16, 8), device="cuda", dtype=torch.float32)
    zdw = torch.empty_like(zdx)
    m = torch.empty((6, n, 4, 16, 8), device="cuda", dtype=torch.int32)
    f = be.lib.sp_diag_proj_c8
    f.restype = ctypes.c_int
    err = f(ctypes.byref(a), _ptr(items), n, _ptr(zdx), _ptr(zdw), _ptr(m),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    assert err == 0, err
    torch.cuda.synchronize()
    nz = (m != 0).to(torch.uint8)
    mdx, mdwf, mdwm = nz[0] | (nz[1] << 1), nz[2] | (nz[3] << 1), nz[4] | (nz[5] << 1)
    return dict(zdx=zdx, zdw=zdw, mdx=mdx, mdwf=mdwf, mdwm=mdwm)


def reference_z(lay, x, items):
    """FP64 z[t, n] = sum_k x[t, f_k] w[n, k] + b[n] of the BF16 operands, canonical layout."""
    L = lay.layout
    G, NB = L.groups, L.blocks_per_group
    NU = NB // L.unit_blocks
    idx = L.unpack_indices(lay.indices).view(G, NB, 8).long()
    W = L.unpack_weights(lay.weight.detach()).view(G, NB, 8, 8).double()  # [g, j, neuron, k]
    bb = L.unpack_bias(lay.bias.detach()).view(G, NB, 8).double()
    u = items[:, 0].long()
    g, lu = u // NU, u % NU
    tok = items[:, 1].long()[:, None] * 16 + torch.arange(16, device="cuda")[None]  # [n, 16]
    tokc = tok.clamp(max=x.shape[0] - 1)
    xd = x.double()
    out = torch.empty((len(u), 4, 16, 8), dtype=torch.float64, device="cuda")
    for bu in range(4):
        j = lu * 4 + bu
        f = idx[g, j]  # [n, 8]
        xs = xd[tokc[:, :, None], f[:, None, :]]  # [n, 16, 8 k]
        out[:, bu] = torch.einsum("ntk,nck->ntc", xs, W[g, j]) + bb[g, j][:, None, :]
    return out


def pick_items(lay, M, n_units, gen):
    L = lay.layout
    NU = L.blocks_per_group // L.unit_blocks
    counts = lay._runtime().units.counts.long()
    G = L.groups
    u = torch.randint(0, G * NU, (n_units,), generator=gen, device="cpu")
    g = u // NU
    # keep units holding at least one valid neuron (block slots j * 8 < n_g)
    ok = (u % NU) * 32 < counts.cpu()[g]
    u, g = u[ok], g[ok]
    tt = torch.randint(0, (M + 15) // 16, (len(u),), generator=gen)
    return torch.stack([u, tt, counts.cpu()[g]], 1).to(torch.int32).cuda()


def stats(out, name):
    zdx, zdw = out["zdx"], out["zdw"]
    vd = out["valid"]  # [n, 4, 16, 8] bool: real (token < B) entries
    bdx, bdw = zdx.view(torch.int32)[vd], zdw.view(torch.int32)[vd]
    zx = zdx[vd]
    eqbits = bdx == bdw
    both_nan = torch.isnan(zdx[vd]) & torch.isnan(zdw[vd])
    r = {"regime": name, "elements": int(vd.sum()), "z_bits_equal": int(eqbits.sum()),
         "z_bits_differ_nonnan": int((~eqbits & ~both_nan).sum()), "z_both_nan": int(both_nan.sum()),
         "z_pos_zero": int((bdx == 0).sum()), "z_neg_zero": int((bdx == -2**31).sum()),
         "z_subnormal": int(((zx != 0) & (zx.abs() < 2.0**-126)).sum()),
         "z_inf": int(torch.isinf(zx).sum()), "z_nan": int(torch.isnan(zx).sum())}
    for k, lab in (("mdx", "dx_mask"), ("mdwf", "dw_fast"), ("mdwm", "dw_mask")):
        m = out[k][vd]
        r[f"{lab}_active"] = int((m & 1).sum())
        r[f"{lab}_masked_active"] = int(((m >> 1) & 1).sum())
    for k in ("mdwf", "mdwm"):
        r[f"decisions_dx_vs_{k}_differ"] = int(((out["mdx"][vd] & 1) != (out[k][vd] & 1)).sum())
        r[f"masked_decisions_dx_vs_{k}_differ"] = int((((out["mdx"][vd] >> 1) & 1) != ((out[k][vd] >> 1) & 1)).sum())
    if "zref" in out:
        zr = out["zref"][vd]
        r["fp64_positive"] = int((zr > 0).sum())
        r["dx_decision_vs_fp64_sign_differ"] = int(((out["mdx"][vd] & 1).bool() != (zr > 0)).sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D, G, M, S = 256, 16, 136, 16
    H = 50_000 + 13  # ragged outputs: partial units / blocks
    gen = torch.Generator().manual_seed(0)
    results = []
    lay = layer(D, H, G, S)
    L = lay.layout
    w0, b0 = lay.weight.detach().clone(), lay.bias.detach().clone()
    gx = torch.Generator(device="cuda").manual_seed(1)
    items = pick_items(lay, M, args.units, gen)

    def regime(name, x, w=None, b=None):
        with torch.no_grad():
            lay.weight.copy_(w0 if w is None else w)
            lay.bias.copy_(b0 if b is None else b)
        out = run_diag(lay, x, items)
        out["zref"] = reference_z(lay, x, items)
        tok = items[:, 1].long()[:, None] * 16 + torch.arange(16, device="cuda")[None]
        out["valid"] = (tok < M)[:, None, :, None].expand(-1, 4, -1, 8).contiguous()
        r = stats(out, name)
        results.append(r)
        print({k: v for k, v in r.items()}, flush=True)
        return out

    def bf(t):
        return t.to(BF)

    xr = bf(torch.randn(M, D, device="cuda", generator=gx))
    regime("random", xr)
    # signed zeros: half of the inputs +-0, a quarter of weights / biases +-0
    xz = xr.clone()
    sel = torch.rand(M, D, device="cuda", generator=gx) < 0.5
    xz[sel] = bf(torch.where(torch.rand(M, D, device="cuda", generator=gx) < 0.5, 0.0, -0.0))[sel]
    wz = w0.clone()
    wz[torch.rand(wz.shape, device="cuda", generator=gx) < 0.25] = -0.0
    bz = b0.clone()
    bz[torch.rand(bz.shape, device="cuda", generator=gx) < 0.25] = 0.0
    bz[torch.rand(bz.shape, device="cuda", generator=gx) < 0.25] = -0.0
    regime("signed_zeros", xz, wz, bz)
    # all-zero inputs: z = bias exactly; bias +0 / -0 / tiny / subnormal
    x0 = torch.zeros(M, D, device="cuda", dtype=BF)
    bs = b0.clone()
    r_ = torch.rand(bs.shape, device="cuda", generator=gx)
    bs[r_ < 0.2] = 0.0
    bs[(r_ >= 0.2) & (r_ < 0.4)] = -0.0
    bs[(r_ >= 0.4) & (r_ < 0.6)] = bf(torch.tensor(2.0**-130, device="cuda"))  # BF16 subnormal bias
    bs[(r_ >= 0.6) & (r_ < 0.8)] = bf(torch.tensor(-2.0**-130, device="cuda"))
    regime("zero_input_bias_zero_subnormal", x0, None, bs)
    # subnormal FP32 products: x, w ~ 2^-70 (normal BF16), x * w ~ 2^-140
    xs = bf(torch.randn(M, D, device="cuda", generator=gx) * 2.0**-70)
    ws = bf(w0.float() * 2.0**-70 / (w0.float().abs().max() + 1e-30))
    regime("subnormal_products_bias0", xs, ws, torch.zeros_like(b0))
    regime("subnormal_products_bias_tiny", xs, ws, bf(b0.float() * 2.0**-138))
    # huge: products overflow -> inf / NaN z
    xh = bf(torch.randn(M, D, device="cuda", generator=gx) * 2.0**60)
    wh = bf(w0.float() * 2.0**70)
    regime("huge_inf_nan", xh, wh, b0)
    # NaN / inf inputs: x = NaN on some features, x = +-inf on others (inf * 0 weights -> NaN)
    xn = xr.clone()
    r_ = torch.rand(M, D, device="cuda", generator=gx)
    xn[r_ < 0.01] = float("nan")
    xn[(r_ >= 0.01) & (r_ < 0.02)] = float("inf")
    xn[(r_ >= 0.02) & (r_ < 0.03)] = -float("inf")
    regime("nan_inf_inputs", xn, wz, b0)
    # near-zero / exact cancellation: bias = -round_bf16(x . w) for each (token, neuron) is not
    # expressible per token; use integer-valued operands so x.w is exact and pick bias = -(x.w) of
    # token 0 in each tile -> z = +-0 exactly for those tokens, small integers elsewhere
    xi = bf(torch.randint(-3, 4, (M, D), device="cuda", generator=gx).float())
    wi = bf(torch.randint(-3, 4, w0.shape, device="cuda", generator=gx).float())
    regime("integer_operands_bias0", xi, wi, torch.zeros_like(b0))
    # near-zero: x.w + b with b = -(x.w) computed for one shared input row (all tokens equal)
    xrow = bf(torch.randn(1, D, device="cuda", generator=gx)).expand(M, D).contiguous()
    with torch.no_grad():
        lay.weight.copy_(w0)
    # z without bias for this row: z_dx - bias; set bias = -bf16(z_nobias) at the block level
    # (computed from the dX output of a zero-bias pass), giving tiny residual z of either sign
    with torch.no_grad():
        lay.bias.zero_()
    out0 = run_diag(lay, xrow, items)
    zb = out0["zdx"][:, :, 0, :]  # [n, 4, 8]: token-independent
    # write -bf16(z) into the packed bias of exactly these blocks via the canonical bias view
    bcan = L.unpack_bias(lay.bias.detach().clone()) if hasattr(L, "unpack_bias") else None
    if bcan is not None:
        NU = L.blocks_per_group // L.unit_blocks
        u = items[:, 0].long()
        g, lu = u // NU, u % NU
        bc = bcan.view(L.groups, L.blocks_per_group, 8)
        for bu in range(4):
            bc[g, lu * 4 + bu] = (-zb[:, bu]).to(bc.dtype)
        with torch.no_grad():
            lay.bias.copy_(L.pack_bias(bc.reshape(bcan.shape)))
        r = regime("near_zero_cancellation", xrow, w0, lay.bias.detach().clone())
        del r
    else:
        results.append({"regime": "near_zero_cancellation", "skipped": "layout has no unpack_bias / pack_bias"})
    save(args.out, {"args": vars(args), "shape": dict(D=D, G=G, H=H, M=M, S=S, C=8), "items": int(items.shape[0]),
                    "results": results})


if __name__ == "__main__":
    main()
