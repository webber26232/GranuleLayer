"""Triton kernels: fused sparse projection + streaming pooling, and recompute
backward.

Pool ids (tl.constexpr POOL): 0 max, 1 topk_mean, 2 mean, 3 logmeanexp,
4 power_mean, 5 softmax_weighted.  Activation ids: 0 identity, 1 relu, 2 gelu.

Tiling (Mode B, block-shared indices): one iteration consumes a *pair* of index
blocks (2*C neurons).  The two blocks' gathered inputs form A [BM, 16] and the
weights form a block-diagonal B [16, 2C] (zeros off the diagonal, so neurons only
ever see their own 8 inputs).  K = 16 is the smallest BF16 tl.dot, so half of
every MMA is structural zeros -- this is the cost of Triton's minimum K and is
why the native m16n8k8 CUDA path exists.

Mode A (independent indices) is SIMT: x [BM, 2C, 8] gathered, z = sum_k x*w.

The hidden activations exist only inside a tile; only compact pooling states
are written to global memory.
"""
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# elementwise helpers
# ---------------------------------------------------------------------------
@triton.jit
def _act(z, ACT: tl.constexpr):
    if ACT == 0:
        return z
    elif ACT == 1:
        return tl.maximum(z, 0.0)
    else:
        return 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))


@triton.jit
def _dact(z, ACT: tl.constexpr):
    if ACT == 0:
        return tl.full(z.shape, 1.0, tl.float32)
    elif ACT == 1:
        return (z > 0.0).to(tl.float32)
    else:
        cdf = 0.5 * (1.0 + tl.erf(z * 0.7071067811865476))
        pdf = 0.3989422804014327 * tl.exp(-0.5 * z * z)
        return cdf + z * pdf


@triton.jit
def _topk_key(z, slot):
    """Order-preserving int64 key: (value, -id).  Larger key = preferred."""
    zb = z.to(tl.int32, bitcast=True)
    s = tl.where(zb >= 0, zb, zb ^ 0x7FFFFFFF)
    return (s.to(tl.int64) << 32) | (4294967295 - slot.to(tl.int64))


@triton.jit
def _topk_reduce_to(keys, BM: tl.constexpr, W: tl.constexpr, KP: tl.constexpr):
    """keys [BM, W] -> top KP per row [BM, KP] (descending), W >= KP."""
    if W == KP:
        return tl.sort(keys, dim=1, descending=True)
    else:
        srt = tl.sort(keys, dim=1, descending=True)
        return tl.max(tl.reshape(srt, [BM, W // KP, KP]), axis=1)


@triton.jit
def _widen(keys, BM: tl.constexpr, W: tl.constexpr, KP: tl.constexpr):
    """Pad [BM, W] with -inf keys up to [BM, KP] (KP >= W, ratio <= 8)."""
    if W >= KP:
        return keys
    else:
        neg = tl.full([BM, W], -9223372036854775807 - 1, tl.int64)
        k2 = tl.reshape(tl.join(keys, neg), [BM, 2 * W])
        if 2 * W >= KP:
            return k2
        else:
            neg2 = tl.full([BM, 2 * W], -9223372036854775807 - 1, tl.int64)
            k4 = tl.reshape(tl.join(k2, neg2), [BM, 4 * W])
            if 4 * W >= KP:
                return k4
            else:
                neg4 = tl.full([BM, 4 * W], -9223372036854775807 - 1, tl.int64)
                return tl.reshape(tl.join(k4, neg4), [BM, 8 * W])



# ---------------------------------------------------------------------------
# parameter addressing (canonical K-contiguous or fragment-packed, see layout.py)
# fblk is the flat block index g * NB + blk
# ---------------------------------------------------------------------------
@triton.jit
def _w_off(fblk, c, k, C: tl.constexpr, FRAG: tl.constexpr):
    if FRAG:
        f = fblk * (C // 8) + c // 8
        return ((f // 4) * 32 + (c % 8) * 4 + k // 2) * 8 + (f % 4) * 2 + k % 2
    else:
        return (fblk * C + c) * 8 + k


@triton.jit
def _b_off(fblk, c, C: tl.constexpr, FRAG: tl.constexpr):
    if FRAG:
        f = fblk * (C // 8) + c // 8
        return ((f // 4) * 4 + (c % 8) // 2) * 8 + (f % 4) * 2 + c % 2
    else:
        return fblk * C + c


@triton.jit
def _i_off(fblk, k, FRAG: tl.constexpr, UB: tl.constexpr):
    if FRAG:
        return ((fblk // UB) * 8 + k) * UB + fblk % UB
    else:
        return fblk * 8 + k

# ---------------------------------------------------------------------------
# streaming state update (one tile)
# ---------------------------------------------------------------------------
@triton.jit
def _update(
    z, slot, valid, m, s, t, wid, keys, scale, pparam,
    POOL: tl.constexpr, ACT: tl.constexpr, MASKED: tl.constexpr, TRACK_IDS: tl.constexpr,
    KP: tl.constexpr, BM: tl.constexpr, N: tl.constexpr, P_INT: tl.constexpr,
    TOPK_SORT: tl.constexpr = False,
):
    """Merge tile z [BM, N] (slots [N], validity [N]) into the running state."""
    if POOL == 0:  # ---- max
        a = _act(z, ACT)
        if MASKED:
            a = tl.where(valid[None, :], a, float("-inf"))
        tm = tl.max(a, axis=1)
        if TRACK_IDS:
            # smallest slot attaining the tile max; tiles arrive in increasing
            # slot order, so a tie with the running max keeps the old winner
            tid = tl.min(tl.where(a == tm[:, None], slot[None, :], 2147483647), axis=1)
            better = tm > m
            wid = tl.where(better, tid, wid)
        m = tl.maximum(m, tm)
    elif POOL == 1:  # ---- top-k (ranks raw pre-activations)
        zz = z
        if MASKED:
            zz = tl.where(valid[None, :], z, float("-inf"))
        tkeys = _topk_key(zz, slot[None, :] + tl.zeros([BM, N], tl.int32))
        if TOPK_SORT:
            # bitonic merge of (state U tile); cost independent of #insertions
            thr = tl.min(keys, axis=1)
            hit = tl.max((tkeys > thr[:, None]).to(tl.int32), axis=1)
            if tl.max(hit, axis=0) > 0:
                if N >= KP:
                    tk = _topk_reduce_to(tkeys, BM, N, KP)
                else:
                    tk = _widen(tkeys, BM, N, KP)
                both = tl.reshape(tl.join(keys, tk), [BM, 2 * KP])
                keys = _topk_reduce_to(both, BM, 2 * KP, KP)
        else:
            # incremental insertion: the state is an unordered set of KP keys;
            # each round inserts every row's best remaining candidate in place of
            # that row's current minimum.  Rounds = max #insertions in any row
            # of this tile (usually 0 or 1 after warm-up).
            col = tl.arange(0, KP)
            neg = -9223372036854775807 - 1
            thr = tl.min(keys, axis=1)
            nc = tl.max(tl.max((tkeys > thr[:, None]).to(tl.int32), axis=1), axis=0)
            while nc > 0:
                best = tl.max(tl.where(tkeys > thr[:, None], tkeys, neg), axis=1)
                has = best > thr
                pos = tl.min(tl.where(keys == thr[:, None], col[None, :], KP), axis=1)
                keys = tl.where((col[None, :] == pos[:, None]) & has[:, None], best[:, None], keys)
                tkeys = tl.where(tkeys == best[:, None], neg, tkeys)
                thr = tl.min(keys, axis=1)
                nc = tl.max(tl.max((tkeys > thr[:, None]).to(tl.int32), axis=1), axis=0)
    elif POOL == 2:  # ---- mean(act)
        a = _act(z, ACT)
        if MASKED:
            a = tl.where(valid[None, :], a, 0.0)
        s += tl.sum(a, axis=1)
    elif POOL == 3 or POOL == 5:  # ---- logmeanexp / softmax-weighted (base-2 online LSE)
        a = _act(z, ACT)
        u = a * scale
        if MASKED:
            u = tl.where(valid[None, :], u, float("-inf"))
        mn = tl.maximum(m, tl.max(u, axis=1))
        r = tl.exp2(m - mn)
        e = tl.exp2(u - mn[:, None])
        s = s * r + tl.sum(e, axis=1)
        if POOL == 5:
            t = t * r + tl.sum(e * a, axis=1)
        m = mn
    elif POOL == 4:  # ---- power mean on relu activations, max-scaled
        a = tl.maximum(z, 0.0)
        if MASKED:
            a = tl.where(valid[None, :], a, 0.0)
        mn = tl.maximum(m, tl.max(a, axis=1))
        inv = tl.where(mn > 0.0, 1.0 / mn, 0.0)
        ratio_old = m * inv
        q = a * inv[:, None]
        if P_INT == 1:
            ro = ratio_old
            qp = q
        elif P_INT == 2:
            ro = ratio_old * ratio_old
            qp = q * q
        elif P_INT == 4:
            ro = ratio_old * ratio_old
            ro = ro * ro
            qp = q * q
            qp = qp * qp
        elif P_INT == 8:
            ro = ratio_old * ratio_old
            ro = ro * ro
            ro = ro * ro
            qp = q * q
            qp = qp * qp
            qp = qp * qp
        else:
            ro = tl.exp2(pparam * tl.log2(ratio_old))
            qp = tl.exp2(pparam * tl.log2(q))
        s = s * ro + tl.sum(qp, axis=1)
        m = mn
    return m, s, t, wid, keys


# ---------------------------------------------------------------------------
# tile computation
# ---------------------------------------------------------------------------
@triton.jit
def _tile_z(
    X, sxf, sxb, IDX, W, BIAS, gbase, pair, tok, tmask,
    MODE_A: tl.constexpr, C: tl.constexpr, BM: tl.constexpr, EVEN_B: tl.constexpr,
    FRAG: tl.constexpr, UB: tl.constexpr,
):
    """Pre-activations of one block pair: z [BM, 2C] fp32 (+ gathered operands)."""
    N: tl.constexpr = 2 * C
    n = tl.arange(0, N)
    blk0 = (gbase + 2 * pair).to(tl.int64)  # flat block index of the first block
    bias = tl.load(BIAS + _b_off(blk0 + n // C, n % C, C, FRAG)).to(tl.float32)
    if not MODE_A:
        r = tl.arange(0, 16)
        idx16 = tl.load(IDX + _i_off(blk0 + r // 8, r % 8, FRAG, UB)).to(tl.int64)
        aptr = X + idx16[None, :] * sxf + tok[:, None].to(tl.int64) * sxb
        if EVEN_B:
            a = tl.load(aptr)
        else:
            a = tl.load(aptr, mask=tmask[:, None], other=0.0)
        wptr = W + _w_off(blk0 + n[None, :] // C, n[None, :] % C, r[:, None] % 8, C, FRAG)
        wmask = (r[:, None] // 8) == (n[None, :] // C)
        wt = tl.load(wptr, mask=wmask, other=0.0)
        acc = tl.zeros([BM, N], tl.float32) + bias[None, :]
        z = tl.dot(a, wt, acc=acc)
        return z, a, wt
    else:
        k = tl.arange(0, 8)
        nk = (blk0 * C + n[:, None]) * 8 + k[None, :]  # [N, 8] (Mode A: canonical)
        idx = tl.load(IDX + nk).to(tl.int64)
        w = tl.load(W + nk).to(tl.float32)
        xptr = X + idx[None, :, :] * sxf + tok[:, None, None].to(tl.int64) * sxb
        if EVEN_B:
            xg = tl.load(xptr).to(tl.float32)
        else:
            xg = tl.load(xptr, mask=tmask[:, None, None], other=0.0).to(tl.float32)
        z = tl.sum(xg * w[None, :, :], axis=2) + bias[None, :]
        return z, xg, w


@triton.jit
def _range_for_program(u, ps, COUNTS, S_SEM, NB, BPS, BPSP, C: tl.constexpr):
    """Pair range [p_lo, p_hi) and the first pair containing padding."""
    g = u // S_SEM
    sh = u % S_SEM
    n_g = tl.load(COUNTS + g)
    b_lo = sh * BPS + ps * BPSP
    b_hi = tl.minimum(tl.minimum(b_lo + BPSP, (sh + 1) * BPS), NB)
    p_lo = b_lo // 2
    p_hi = b_hi // 2
    # pairs whose 2C slots are all valid
    full_end = n_g // (2 * C)
    # pairs with at least one valid slot
    any_end = (n_g + 2 * C - 1) // (2 * C)
    p_full = tl.minimum(tl.maximum(full_end, p_lo), p_hi)
    p_any = tl.minimum(tl.maximum(any_end, p_lo), p_hi)
    return g, n_g, p_lo, p_full, p_any


# ---------------------------------------------------------------------------
# forward: fused projection + local pooling, writes compact states
# ---------------------------------------------------------------------------
@triton.jit
def fwd_pool_kernel(
    X, sxf, sxb, IDX, W, BIAS, COUNTS,
    ST_M, ST_S, ST_T, ST_ID, ST_KEYS,
    B, U, S_SEM, NB, BPS, BPSP, n_tiles,
    scale, pparam,
    POOL: tl.constexpr, ACT: tl.constexpr, MODE_A: tl.constexpr, C: tl.constexpr,
    KP: tl.constexpr, TRACK_IDS: tl.constexpr, BM: tl.constexpr, TT: tl.constexpr,
    EVEN_B: tl.constexpr, P_INT: tl.constexpr, TOPK_SORT: tl.constexpr = False,
    FRAG: tl.constexpr = False, UB: tl.constexpr = 1, TOPK_FINAL: tl.constexpr = False,
):
    N: tl.constexpr = 2 * C
    pid_t = tl.program_id(0)
    u = tl.program_id(1)
    ps = tl.program_id(2)
    PS = tl.num_programs(2)
    g, n_g, p_lo, p_full, p_any = _range_for_program(u, ps, COUNTS, S_SEM, NB, BPS, BPSP, C)
    gbase = g * NB
    n = tl.arange(0, N)
    for tt in range(TT):
        tile = pid_t * TT + tt
        tok = tile * BM + tl.arange(0, BM)
        tmask = tok < B
        m = tl.full([BM], float("-inf"), tl.float32)
        if POOL == 2 or POOL == 4:
            m = tl.zeros([BM], tl.float32)
        s = tl.zeros([BM], tl.float32)
        t = tl.zeros([BM], tl.float32)
        wid = tl.full([BM], 2147483647, tl.int32)
        keys = tl.full([BM, KP], -9223372036854775807 - 1, tl.int64)
        if tile < n_tiles:
            for pair in range(p_lo, p_full):
                z, _a, _w = _tile_z(X, sxf, sxb, IDX, W, BIAS, gbase, pair, tok, tmask, MODE_A, C, BM, EVEN_B, FRAG, UB)
                slot = pair * N + n
                m, s, t, wid, keys = _update(z, slot, n < N, m, s, t, wid, keys, scale, pparam,
                                             POOL, ACT, False, TRACK_IDS, KP, BM, N, P_INT, TOPK_SORT)
            for pair in range(p_full, p_any):
                z, _a, _w = _tile_z(X, sxf, sxb, IDX, W, BIAS, gbase, pair, tok, tmask, MODE_A, C, BM, EVEN_B, FRAG, UB)
                slot = pair * N + n
                m, s, t, wid, keys = _update(z, slot, slot < n_g, m, s, t, wid, keys, scale, pparam,
                                             POOL, ACT, True, TRACK_IDS, KP, BM, N, P_INT, TOPK_SORT)
            off = (ps.to(tl.int64) * B + tok) * U + u
            if POOL == 1:
                kk = tl.arange(0, KP)
                if TOPK_FINAL:
                    # single split: finalize in registers (no int64 key tensor in HBM)
                    neg = -9223372036854775807 - 1
                    valid = keys > neg
                    sb = (keys >> 32).to(tl.int32)
                    vbits = tl.where(sb >= 0, sb, sb ^ 0x7FFFFFFF)
                    val = vbits.to(tl.float32, bitcast=True)
                    slot = (4294967295 - (keys & 4294967295)).to(tl.int32)
                    sh = u % S_SEM
                    n_u = tl.minimum(tl.maximum(n_g - sh * BPS * C, 0), BPS * C)
                    den = tl.minimum(n_u, KP).to(tl.float32)
                    pre = tl.sum(tl.where(valid, _act(val, ACT), 0.0), axis=1) / den
                    tl.store(ST_M + off, pre, mask=tmask)
                    if TRACK_IDS:
                        tl.store(ST_ID + off[:, None] * KP + kk[None, :], tl.where(valid, slot, -1),
                                 mask=tmask[:, None])
                else:
                    tl.store(ST_KEYS + off[:, None] * KP + kk[None, :], keys, mask=tmask[:, None])
            else:
                tl.store(ST_M + off, m, mask=tmask)
                if POOL != 0:
                    tl.store(ST_S + off, s, mask=tmask)
                if POOL == 5:
                    tl.store(ST_T + off, t, mask=tmask)
                if TRACK_IDS:
                    tl.store(ST_ID + off, wid, mask=tmask)


# ---------------------------------------------------------------------------
# dense recompute backward (generic; FP32 atomics)
# ---------------------------------------------------------------------------
@triton.jit
def _dz_from_z(z, c1, c2, c3, scale, pparam, POOL: tl.constexpr, ACT: tl.constexpr, P_INT: tl.constexpr):
    """dL/dz for smooth operators from compact per-row coefficients.

    mean:      c1 = g / n
    lme:       c1 = g,  c2 = log2-normaliser L2
    power:     c1 = g / n,  c2 = 1 / y  (c1 = 0 when y == 0)
    softmax_w: c1 = g,  c2 = L2,  c3 = y
    """
    if POOL == 2:
        return c1[:, None] * _dact(z, ACT)
    elif POOL == 3:
        a = _act(z, ACT)
        p = tl.exp2(a * scale - c2[:, None])
        return c1[:, None] * p * _dact(z, ACT)
    elif POOL == 4:
        a = tl.maximum(z, 0.0)
        q = a * c2[:, None]
        if P_INT == 1:
            qp = tl.full(q.shape, 1.0, tl.float32)
        elif P_INT == 2:
            qp = q
        elif P_INT == 4:
            qp = q * q * q
        elif P_INT == 8:
            q2 = q * q
            q4 = q2 * q2
            qp = q4 * q2 * q
        else:
            qp = tl.exp2((pparam - 1.0) * tl.log2(q))
        return tl.where(z > 0.0, c1[:, None] * qp, 0.0)
    else:  # POOL == 5
        a = _act(z, ACT)
        p = tl.exp2(a * scale - c2[:, None])
        beta = scale * 0.6931471805599453
        return c1[:, None] * p * (1.0 + beta * (a - c3[:, None])) * _dact(z, ACT)


@triton.jit
def _bwd_tile(
    X, sxf, sxb, IDX, W, BIAS, DX, sdf, sdb, DW, DB,
    gbase, pair, n_g, tok, tmask, c1, c2, c3, scale, pparam,
    POOL: tl.constexpr, ACT: tl.constexpr, MODE_A: tl.constexpr, C: tl.constexpr,
    BM: tl.constexpr, EVEN_B: tl.constexpr, P_INT: tl.constexpr, MASKED: tl.constexpr,
    FRAG: tl.constexpr, UB: tl.constexpr,
):
    N: tl.constexpr = 2 * C
    n = tl.arange(0, N)
    z, a, w = _tile_z(X, sxf, sxb, IDX, W, BIAS, gbase, pair, tok, tmask, MODE_A, C, BM, EVEN_B, FRAG, UB)
    dz = _dz_from_z(z, c1, c2, c3, scale, pparam, POOL, ACT, P_INT)
    slot = pair * N + n
    vmask = slot < n_g
    if MASKED:
        dz = tl.where(vmask[None, :], dz, 0.0)
    if not EVEN_B:
        dz = tl.where(tmask[:, None], dz, 0.0)
    blk0 = (gbase + 2 * pair).to(tl.int64)
    tl.atomic_add(DB + _b_off(blk0 + n // C, n % C, C, FRAG), tl.sum(dz, axis=0), mask=vmask, sem="relaxed")
    if not MODE_A:
        r = tl.arange(0, 16)
        # dz as an unevaluated sum of two bf16 numbers (~16-bit mantissa)
        dz_hi = dz.to(a.dtype)
        dz_lo = (dz - dz_hi.to(tl.float32)).to(a.dtype)
        at = tl.trans(a)
        dwbd = tl.dot(at, dz_hi) + tl.dot(at, dz_lo)  # [16, N]
        wptr = DW + _w_off(blk0 + n[None, :] // C, n[None, :] % C, r[:, None] % 8, C, FRAG)
        wmask = ((r[:, None] // 8) == (n[None, :] // C)) & vmask[None, :]
        tl.atomic_add(wptr, dwbd, mask=wmask, sem="relaxed")
        wtt = tl.trans(w)  # [N, 16] block diagonal
        da = tl.dot(dz_hi, wtt) + tl.dot(dz_lo, wtt)  # [BM, 16]
        idx16 = tl.load(IDX + _i_off(blk0 + r // 8, r % 8, FRAG, UB)).to(tl.int64)
        dptr = DX + idx16[None, :] * sdf + tok[:, None].to(tl.int64) * sdb
        tl.atomic_add(dptr, da, mask=tmask[:, None], sem="relaxed")
    else:
        k = tl.arange(0, 8)
        nk = (blk0 * C + n[:, None]) * 8 + k[None, :]
        dw = tl.sum(dz[:, :, None] * a, axis=0)  # [N, 8]
        tl.atomic_add(DW + nk, dw, mask=vmask[:, None], sem="relaxed")
        idx = tl.load(IDX + nk).to(tl.int64)
        dxg = dz[:, :, None] * w[None, :, :]  # [BM, N, 8]
        dptr = DX + idx[None, :, :] * sdf + tok[:, None, None].to(tl.int64) * sdb
        tl.atomic_add(dptr, dxg, mask=tmask[:, None, None] & vmask[None, :, None], sem="relaxed")


@triton.jit
def bwd_dense_kernel(
    X, sxf, sxb, IDX, W, BIAS, COUNTS, DX, sdf, sdb, DW, DB,
    C1, C2, C3, B, U, S_SEM, NB, BPS, BPSP, scale, pparam,
    POOL: tl.constexpr, ACT: tl.constexpr, MODE_A: tl.constexpr, C: tl.constexpr,
    BM: tl.constexpr, EVEN_B: tl.constexpr, P_INT: tl.constexpr,
    FRAG: tl.constexpr = False, UB: tl.constexpr = 1,
):
    pid_t = tl.program_id(0)
    u = tl.program_id(1)
    ps = tl.program_id(2)
    g, n_g, p_lo, p_full, p_any = _range_for_program(u, ps, COUNTS, S_SEM, NB, BPS, BPSP, C)
    gbase = g * NB
    tok = pid_t * BM + tl.arange(0, BM)
    tmask = tok < B
    off = tok.to(tl.int64) * U + u
    c1 = tl.load(C1 + off, mask=tmask, other=0.0)
    c2 = tl.load(C2 + off, mask=tmask, other=0.0)
    c3 = tl.load(C3 + off, mask=tmask, other=0.0)
    for pair in range(p_lo, p_full):
        _bwd_tile(X, sxf, sxb, IDX, W, BIAS, DX, sdf, sdb, DW, DB, gbase, pair, n_g, tok, tmask,
                  c1, c2, c3, scale, pparam, POOL, ACT, MODE_A, C, BM, EVEN_B, P_INT, False, FRAG, UB)
    for pair in range(p_full, p_any):
        _bwd_tile(X, sxf, sxb, IDX, W, BIAS, DX, sdf, sdb, DW, DB, gbase, pair, n_g, tok, tmask,
                  c1, c2, c3, scale, pparam, POOL, ACT, MODE_A, C, BM, EVEN_B, P_INT, True, FRAG, UB)


# ---------------------------------------------------------------------------
# sparse winner-event backward (Max / Top-K)
# ---------------------------------------------------------------------------
@triton.jit
def bwd_events_kernel(
    X, sxf, sxb, IDX, W, BIAS, DX, sdf, sdb, DW, DB,
    EV_SLOT, EV_COEF, n_events, KPE, U, S_SEM, NB,
    ACT: tl.constexpr, MODE_A: tl.constexpr, C: tl.constexpr, BE: tl.constexpr,
    FRAG: tl.constexpr = False, UB: tl.constexpr = 1,
):
    """Each event (b, u, slot, coef): dz = coef * act'(z) for one neuron.

    EV_SLOT / EV_COEF are [B, U, KPE] (KPE = 1 for max, k for top-k); slot < 0
    marks an empty event.  z is recomputed (8 MACs) - only needed for act'.
    """
    e = tl.program_id(0) * BE + tl.arange(0, BE)
    emask = e < n_events
    slot = tl.load(EV_SLOT + e, mask=emask, other=-1)
    coef = tl.load(EV_COEF + e, mask=emask, other=0.0)
    emask = emask & (slot >= 0) & (coef != 0.0)
    slot = tl.maximum(slot, 0)
    bu = e // KPE
    b = bu // U
    u = bu % U
    g = u // S_SEM
    blk = g.to(tl.int64) * NB + slot // C
    c = slot % C
    k = tl.arange(0, 8)
    if MODE_A:
        idx = tl.load(IDX + (blk * C + c)[:, None] * 8 + k[None, :], mask=emask[:, None], other=0).to(tl.int64)
    else:
        idx = tl.load(IDX + _i_off(blk[:, None], k[None, :], FRAG, UB), mask=emask[:, None], other=0).to(tl.int64)
    woff = _w_off(blk[:, None], c[:, None], k[None, :], C, FRAG)
    boff = _b_off(blk, c, C, FRAG)
    w = tl.load(W + woff, mask=emask[:, None], other=0.0).to(tl.float32)
    bias = tl.load(BIAS + boff, mask=emask, other=0.0).to(tl.float32)
    xg = tl.load(X + idx * sxf + b[:, None].to(tl.int64) * sxb, mask=emask[:, None], other=0.0).to(tl.float32)
    z = tl.sum(xg * w, axis=1) + bias
    dz = coef * _dact(z, ACT)
    dz = tl.where(emask, dz, 0.0)
    tl.atomic_add(DB + boff, dz, mask=emask, sem="relaxed")
    tl.atomic_add(DW + woff, dz[:, None] * xg, mask=emask[:, None], sem="relaxed")
    tl.atomic_add(DX + idx * sdf + b[:, None].to(tl.int64) * sdb, dz[:, None] * w, mask=emask[:, None], sem="relaxed")
