// Input-supergroup ("grouped") kernels for Mean(act) pooling (supergroups.py).
//
// Every index record of supergroup q uses features of one S-feature set feat[q], so a CTA
// whose work all belongs to q stages only those S rows of X (instead of all D) and can use
// a larger token tile.  The global int16 indices of the existing packed parameters are
// translated to local slab rows through a D-entry shared-memory map (no extra index
// tensor); records keep their fan-in order, so weights pair with the same inputs and the
// MMAs see the same operands as in the generic kernels.
//
// Work lists (host-built, supergroups.work_items / chunk_table / unit_list):
//   items  [n, 3]  (g, first flat unit, unit count): a bucket = contiguous units of output g
//   chunks [m, 3]  (q, lo, hi): a CTA's range of items (fwd, dX) or of q-ordered units (dW)
//
//   sg_fwd  CTA = (chunk, token tile of BM = 16 MT tokens).  A warp takes whole items: the
//           Mean sum of its tokens over the item's neurons stays in registers, then one FP32
//           atomicAdd per (token, item) into ysum [G, B] (token-contiguous -> 32 B sectors).
//   sg_dx   CTA = (chunk, BM tokens).  dA = dZ W^T (as k_dx) is accumulated into FP32 slab
//           copies [ncopy][S][BM]: dx_group_warps warps share one copy (1 = warp-private,
//           plain read-modify-write, no CAS; otherwise the 64-bit CAS of k_dx).  The copies
//           are summed and flushed with one FP32 atomicAdd per (feature, token) per CTA into
//           dXT [D, B] (token-contiguous).
//   sg_dw   CTA = (chunk of units of one supergroup), all tokens; exclusive ownership of its
//           units' dW/db (register accumulators, plain RMW write-back, as k_dw_idx).  The S
//           rows of TS 16-token tiles are staged per pipeline stage into a CTA-shared ring.
//           dw_block / the token loop order are those of k_dw_idx: dW/db are bitwise
//           identical to the generic kernels for the same epilogue.
#pragma once
#include "sparse_pool_bwd.cuh"

// Attribution ablations (benchmarks only, WRONG results; separate builds through
// SPARSE_POOL_NVCC_EXTRA=-DSP_ABLATE_SG=n):
//   1: sg_dx without the global dX flush (no FP32 atomics to dXT)
//   2: sg_dx without the local slab scatter (products kept live, never stored)
//   3: sg_fwd without the per-(token, item) global atomics
#ifndef SP_ABLATE_SG
#define SP_ABLATE_SG 0
#endif

struct SgArgs {
  const bf16* X;  // feature-major X^T: element (b, f) at X[f * sxf + b]
  long long sxf;
  const int16_t* idx;
  const bf16* W;
  const bf16* bias;
  const int* counts;  // [G]
  const int16_t* feat;  // [Q, S] sorted global features
  const int* items;     // [n_items, 3]
  const int* chunks;    // [n_chunks, 3]
  const int* units;     // dW: flat unit ids ordered by supergroup
  float* ysum;          // fwd: [G, B]
  const float* c1T;     // bwd: Mean coefficients [G, B]
  float* dXT;           // [D, B]
  float* dW;
  float* dB;
  int B, G, D, S, NB, NU;  // NU = NB / UB units per output
  int dx_group_warps;
  int flags;
  const unsigned char* nonfinite;
};

__device__ __forceinline__ int sg_align16(int bytes) { return (bytes + 15) & ~15; }

// stage X[t0 : t0 + 16 MT, feat[q, :]] into the swizzled [S][16 MT] slab and build the
// global -> local row map (Swz<MT> is conflict-free for every MT: CH >= 8 xors c with f & 7)
template <int MT>
__device__ __forceinline__ void sg_stage(const SgArgs& a, const int16_t* fq, int16_t* smap, bf16* slab, int t0) {
  constexpr int BM = 16 * MT, CH = BM / 8;
  const int nthr = blockDim.x;
  for (int i = threadIdx.x; i < a.S; i += nthr) smap[fq[i]] = (int16_t)i;
  for (int i = threadIdx.x; i < a.S * CH; i += nthr) {
    const int f = i / CH, c = i % CH;
    const int tok = t0 + c * 8;
    const bf16* src = a.X + (long long)fq[f] * a.sxf + tok;
    const int bytes = (tok + 8 <= a.B) ? 16 : (tok < a.B ? 2 * (a.B - tok) : 0);
    if (bytes == 0) src = a.X;
    cp_async16(slab + Swz<MT>::off(f, c), src, bytes);
  }
  cp_async_wait_all();
}

// A fragments of all MT m16 tiles of one block: af[2 mt], af[2 mt + 1] (rows = tokens)
template <int MT>
__device__ __forceinline__ void sg_gather(uint32_t* af, uint32_t slab_s, int fidx, int lane) {
  const int mat = lane >> 3;
  if constexpr (MT == 1) {
    ldmatrix_x2_trans(af, slab_s + 2 * Swz<1>::off(fidx, mat & 1));
  } else {
#pragma unroll
    for (int j = 0; j < MT / 2; j++) ldmatrix_x4_trans(af + 4 * j, slab_s + 2 * Swz<MT>::off(fidx, 4 * j + mat));
  }
}

// translate the packed unit's block bu to the per-block register form with a local row
template <int C>
__device__ __forceinline__ void sg_block(BlockRegs<C>& blk, const Unit<C>& un, int bu, const int16_t* smap) {
  unit_block<C>(blk, un, bu);
  blk.fidx = smap[blk.fidx];
}

// per-warp walk over items it = lo + warp, lo + warp + NW, ...: every unit of an item, with a
// one-unit register prefetch that crosses item boundaries
#define SG_ITEM_LOOP_BEGIN                                                         \
  int it = lo + warp;                                                              \
  Unit<C> nxt;                                                                     \
  if (it < hi) load_unit<C>(nxt, fa, a.items[3 * it + 1], lane);                   \
  for (; it < hi; it += NW) {                                                      \
    const int g = a.items[3 * it], u0 = a.items[3 * it + 1], nu = a.items[3 * it + 2]; \
    const int n_g = a.counts[g];                                                   \
    const int lu0 = u0 - g * a.NU;

#define SG_UNIT_LOOP_BEGIN                                                         \
    for (int k = 0; k < nu; k++) {                                                 \
      Unit<C> cur = nxt;                                                           \
      {                                                                            \
        int nit = it, nk = k + 1;                                                  \
        if (nk == nu) { nit = it + NW; nk = 0; }                                   \
        if (nit < hi) load_unit<C>(nxt, fa, a.items[3 * nit + 1] + nk, lane);      \
      }                                                                            \
      const int jb = (lu0 + k) * UB;                    /* first block of the unit */ \
      const bool full = (jb + UB) * C <= n_g;

#define SG_UNIT_LOOP_END }
#define SG_ITEM_LOOP_END }

// ============================================================================
// forward: Mean(act)
// ============================================================================
template <int ACT, int C, int MT, bool MASKED>
__device__ __forceinline__ void sg_fwd_block(float* s, const BlockRegs<C>& cur, int j, int n_g, uint32_t slab_s,
                                             int lane) {
  constexpr int NT = C / 8;
  const int col0 = 2 * (lane & 3);
  uint32_t af[2 * MT];
  sg_gather<MT>(af, slab_s, cur.fidx, lane);
#pragma unroll
  for (int n0 = 0; n0 < NT; n0 += (NT < 2 ? 1 : 2)) {
    constexpr int NTC = NT < 2 ? NT : 2;
    bool valid[2 * NTC];
#pragma unroll
    for (int q = 0; q < NTC; q++) {
      const int slot = j * C + (n0 + q) * 8 + col0;
      valid[2 * q] = !MASKED || slot < n_g;
      valid[2 * q + 1] = !MASKED || slot + 1 < n_g;
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float d[NTC][4];
#pragma unroll
      for (int q = 0; q < NTC; q++) {
        const float2 bf = bf16x2_to_float2(cur.bb[n0 + q]);
        mma_m16n8k8_bf16(d[q], af[2 * mt], af[2 * mt + 1], cur.w[n0 + q], bf.x, bf.y, bf.x, bf.y);
      }
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < NTC; q++)
#pragma unroll
          for (int e = 0; e < 2; e++)
            if (!MASKED || valid[2 * q + e]) s[2 * mt + h] += act_f<ACT>(d[q][2 * h + e]);
    }
  }
}

template <int ACT, int C, int MT, int NW>
__global__ void __launch_bounds__(NW * 32) sg_fwd_kernel(SgArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int BM = 16 * MT, UB = Unit<C>::UB;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int t0 = blockIdx.y * BM;
  FwdArgs fa;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;
  sg_stage<MT>(a, a.feat + (long long)q * a.S, smap, slab, t0);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);

  SG_ITEM_LOOP_BEGIN
  float s[2 * MT];
#pragma unroll
  for (int i = 0; i < 2 * MT; i++) s[i] = 0.f;
  SG_UNIT_LOOP_BEGIN
#pragma unroll
  for (int bu = 0; bu < UB; bu++) {
    const int j = jb + bu;
    BlockRegs<C> blk;
    sg_block<C>(blk, cur, bu, smap);
    if (full) sg_fwd_block<ACT, C, MT, false>(s, blk, j, n_g, slab_s, lane);
    else if (j * C < n_g) sg_fwd_block<ACT, C, MT, true>(s, blk, j, n_g, slab_s, lane);
  }
  SG_UNIT_LOOP_END
  // quad lanes hold the same rows: reduce, then one atomic per (token, item)
#pragma unroll
  for (int i = 0; i < 2 * MT; i++) {
    s[i] += __shfl_xor_sync(0xffffffff, s[i], 1);
    s[i] += __shfl_xor_sync(0xffffffff, s[i], 2);
  }
  if ((lane & 3) == 0) {
#pragma unroll
    for (int i = 0; i < 2 * MT; i++) {
      const int tok = t0 + (i >> 1) * 16 + (i & 1) * 8 + (lane >> 2);
      if (SP_ABLATE_SG == 3) {
        if (s[i] == 1.2345e-30f) a.ysum[g] = s[i];
      } else if (tok < a.B) {
        atomicAdd(a.ysum + (long long)g * a.B + tok, s[i]);
      }
    }
  }
  SG_ITEM_LOOP_END
}

// ============================================================================
// dX
// ============================================================================
// FP32 slab [S][BM]: tokens t, t + 8 of an m16 tile adjacent (one 64-bit CAS / RMW per pair);
// the m16 segment of odd features is swapped so features alternate bank halves (as k_dx)
template <int MT>
__device__ __forceinline__ int sg_dxs_off(int f, int t) {
  const int seg = t >> 4;
  const int sw = (MT > 1) ? (seg ^ (f & 1)) : seg;
  return f * 16 * MT + 16 * sw + (t & 7) * 2 + ((t >> 3) & 1);
}

__device__ __forceinline__ void sg_rowcoef(RowCoef& rc, const float* c1g, int B, int t0, int lane) {
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int tok = t0 + (lane >> 2) + 8 * h;
    rc.tv[h] = tok < B;
    rc.c1[h] = rc.tv[h] ? __ldg(c1g + tok) : 0.f;  // c1 = 0 -> dz = 0 for padded tokens
    rc.c2[h] = 0.f;
    rc.c3[h] = 0.f;
    rc.one[h] = mul_ftz(rc.c1[h], 1.f);
    rc.c2f[h] = rc.tv[h] ? 0.f : INFINITY;
    split_bf16(rc.one[h], rc.one[h], rc.hp[h], rc.lp[h]);
  }
}

// dA (16 tokens x 8 inputs, accumulator layout) of one block for one m16 tile: the
// arithmetic of dx_block (sparse_pool_bwd.cuh) without its gather and scatter
// wt: W^T fragments of the block (movmatrix once per block, shared by its MT token tiles)
template <int ACT, int C, bool SPLIT, bool MASKED, int EPI>
__device__ __forceinline__ void sg_dx_tile(float* da, const uint32_t* af, const BlockRegs<C>& blk, const uint32_t* wt,
                                           int j, int n_g, const RowCoef& rc, int lane) {
  constexpr int NT = C / 8, POOL = P_MEAN, P_INT = 0;
  const int col0 = 2 * (lane & 3);
  if constexpr (EPI == 2 && FastDz<POOL, ACT, P_INT>::mask) {
#pragma unroll
    for (int nt = 0; nt < NT; nt += (NT == 1 ? 1 : 2)) {
      uint32_t m0[4];
      tile_mask<ACT, MASKED>(m0, af, blk.w[nt], blk.bb[nt], j * C + nt * 8 + col0, n_g);
      const uint32_t q00 = pair_mask(m0[0], m0[1]), q01 = pair_mask(m0[2], m0[3]);
      if constexpr (NT == 1) {
        mma_m16n8k8_bf16(da, rc.hp[0] & q00, rc.hp[1] & q01, wt[0], da[0], da[1], da[2], da[3]);
        if (SPLIT) mma_m16n8k8_bf16(da, rc.lp[0] & q00, rc.lp[1] & q01, wt[0], da[0], da[1], da[2], da[3]);
      } else {
        uint32_t m1[4];
        tile_mask<ACT, MASKED>(m1, af, blk.w[nt + 1], blk.bb[nt + 1], j * C + (nt + 1) * 8 + col0, n_g);
        const uint32_t q10 = pair_mask(m1[0], m1[1]), q11 = pair_mask(m1[2], m1[3]);
        uint32_t ah[4] = {rc.hp[0] & q00, rc.hp[1] & q01, rc.hp[0] & q10, rc.hp[1] & q11};
        const uint32_t b0 = wt[nt], b1 = wt[nt + 1];
        mma_m16n8k16_bf16(da, ah, b0, b1);
        if (SPLIT) {
          uint32_t al[4] = {rc.lp[0] & q00, rc.lp[1] & q01, rc.lp[0] & q10, rc.lp[1] & q11};
          mma_m16n8k16_bf16(da, al, b0, b1);
        }
      }
    }
  } else if constexpr (NT == 1) {
    float dz[4];
    tile_dz<POOL, ACT, P_INT, MASKED, EPI>(dz, af, blk.w[0], blk.bb[0], rc, j * C + col0, n_g, 1.f, 0.f);
    uint32_t h0, l0, h1, l1;
    split_bf16(dz[0], dz[1], h0, l0);
    split_bf16(dz[2], dz[3], h1, l1);
    mma_m16n8k8_bf16(da, h0, h1, wt[0], da[0], da[1], da[2], da[3]);
    if (SPLIT) mma_m16n8k8_bf16(da, l0, l1, wt[0], da[0], da[1], da[2], da[3]);
  } else {
#pragma unroll
    for (int nt = 0; nt < NT; nt += 2) {
      float dz0[4], dz1[4];
      tile_dz<POOL, ACT, P_INT, MASKED, EPI>(dz0, af, blk.w[nt], blk.bb[nt], rc, j * C + nt * 8 + col0, n_g, 1.f, 0.f);
      tile_dz<POOL, ACT, P_INT, MASKED, EPI>(dz1, af, blk.w[nt + 1], blk.bb[nt + 1], rc, j * C + (nt + 1) * 8 + col0,
                                             n_g, 1.f, 0.f);
      uint32_t ah[4], al[4];
      split_bf16(dz0[0], dz0[1], ah[0], al[0]);
      split_bf16(dz0[2], dz0[3], ah[1], al[1]);
      split_bf16(dz1[0], dz1[1], ah[2], al[2]);
      split_bf16(dz1[2], dz1[3], ah[3], al[3]);
      const uint32_t b0 = wt[nt], b1 = wt[nt + 1];
      mma_m16n8k16_bf16(da, ah, b0, b1);
      if (SPLIT) mma_m16n8k16_bf16(da, al, b0, b1);
    }
  }
}

template <int ACT, int C, int MT, int NW, bool SPLIT, int EPI>
__device__ __forceinline__ void sg_dx_body(const SgArgs& a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int BM = 16 * MT, UB = Unit<C>::UB;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  float* dxs = reinterpret_cast<float*>(smem_raw + sg_align16(a.D * 2) + a.S * BM * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int t0 = blockIdx.y * BM;
  const int WG = a.dx_group_warps;  // warps per FP32 slab copy
  const int ncopy = NW / WG;
  const bool priv = WG == 1;  // exclusive copy: plain read-modify-write
  const int16_t* fq = a.feat + (long long)q * a.S;
  FwdArgs fa;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;
  for (int i = threadIdx.x; i < ncopy * a.S * BM; i += NW * 32) dxs[i] = 0.f;
  sg_stage<MT>(a, fq, smap, slab, t0);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  float* mys = dxs + (warp / WG) * a.S * BM;

  SG_ITEM_LOOP_BEGIN
  RowCoef rc[MT];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) sg_rowcoef(rc[mt], a.c1T + (long long)g * a.B, a.B, t0 + 16 * mt, lane);
  SG_UNIT_LOOP_BEGIN
#pragma unroll
  for (int bu = 0; bu < UB; bu++) {
    const int j = jb + bu;
    if (!full && j * C >= n_g) continue;
    BlockRegs<C> blk;
    sg_block<C>(blk, cur, bu, smap);
    uint32_t af[2 * MT];
    sg_gather<MT>(af, slab_s, blk.fidx, lane);
    const int col0 = 2 * (lane & 3);
    const int f0 = __shfl_sync(0xffffffff, blk.fidx, col0);
    const int f1 = __shfl_sync(0xffffffff, blk.fidx, col0 + 1);
    const int t = lane >> 2;
    uint32_t wt[C / 8];  // W^T fragment: B[k = neuron][n = input k]
#pragma unroll
    for (int nt = 0; nt < C / 8; nt++) wt[nt] = movmatrix_trans(blk.w[nt]);
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float da[4] = {0.f, 0.f, 0.f, 0.f};
      if (full) sg_dx_tile<ACT, C, SPLIT, false, EPI>(da, af + 2 * mt, blk, wt, j, n_g, rc[mt], lane);
      else sg_dx_tile<ACT, C, SPLIT, true, EPI>(da, af + 2 * mt, blk, wt, j, n_g, rc[mt], lane);
      float* p0 = mys + sg_dxs_off<MT>(f0, 16 * mt + t);
      float* p1 = mys + sg_dxs_off<MT>(f1, 16 * mt + t);
      if (SP_ABLATE_SG == 2) {
        if (da[0] + da[1] + da[2] + da[3] == 1.2345e-30f) mys[f0 & 7] = (float)f1;
      } else if (priv) {  // this warp owns the copy and a record's features are distinct: no races
        float2 v0 = *reinterpret_cast<float2*>(p0), v1 = *reinterpret_cast<float2*>(p1);
        v0.x += da[0]; v0.y += da[2]; v1.x += da[1]; v1.y += da[3];
        *reinterpret_cast<float2*>(p0) = v0;
        *reinterpret_cast<float2*>(p1) = v1;
      } else {
        atomic_add_f32x2(p0, da[0], da[2]);
        atomic_add_f32x2(p1, da[1], da[3]);
      }
    }
  }
  SG_UNIT_LOOP_END
  SG_ITEM_LOOP_END
  __syncthreads();
  // reduce the copies and flush: one global FP32 atomic per (feature, token) per CTA
  for (int i = threadIdx.x; i < a.S * BM; i += NW * 32) {
    const int f = i / BM, tl = i % BM;
    const int o = sg_dxs_off<MT>(f, tl);
    float v = dxs[o];
    for (int c = 1; c < ncopy; c++) v += dxs[c * a.S * BM + o];
    const int tok = t0 + tl;
    if (SP_ABLATE_SG == 1) v = (v == 1.2345e-30f) ? v : 0.f;
    if (tok < a.B && v != 0.f) atomicAdd(a.dXT + (long long)fq[f] * a.B + tok, v);
  }
}

template <int ACT, int C, int MT, int NW, bool SPLIT, int EPI>
__global__ void __launch_bounds__(NW * 32) sg_dx_kernel(SgArgs a) {
  // guarded fallback launch of a fused backward (flags & BWD_FALLBACK): runs only for
  // non-finite coefficients, when the fused kernel exited without contributing
  if ((a.flags & BWD_FALLBACK) && !(a.nonfinite && *a.nonfinite)) return;
  // in-kernel generic fallback for non-finite coefficients (as k_dx)
  if (EPI && !(a.nonfinite && *a.nonfinite)) sg_dx_body<ACT, C, MT, NW, SPLIT, EPI>(a);
  else sg_dx_body<ACT, C, MT, NW, SPLIT, 0>(a);
}

// ============================================================================
// dW
// ============================================================================
__device__ __forceinline__ void sg_colcoef(ColCoef& cc, const float* c1g, int B, int t0, int lane) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int tok = t0 + 2 * (lane & 3) + (i & 1) + 8 * (i >> 1);
    cc.tv[i] = tok < B;
    cc.c1[i] = cc.tv[i] ? __ldg(c1g + tok) : 0.f;
    cc.c2[i] = 0.f;
    cc.c3[i] = 0.f;
    cc.one[i] = mul_ftz(cc.c1[i], 1.f);
    cc.c2f[i] = cc.tv[i] ? 0.f : INFINITY;
  }
#pragma unroll
  for (int th = 0; th < 2; th++) split_bf16(cc.one[2 * th], cc.one[2 * th + 1], cc.hp[th], cc.lp[th]);
}

// write back: exclusive ownership, plain read-modify-write (dw_writeback with explicit units)
template <int C, int WPW>
__device__ __forceinline__ void sg_dw_writeback(const SgArgs& a, DwAcc<C> (&acc)[WPW], const int (&unit)[WPW],
                                                int lane) {
  constexpr int UB = Unit<C>::UB;
  constexpr int NT = C / 8;
  constexpr int NP = DwAcc<C>::NP;
#pragma unroll
  for (int i = 0; i < WPW; i++) {
    if (unit[i] < 0) continue;
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
#pragma unroll
      for (int q = 0; q < NP; q++) {
        float db0 = acc[i].b[bu][q][0], db1 = acc[i].b[bu][q][1];
        db0 += __shfl_xor_sync(0xffffffff, db0, 1);
        db0 += __shfl_xor_sync(0xffffffff, db0, 2);
        db1 += __shfl_xor_sync(0xffffffff, db1, 1);
        db1 += __shfl_xor_sync(0xffffffff, db1, 2);
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int nt = 2 * q + h;
          if (nt >= NT) continue;
          const int f = bu * NT + nt;
          const int qd = f >> 2, fi = f & 3;
          const long long quad = (long long)unit[i] * Unit<C>::NQ4 + qd;
          const float* d = acc[i].w[bu][q];
          float* wp = a.dW + ((quad * 32 + lane) * 4 + fi) * 2;
          float2 old = *reinterpret_cast<float2*>(wp);
          if (NT == 1) {
            old.x += d[0] + d[2];
            old.y += d[1] + d[3];
          } else {
            old.x += d[2 * h];
            old.y += d[2 * h + 1];
          }
          *reinterpret_cast<float2*>(wp) = old;
          if ((lane & 3) == 0) {
            const int gq = lane >> 2;
            float* bp = a.dB + ((quad * 4 + (gq >> 1)) * 4 + fi) * 2 + (gq & 1);
            *bp += (h == 0) ? db0 : db1;
          }
        }
      }
    }
  }
}

// TS 16-token tiles per pipeline stage, ST stages; ring [ST][TS][S][16] bf16 (Swz<1> rows)
template <int ACT, int C, int NW, int WPW, int TS, int ST, bool SPLIT, int EPI>
__device__ __forceinline__ void sg_dw_body(const SgArgs& a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int UB = Unit<C>::UB, NT = C / 8, NP = DwAcc<C>::NP, POOL = P_MEAN, P_INT = 0;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* ring = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int S = a.S;
  FwdArgs fa;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;
  for (int i = threadIdx.x; i < S; i += NW * 32) smap[fq[i]] = (int16_t)i;

  // this warp's WPW consecutive units of the chunk (consecutive units mostly share an output)
  Unit<C> un[WPW];
  DwAcc<C> acc[WPW];
  int unit[WPW], gi[WPW], ng[WPW], jb[WPW];
  float biasT[WPW][UB][2 * NP];
#pragma unroll
  for (int i = 0; i < WPW; i++) {
    const int pos = lo + warp * WPW + i;
    unit[i] = pos < hi ? a.units[pos] : -1;
    gi[i] = unit[i] >= 0 ? unit[i] / a.NU : 0;
    ng[i] = unit[i] >= 0 ? a.counts[gi[i]] : 0;
    jb[i] = unit[i] >= 0 ? (unit[i] - gi[i] * a.NU) * UB : 0;
    if (unit[i] >= 0) load_unit<C>(un[i], fa, unit[i], lane);
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
#pragma unroll
      for (int qq = 0; qq < NP; qq++) {
#pragma unroll
        for (int e = 0; e < 4; e++) acc[i].w[bu][qq][e] = 0.f;
        acc[i].b[bu][qq][0] = acc[i].b[bu][qq][1] = 0.f;
      }
      BlockRegs<C> blk;
      unit_block<C>(blk, un[i], bu);
#pragma unroll
      for (int nt = 0; nt < 2 * NP; nt++) {
        uint32_t word = (nt < NT) ? blk.bb[nt] : 0u;
        const int gq = lane >> 2;
        uint32_t src = __shfl_sync(0xffffffff, word, gq >> 1);
        float2 bf = bf16x2_to_float2(src);
        biasT[i][bu][nt] = (gq & 1) ? bf.y : bf.x;
      }
    }
  }
  __syncthreads();  // smap complete
  int lrow[WPW][UB];  // this lane's local slab row (fan-in k = lane & 7) of each block
#pragma unroll
  for (int i = 0; i < WPW; i++)
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
      BlockRegs<C> blk;
      unit_block<C>(blk, un[i], bu);
      lrow[i][bu] = unit[i] >= 0 ? smap[blk.fidx] : 0;
    }

  const int n_tiles = (a.B + 15) / 16;
  const int n_st = (n_tiles + TS - 1) / TS;
  const int stage_elems = TS * S * 16;
  auto stage = [&](int s) {
    bf16* dst = ring + (s % ST) * stage_elems;
    for (int i = threadIdx.x; i < TS * S * 2; i += NW * 32) {
      const int sub = i / (S * 2), r = i % (S * 2);
      const int f = r >> 1, c = r & 1;
      const int tok = (s * TS + sub) * 16 + c * 8;
      int bytes = (tok + 8 <= a.B) ? 16 : (tok < a.B ? 2 * (a.B - tok) : 0);
      const bf16* src = bytes ? a.X + (long long)fq[f] * a.sxf + tok : a.X;
      cp_async16(dst + sub * S * 16 + Swz<1>::off(f, c), src, bytes);
    }
  };
#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < n_st) stage(s);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int s = 0; s < n_st; s++) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2));
    __syncthreads();  // stage s visible to all; every warp is done with stage s - 1's slot
    if (s + ST - 1 < n_st) stage(s + ST - 1);
    asm volatile("cp.async.commit_group;\n" ::);
    const bf16* base = ring + (s % ST) * stage_elems;
#pragma unroll 1
    for (int sub = 0; sub < TS; sub++) {
      const int tt = s * TS + sub;
      if (tt >= n_tiles) break;
      const uint32_t slab_s = smem_u32(base + sub * S * 16);
      ColCoef cc;
#pragma unroll
      for (int i = 0; i < WPW; i++) {
        if (unit[i] < 0) continue;
        if (i == 0 || gi[i] != gi[i - 1] || unit[i - 1] < 0) sg_colcoef(cc, a.c1T + (long long)gi[i] * a.B, a.B, tt * 16, lane);
#pragma unroll
        for (int bu = 0; bu < UB; bu++) {
          const int j = jb[i] + bu;
          BlockRegs<C> blk;
          unit_block<C>(blk, un[i], bu);
          blk.fidx = lrow[i][bu];
          if ((j + 1) * C <= ng[i])
            dw_block<POOL, ACT, C, SPLIT, P_INT, false, EPI>(acc[i], bu, blk, biasT[i][bu], j, ng[i], slab_s, cc, lane,
                                                            1.f, 0.f);
          else if (j * C < ng[i])
            dw_block<POOL, ACT, C, SPLIT, P_INT, true, EPI>(acc[i], bu, blk, biasT[i][bu], j, ng[i], slab_s, cc, lane,
                                                           1.f, 0.f);
        }
      }
    }
  }
  asm volatile("cp.async.wait_all;\n" ::);
  sg_dw_writeback<C, WPW>(a, acc, unit, lane);
}

template <int ACT, int C, int NW, int WPW, int TS, int ST, bool SPLIT, int MINB, int EPI>
__global__ void __launch_bounds__(NW * 32, MINB) sg_dw_kernel(SgArgs a) {
  // specialised launch runs unless the coefficients are non-finite; the generic launch
  // (flags & BWD_FALLBACK) only then (one body per kernel, as k_dw_idx)
  bool active;
  if (EPI) active = !(a.nonfinite && *a.nonfinite);
  else active = !(a.flags & BWD_FALLBACK) || (a.nonfinite && *a.nonfinite);
  if (active) sg_dw_body<ACT, C, NW, WPW, TS, ST, SPLIT, EPI>(a);
}
