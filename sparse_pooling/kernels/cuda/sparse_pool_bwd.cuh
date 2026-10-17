// Native Ampere recompute backward for smooth pooling operators (Mean(act),
// LogMeanExp, Power mean, Softmax-weighted).  Fragment-packed Mode B storage.
//
// Nothing from the forward pass is reused except compact per-(token, unit)
// coefficients c1, c2, c3 (see ops.dense_coefficients).  Every kernel
// recomputes z = A W + b with mma.m16n8k8 and rebuilds dL/dz from z and the
// coefficients.  Two kernels, because the two gradient reductions want
// opposite loop orders:
//
//   k_dx  token-stationary  CTA = (16 tokens, a range of pooling units).
//         dA = dZ W^T (tensor cores) is scattered into a [D x 16] FP32 slab
//         in shared memory; one flush per CTA to global dX.
//   k_dw  weight-stationary CTA = (a chunk of work units of one pooling unit),
//         loops over all tokens; dW^T = dZ^T X_g (m16n8k16) and db stay in
//         registers and are written once (exclusive ownership -> no atomics,
//         deterministic).  The accumulator fragment of (block, n-tile) lands on
//         exactly the lanes/elements of the packed weight layout.
//
// dZ enters the tensor-core products as an unevaluated sum hi + lo of two
// BF16 numbers (SPLIT = true, ~16-bit mantissa) or as one BF16 (SPLIT = false).
#pragma once
#include "sparse_pool_fwd.cuh"

// Performance-attribution ablations (benchmarks/bench_ablation.py).  Every
// non-zero value produces WRONG gradients and is never part of the default build.
//   SP_ABLATE_DX 1: racy plain RMW instead of the 64-bit CAS scatter
//                2: no scatter (products kept live, never stored)
//                3: CAS scatter but no global flush
//                4: every warp re-loads its first work unit (parameter streaming -> cache hits)
//   SP_ABLATE_DW 1: no X staging inside the token loop (k_dw and k_dw_idx)
//                2: k_dw without the per-tile __syncthreads (copies kept)
//                3: no c1/c2/c3 coefficient loads (constants)
#ifndef SP_ABLATE_DX
#define SP_ABLATE_DX 0
#endif
#ifndef SP_ABLATE_DW
#define SP_ABLATE_DW 0
#endif

struct BwdArgs {
  const bf16* X;
  long long sxf, sxb;
  const int16_t* idx;
  const bf16* W;
  const bf16* bias;
  const int* counts;
  const float* c1;  // [B, U]
  const float* c2;
  const float* c3;
  float* dX;  // FP32, element (b, f) at dX[f * sdf + b * sdb]
  long long sdf, sdb;
  float* dW;  // FP32, packed layout (same shape as W)
  float* dB;  // FP32, packed layout (same shape as bias)
  int B, U, S, NB, BPS, D;
  float scale, p;
  int p_int;
  int units_per_cta;   // k_dx: pooling units per CTA
  int wunits_per_cta;  // k_dw: work units per CTA
};

// dL/dz for one element, from z and the row coefficients
template <int POOL, int ACT, int P_INT>
__device__ __forceinline__ float dz_f(float z, float c1, float c2, float c3, float scale, float p) {
  if (POOL == P_MEAN) return c1 * dact_f<ACT>(z);
  if (POOL == P_LME) {
    float a = act_f<ACT>(z);
    return c1 * exp2_approx(fmaf(a, scale, -c2)) * dact_f<ACT>(z);
  }
  if (POOL == P_POW) {
    if (z <= 0.f) return 0.f;
    float q = z * c2;  // a / y
    float qp;
    if (P_INT == 1) qp = 1.f;
    else if (P_INT == 2) qp = q;
    else if (P_INT == 4) qp = q * q * q;
    else if (P_INT == 8) { float q2 = q * q; float q4 = q2 * q2; qp = q4 * q2 * q; }
    else qp = exp2_approx((p - 1.f) * __log2f(q));
    return c1 * qp;
  }
  // P_SW
  float a = act_f<ACT>(z);
  float pr = exp2_approx(fmaf(a, scale, -c2));
  return c1 * pr * fmaf(scale * 0.6931471805599453f, a - c3, 1.f) * dact_f<ACT>(z);
}

struct RowCoef {
  float c1[2], c2[2], c3[2];  // rows lane/4 and lane/4 + 8
  bool tv[2];                 // token < B
  __device__ __forceinline__ void load(const BwdArgs& a, int t0, int u, int lane) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int tok = t0 + (lane >> 2) + 8 * h;
      tv[h] = tok < a.B;
      if (tok < a.B) {
        long long o = (long long)tok * a.U + u;
        c1[h] = __ldg(a.c1 + o);
        c2[h] = __ldg(a.c2 + o);
        c3[h] = __ldg(a.c3 + o);
      } else {
        c1[h] = 0.f; c2[h] = 0.f; c3[h] = 0.f;  // c1 = 0 -> dz = 0 for padded tokens
      }
    }
  }
};

// z for one n8 tile (16 tokens) -> dz (4 values in accumulator layout)
template <int POOL, int ACT, int P_INT, bool MASKED>
__device__ __forceinline__ void tile_dz(float* dz, const uint32_t* af, uint32_t w, uint32_t bb, const RowCoef& rc,
                                        int slot0, int n_g, float scale, float p) {
  float2 bf = bf16x2_to_float2(bb);
  float d[4];
  mma_m16n8k8_bf16(d, af[0], af[1], w, bf.x, bf.y, bf.x, bf.y);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int h = i >> 1;
    float v = dz_f<POOL, ACT, P_INT>(d[i], rc.c1[h], rc.c2[h], rc.c3[h], scale, p);
    if (MASKED && slot0 + (i & 1) >= n_g) v = 0.f;
    dz[i] = rc.tv[h] ? v : 0.f;  // select (not multiply): padded tokens may give inf
  }
}

__device__ __forceinline__ void split_bf16(float x0, float x1, uint32_t& hi, uint32_t& lo) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
  float2 hf = __bfloat1622float2(h);
  hi = *reinterpret_cast<uint32_t*>(&h);
  lo = pack_bf16x2(x0 - hf.x, x1 - hf.y);
}

// ============================================================================
// k_dx
// ============================================================================
// dX slab [D][16] FP32 with tokens t and t + 8 adjacent
__device__ __forceinline__ int dxs_off(int f, int t) { return f * 16 + (t & 7) * 2 + (t >> 3); }

__device__ __forceinline__ void atomic_add_f32x2(float* p, float a, float b) {
  unsigned long long* q = reinterpret_cast<unsigned long long*>(p);
  unsigned long long old = *q, assumed;
  do {
    assumed = old;
    float2 v = *reinterpret_cast<float2*>(&assumed);
    v.x += a;
    v.y += b;
    old = atomicCAS(q, assumed, *reinterpret_cast<unsigned long long*>(&v));
  } while (old != assumed);
}
template <int POOL, int ACT, int C, bool SPLIT, int P_INT, bool MASKED>
__device__ __forceinline__ void dx_block(const BlockRegs<C>& blk, int j, int n_g, uint32_t slab_s, float* dxs,
                                         const RowCoef& rc, int lane, float scale, float p) {
  constexpr int NT = C / 8;
  const int col0 = 2 * (lane & 3);
  uint32_t af[2];
  ldmatrix_x2_trans(af, slab_s + 2 * Swz<1>::off(blk.fidx, (lane >> 3) & 1));
  float da[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (NT == 1) {
    float dz[4];
    tile_dz<POOL, ACT, P_INT, MASKED>(dz, af, blk.w[0], blk.bb[0], rc, j * C + col0, n_g, scale, p);
    uint32_t wt = movmatrix_trans(blk.w[0]);  // W^T fragment: B[k = neuron][n = input k]
    uint32_t h0, l0, h1, l1;
    split_bf16(dz[0], dz[1], h0, l0);
    split_bf16(dz[2], dz[3], h1, l1);
    mma_m16n8k8_bf16(da, h0, h1, wt, da[0], da[1], da[2], da[3]);
    if (SPLIT) mma_m16n8k8_bf16(da, l0, l1, wt, da[0], da[1], da[2], da[3]);
  } else {
#pragma unroll
    for (int nt = 0; nt < NT; nt += 2) {
      float dz0[4], dz1[4];
      tile_dz<POOL, ACT, P_INT, MASKED>(dz0, af, blk.w[nt], blk.bb[nt], rc, j * C + nt * 8 + col0, n_g, scale, p);
      tile_dz<POOL, ACT, P_INT, MASKED>(dz1, af, blk.w[nt + 1], blk.bb[nt + 1], rc, j * C + (nt + 1) * 8 + col0, n_g,
                                        scale, p);
      uint32_t ah[4], al[4];
      split_bf16(dz0[0], dz0[1], ah[0], al[0]);
      split_bf16(dz0[2], dz0[3], ah[1], al[1]);
      split_bf16(dz1[0], dz1[1], ah[2], al[2]);
      split_bf16(dz1[2], dz1[3], ah[3], al[3]);
      uint32_t b0 = movmatrix_trans(blk.w[nt]), b1 = movmatrix_trans(blk.w[nt + 1]);
      mma_m16n8k16_bf16(da, ah, b0, b1);
      if (SPLIT) mma_m16n8k16_bf16(da, al, b0, b1);
    }
  }
  // da: (token lane/4 [+8], input k = col0 [+1]) -> scatter into the dX slab.
  // GA102 has no native FP32 shared-memory atomics (atomicAdd is a CAS loop),
  // so tokens t and t+8 of a feature are interleaved and updated together
  // with one 64-bit CAS: two CAS loops per lane instead of four.
  int f0 = __shfl_sync(0xffffffff, blk.fidx, col0);
  int f1 = __shfl_sync(0xffffffff, blk.fidx, col0 + 1);
  int t = lane >> 2;
#if SP_ABLATE_DX == 1
  float2* p0 = reinterpret_cast<float2*>(dxs + dxs_off(f0, t));
  float2* p1 = reinterpret_cast<float2*>(dxs + dxs_off(f1, t));
  float2 v0 = *p0, v1 = *p1;
  v0.x += da[0]; v0.y += da[2]; v1.x += da[1]; v1.y += da[3];
  *p0 = v0; *p1 = v1;
#elif SP_ABLATE_DX == 2
  if (da[0] + da[1] + da[2] + da[3] == 1.2345e-30f) dxs[f0 & 7] = (float)f1;
#else
  atomic_add_f32x2(dxs + dxs_off(f0, t), da[0], da[2]);
  atomic_add_f32x2(dxs + dxs_off(f1, t), da[1], da[3]);
#endif
}

template <int POOL, int ACT, int C, int NW, bool SPLIT, int P_INT>
__global__ void __launch_bounds__(NW * 32) k_dx(BwdArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* slab = reinterpret_cast<bf16*>(smem_raw);                        // [D][16] bf16, swizzled
  float* dxs = reinterpret_cast<float*>(smem_raw + (size_t)a.D * 16 * 2);  // [D][16] fp32
  constexpr int UB = Unit<C>::UB;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int t0 = blockIdx.x * 16;
  const int u_lo = blockIdx.y * a.units_per_cta;
  const int u_hi = min(u_lo + a.units_per_cta, a.U);

  FwdArgs fa;
  fa.X = a.X; fa.sxf = a.sxf; fa.sxb = a.sxb; fa.B = a.B; fa.D = a.D;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;
  stage_slab<1>(fa, slab, t0);
  for (int i = threadIdx.x; i < a.D * 16; i += NW * 32) dxs[i] = 0.f;
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);

  for (int u = u_lo; u < u_hi; u++) {
    const int g = u / a.S, sh = u % a.S;
    const int n_g = a.counts[g];
    const int b_lo = sh * a.BPS;
    const int b_hi = min((sh + 1) * a.BPS, a.NB);
    const int b_full = min(max(n_g / C, b_lo), b_hi);
    const int b_any = min(max((n_g + C - 1) / C, b_lo), b_hi);
    const long long ubase = ((long long)g * a.NB + b_lo) / UB;
    const int nu_full = (b_full - b_lo) / UB;
    RowCoef rc;
    rc.load(a, t0, u, lane);
    Unit<C> nxt;
    if (warp < nu_full) load_unit<C>(nxt, fa, ubase + warp, lane);
    for (int uu = warp; uu < nu_full; uu += NW) {
      Unit<C> cur = nxt;
      if (uu + NW < nu_full) load_unit<C>(nxt, fa, ubase + (SP_ABLATE_DX == 4 ? warp : uu + NW), lane);
#pragma unroll
      for (int bu = 0; bu < UB; bu++) {
        BlockRegs<C> blk;
        unit_block<C>(blk, cur, bu);
        dx_block<POOL, ACT, C, SPLIT, P_INT, false>(blk, b_lo + uu * UB + bu, n_g, slab_s, dxs, rc, lane, a.scale, a.p);
      }
    }
    const int bt = b_lo + nu_full * UB;
    if (bt < b_any && warp == nu_full % NW) {
      Unit<C> cur;
      load_unit<C>(cur, fa, ubase + nu_full, lane);
#pragma unroll
      for (int bu = 0; bu < UB; bu++) {
        if (bt + bu < b_any) {
          BlockRegs<C> blk;
          unit_block<C>(blk, cur, bu);
          dx_block<POOL, ACT, C, SPLIT, P_INT, true>(blk, bt + bu, n_g, slab_s, dxs, rc, lane, a.scale, a.p);
        }
      }
    }
  }
  __syncthreads();
  // flush: one global reduction per (feature, token) per CTA
  for (int i = threadIdx.x; i < a.D * 16; i += NW * 32) {
    int f = i >> 4, tl = i & 15;
    int tok = t0 + tl;
    float v = dxs[dxs_off(f, tl)];
    if (SP_ABLATE_DX == 3) v = (v == 1.2345e-30f) ? v : 0.f;
    if (tok < a.B && v != 0.f) atomicAdd(a.dX + (long long)f * a.sdf + (long long)tok * a.sdb, v);
  }
}

// ============================================================================
// k_dw
// ============================================================================
// Transposed recompute for the weight-stationary kernel:
//   z^T [16 neurons x 8 tokens] = W [16 neurons x 8 inputs] * X_g^T [8 inputs x 8 tokens] + b
// The packed weight fragment is directly the m16n8k8 A operand (rows = neurons)
// and the forward's ldmatrix.trans registers are directly the B operand
// (cols = tokens).  dZ^T then comes out in exactly the A layout of the dW
// product dW^T = dZ^T X_g, so no register transposes are needed.
struct ColCoef {  // coefficients for this lane's token columns 2t, 2t+1 (+8)
  float c1[4], c2[4], c3[4];
  bool tv[4];
  __device__ __forceinline__ void load(const BwdArgs& a, int t0, int u, int lane) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      if (SP_ABLATE_DW == 3) {
        int tok = t0 + 2 * (lane & 3) + (i & 1) + 8 * (i >> 1);
        tv[i] = tok < a.B;
        c1[i] = 1e-3f * (float)(tok & 7); c2[i] = 0.5f; c3[i] = 0.25f;
        continue;
      }
      int tok = t0 + 2 * (lane & 3) + (i & 1) + 8 * (i >> 1);
      tv[i] = tok < a.B;
      if (tv[i]) {
        long long o = (long long)tok * a.U + u;
        c1[i] = __ldg(a.c1 + o);
        c2[i] = __ldg(a.c2 + o);
        c3[i] = __ldg(a.c3 + o);
      } else {
        c1[i] = 0.f; c2[i] = 0.f; c3[i] = 0.f;
      }
    }
  }
};

template <int C>
struct DwAcc {
  static constexpr int NT = C / 8;
  static constexpr int NP = (NT + 1) / 2;  // m16 neuron row tiles per block
  float w[Unit<C>::UB][NP][4];
  float b[Unit<C>::UB][NP][2];  // rows lane/4 and lane/4 + 8, partial over this lane's tokens
};

template <int POOL, int ACT, int C, bool SPLIT, int P_INT, bool MASKED>
__device__ __forceinline__ void dw_block(DwAcc<C>& acc, int bu, const BlockRegs<C>& blk, const float* bT, int j,
                                         int n_g, uint32_t slab_s, const ColCoef& cc, int lane, float scale, float p) {
  constexpr int NT = C / 8;
  const int c = (lane >> 3) & 1;
  uint32_t xa[2], xb[2];
  const uint32_t addr = slab_s + 2 * Swz<1>::off(blk.fidx, c);
  ldmatrix_x2_trans(xa, addr);  // (token g, inputs 2t..): B operand of z^T, tokens 0-7 / 8-15
  ldmatrix_x2(xb, addr);        // (input g, tokens 2t..): B operand of dW^T
#pragma unroll
  for (int q = 0; q < (NT + 1) / 2; q++) {
    const int nt0 = 2 * q;
    const uint32_t w0 = blk.w[nt0];
    const uint32_t w1 = (nt0 + 1 < NT) ? blk.w[nt0 + 1] : 0u;
    const float b0 = bT[2 * q], b1 = bT[2 * q + 1];  // bias of neuron rows g, g+8
    const int r0 = j * C + nt0 * 8 + (lane >> 2), r1 = r0 + 8;
    const bool v0 = !MASKED || r0 < n_g, v1 = (nt0 + 1 < NT) && (!MASKED || r1 < n_g);
    uint32_t ah[4], al[4];
#pragma unroll
    for (int th = 0; th < 2; th++) {  // token halves 0-7, 8-15
      float d[4];
      mma_m16n8k8_bf16(d, w0, w1, xa[th], b0, b0, b1, b1);
      float dz[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if (i >= 2 && nt0 + 1 >= NT) {  // C = 8: rows g+8 are structural padding
          dz[i] = 0.f;
          continue;
        }
        const int ci = (i & 1) + 2 * th;  // token column index into ColCoef
        float v = dz_f<POOL, ACT, P_INT>(d[i], cc.c1[ci], cc.c2[ci], cc.c3[ci], scale, p);
        const bool ok = cc.tv[ci] && ((i < 2) ? v0 : v1);
        dz[i] = ok ? v : 0.f;
      }
      acc.b[bu][q][0] += dz[0] + dz[1];
      acc.b[bu][q][1] += dz[2] + dz[3];
      // A of m16n8k16: {rows g, tok 0-7}, {rows g+8, tok 0-7}, {rows g, tok 8-15}, {rows g+8, tok 8-15}
      split_bf16(dz[0], dz[1], ah[2 * th], al[2 * th]);
      split_bf16(dz[2], dz[3], ah[2 * th + 1], al[2 * th + 1]);
    }
    if constexpr (NT == 1) {
      // C = 8: neuron rows 8-15 of the m16 tile are structural zeros; put the
      // lo correction there so ONE MMA yields hi (rows g) and lo (rows g+8)
      // partial products; they are summed at write-back.
      uint32_t A[4] = {ah[0], SPLIT ? al[0] : 0u, ah[2], SPLIT ? al[2] : 0u};
      mma_m16n8k16_bf16(acc.w[bu][q], A, xb[0], xb[1]);
    } else {
      uint32_t A[4] = {ah[0], ah[1], ah[2], ah[3]};
      mma_m16n8k16_bf16(acc.w[bu][q], A, xb[0], xb[1]);
      if (SPLIT) {
        uint32_t L[4] = {al[0], al[1], al[2], al[3]};
        mma_m16n8k16_bf16(acc.w[bu][q], L, xb[0], xb[1]);
      }
    }
  }
}

template <int C, int WPW>
__device__ __forceinline__ void dw_writeback(const BwdArgs& a, DwAcc<C> (&acc)[WPW], const int (&wid)[WPW],
                                             long long ubase, int lane);

template <int POOL, int ACT, int C, int NW, int WPW, bool SPLIT, int P_INT>
__global__ void __launch_bounds__(NW * 32) k_dw(BwdArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* slab0 = reinterpret_cast<bf16*>(smem_raw);
  bf16* slab1 = slab0 + (size_t)a.D * 16;
  constexpr int UB = Unit<C>::UB;
  constexpr int NT = C / 8;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int u = blockIdx.y;
  const int g = u / a.S, sh = u % a.S;
  const int n_g = a.counts[g];
  const int b_lo = sh * a.BPS;
  const int b_hi = min((sh + 1) * a.BPS, a.NB);
  const int b_any = min(max((n_g + C - 1) / C, b_lo), b_hi);
  const long long ubase = ((long long)g * a.NB + b_lo) / UB;
  const int nu_any = (b_any - b_lo + UB - 1) / UB;  // work units with any valid block
  const int wu0 = blockIdx.x * a.wunits_per_cta;
  if (wu0 >= nu_any) return;

  FwdArgs fa;
  fa.X = a.X; fa.sxf = a.sxf; fa.sxb = a.sxb; fa.B = a.B; fa.D = a.D;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;

  // this warp's work units (registers for the whole kernel)
  Unit<C> un[WPW];
  DwAcc<C> acc[WPW];
  int wid[WPW];
  constexpr int NP = DwAcc<C>::NP;
  float biasT[WPW][UB][2 * NP];  // bias of neuron rows (lane/4, lane/4 + 8) per row tile
#pragma unroll
  for (int i = 0; i < WPW; i++) {
    wid[i] = wu0 + warp + i * NW;
    if (wid[i] < nu_any && wid[i] < wu0 + a.wunits_per_cta) load_unit<C>(un[i], fa, ubase + wid[i], lane);
    else wid[i] = -1;
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
#pragma unroll
      for (int q = 0; q < DwAcc<C>::NP; q++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[i].w[bu][q][e] = 0.f;
#pragma unroll
      for (int q = 0; q < NP; q++) acc[i].b[bu][q][0] = acc[i].b[bu][q][1] = 0.f;
    }
    // transpose the bias pairs: lane holds bias(cols 2t, 2t+1); we need bias(row g)
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
      BlockRegs<C> blk;
      unit_block<C>(blk, un[i], bu);
#pragma unroll
      for (int nt = 0; nt < 2 * NP; nt++) {
        uint32_t word = (nt < NT) ? blk.bb[nt] : 0u;
        const int g = lane >> 2;
        uint32_t src = __shfl_sync(0xffffffff, word, g >> 1);  // lane (g/2) holds cols 2(g/2), +1
        float2 bf = bf16x2_to_float2(src);
        biasT[i][bu][nt] = (g & 1) ? bf.y : bf.x;
      }
    }
  }

  const int n_tiles = (a.B + 15) / 16;
  stage_slab<1>(fa, slab0, 0);  // (waits for its own copies)
  __syncthreads();
  for (int tt = 0; tt < n_tiles; tt++) {
    bf16* cur_slab = (tt & 1) ? slab1 : slab0;
    bf16* nxt_slab = (tt & 1) ? slab0 : slab1;
    // prefetch the next token tile while computing this one
    if (tt + 1 < n_tiles && a.sxb == 1 && SP_ABLATE_DW != 1) {
      for (int i = threadIdx.x; i < a.D * 2; i += NW * 32) {
        int f = i >> 1, c = i & 1;
        int tok = (tt + 1) * 16 + c * 8;
        const bf16* src = a.X + (long long)f * a.sxf + tok;
        int bytes = (tok + 8 <= a.B) ? 16 : (tok < a.B ? 2 * (a.B - tok) : 0);
        if (bytes == 0) src = a.X;
        cp_async16(nxt_slab + Swz<1>::off(f, c), src, bytes);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    ColCoef cc;
    cc.load(a, tt * 16, u, lane);
    const uint32_t slab_s = smem_u32(cur_slab);
#pragma unroll
    for (int i = 0; i < WPW; i++) {
      if (wid[i] < 0) continue;
#pragma unroll
      for (int bu = 0; bu < UB; bu++) {
        const int j = b_lo + wid[i] * UB + bu;
        BlockRegs<C> blk;
        unit_block<C>(blk, un[i], bu);
        if ((j + 1) * C <= n_g)
          dw_block<POOL, ACT, C, SPLIT, P_INT, false>(acc[i], bu, blk, biasT[i][bu], j, n_g, slab_s, cc, lane,
                                                      a.scale, a.p);
        else if (j < b_any)
          dw_block<POOL, ACT, C, SPLIT, P_INT, true>(acc[i], bu, blk, biasT[i][bu], j, n_g, slab_s, cc, lane,
                                                     a.scale, a.p);
      }
    }
    if (tt + 1 < n_tiles && a.sxb != 1) {
      __syncthreads();
      stage_slab<1>(fa, nxt_slab, (tt + 1) * 16);
    }
    asm volatile("cp.async.wait_all;\n" ::);
    if (SP_ABLATE_DW != 2) __syncthreads();
  }

  dw_writeback<C, WPW>(a, acc, wid, ubase, lane);
}

// ============================================================================
// k_dw_idx: k_dw with indexed, warp-local input staging (feature-major X only)
// ============================================================================
// k_dw stages all D feature rows of every 16-token tile, but a warp only ever
// reads the WPW * UB * 8 rows named by its own index records.  Here each warp
// stages exactly those rows into a private ring of ST compact slabs
// [R rows][16 tokens] (same Swz<1> row layout, so ldmatrix stays conflict-free)
// and dw_block addresses them by local row (i, bu, k) instead of by feature.
// Warps never share slab data: one cp.async group per tile, wait_group +
// __syncwarp, no CTA barriers in the token loop.  Arithmetic, accumulation
// order and write-back are identical to k_dw.
// MINB > 1 asks ptxas for MINB resident CTAs per SM (register cap 64K / (MINB * NW * 32))
template <int POOL, int ACT, int C, int NW, int WPW, int ST, bool SPLIT, int P_INT, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) k_dw_idx(BwdArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int UB = Unit<C>::UB;
  constexpr int R = WPW * UB * 8;               // staged rows per warp
  constexpr int CPL = (2 * R + 31) / 32;        // 16-byte copies per lane per tile
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  bf16* ring = reinterpret_cast<bf16*>(smem_raw) + (size_t)warp * ST * R * 16;
  const int u = blockIdx.y;
  const int g = u / a.S, sh = u % a.S;
  const int n_g = a.counts[g];
  const int b_lo = sh * a.BPS;
  const int b_hi = min((sh + 1) * a.BPS, a.NB);
  const int b_any = min(max((n_g + C - 1) / C, b_lo), b_hi);
  const long long ubase = ((long long)g * a.NB + b_lo) / UB;
  const int nu_any = (b_any - b_lo + UB - 1) / UB;
  const int wu0 = blockIdx.x * a.wunits_per_cta;
  if (wu0 >= nu_any) return;

  FwdArgs fa;
  fa.X = a.X; fa.sxf = a.sxf; fa.sxb = a.sxb; fa.B = a.B; fa.D = a.D;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;

  Unit<C> un[WPW];
  DwAcc<C> acc[WPW];
  int wid[WPW];
  constexpr int NT = C / 8;
  constexpr int NP = DwAcc<C>::NP;
  float biasT[WPW][UB][2 * NP];
#pragma unroll
  for (int i = 0; i < WPW; i++) {
    wid[i] = wu0 + warp + i * NW;
    if (wid[i] < nu_any && wid[i] < wu0 + a.wunits_per_cta) load_unit<C>(un[i], fa, ubase + wid[i], lane);
    else wid[i] = -1;
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
#pragma unroll
      for (int q = 0; q < NP; q++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[i].w[bu][q][e] = 0.f;
#pragma unroll
      for (int q = 0; q < NP; q++) acc[i].b[bu][q][0] = acc[i].b[bu][q][1] = 0.f;
    }
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
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

  // staging plan: copy j = lane + 32 q moves (local row j / 2, token chunk j & 1).
  // Local row r = (i * UB + bu) * 8 + k holds input feature idx[unit i, block bu, k],
  // held as int16 by lane k of un[i]; -1 marks rows of absent work units (zero-filled).
  int src_off[CPL];
#pragma unroll
  for (int q = 0; q < CPL; q++) {
    const int r = (lane + 32 * q) >> 1;
    const int k = r & 7, bu = (r >> 3) % UB, ui = r / (8 * UB);
    int f = -1;
#pragma unroll
    for (int i = 0; i < WPW; i++) {
#pragma unroll
      for (int b2 = 0; b2 < UB; b2++) {
        uint32_t iw = b2 < 2 ? un[i].i0 : un[i].i1;
        iw = __shfl_sync(0xffffffff, iw, k);
        const int fi = (int)((iw >> (16 * (b2 & 1))) & 0xffffu);
        if (ui == i && bu == b2 && wid[i] >= 0) f = fi;
      }
    }
    src_off[q] = (f >= 0 && lane + 32 * q < 2 * R) ? f * (int)a.sxf : -1;
  }
  auto stage = [&](int slot, int t0) {
    bf16* dst = ring + slot * R * 16;
#pragma unroll
    for (int q = 0; q < CPL; q++) {
      const int j = lane + 32 * q;
      if (j < 2 * R) {
        const int r = j >> 1, c = j & 1;
        const int tok = t0 + c * 8;
        int bytes = (tok + 8 <= a.B) ? 16 : (tok < a.B ? 2 * (a.B - tok) : 0);
        if (src_off[q] < 0) bytes = 0;
        const bf16* src = bytes ? a.X + src_off[q] + tok : a.X;
        cp_async16(dst + Swz<1>::off(r, c), src, bytes);
      }
    }
  };

  const int n_tiles = (a.B + 15) / 16;
#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < n_tiles) stage(s, s * 16);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int tt = 0; tt < n_tiles; tt++) {
    // refill the slot consumed in iteration tt - 1 (all lanes left it at the
    // trailing __syncwarp), then wait for tile tt: ST - 1 groups stay in flight
    if (tt + ST - 1 < n_tiles && SP_ABLATE_DW != 1) stage((tt + ST - 1) % ST, (tt + ST - 1) * 16);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 1));
    __syncwarp();
    ColCoef cc;
    cc.load(a, tt * 16, u, lane);
    const uint32_t slab_s = smem_u32(ring + (tt % ST) * R * 16);
#pragma unroll
    for (int i = 0; i < WPW; i++) {
      if (wid[i] < 0) continue;
#pragma unroll
      for (int bu = 0; bu < UB; bu++) {
        const int j = b_lo + wid[i] * UB + bu;
        BlockRegs<C> blk;
        unit_block<C>(blk, un[i], bu);
        blk.fidx = (i * UB + bu) * 8 + (lane & 7);  // local slab row
        if ((j + 1) * C <= n_g)
          dw_block<POOL, ACT, C, SPLIT, P_INT, false>(acc[i], bu, blk, biasT[i][bu], j, n_g, slab_s, cc, lane,
                                                      a.scale, a.p);
        else if (j < b_any)
          dw_block<POOL, ACT, C, SPLIT, P_INT, true>(acc[i], bu, blk, biasT[i][bu], j, n_g, slab_s, cc, lane,
                                                     a.scale, a.p);
      }
    }
    __syncwarp();
  }
  asm volatile("cp.async.wait_all;\n" ::);
  dw_writeback<C, WPW>(a, acc, wid, ubase, lane);
}

// write back: exclusive ownership, plain read-modify-write
template <int C, int WPW>
__device__ __forceinline__ void dw_writeback(const BwdArgs& a, DwAcc<C> (&acc)[WPW], const int (&wid)[WPW],
                                             long long ubase, int lane) {
  constexpr int UB = Unit<C>::UB;
  constexpr int NT = C / 8;
  constexpr int NP = DwAcc<C>::NP;
#pragma unroll
  for (int i = 0; i < WPW; i++) {
    if (wid[i] < 0) continue;
    const long long unit = ubase + wid[i];
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
#pragma unroll
      for (int q = 0; q < NP; q++) {
        // db: reduce over the 4 lanes of the quad (they hold different tokens)
        float db0 = acc[i].b[bu][q][0], db1 = acc[i].b[bu][q][1];
        db0 += __shfl_xor_sync(0xffffffff, db0, 1);
        db0 += __shfl_xor_sync(0xffffffff, db0, 2);
        db1 += __shfl_xor_sync(0xffffffff, db1, 1);
        db1 += __shfl_xor_sync(0xffffffff, db1, 2);
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int nt = 2 * q + h;
          if (nt >= NT) continue;
          const int f = bu * NT + nt;  // fragment within the unit
          const int qd = f >> 2, fi = f & 3;
          const long long quad = unit * Unit<C>::NQ4 + qd;
          // accumulator rows g (h = 0) / g + 8 (h = 1) = neuron g of n-tile nt,
          // cols 2t, 2t+1 = inputs: exactly the packed layout lane / element
          const float* d = acc[i].w[bu][q];
          float* wp = a.dW + ((quad * 32 + lane) * 4 + fi) * 2;
          float2 old = *reinterpret_cast<float2*>(wp);
          if (NT == 1) {  // rows g (hi) + rows g+8 (lo correction)
            old.x += d[0] + d[2];
            old.y += d[1] + d[3];
          } else {
            old.x += d[2 * h];
            old.y += d[2 * h + 1];
          }
          *reinterpret_cast<float2*>(wp) = old;
          // db of neuron g lives in lanes 4g..4g+3; packed bias wants (lane4 = g/2, e = g&1)
          if ((lane & 3) == 0) {
            const int g = lane >> 2;
            float* bp = a.dB + ((quad * 4 + (g >> 1)) * 4 + fi) * 2 + (g & 1);
            *bp += (h == 0) ? db0 : db1;
          }
        }
      }
    }
  }
}
