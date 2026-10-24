// EXPERIMENTAL width-preserving sparse kernels of the K = 64 S = K index-free layout ("kgx" bundle;
// experimental/kgx_wide.py, results/WIDTH_PRESERVING_SPARSITY.md).  New kernels next to the measured
// kgx.cuh ones (whose code is unchanged): compact input tiles of width TB = 16 or 8 per 16-neuron row
// group, optionally plus a protected two-input BACKBONE (BB) per row group.
//
// Model (unchanged): z_i = bias_i + sum_{k in support(i)} w_ik x_k, ONE FP32 accumulator per (token,
// neuron) before one ReLU, y_g = sum_valid ReLU(z_i) / n_g; backward recomputes the complete z.
//
//   tiles     row group u keeps an NKT = 64 / TB bit mask of its active TB-input tiles and the compact
//             index of its first active tile: tmeta[u] = (first << NKT) | mask; tile values
//             W[n_tiles][16 neurons][TB inputs] (BF16, the trainable parameter).  TB = 8: every active
//             8-input tile runs m16n8k8 MMAs in the forward / z recomputations (two of them where a whole
//             k16 is active); the dX products (k = 16 neurons, n = 8 inputs) and dW products are
//             m16n8k16 per active 8-input tile as for TB = 16.
//   backbone  bbk[u] = a0 | a1 << 8 (two local inputs of the group, a0 < a1), bbw[slot][2] BF16 weights
//             per neuron slot (independent per neuron).  z gets v0 x[a0] + v1 x[a1] added in FP32 on CUDA
//             cores (exact BF16 x BF16 products, FP32 sums) to the same accumulator as the tile MMAs, so
//             there is exactly one activation decision.  A tile overlapping a0 / a1 stores a structural
//             zero there (masked in the optimizer): no edge has two trainable copies.
//             dX: dA[a] += sum_i dz_i v_ia with dz in FP32 (scalar, NOT split); dW: dv_ia = sum_t dz x_a
//             in FP32.  With BB every valid row group runs (no all-zero group exists).
// Numerics label (BB): "mixed: tile products BF16 MMA (dZ split hi + lo or BF16), backbone products FP32
// scalar (dz FP32)".  Fast FP32-atomic cross-CTA reductions only (no deterministic mode).
// Work lists: kgx.cuh (items / chunks / tasks; for non-BB layers only groups with an active tile).
#pragma once
#include "kgx.cuh"

#ifndef SP_ABLATE_KW_BB
// ATTRIBUTION ONLY (build kind "ablation", deliberately WRONG results): -DSP_ABLATE_KW_BB=1 removes every
// backbone load / product / gradient while every row group still runs (all neurons live), so
// time(hyb16) - time(ablated hyb16) is the cost of the scalar backbone itself.
#define SP_ABLATE_KW_BB 0
#endif

struct KwArgs {
  KxArgs a;
  const int* bbk;   // [G NB C / 16] backbone coordinates (BB)
  const bf16* bbw;  // [G NB C, 2] backbone weights (BB)
  float* dBB;       // [G NB C, 2] FP32 backbone main_grad (BB)
  int tb;           // tile width (checked against the template)
  int bb;           // backbone flag (checked against the template)
};

template <int TB>
struct KwTile {
  static_assert(TB == 16 || TB == 8, "TB");
  static constexpr int NKT = 64 / TB;          // tiles per row group
  static constexpr int TE = 16 * TB;           // elements per tile
  static constexpr int K8 = TB / 8;            // k8 slices per tile
  static constexpr int MASK = (1 << NKT) - 1;
  // 8-bit k8-slice activity of a tile mask
  __device__ __forceinline__ static int k8mask(int m) {
    if constexpr (TB == 8) {
      return m;
    } else {
      int r = 0;
#pragma unroll
      for (int t = 0; t < 4; t++) r |= ((m >> t) & 1) * (3 << (2 * t));
      return r;
    }
  }
  // element offset of (neuron row, k8 slice kn, pair 2 t4) inside the compact tiles of a group
  __device__ __forceinline__ static long long off(long long first, int m, int row, int kn, int t4) {
    const int ti = kn / K8;
    const int rk = __popc(m & ((1 << ti) - 1));
    return (first + rk) * TE + row * TB + (kn % K8) * 8 + 2 * t4;
  }
};

// Warp-uniform tile masks are dispatched through a switch over each 4-bit nibble of the 8-bit k8-slice
// mask (slices 0-3 / 4-7): every case is straight-line code with exactly its active MMAs.  (ptxas
// if-converted per-slice / per-tile tests into predicated MMAs, which issue for inactive tiles too:
// measured 1.4x slower dW on one-tile supports.)
#define KW_NIB_CASES(F) \
  F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11) F(12) F(13) F(14) F(15)
#define KW_SWITCH_NIB(v, BODY)                          \
  switch (v) {                                          \
    KW_NIB_CASES(BODY)                                  \
    default:                                            \
      break;                                            \
  }

// x[token][a] of the staged slab (Swz<MT> layout of kx_stage) as FP32
template <int MT>
__device__ __forceinline__ float kw_x(const bf16* slab, int a, int tok) {
  return __bfloat162float(slab[Swz<MT>::off(a, tok >> 3) + (tok & 7)]);
}

__device__ __forceinline__ float2 kw_bbw(const KwArgs& w, long long slot) {
  const uint32_t v = __ldg(reinterpret_cast<const unsigned int*>(w.bbw) + slot);
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
}

// wf[kn] = W[c][8 kn + 2 (lane % 4) .. + 1] (0 for inactive slices); returns the k8 mask
template <int TB>
__device__ __forceinline__ int kw_wload(uint32_t* wf, const KxArgs& a, long long fb, int c, int t4, bool on, int& m) {
  using T = KwTile<TB>;
  const int meta = __ldg(a.tmeta + (fb * a.C + c) / 16);
  m = meta & T::MASK;
  const long long first = (unsigned)meta >> T::NKT;
  const int k8 = on ? T::k8mask(m) : 0;
#pragma unroll
  for (int kn = 0; kn < KX_KN; kn++)
    wf[kn] = ((k8 >> kn) & 1) ? kx_ld2(a.W + T::off(first, m, c & 15, kn, t4)) : 0u;
  return T::k8mask(m);
}

// d[mt] += A(tokens x 64) W^T over the k8 slices of nibble NIB (slices 4 H .. 4 H + 3) for all MT token
// tiles.  A pair of active slices of one k16 runs one m16n8k16; a single slice one m16n8k8 (A halves).
template <int MT, int H, int NIB>
__device__ __forceinline__ void kw_mma_nib(float (*d)[4], const KxA<MT>& af, const uint32_t* wf) {
#pragma unroll
  for (int p = 0; p < 2; p++) {
    const int kt = 2 * H + p;
    const int s2 = (NIB >> (2 * p)) & 3;
    if (s2 == 3) {
#pragma unroll
      for (int mt = 0; mt < MT; mt++) mma_m16n8k16_bf16(d[mt], af.r[mt][kt], wf[2 * kt], wf[2 * kt + 1]);
    } else if (s2 == 1) {
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        float* o = d[mt];
        mma_m16n8k8_bf16(o, af.r[mt][kt][0], af.r[mt][kt][1], wf[2 * kt], o[0], o[1], o[2], o[3]);
      }
    } else if (s2 == 2) {
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        float* o = d[mt];
        mma_m16n8k8_bf16(o, af.r[mt][kt][2], af.r[mt][kt][3], wf[2 * kt + 1], o[0], o[1], o[2], o[3]);
      }
    }
  }
}

template <int MT, int TB>
__device__ __forceinline__ void kw_mma_z(float (*d)[4], const KxA<MT>& af, const uint32_t* wf, int k8) {
#define KW_Z0(N) \
  case N: kw_mma_nib<MT, 0, N>(d, af, wf); break;
#define KW_Z1(N) \
  case N: kw_mma_nib<MT, 1, N>(d, af, wf); break;
  KW_SWITCH_NIB(k8 & 15, KW_Z0)
  KW_SWITCH_NIB(k8 >> 4, KW_Z1)
#undef KW_Z0
#undef KW_Z1
}

// da[mt][kn] += dZ (16 neurons, split hi / lo) W over the k8 slices of nibble NIB (dX: k = neurons, n = inputs)
template <int MT, int H, int NIB, bool SPLIT>
__device__ __forceinline__ void kw_da_nib(float (*da)[KX_KN][4], const uint32_t (*ah)[4], const uint32_t (*al)[4],
                                          const uint32_t (*wt)[KX_KN]) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    if ((NIB >> j) & 1) {
      const int kn = 4 * H + j;
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        mma_m16n8k16_bf16(da[mt][kn], ah[mt], wt[0][kn], wt[1][kn]);
        if (SPLIT) mma_m16n8k16_bf16(da[mt][kn], al[mt], wt[0][kn], wt[1][kn]);
      }
    }
  }
}

// acc[kn] += dZ^T (16 neurons x 16 tokens, split) X over the k8 slices of nibble NIB (dW)
template <int H, int NIB, bool SPLIT>
__device__ __forceinline__ void kw_dw_nib(float (*acc)[4], const uint32_t* ah, const uint32_t* al,
                                          const uint32_t (*bn)[2]) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    if ((NIB >> j) & 1) {
      const int kn = 4 * H + j;
      mma_m16n8k16_bf16(acc[kn], ah, bn[kn][0], bn[kn][1]);
      if (SPLIT) mma_m16n8k16_bf16(acc[kn], al, bn[kn][0], bn[kn][1]);
    }
  }
}

// ============================================================================
// forward: Mean(ReLU)
// ============================================================================
template <int MT, int NW, int TB, bool BB>
__global__ void __launch_bounds__(NW * 32) kw_fwd_kernel(KwArgs w) {
  constexpr bool BBX = BB && !SP_ABLATE_KW_BB;  // backbone arithmetic (BB: all groups run)
  const KxArgs& a = w.a;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * 16 * MT;
  kx_stage<MT>(a, a.feat + (long long)q * a.S, nullptr, slab, t0, false);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  const int C = a.C;
  KxA<MT> af;
  kx_gather<MT, true>(af, nullptr, nullptr, slab_s, lane);
  for (int it = lo + warp; it < hi; it += NW) {
    int g, fb0, nb, c0, c1;
    kx_item(a, it, g, fb0, nb, c0, c1);
    const int n_g = a.counts[g];
    const int j0 = fb0 - g * a.NB;
    float s[2 * MT];
#pragma unroll
    for (int i = 0; i < 2 * MT; i++) s[i] = 0.f;
    for (int bi = 0; bi < nb; bi++) {
      const long long fb = fb0 + bi;
      const int cend = min(c1, n_g - (j0 + bi) * C);
      const bf16* bb = a.bias + fb * C;
      for (int nt = c0 / 8; nt * 8 < cend; nt++) {
        const int c = nt * 8 + g4;
        uint32_t wf[KX_KN];
        int m;
        const int k8 = kw_wload<TB>(wf, a, fb, c, t4, true, m);
        if (!BB && m == 0) continue;  // no active tile: bias-only neurons (handled outside, exactly)
        const int cc = nt * 8 + 2 * t4;
        const float bx = __bfloat162float(bb[cc]), by = __bfloat162float(bb[cc + 1]);
        const bool v0 = cc < cend, v1 = cc + 1 < cend;
        int a0 = 0, a1 = 0;
        float2 w0 = make_float2(0.f, 0.f), w1 = w0;
        if constexpr (BBX) {
          const int k = __ldg(w.bbk + (fb * C + nt * 8) / 16);
          a0 = k & 0xFF;
          a1 = k >> 8;
          w0 = kw_bbw(w, fb * C + cc);
          w1 = kw_bbw(w, fb * C + cc + 1);
        }
        float dd[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) dd[mt][0] = dd[mt][2] = bx, dd[mt][1] = dd[mt][3] = by;
        kw_mma_z<MT, TB>(dd, af, wf, k8);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          float* d = dd[mt];
          if constexpr (BBX) {
            const float xa0 = kw_x<MT>(slab, a0, 16 * mt + g4), xa1 = kw_x<MT>(slab, a1, 16 * mt + g4);
            const float xb0 = kw_x<MT>(slab, a0, 16 * mt + 8 + g4), xb1 = kw_x<MT>(slab, a1, 16 * mt + 8 + g4);
            d[0] = fmaf(w0.y, xa1, fmaf(w0.x, xa0, d[0]));
            d[1] = fmaf(w1.y, xa1, fmaf(w1.x, xa0, d[1]));
            d[2] = fmaf(w0.y, xb1, fmaf(w0.x, xb0, d[2]));
            d[3] = fmaf(w1.y, xb1, fmaf(w1.x, xb0, d[3]));
          }
          s[2 * mt] += (v0 ? fmaxf(d[0], 0.f) : 0.f) + (v1 ? fmaxf(d[1], 0.f) : 0.f);
          s[2 * mt + 1] += (v0 ? fmaxf(d[2], 0.f) : 0.f) + (v1 ? fmaxf(d[3], 0.f) : 0.f);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 2 * MT; i++) {
      s[i] += __shfl_xor_sync(0xffffffff, s[i], 1);
      s[i] += __shfl_xor_sync(0xffffffff, s[i], 2);
    }
    if (t4 == 0) {
#pragma unroll
      for (int i = 0; i < 2 * MT; i++) {
        const int tok = t0 + (i >> 1) * 16 + (i & 1) * 8 + g4;
        if (tok < a.B) atomicAdd(a.ysum + (long long)g * a.B + tok, s[i]);
      }
    }
  }
}

// ============================================================================
// dX: register-accumulated dA over all records of the warp (the 64 set inputs), warp-private slab
// copies summed in a fixed order; BB adds the scalar backbone dA directly into the warp's copy (the cell
// (input, token) is always written by the same lane), so the copies are zeroed first and the register
// accumulator is ADDED at the end.
// ============================================================================
template <int MT, int NW, int TB, bool BB, bool SPLIT>
__global__ void __launch_bounds__(NW * 32) kw_dx_kernel(KwArgs w) {
  constexpr bool BBX = BB && !SP_ABLATE_KW_BB;  // backbone arithmetic (BB: all groups run)
  const KxArgs& a = w.a;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int BM = 16 * MT;
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  float* dxs = reinterpret_cast<float*>(smem_raw + sg_align16(a.D * 2) + a.S * BM * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * BM;
  const int16_t* fq = a.feat + (long long)q * a.S;
  if constexpr (BB)
    for (int i = threadIdx.x; i < NW * a.S * BM; i += NW * 32) dxs[i] = 0.f;
  kx_stage<MT>(a, fq, nullptr, slab, t0, false);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  float* mys = dxs + warp * a.S * BM;
  const int C = a.C;
  KxA<MT> af;
  kx_gather<MT, true>(af, nullptr, nullptr, slab_s, lane);
  float da[MT][KX_KN][4];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int kn = 0; kn < KX_KN; kn++) da[mt][kn][0] = da[mt][kn][1] = da[mt][kn][2] = da[mt][kn][3] = 0.f;
  for (int it = lo + warp; it < hi; it += NW) {
    int g, fb0, nb, c0, c1;
    kx_item(a, it, g, fb0, nb, c0, c1);
    const int n_g = a.counts[g];
    const int j0 = fb0 - g * a.NB;
    float c1v[MT][2];
    bool tv[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int tok = t0 + 16 * mt + 8 * h + g4;
        tv[mt][h] = tok < a.B;
        c1v[mt][h] = tv[mt][h] ? __ldg(a.c1T + (long long)g * a.B + tok) : 0.f;
      }
    for (int bi = 0; bi < nb; bi++) {
      const long long fb = fb0 + bi;
      const int cend = min(c1, n_g - (j0 + bi) * C);
      const bf16* bb = a.bias + fb * C;
      for (int c16 = c0 / 16; c16 * 16 < cend; c16++) {
        const bool two = c16 * 16 + 8 < cend;
        uint32_t wf[2][KX_KN], wt[2][KX_KN];
        float bx[2], by[2];
        int k8 = 0, m = 0;
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int nt = 2 * c16 + h;
          const bool on = h == 0 || two;
          k8 = kw_wload<TB>(wf[h], a, fb, nt * 8 + g4, t4, on, m);
#pragma unroll
          for (int kn = 0; kn < KX_KN; kn++) wt[h][kn] = movmatrix_trans(wf[h][kn]);
          const int cc = nt * 8 + 2 * t4;
          bx[h] = on ? __bfloat162float(bb[cc]) : 0.f;
          by[h] = on ? __bfloat162float(bb[cc + 1]) : 0.f;
        }
        if (!BB && m == 0) continue;
        int a0 = 0, a1 = 0;
        float2 vw[2][2];
        if constexpr (BBX) {
          const int k = __ldg(w.bbk + (fb * C + c16 * 16) / 16);
          a0 = k & 0xFF;
          a1 = k >> 8;
#pragma unroll
          for (int h = 0; h < 2; h++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
              const int cc = (2 * c16 + h) * 8 + 2 * t4 + e;
              vw[h][e] = (h == 0 || two) ? kw_bbw(w, fb * C + cc) : make_float2(0.f, 0.f);
            }
        }
        float zz[2][MT][4];
#pragma unroll
        for (int h = 0; h < 2; h++) {
#pragma unroll
          for (int mt = 0; mt < MT; mt++) zz[h][mt][0] = zz[h][mt][2] = bx[h], zz[h][mt][1] = zz[h][mt][3] = by[h];
          if (h == 0 || two) kw_mma_z<MT, TB>(zz[h], af, wf[h], k8);
        }
        uint32_t ah[MT][4], al[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          float xv[2][2];  // [token half][coordinate]
          if constexpr (BBX) {
#pragma unroll
            for (int r = 0; r < 2; r++) {
              xv[r][0] = kw_x<MT>(slab, a0, 16 * mt + 8 * r + g4);
              xv[r][1] = kw_x<MT>(slab, a1, 16 * mt + 8 * r + g4);
            }
          }
          float dz[2][4];
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int cc = (2 * c16 + h) * 8 + 2 * t4;
            float* d = zz[h][mt];
            if constexpr (BBX) {
#pragma unroll
              for (int i = 0; i < 4; i++) {
                const int r = i >> 1, e = i & 1;
                d[i] = fmaf(vw[h][e].y, xv[r][1], fmaf(vw[h][e].x, xv[r][0], d[i]));
              }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
              const int r = i >> 1;
              float v = dz_f<P_MEAN, A_RELU, 0>(d[i], c1v[mt][r], 0.f, 0.f, 1.f, 0.f);
              if (cc + (i & 1) >= cend || (h == 1 && !two)) v = 0.f;
              dz[h][i] = tv[mt][r] ? v : 0.f;
            }
          }
          if constexpr (BBX) {  // scalar backbone dA (FP32 dz), summed over the group's 16 neurons
            float p[2][2];
#pragma unroll
            for (int r = 0; r < 2; r++)
#pragma unroll
              for (int j = 0; j < 2; j++) {
                float acc = 0.f;
#pragma unroll
                for (int h = 0; h < 2; h++)
#pragma unroll
                  for (int e = 0; e < 2; e++) acc = fmaf(dz[h][2 * r + e], j ? vw[h][e].y : vw[h][e].x, acc);
                acc += __shfl_xor_sync(0xffffffff, acc, 1);
                acc += __shfl_xor_sync(0xffffffff, acc, 2);
                p[r][j] = acc;
              }
            if (t4 < 2) {  // lane (g4, t4 = r) owns the cells of token 16 mt + 8 r + g4
              const int r = t4, tl = 16 * mt + 8 * r + g4;
              mys[sg_dxs_off<MT>(a0, tl)] += p[r][0];
              mys[sg_dxs_off<MT>(a1, tl)] += p[r][1];
            }
          }
          split_bf16(dz[0][0], dz[0][1], ah[mt][0], al[mt][0]);
          split_bf16(dz[0][2], dz[0][3], ah[mt][1], al[mt][1]);
          split_bf16(dz[1][0], dz[1][1], ah[mt][2], al[mt][2]);
          split_bf16(dz[1][2], dz[1][3], ah[mt][3], al[mt][3]);
        }
        // dA += dZ W over the active slices (k = 16 neurons: a group whose second n-tile holds no valid
        // neuron has dz = 0 and W = 0 there; k8 == 0: backbone-only group)
#define KW_A0(N) \
  case N: kw_da_nib<MT, 0, N, SPLIT>(da, ah, al, wt); break;
#define KW_A1(N) \
  case N: kw_da_nib<MT, 1, N, SPLIT>(da, ah, al, wt); break;
        KW_SWITCH_NIB(k8 & 15, KW_A0)
        KW_SWITCH_NIB(k8 >> 4, KW_A1)
#undef KW_A0
#undef KW_A1
      }
    }
  }
  if constexpr (BB) __syncwarp();  // backbone cells (other lanes) before the register flush
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int kn = 0; kn < KX_KN; kn++) {
      const float* d = da[mt][kn];
      float* p0 = mys + sg_dxs_off<MT>(8 * kn + 2 * t4, 16 * mt + g4);
      float* p1 = mys + sg_dxs_off<MT>(8 * kn + 2 * t4 + 1, 16 * mt + g4);
      if constexpr (BB) {
        float2 v0 = *reinterpret_cast<float2*>(p0), v1 = *reinterpret_cast<float2*>(p1);
        v0.x += d[0]; v0.y += d[2]; v1.x += d[1]; v1.y += d[3];
        *reinterpret_cast<float2*>(p0) = v0;
        *reinterpret_cast<float2*>(p1) = v1;
      } else {
        *reinterpret_cast<float2*>(p0) = make_float2(d[0], d[2]);
        *reinterpret_cast<float2*>(p1) = make_float2(d[1], d[3]);
      }
    }
  __syncthreads();
  for (int i = threadIdx.x; i < a.S * BM; i += NW * 32) {
    const int f = i / BM, tl = i % BM;
    const int o = sg_dxs_off<MT>(f, tl);
    float v = dxs[o];
    for (int c = 1; c < NW; c++) v += dxs[c * a.S * BM + o];
    const int tok = t0 + tl;
    if (tok < a.B && v != 0.f) atomicAdd(a.dXT + (long long)fq[f] * a.B + tok, v);
  }
}

// ============================================================================
// dW / db (+ backbone dv): a warp owns one 16-neuron row group (exclusive writes), all tokens
// ============================================================================
template <int NW, int TS, int ST, int TB, bool BB, bool SPLIT>
__global__ void __launch_bounds__(NW * 32) kw_dw_kernel(KwArgs w) {
  constexpr bool BBX = BB && !SP_ABLATE_KW_BB;  // backbone arithmetic (BB: all groups run)
  using T = KwTile<TB>;
  const KxArgs& a = w.a;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* ring = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int S = a.S, C = a.C;

  uint32_t aw[KX_KT][4];
  float bz[2];
  bool rv[2];
  int lt[KX_KT], ln[KX_KT];
  float acc[KX_KN][4], db[2], dv[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  float2 vr[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
  int a0 = 0, a1 = 0;
  const int pos = lo + warp;
  const long long fbt = pos < hi ? a.tasks[2 * pos] : -1;
  const int tc0 = pos < hi ? 16 * a.tasks[2 * pos + 1] : 0;
  const int tg = fbt >= 0 ? (int)(fbt / a.NB) : 0;
  const long long fbs = fbt >= 0 ? fbt : 0;
  int m = 0, k8 = 0;
  long long first = 0;
  {
    const int n_g = fbt >= 0 ? a.counts[tg] : 0;
    const int j = fbt >= 0 ? (int)(fbt - (long long)tg * a.NB) : 0;
#pragma unroll
    for (int kn = 0; kn < KX_KN; kn++) acc[kn][0] = acc[kn][1] = acc[kn][2] = acc[kn][3] = 0.f;
    db[0] = db[1] = 0.f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int c = tc0 + g4 + 8 * h;
      rv[h] = fbt >= 0 && c < C && j * C + c < n_g;
      bz[h] = rv[h] ? __bfloat162float(a.bias[fbt * C + c]) : 0.f;
      if constexpr (BBX) vr[h] = rv[h] ? kw_bbw(w, fbt * C + c) : make_float2(0.f, 0.f);
    }
    const int meta = fbt >= 0 ? a.tmeta[(fbs * C + tc0) / 16] : 0;
    m = meta & T::MASK;
    first = (unsigned)meta >> T::NKT;
    k8 = T::k8mask(m);
    if constexpr (BBX) {
      const int k = fbt >= 0 ? __ldg(w.bbk + (fbs * C + tc0) / 16) : 0;
      a0 = k & 0xFF;
      a1 = k >> 8;
    }
#pragma unroll
    for (int kt = 0; kt < KX_KT; kt++) {
      const bool lo8 = (k8 >> (2 * kt)) & 1, hi8 = (k8 >> (2 * kt + 1)) & 1;
      aw[kt][0] = lo8 && rv[0] ? kx_ld2(a.W + T::off(first, m, g4, 2 * kt, t4)) : 0u;
      aw[kt][1] = lo8 && rv[1] ? kx_ld2(a.W + T::off(first, m, g4 + 8, 2 * kt, t4)) : 0u;
      aw[kt][2] = hi8 && rv[0] ? kx_ld2(a.W + T::off(first, m, g4, 2 * kt + 1, t4)) : 0u;
      aw[kt][3] = hi8 && rv[1] ? kx_ld2(a.W + T::off(first, m, g4 + 8, 2 * kt + 1, t4)) : 0u;
      const int mm = lane >> 3, r = lane & 7;
      lt[kt] = 16 * kt + (mm & 1) * 8 + r;
      ln[kt] = 8 * (2 * kt + (mm >> 1)) + r;
    }
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
    cp_async_commit();
  }
  for (int s = 0; s < n_st; s++) {
    cp_async_wait_group<ST - 2>();
    __syncthreads();
    if (s + ST - 1 < n_st) stage(s + ST - 1);
    cp_async_commit();
    const bf16* base = ring + (s % ST) * stage_elems;
#pragma unroll 1
    for (int sub = 0; sub < TS; sub++) {
      const int tt = s * TS + sub;
      if (tt >= n_tiles) break;
      if (fbt < 0) continue;
      const bf16* sb = base + sub * S * 16;
      const uint32_t slab_s = smem_u32(sb);
      float cc[2][2];
      bool tv[2][2];
#pragma unroll
      for (int n = 0; n < 2; n++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int tok = tt * 16 + 8 * n + 2 * t4 + e;
          tv[n][e] = tok < a.B;
          cc[n][e] = tv[n][e] ? __ldg(a.c1T + (long long)tg * a.B + tok) : 0.f;
        }
      float zt[2][4];  // rows = neurons (lane / 4, + 8), cols = tokens of n-tile n
#pragma unroll
      for (int n = 0; n < 2; n++) {
        zt[n][0] = zt[n][1] = bz[0];
        zt[n][2] = zt[n][3] = bz[1];
      }
      uint32_t bn[KX_KN][2];
#pragma unroll
      for (int kt = 0; kt < KX_KT; kt++) {
        const int s2 = (k8 >> (2 * kt)) & 3;
        if (!s2) {
          bn[2 * kt][0] = bn[2 * kt][1] = bn[2 * kt + 1][0] = bn[2 * kt + 1][1] = 0u;
          continue;
        }
        uint32_t bx[4];  // (k lo, chunk 0), (k hi, chunk 0), (k lo, chunk 1), (k hi, chunk 1)
        ldmatrix_x4_trans(bx, slab_s + 2 * Swz<1>::off(lt[kt], lane >> 4));
        if constexpr (TB == 16) {
          mma_m16n8k16_bf16(zt[0], aw[kt], bx[0], bx[1]);
          mma_m16n8k16_bf16(zt[1], aw[kt], bx[2], bx[3]);
        } else {
          if (s2 & 1) {
            mma_m16n8k8_bf16(zt[0], aw[kt][0], aw[kt][1], bx[0], zt[0][0], zt[0][1], zt[0][2], zt[0][3]);
            mma_m16n8k8_bf16(zt[1], aw[kt][0], aw[kt][1], bx[2], zt[1][0], zt[1][1], zt[1][2], zt[1][3]);
          }
          if (s2 & 2) {
            mma_m16n8k8_bf16(zt[0], aw[kt][2], aw[kt][3], bx[1], zt[0][0], zt[0][1], zt[0][2], zt[0][3]);
            mma_m16n8k8_bf16(zt[1], aw[kt][2], aw[kt][3], bx[3], zt[1][0], zt[1][1], zt[1][2], zt[1][3]);
          }
        }
        uint32_t b4[4];
        ldmatrix_x4(b4, slab_s + 2 * Swz<1>::off(ln[kt], (lane >> 3) & 1));
        bn[2 * kt][0] = b4[0];
        bn[2 * kt][1] = b4[1];
        bn[2 * kt + 1][0] = b4[2];
        bn[2 * kt + 1][1] = b4[3];
      }
      float xb[2][2][2];  // [coordinate][n-tile][col]: x of the lane's 4 tokens at a0 / a1
      if constexpr (BBX) {
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const uint32_t u = *reinterpret_cast<const uint32_t*>(sb + Swz<1>::off(j ? a1 : a0, n) + 2 * t4);
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
            xb[j][n][0] = f.x;
            xb[j][n][1] = f.y;
          }
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const int r = e >> 1, col = e & 1;
            zt[n][e] = fmaf(vr[r].y, xb[1][n][col], fmaf(vr[r].x, xb[0][n][col], zt[n][e]));
          }
      }
      float dz[2][4];
#pragma unroll
      for (int n = 0; n < 2; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = e >> 1, col = e & 1;
          float v = dz_f<P_MEAN, A_RELU, 0>(zt[n][e], cc[n][col], 0.f, 0.f, 1.f, 0.f);
          if (!rv[r]) v = 0.f;
          dz[n][e] = tv[n][col] ? v : 0.f;
        }
      db[0] += (dz[0][0] + dz[0][1]) + (dz[1][0] + dz[1][1]);
      db[1] += (dz[0][2] + dz[0][3]) + (dz[1][2] + dz[1][3]);
      if constexpr (BBX) {
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int n = 0; n < 2; n++)
              dv[r][j] = fmaf(dz[n][2 * r + 1], xb[j][n][1], fmaf(dz[n][2 * r], xb[j][n][0], dv[r][j]));
      }
      if constexpr (BB)
        if (m == 0) continue;  // backbone-only group
      uint32_t ah[4], al[4];
      split_bf16(dz[0][0], dz[0][1], ah[0], al[0]);
      split_bf16(dz[0][2], dz[0][3], ah[1], al[1]);
      split_bf16(dz[1][0], dz[1][1], ah[2], al[2]);
      split_bf16(dz[1][2], dz[1][3], ah[3], al[3]);
#define KW_W0(N) \
  case N: kw_dw_nib<0, N, SPLIT>(acc, ah, al, bn); break;
#define KW_W1(N) \
  case N: kw_dw_nib<1, N, SPLIT>(acc, ah, al, bn); break;
      KW_SWITCH_NIB(k8 & 15, KW_W0)
      KW_SWITCH_NIB(k8 >> 4, KW_W1)
#undef KW_W0
#undef KW_W1
    }
  }
  cp_async_wait_all();
  float b0 = db[0], b1 = db[1];
  b0 += __shfl_xor_sync(0xffffffff, b0, 1);
  b0 += __shfl_xor_sync(0xffffffff, b0, 2);
  b1 += __shfl_xor_sync(0xffffffff, b1, 1);
  b1 += __shfl_xor_sync(0xffffffff, b1, 2);
  if constexpr (BBX) {
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int j = 0; j < 2; j++) {
        dv[r][j] += __shfl_xor_sync(0xffffffff, dv[r][j], 1);
        dv[r][j] += __shfl_xor_sync(0xffffffff, dv[r][j], 2);
      }
  }
  if (fbt < 0) return;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    if (!rv[h]) continue;
    const int c = tc0 + g4 + 8 * h;
#pragma unroll
    for (int kn = 0; kn < KX_KN; kn++) {
      if (!((k8 >> kn) & 1)) continue;
      float* wp = a.dW + T::off(first, m, g4 + 8 * h, kn, t4);
      float2 old = *reinterpret_cast<float2*>(wp);
      old.x += acc[kn][2 * h];
      old.y += acc[kn][2 * h + 1];
      *reinterpret_cast<float2*>(wp) = old;
    }
    if (t4 == 0) {
      a.dB[fbt * C + c] += h ? b1 : b0;
      if constexpr (BBX) {
        float2* p = reinterpret_cast<float2*>(w.dBB + 2 * (fbt * C + c));
        float2 o = *p;
        o.x += dv[h][0];
        o.y += dv[h][1];
        *p = o;
      }
    }
  }
}
