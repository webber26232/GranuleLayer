// EXPERIMENTAL generalized-fan-in input-supergroup kernels ("kgen" bundle; sparse_pooling/experimental).
//
// Model (identical to the production layer, any fan-in K in {8, 16, 32, 64}): logical block j of
// output g holds C neurons (C % 8 == 0, C <= 256) sharing ONE K-input index record; every neuron has
// its own K weights and bias:
//     z = bias + sum_{k < K} w[k] x[idx[k]],   a = act(z),   y_g = mean over the n_g valid neurons.
// The K products are accumulated into one FP32 accumulator (K / 16 m16n8k16 tiles, or one m16n8k8
// at K = 8) before the single activation; backward recomputes the complete z and uses its one
// activation decision for every dX / dW / db term (no K8 sub-neurons).
//
// Storage (canonical, unpacked; experimental/kgen_layout.py):
//     idx [G * NB, K] int16 (sorted, distinct)   W [G * NB, C, K] bf16   bias [G * NB, C] bf16
//     dW  [G * NB, C, K] fp32                     dB [G * NB, C] fp32 (main_grad, same order)
// Work lists (host-built, experimental/kgen_graph.py), every logical block in exactly one bucket:
//     items  [n, 3]  (g, first flat block g * NB + j, blocks): a bucket of output g in supergroup q
//     chunks [m, 3]  (q, lo, hi): a CTA's item range (fwd, dX) / task range (dW), one supergroup
//     tasks  [n, 2]  (flat block, 16-neuron group): dW ownership units, ordered by supergroup
//
// kg_fwd  CTA = (token tile of 16 MT tokens = blockIdx.x, chunk = blockIdx.y).  The S rows of X of
//         the chunk's supergroup are staged in shared memory (Swz<MT> slab of sparse_pool_fwd.cuh);
//         the K inputs of a record are gathered with one ldmatrix.x4.trans per (m16 tile, k16 tile).
//         A warp takes whole items; one FP32 atomicAdd per (token, item) into ysum [G, B].
// kg_dx   same CTA; dA = dZ W^T (W^T fragments by movmatrix), dZ = c1 act'(z) as BF16 hi (+ lo with
//         SPLIT) operands, accumulated over the block's 16-neuron groups, then added into FP32 slab
//         copies [S][16 MT] (dx_group_warps warps per copy: 1 = private read-modify-write, else the
//         64-bit CAS of the production kernels); one FP32 atomicAdd per (feature, token) per CTA into
//         dXT [D, B].
// kg_dw   CTA = chunk of tasks of one supergroup, all tokens (TS 16-token tiles per ring stage, ST
//         stages).  Per task: z^T = W X^T (W rows are the A operand), dZ^T, dW^T = dZ^T X (hi + lo),
//         db = sum dZ; register accumulators, exclusive ownership, one plain read-modify-write.
// Numerics: BF16 operands, FP32 accumulation; dZ enters the gradient products as one BF16 (bf16) or
// a hi + lo BF16 pair (split), as the production kernels; the summation order differs from them.
// dZ uses the production generic expression dz_f (c1 * act'(z); padded tokens / neurons selected to
// 0), so non-finite coefficients propagate as in the production generic path (no specialised
// epilogue, no fallback launch needed).  Fast reduction only (FP32 atomics): no deterministic mode.
#pragma once
#include "../../sparse_pool_sg.cuh"

struct KgArgs {
  const bf16* X;  // feature-major X^T: element (b, f) at X[f * sxf + b]
  long long sxf;
  const int16_t* idx;
  const bf16* W;
  const bf16* bias;
  const int* counts;    // [G]
  const int16_t* feat;  // [Q, S]
  const int* items;     // [n_items, 3]
  const int* chunks;    // [n_chunks, 3]
  const int* tasks;     // [n_tasks, 2]
  float* ysum;          // [G, B]
  const float* c1T;     // [G, B] Mean coefficients g / n_g
  float* dXT;           // [D, B]
  float* dW;
  float* dB;
  int B, G, D, S, NB, C;
  int dx_group_warps;
  int flags;
};

// compile-time tilings per K (also reported to the host by sp_kg_config)
template <int K>
struct KgCfg {
  static constexpr int FWD_MT = K == 64 ? 2 : 4, FWD_NW = 8;
  static constexpr int DX_MT = K >= 32 ? 2 : 4, DX_NW = 8;
  static constexpr int DW_WPT = K == 8 ? 8 : (K == 16 ? 4 : (K == 32 ? 2 : 1)), DW_NW = 8, DW_TS = 2, DW_ST = 3;
};

template <int MT>
__device__ __forceinline__ void kg_stage(const KgArgs& a, const int16_t* fq, int16_t* smap, bf16* slab, int t0) {
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

// A fragments (rows = tokens, cols = fan-in) of one record for all MT m16 tiles.
// K >= 16: af[mt][kt][0..3] of m16n8k16; lane L supplies the slab row of fan-in
//          16 kt + 8 ((L >> 4) & 1) + (L & 7) at token chunk 2 mt + ((L >> 3) & 1).
// K == 8:  af[mt][0][0..1] of m16n8k8 (the production sg_gather).
template <int K, int MT>
struct KgA {
  static constexpr int KT = K >= 16 ? K / 16 : 1;
  uint32_t r[MT][KT][4];
};

template <int K, int MT>
__device__ __forceinline__ void kg_gather(KgA<K, MT>& af, const int16_t* rec, const int16_t* smap, uint32_t slab_s,
                                          int lane) {
  if constexpr (K == 8) {
    const int loc = smap[rec[lane & 7]];
    const int mat = lane >> 3;
    if constexpr (MT == 1) {
      ldmatrix_x2_trans(af.r[0][0], slab_s + 2 * Swz<1>::off(loc, mat & 1));
    } else {
#pragma unroll
      for (int j = 0; j < MT / 2; j++) {
        uint32_t t[4];
        ldmatrix_x4_trans(t, slab_s + 2 * Swz<MT>::off(loc, 4 * j + mat));
        af.r[2 * j][0][0] = t[0];
        af.r[2 * j][0][1] = t[1];
        af.r[2 * j + 1][0][0] = t[2];
        af.r[2 * j + 1][0][1] = t[3];
      }
    }
  } else {
    const int kk = ((lane >> 4) & 1) * 8 + (lane & 7), co = (lane >> 3) & 1;
#pragma unroll
    for (int kt = 0; kt < K / 16; kt++) {
      const int loc = smap[rec[16 * kt + kk]];
#pragma unroll
      for (int mt = 0; mt < MT; mt++) ldmatrix_x4_trans(af.r[mt][kt], slab_s + 2 * Swz<MT>::off(loc, 2 * mt + co));
    }
  }
}

__device__ __forceinline__ uint32_t ld_w2(const bf16* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }

// z (16 tokens x 8 neurons, accumulator layout) of n-tile wf (wf[k8] = W[c = 8 nt + lane / 4][8 k8 + 2 (lane % 4)]
// as a BF16 pair) for m16 tile mt; d initialised with the bias of the lane's columns
template <int K, int MT>
__device__ __forceinline__ void kg_z(float* d, const KgA<K, MT>& af, int mt, const uint32_t* wf, float bx, float by) {
  d[0] = bx; d[1] = by; d[2] = bx; d[3] = by;
  if constexpr (K == 8) {
    mma_m16n8k8_bf16(d, af.r[mt][0][0], af.r[mt][0][1], wf[0], d[0], d[1], d[2], d[3]);
  } else {
#pragma unroll
    for (int kt = 0; kt < K / 16; kt++) mma_m16n8k16_bf16(d, af.r[mt][kt], wf[2 * kt], wf[2 * kt + 1]);
  }
}

// ============================================================================
// forward: Mean(act)
// ============================================================================
template <int K, int MT, int NW, int ACT>
__global__ void __launch_bounds__(NW * 32) kg_fwd_kernel(KgArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * 16 * MT;
  kg_stage<MT>(a, a.feat + (long long)q * a.S, smap, slab, t0);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  const int C = a.C;
  for (int it = lo + warp; it < hi; it += NW) {
    const int g = a.items[3 * it], fb0 = a.items[3 * it + 1], nb = a.items[3 * it + 2];
    const int n_g = a.counts[g];
    const int j0 = fb0 - g * a.NB;
    float s[2 * MT];
#pragma unroll
    for (int i = 0; i < 2 * MT; i++) s[i] = 0.f;
    for (int bi = 0; bi < nb; bi++) {
      const long long fb = fb0 + bi;
      const int nval = min(C, n_g - (j0 + bi) * C);  // valid neurons of the block (> 0: items hold valid blocks)
      KgA<K, MT> af;
      kg_gather<K, MT>(af, a.idx + fb * K, smap, slab_s, lane);
      const bf16* wb = a.W + fb * C * K;
      const bf16* bb = a.bias + fb * C;
      for (int nt = 0; nt * 8 < nval; nt++) {
        uint32_t wf[K / 8];
#pragma unroll
        for (int k8 = 0; k8 < K / 8; k8++) wf[k8] = ld_w2(wb + (nt * 8 + g4) * K + 8 * k8 + 2 * t4);
        const int cc = nt * 8 + 2 * t4;
        const float bx = __bfloat162float(bb[cc]), by = __bfloat162float(bb[cc + 1]);
        const bool v0 = cc < nval, v1 = cc + 1 < nval;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          float d[4];
          kg_z<K, MT>(d, af, mt, wf, bx, by);
          s[2 * mt] += (v0 ? act_f<ACT>(d[0]) : 0.f) + (v1 ? act_f<ACT>(d[1]) : 0.f);
          s[2 * mt + 1] += (v0 ? act_f<ACT>(d[2]) : 0.f) + (v1 ? act_f<ACT>(d[3]) : 0.f);
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
// dX
// ============================================================================
template <int K, int MT, int NW, int ACT, bool SPLIT>
__global__ void __launch_bounds__(NW * 32) kg_dx_kernel(KgArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int BM = 16 * MT, KN = K / 8;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  float* dxs = reinterpret_cast<float*>(smem_raw + sg_align16(a.D * 2) + a.S * BM * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * BM;
  const int WG = a.dx_group_warps, ncopy = NW / WG;
  const bool priv = WG == 1;
  const int16_t* fq = a.feat + (long long)q * a.S;
  for (int i = threadIdx.x; i < ncopy * a.S * BM; i += NW * 32) dxs[i] = 0.f;
  kg_stage<MT>(a, fq, smap, slab, t0);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  float* mys = dxs + (warp / WG) * a.S * BM;
  const int C = a.C;
  for (int it = lo + warp; it < hi; it += NW) {
    const int g = a.items[3 * it], fb0 = a.items[3 * it + 1], nb = a.items[3 * it + 2];
    const int n_g = a.counts[g];
    const int j0 = fb0 - g * a.NB;
    float c1[MT][2];
    bool tv[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int tok = t0 + 16 * mt + 8 * h + g4;
        tv[mt][h] = tok < a.B;
        c1[mt][h] = tv[mt][h] ? __ldg(a.c1T + (long long)g * a.B + tok) : 0.f;
      }
    for (int bi = 0; bi < nb; bi++) {
      const long long fb = fb0 + bi;
      const int nval = min(C, n_g - (j0 + bi) * C);
      const int16_t* rec = a.idx + fb * K;
      KgA<K, MT> af;
      kg_gather<K, MT>(af, rec, smap, slab_s, lane);
      int f0[KN], f1[KN];  // local slab rows of this lane's two dA columns per k8 tile
#pragma unroll
      for (int kn = 0; kn < KN; kn++) {
        f0[kn] = smap[rec[8 * kn + 2 * t4]];
        f1[kn] = smap[rec[8 * kn + 2 * t4 + 1]];
      }
      float da[MT][KN][4];
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int kn = 0; kn < KN; kn++) da[mt][kn][0] = da[mt][kn][1] = da[mt][kn][2] = da[mt][kn][3] = 0.f;
      const bf16* wb = a.W + fb * C * K;
      const bf16* bb = a.bias + fb * C;
      for (int c16 = 0; c16 * 16 < nval; c16++) {
        const bool two = c16 * 16 + 8 < nval;  // second n-tile holds valid neurons
        uint32_t wf[2][KN], wt[2][KN];
        float bx[2], by[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int nt = 2 * c16 + h;
          const bool on = h == 0 || two;
#pragma unroll
          for (int kn = 0; kn < KN; kn++) {
            wf[h][kn] = on ? ld_w2(wb + (nt * 8 + g4) * K + 8 * kn + 2 * t4) : 0u;
            wt[h][kn] = movmatrix_trans(wf[h][kn]);  // B[k = neuron][n = fan-in]
          }
          const int cc = nt * 8 + 2 * t4;
          bx[h] = on ? __bfloat162float(bb[cc]) : 0.f;
          by[h] = on ? __bfloat162float(bb[cc + 1]) : 0.f;
        }
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          float dz[2][4];
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int cc = (2 * c16 + h) * 8 + 2 * t4;
            if (h == 1 && !two) {
              dz[1][0] = dz[1][1] = dz[1][2] = dz[1][3] = 0.f;
              continue;
            }
            float d[4];
            kg_z<K, MT>(d, af, mt, wf[h], bx[h], by[h]);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              const int r = i >> 1;
              float v = dz_f<P_MEAN, ACT, 0>(d[i], c1[mt][r], 0.f, 0.f, 1.f, 0.f);
              if (cc + (i & 1) >= nval) v = 0.f;
              dz[h][i] = tv[mt][r] ? v : 0.f;  // select (not multiply): padded tokens may give inf
            }
          }
          uint32_t ah[4], al[4];
          split_bf16(dz[0][0], dz[0][1], ah[0], al[0]);
          split_bf16(dz[0][2], dz[0][3], ah[1], al[1]);
          split_bf16(dz[1][0], dz[1][1], ah[2], al[2]);
          split_bf16(dz[1][2], dz[1][3], ah[3], al[3]);
#pragma unroll
          for (int kn = 0; kn < KN; kn++) {
            float* o = da[mt][kn];
            if (two) {
              mma_m16n8k16_bf16(o, ah, wt[0][kn], wt[1][kn]);
              if (SPLIT) mma_m16n8k16_bf16(o, al, wt[0][kn], wt[1][kn]);
            } else {
              mma_m16n8k8_bf16(o, ah[0], ah[1], wt[0][kn], o[0], o[1], o[2], o[3]);
              if (SPLIT) mma_m16n8k8_bf16(o, al[0], al[1], wt[0][kn], o[0], o[1], o[2], o[3]);
            }
          }
        }
      }
      // add the block's dA into the slab copy (a record's inputs are distinct: no intra-warp races)
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int kn = 0; kn < KN; kn++) {
          const float* d = da[mt][kn];
          float* p0 = mys + sg_dxs_off<MT>(f0[kn], 16 * mt + g4);
          float* p1 = mys + sg_dxs_off<MT>(f1[kn], 16 * mt + g4);
          if (priv) {
            float2 v0 = *reinterpret_cast<float2*>(p0), v1 = *reinterpret_cast<float2*>(p1);
            v0.x += d[0]; v0.y += d[2]; v1.x += d[1]; v1.y += d[3];
            *reinterpret_cast<float2*>(p0) = v0;
            *reinterpret_cast<float2*>(p1) = v1;
          } else {
            atomic_add_f32x2(p0, d[0], d[2]);
            atomic_add_f32x2(p1, d[1], d[3]);
          }
        }
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < a.S * BM; i += NW * 32) {
    const int f = i / BM, tl = i % BM;
    const int o = sg_dxs_off<MT>(f, tl);
    float v = dxs[o];
    for (int c = 1; c < ncopy; c++) v += dxs[c * a.S * BM + o];
    const int tok = t0 + tl;
    if (tok < a.B && v != 0.f) atomicAdd(a.dXT + (long long)fq[f] * a.B + tok, v);
  }
}

// ============================================================================
// dW / db
// ============================================================================
// one task = 16 neurons (rows) of one block: W A fragments (all K), bias, row validity, slab rows
template <int K>
struct KgTask {
  static constexpr int KT = K >= 16 ? K / 16 : 1;
  uint32_t aw[KT][4];  // K == 8: aw[0][0..1] (m16n8k8)
  float bz[2];         // bias of rows lane / 4, lane / 4 + 8
  bool rv[2];          // row valid (neuron < C and slot < n_g)
  int lt[KT];          // z^T B operand (ldmatrix.trans): slab row of this lane's fan-in, per k16 tile
  int ln[KT];          // dW B operand (ldmatrix): slab row per pair of k8 tiles (K == 8: unused)
  long long fb;        // flat block (-1: no task)
  int c0, g;
};

template <int K, int WPT, int NW, int TS, int ST, int ACT, bool SPLIT>
__global__ void __launch_bounds__(NW * 32) kg_dw_kernel(KgArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int KT = KgTask<K>::KT, KN = K / 8;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* ring = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int S = a.S, C = a.C;
  for (int i = threadIdx.x; i < S; i += NW * 32) smap[fq[i]] = (int16_t)i;
  __syncthreads();

  KgTask<K> tk[WPT];
  float acc[WPT][KN][4], db[WPT][2];
#pragma unroll
  for (int i = 0; i < WPT; i++) {
    const int pos = lo + warp * WPT + i;
    KgTask<K>& t = tk[i];
    t.fb = pos < hi ? a.tasks[2 * pos] : -1;
    t.c0 = pos < hi ? 16 * a.tasks[2 * pos + 1] : 0;
    t.g = t.fb >= 0 ? (int)(t.fb / a.NB) : 0;
    const int n_g = t.fb >= 0 ? a.counts[t.g] : 0;
    const int j = t.fb >= 0 ? (int)(t.fb - (long long)t.g * a.NB) : 0;
#pragma unroll
    for (int kn = 0; kn < KN; kn++) acc[i][kn][0] = acc[i][kn][1] = acc[i][kn][2] = acc[i][kn][3] = 0.f;
    db[i][0] = db[i][1] = 0.f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int c = t.c0 + g4 + 8 * h;
      t.rv[h] = t.fb >= 0 && c < C && j * C + c < n_g;
      t.bz[h] = t.rv[h] ? __bfloat162float(a.bias[t.fb * C + c]) : 0.f;
    }
    const long long fbs = t.fb >= 0 ? t.fb : 0;
    const bf16* w0 = a.W + (fbs * C + t.c0 + g4) * K;  // row c
    const bf16* w1 = w0 + 8 * K;                        // row c + 8
    const int16_t* rec = a.idx + fbs * K;
    if constexpr (K == 8) {
      t.aw[0][0] = t.rv[0] ? ld_w2(w0 + 2 * t4) : 0u;
      t.aw[0][1] = t.rv[1] ? ld_w2(w1 + 2 * t4) : 0u;
      t.lt[0] = smap[rec[lane & 7]];
      t.ln[0] = t.lt[0];
    } else {
#pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        t.aw[kt][0] = t.rv[0] ? ld_w2(w0 + 16 * kt + 2 * t4) : 0u;
        t.aw[kt][1] = t.rv[1] ? ld_w2(w1 + 16 * kt + 2 * t4) : 0u;
        t.aw[kt][2] = t.rv[0] ? ld_w2(w0 + 16 * kt + 8 + 2 * t4) : 0u;
        t.aw[kt][3] = t.rv[1] ? ld_w2(w1 + 16 * kt + 8 + 2 * t4) : 0u;
        const int m = lane >> 3, r = lane & 7;
        t.lt[kt] = smap[rec[16 * kt + (m & 1) * 8 + r]];       // matrices (k lo/hi) x (token chunk m >> 1)
        t.ln[kt] = smap[rec[8 * (2 * kt + (m >> 1)) + r]];     // matrices (k8 tile 2 kt + m >> 1) x (chunk m & 1)
      }
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
      const uint32_t slab_s = smem_u32(base + sub * S * 16);
#pragma unroll
      for (int i = 0; i < WPT; i++) {
        const KgTask<K>& t = tk[i];
        if (t.fb < 0) continue;
        // coefficients of this lane's z^T columns: tokens 16 tt + 8 n + 2 (lane % 4) + e
        float cc[2][2];
        bool tv[2][2];
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const int tok = tt * 16 + 8 * n + 2 * t4 + e;
            tv[n][e] = tok < a.B;
            cc[n][e] = tv[n][e] ? __ldg(a.c1T + (long long)t.g * a.B + tok) : 0.f;
          }
        float zt[2][4];  // z^T: rows = neurons (lane / 4, + 8), cols = tokens of n-tile n
#pragma unroll
        for (int n = 0; n < 2; n++) {
          zt[n][0] = zt[n][1] = t.bz[0];
          zt[n][2] = zt[n][3] = t.bz[1];
        }
        uint32_t bn[KN][2];  // dW B operand: X[token][fan-in] per k8 tile, token chunks 0 / 1
        if constexpr (K == 8) {
          uint32_t bx[2];
          ldmatrix_x2_trans(bx, slab_s + 2 * Swz<1>::off(t.lt[0], (lane >> 3) & 1));
          mma_m16n8k8_bf16(zt[0], t.aw[0][0], t.aw[0][1], bx[0], zt[0][0], zt[0][1], zt[0][2], zt[0][3]);
          mma_m16n8k8_bf16(zt[1], t.aw[0][0], t.aw[0][1], bx[1], zt[1][0], zt[1][1], zt[1][2], zt[1][3]);
          ldmatrix_x2(bn[0], slab_s + 2 * Swz<1>::off(t.ln[0], (lane >> 3) & 1));
        } else {
#pragma unroll
          for (int kt = 0; kt < KT; kt++) {
            uint32_t bx[4];  // (k lo, chunk 0), (k hi, chunk 0), (k lo, chunk 1), (k hi, chunk 1)
            ldmatrix_x4_trans(bx, slab_s + 2 * Swz<1>::off(t.lt[kt], lane >> 4));
            mma_m16n8k16_bf16(zt[0], t.aw[kt], bx[0], bx[1]);
            mma_m16n8k16_bf16(zt[1], t.aw[kt], bx[2], bx[3]);
            uint32_t b4[4];  // (k8 2 kt, chunk 0), (k8 2 kt, chunk 1), (k8 2 kt + 1, chunk 0), (.., chunk 1)
            ldmatrix_x4(b4, slab_s + 2 * Swz<1>::off(t.ln[kt], (lane >> 3) & 1));
            bn[2 * kt][0] = b4[0];
            bn[2 * kt][1] = b4[1];
            bn[2 * kt + 1][0] = b4[2];
            bn[2 * kt + 1][1] = b4[3];
          }
        }
        float dz[2][4];
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const int r = e >> 1, col = e & 1;
            float v = dz_f<P_MEAN, ACT, 0>(zt[n][e], cc[n][col], 0.f, 0.f, 1.f, 0.f);
            if (!t.rv[r]) v = 0.f;
            dz[n][e] = tv[n][col] ? v : 0.f;
          }
        db[i][0] += (dz[0][0] + dz[0][1]) + (dz[1][0] + dz[1][1]);
        db[i][1] += (dz[0][2] + dz[0][3]) + (dz[1][2] + dz[1][3]);
        uint32_t ah[4], al[4];  // A = dZ^T [16 neurons x 16 tokens]
        split_bf16(dz[0][0], dz[0][1], ah[0], al[0]);
        split_bf16(dz[0][2], dz[0][3], ah[1], al[1]);
        split_bf16(dz[1][0], dz[1][1], ah[2], al[2]);
        split_bf16(dz[1][2], dz[1][3], ah[3], al[3]);
#pragma unroll
        for (int kn = 0; kn < KN; kn++) {
          mma_m16n8k16_bf16(acc[i][kn], ah, bn[kn][0], bn[kn][1]);
          if (SPLIT) mma_m16n8k16_bf16(acc[i][kn], al, bn[kn][0], bn[kn][1]);
        }
      }
    }
  }
  cp_async_wait_all();
  // write back (exclusive ownership): rows lane / 4 (acc[0..1]) and lane / 4 + 8 (acc[2..3])
#pragma unroll
  for (int i = 0; i < WPT; i++) {
    const KgTask<K>& t = tk[i];
    float b0 = db[i][0], b1 = db[i][1];
    b0 += __shfl_xor_sync(0xffffffff, b0, 1);
    b0 += __shfl_xor_sync(0xffffffff, b0, 2);
    b1 += __shfl_xor_sync(0xffffffff, b1, 1);
    b1 += __shfl_xor_sync(0xffffffff, b1, 2);
    if (t.fb < 0) continue;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      if (!t.rv[h]) continue;
      const int c = t.c0 + g4 + 8 * h;
      float* wp = a.dW + (t.fb * C + c) * K + 2 * t4;
#pragma unroll
      for (int kn = 0; kn < KN; kn++) {
        float2 old = *reinterpret_cast<float2*>(wp + 8 * kn);
        old.x += acc[i][kn][2 * h];
        old.y += acc[i][kn][2 * h + 1];
        *reinterpret_cast<float2*>(wp + 8 * kn) = old;
      }
      if (t4 == 0) a.dB[t.fb * C + c] += h ? b1 : b0;
    }
  }
}
