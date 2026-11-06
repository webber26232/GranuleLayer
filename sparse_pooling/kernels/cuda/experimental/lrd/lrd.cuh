// EXPERIMENTAL learned sliding-window reduction kernels ("lrd" bundle; sparse_pooling/experimental/learned_reduction.py,
// docs/experiments/learned_sliding_window_reduction.md).
//
// Model: z = bias + sum_k w[k] x[idx[k]] (K = 16, C neurons share a record), h = ReLU(z), and instead of the kgen
// shared per-output Mean coefficient every neuron carries its own coefficients P[neuron, f], f < F:
//     U[w0 + f, token] += sum_{neurons of the block} P[neuron, f] h[token, neuron]
//     dh[token, neuron] = sum_{f} Delta[w0 + f, token] P[neuron, f]            (Delta = dLoss / dU)
//     dP[neuron, f]     = sum_{tokens} Delta[w0 + f, token] h[token, neuron]
// Every kernel recomputes z with the kgen input-supergroup machinery (kg_stage, kg_gather, kg_z); no M x H tensor
// exists.  Hidden tiles stay in registers: their 16 tokens x 16 neurons activation fragments are exactly the A operand
// of the dX product, so no staging / rearrangement through shared memory is needed.
//
// Storage: P [G * NB * C, F] bf16 (neuron-major, zero at non-member slots); dP the same shape in fp32 (main_grad);
//          U [J, B] fp32; Dl [J + 2 PAD, B] fp32: Delta with PAD zero rows on both sides so a block's slots that fall
//          outside [0, J) read zero without a branch.
// Work lists: items [n, 4] (g, first flat block, blocks, w0), chunks [m, 3] as kgen, tasks [n, 3] (flat block, 16-neuron
//          group, w0).  Every block of an item shares w0 (host-built; experimental/learned_reduction.ReductionExec).
//
// Forward epilogues (EPI):
//   LR_MMA_BF16   h rounded to one BF16; U += h @ P^T with one m16n8k16 per (16 tokens, 16 neurons): a numerical change
//   LR_MMA_SPLIT  h as a BF16 hi + lo pair, two m16n8k16: within about 2^-16 of the FP32 hidden value
//   LR_SIMT<FX>   FP32 h, FP32 per-lane multiply-adds, shuffles to reduce over the warp: the FP32-hidden reference path
// Backward: Delta enters as a BF16 hi + lo pair when SPLIT (one BF16 otherwise); P and W operands are BF16 (exact).
// dZ uses the production generic expression dz_f.  Cross-CTA sums use FP32 atomics: results are not bitwise deterministic.
#pragma once
#include "../kgen/kgen.cuh"

struct LrArgs {
  KgArgs k;           // items / tasks use the stride-4 / stride-3 tables above; k.ysum, k.c1T unused
  const bf16* P;
  float* U;
  const float* Dl;
  float* dP;
  int F, J, PAD, pad0;
};

enum { LR_MMA_BF16 = 0, LR_MMA_SPLIT = 1, LR_SIMT = 2 };
enum { LR_DO_DW = 1, LR_DO_DP = 2 };

// two consecutive coefficients (P[n, f0], P[n, f0 + 1]) as one BF16 pair; zero past F (no divergence: clamped loads)
__device__ __forceinline__ uint32_t lr_ld_p(const bf16* p, long long n, int f0, int F) {
  const unsigned short* q = reinterpret_cast<const unsigned short*>(p) + n * F;
  const unsigned int lo = __ldg(q + (f0 < F ? f0 : F - 1));
  const unsigned int hi = __ldg(q + (f0 + 1 < F ? f0 + 1 : F - 1));
  return (f0 < F ? lo : 0u) | ((f0 + 1 < F ? hi : 0u) << 16);
}

// ============================================================================
// forward
// ============================================================================
template <int K, int MT, int NW, int EPI, int FX>
__global__ void __launch_bounds__(NW * 32) lr_fwd_kernel(LrArgs ar) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const KgArgs& a = ar.k;
  constexpr int KN = K / 8;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * 16 * MT;
  kg_stage<MT>(a, a.feat + (long long)q * a.S, smap, slab, t0);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  const int C = a.C, F = ar.F, J = ar.J;
  for (int it = lo + warp; it < hi; it += NW) {
    const int g = a.items[4 * it], fb0 = a.items[4 * it + 1], nb = a.items[4 * it + 2], w0 = a.items[4 * it + 3];
    const int n_g = a.counts[g];
    const int j0 = fb0 - g * a.NB;
    float u[MT][4];    // MMA: accumulator layout (rows g4, g4 + 8; cols 2 t4, 2 t4 + 1)
    float us[MT][2][EPI == LR_SIMT ? FX : 1];  // SIMT: per-lane partial sums (token rows x slots), reduced over t4
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      u[mt][0] = u[mt][1] = u[mt][2] = u[mt][3] = 0.f;
      if constexpr (EPI == LR_SIMT) {
#pragma unroll
        for (int f = 0; f < FX; f++) us[mt][0][f] = us[mt][1][f] = 0.f;
      }
    }
    for (int bi = 0; bi < nb; bi++) {
      const long long fb = fb0 + bi;
      const int nval = min(C, n_g - (j0 + bi) * C);
      KgA<K, MT> af;
      kg_gather<K, MT>(af, a.idx + fb * K, smap, slab_s, lane);
      const bf16* wb = a.W + fb * C * K;
      const bf16* bb = a.bias + fb * C;
      const bf16* pb = ar.P + fb * C * F;
      for (int c16 = 0; c16 * 16 < nval; c16++) {
        const bool two = c16 * 16 + 8 < nval;
        uint32_t wf[2][KN], pt[2];
        float bx[2], by[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int nt = 2 * c16 + h;
          const bool on = h == 0 || two;
#pragma unroll
          for (int kn = 0; kn < KN; kn++) wf[h][kn] = on ? ld_w2(wb + (nt * 8 + g4) * K + 8 * kn + 2 * t4) : 0u;
          const int cc = nt * 8 + 2 * t4;
          bx[h] = on ? __bfloat162float(bb[cc]) : 0.f;
          by[h] = on ? __bfloat162float(bb[cc + 1]) : 0.f;
          if constexpr (EPI != LR_SIMT) pt[h] = on ? movmatrix_trans(lr_ld_p(pb, nt * 8 + g4, 2 * t4, F)) : 0u;
        }
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          float hv[2][4];
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int cc = (2 * c16 + h) * 8 + 2 * t4;
            if (h == 1 && !two) {
              hv[1][0] = hv[1][1] = hv[1][2] = hv[1][3] = 0.f;
              continue;
            }
            float d[4];
            kg_z<K, MT>(d, af, mt, wf[h], bx[h], by[h]);
            const bool v0 = cc < nval, v1 = cc + 1 < nval;
            hv[h][0] = v0 ? fmaxf(d[0], 0.f) : 0.f;
            hv[h][1] = v1 ? fmaxf(d[1], 0.f) : 0.f;
            hv[h][2] = v0 ? fmaxf(d[2], 0.f) : 0.f;
            hv[h][3] = v1 ? fmaxf(d[3], 0.f) : 0.f;
          }
          if constexpr (EPI == LR_SIMT) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
              if (h == 1 && !two) continue;
              const int cc = (2 * c16 + h) * 8 + 2 * t4;
#pragma unroll
              for (int f = 0; f < FX; f++) {
                const float p0 = __bfloat162float(pb[(long long)cc * F + min(f, F - 1)]);
                const float p1 = __bfloat162float(pb[(long long)(cc + 1) * F + min(f, F - 1)]);
                const float q0 = f < F ? p0 : 0.f, q1 = f < F ? p1 : 0.f;
                us[mt][0][f] += hv[h][0] * q0 + hv[h][1] * q1;
                us[mt][1][f] += hv[h][2] * q0 + hv[h][3] * q1;
              }
            }
          } else {
            uint32_t ah[4], al[4];
            split_bf16(hv[0][0], hv[0][1], ah[0], al[0]);
            split_bf16(hv[0][2], hv[0][3], ah[1], al[1]);
            split_bf16(hv[1][0], hv[1][1], ah[2], al[2]);
            split_bf16(hv[1][2], hv[1][3], ah[3], al[3]);
            mma_m16n8k16_bf16(u[mt], ah, pt[0], pt[1]);
            if constexpr (EPI == LR_MMA_SPLIT) mma_m16n8k16_bf16(u[mt], al, pt[0], pt[1]);
          }
        }
      }
    }
    // merge the item's bounded partial sums (one FP32 atomic per (token, destination window))
    if constexpr (EPI == LR_SIMT) {
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int f = 0; f < FX; f++) {
            float s = us[mt][r][f];
            s += __shfl_xor_sync(0xffffffff, s, 1);
            s += __shfl_xor_sync(0xffffffff, s, 2);
            const int tok = t0 + 16 * mt + 8 * r + g4, w = w0 + f;
            if (t4 == 0 && f < F && tok < a.B && w >= 0 && w < J) atomicAdd(ar.U + (long long)w * a.B + tok, s);
          }
    } else {
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const int tok = t0 + 16 * mt + 8 * r + g4, f = 2 * t4 + e, w = w0 + f;
            if (f < F && tok < a.B && w >= 0 && w < J) atomicAdd(ar.U + (long long)w * a.B + tok, u[mt][2 * r + e]);
          }
    }
  }
}

// ============================================================================
// dX: dh from Delta and P, then the kgen dA / slab / atomic path
// ============================================================================
template <int K, int MT, int NW, bool SPLIT>
__global__ void __launch_bounds__(NW * 32) lr_dx_kernel(LrArgs ar) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const KgArgs& a = ar.k;
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
  const int C = a.C, F = ar.F;
  for (int it = lo + warp; it < hi; it += NW) {
    const int g = a.items[4 * it], fb0 = a.items[4 * it + 1], nb = a.items[4 * it + 2], w0 = a.items[4 * it + 3];
    const int n_g = a.counts[g];
    const int j0 = fb0 - g * a.NB;
    // Delta fragments of the item: A operand of m16n8k8 (rows = tokens, k = slots), hi + lo per m16 tile
    uint32_t dh_[MT][2], dl_[MT][2];
    bool tv[MT][2];
    const int f0 = 2 * t4;
    const float* d0 = ar.Dl + (long long)(w0 + min(f0, F - 1) + ar.PAD) * a.B;
    const float* d1 = ar.Dl + (long long)(w0 + min(f0 + 1, F - 1) + ar.PAD) * a.B;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int tok = t0 + 16 * mt + 8 * h + g4;
        tv[mt][h] = tok < a.B;
        const int ts = tv[mt][h] ? tok : 0;
        const float v0 = __ldg(d0 + ts), v1 = __ldg(d1 + ts);
        split_bf16(tv[mt][h] && f0 < F ? v0 : 0.f, tv[mt][h] && f0 + 1 < F ? v1 : 0.f, dh_[mt][h], dl_[mt][h]);
      }
    for (int bi = 0; bi < nb; bi++) {
      const long long fb = fb0 + bi;
      const int nval = min(C, n_g - (j0 + bi) * C);
      const int16_t* rec = a.idx + fb * K;
      KgA<K, MT> af;
      kg_gather<K, MT>(af, rec, smap, slab_s, lane);
      int f0s[KN], f1s[KN];
#pragma unroll
      for (int kn = 0; kn < KN; kn++) {
        f0s[kn] = smap[rec[8 * kn + 2 * t4]];
        f1s[kn] = smap[rec[8 * kn + 2 * t4 + 1]];
      }
      float da[MT][KN][4];
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int kn = 0; kn < KN; kn++) da[mt][kn][0] = da[mt][kn][1] = da[mt][kn][2] = da[mt][kn][3] = 0.f;
      const bf16* wb = a.W + fb * C * K;
      const bf16* bb = a.bias + fb * C;
      const bf16* pb = ar.P + fb * C * F;
      for (int c16 = 0; c16 * 16 < nval; c16++) {
        const bool two = c16 * 16 + 8 < nval;
        uint32_t wf[2][KN], wt[2][KN], pf[2];
        float bx[2], by[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int nt = 2 * c16 + h;
          const bool on = h == 0 || two;
#pragma unroll
          for (int kn = 0; kn < KN; kn++) {
            wf[h][kn] = on ? ld_w2(wb + (nt * 8 + g4) * K + 8 * kn + 2 * t4) : 0u;
            wt[h][kn] = movmatrix_trans(wf[h][kn]);
          }
          const int cc = nt * 8 + 2 * t4;
          bx[h] = on ? __bfloat162float(bb[cc]) : 0.f;
          by[h] = on ? __bfloat162float(bb[cc + 1]) : 0.f;
          pf[h] = on ? lr_ld_p(pb, nt * 8 + g4, 2 * t4, F) : 0u;
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
            float d[4], e[4] = {0.f, 0.f, 0.f, 0.f};
            kg_z<K, MT>(d, af, mt, wf[h], bx[h], by[h]);
            mma_m16n8k8_bf16(e, dh_[mt][0], dh_[mt][1], pf[h], e[0], e[1], e[2], e[3]);
            if (SPLIT) mma_m16n8k8_bf16(e, dl_[mt][0], dl_[mt][1], pf[h], e[0], e[1], e[2], e[3]);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              float v = dz_f<P_MEAN, A_RELU, 0>(d[i], e[i], 0.f, 0.f, 1.f, 0.f);
              if (cc + (i & 1) >= nval) v = 0.f;
              dz[h][i] = tv[mt][i >> 1] ? v : 0.f;
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
            mma_m16n8k16_bf16(o, ah, wt[0][kn], wt[1][kn]);
            if (SPLIT) mma_m16n8k16_bf16(o, al, wt[0][kn], wt[1][kn]);
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int kn = 0; kn < KN; kn++) {
          const float* d = da[mt][kn];
          float* p0 = mys + sg_dxs_off<MT>(f0s[kn], 16 * mt + g4);
          float* p1 = mys + sg_dxs_off<MT>(f1s[kn], 16 * mt + g4);
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
// dW / db and / or dP: MODE bits LR_DO_DW, LR_DO_DP (fused or separately scheduled)
// ============================================================================
template <int K>
struct LrTask {
  KgTask<K> t;
  uint32_t ap[2];  // P A operand of dh^T: rows lane / 4 (, + 8), k = 2 (lane % 4), + 1
  int w0;
};

template <int K, int WPT, int NW, int TS, int ST, bool SPLIT, int MODE>
__global__ void __launch_bounds__(NW * 32) lr_dw_kernel(LrArgs ar) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const KgArgs& a = ar.k;
  constexpr int KT = KgTask<K>::KT, KN = K / 8;
  constexpr bool DO_W = MODE & LR_DO_DW, DO_P = MODE & LR_DO_DP;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* ring = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int S = a.S, C = a.C, F = ar.F;
  const int f0 = 2 * t4;
  for (int i = threadIdx.x; i < S; i += NW * 32) smap[fq[i]] = (int16_t)i;
  __syncthreads();

  LrTask<K> tk[WPT];
  float acc[WPT][KN][4], db[WPT][2], accp[WPT][4];
#pragma unroll
  for (int i = 0; i < WPT; i++) {
    const int pos = lo + warp * WPT + i;
    KgTask<K>& t = tk[i].t;
    t.fb = pos < hi ? a.tasks[3 * pos] : -1;
    t.c0 = pos < hi ? 16 * a.tasks[3 * pos + 1] : 0;
    tk[i].w0 = pos < hi ? a.tasks[3 * pos + 2] : 0;
    t.g = t.fb >= 0 ? (int)(t.fb / a.NB) : 0;
    const int n_g = t.fb >= 0 ? a.counts[t.g] : 0;
    const int j = t.fb >= 0 ? (int)(t.fb - (long long)t.g * a.NB) : 0;
#pragma unroll
    for (int kn = 0; kn < KN; kn++) acc[i][kn][0] = acc[i][kn][1] = acc[i][kn][2] = acc[i][kn][3] = 0.f;
    db[i][0] = db[i][1] = 0.f;
    accp[i][0] = accp[i][1] = accp[i][2] = accp[i][3] = 0.f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int c = t.c0 + g4 + 8 * h;
      t.rv[h] = t.fb >= 0 && c < C && j * C + c < n_g;
      t.bz[h] = t.rv[h] ? __bfloat162float(a.bias[t.fb * C + c]) : 0.f;
      const long long fbs_ = t.fb >= 0 ? t.fb : 0;
      tk[i].ap[h] = t.rv[h] ? lr_ld_p(ar.P + fbs_ * C * F, c, f0, F) : 0u;
    }
    const long long fbs = t.fb >= 0 ? t.fb : 0;
    const bf16* w0 = a.W + (fbs * C + t.c0 + g4) * K;
    const bf16* w1 = w0 + 8 * K;
    const int16_t* rec = a.idx + fbs * K;
#pragma unroll
    for (int kt = 0; kt < KT; kt++) {
      t.aw[kt][0] = t.rv[0] ? ld_w2(w0 + 16 * kt + 2 * t4) : 0u;
      t.aw[kt][1] = t.rv[1] ? ld_w2(w1 + 16 * kt + 2 * t4) : 0u;
      t.aw[kt][2] = t.rv[0] ? ld_w2(w0 + 16 * kt + 8 + 2 * t4) : 0u;
      t.aw[kt][3] = t.rv[1] ? ld_w2(w1 + 16 * kt + 8 + 2 * t4) : 0u;
      const int m = lane >> 3, r = lane & 7;
      t.lt[kt] = smap[rec[16 * kt + (m & 1) * 8 + r]];
      t.ln[kt] = smap[rec[8 * (2 * kt + (m >> 1)) + r]];
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
        const KgTask<K>& t = tk[i].t;
        if (t.fb < 0) continue;
        bool tv[2][2];
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) tv[n][e] = tt * 16 + 8 * n + 2 * t4 + e < a.B;
        // Delta: B operand of m16n8k8 (n = token lane / 4, k = slots), hi + lo per token n-tile
        const float* d0 = ar.Dl + (long long)(tk[i].w0 + min(f0, F - 1) + ar.PAD) * a.B;
        const float* d1 = ar.Dl + (long long)(tk[i].w0 + min(f0 + 1, F - 1) + ar.PAD) * a.B;
        uint32_t bdh[2], bdl[2];
#pragma unroll
        for (int n = 0; n < 2; n++) {
          const int tok = tt * 16 + 8 * n + g4;
          const bool ok = tok < a.B;
          const int ts = ok ? tok : 0;
          const float v0 = __ldg(d0 + ts), v1 = __ldg(d1 + ts);
          split_bf16(ok && f0 < F ? v0 : 0.f, ok && f0 + 1 < F ? v1 : 0.f, bdh[n], bdl[n]);
        }
        float zt[2][4];
#pragma unroll
        for (int n = 0; n < 2; n++) {
          zt[n][0] = zt[n][1] = t.bz[0];
          zt[n][2] = zt[n][3] = t.bz[1];
        }
        uint32_t bn[KN][2];
#pragma unroll
        for (int kt = 0; kt < KT; kt++) {
          uint32_t bx[4];
          ldmatrix_x4_trans(bx, slab_s + 2 * Swz<1>::off(t.lt[kt], lane >> 4));
          mma_m16n8k16_bf16(zt[0], t.aw[kt], bx[0], bx[1]);
          mma_m16n8k16_bf16(zt[1], t.aw[kt], bx[2], bx[3]);
          if constexpr (DO_W) {
            uint32_t b4[4];
            ldmatrix_x4(b4, slab_s + 2 * Swz<1>::off(t.ln[kt], (lane >> 3) & 1));
            bn[2 * kt][0] = b4[0];
            bn[2 * kt][1] = b4[1];
            bn[2 * kt + 1][0] = b4[2];
            bn[2 * kt + 1][1] = b4[3];
          }
        }
        if constexpr (DO_W) {
          float dz[2][4];
#pragma unroll
          for (int n = 0; n < 2; n++) {
            float e[4] = {0.f, 0.f, 0.f, 0.f};
            mma_m16n8k8_bf16(e, tk[i].ap[0], tk[i].ap[1], bdh[n], e[0], e[1], e[2], e[3]);
            if (SPLIT) mma_m16n8k8_bf16(e, tk[i].ap[0], tk[i].ap[1], bdl[n], e[0], e[1], e[2], e[3]);
#pragma unroll
            for (int x = 0; x < 4; x++) {
              const int r = x >> 1, col = x & 1;
              float v = dz_f<P_MEAN, A_RELU, 0>(zt[n][x], e[x], 0.f, 0.f, 1.f, 0.f);
              if (!t.rv[r]) v = 0.f;
              dz[n][x] = tv[n][col] ? v : 0.f;
            }
          }
          db[i][0] += (dz[0][0] + dz[0][1]) + (dz[1][0] + dz[1][1]);
          db[i][1] += (dz[0][2] + dz[0][3]) + (dz[1][2] + dz[1][3]);
          uint32_t ah[4], al[4];
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
        if constexpr (DO_P) {
          float hv[2][4];
#pragma unroll
          for (int n = 0; n < 2; n++)
#pragma unroll
            for (int x = 0; x < 4; x++)
              hv[n][x] = (t.rv[x >> 1] && tv[n][x & 1]) ? fmaxf(zt[n][x], 0.f) : 0.f;
          uint32_t hh[4], hl[4];
          split_bf16(hv[0][0], hv[0][1], hh[0], hl[0]);
          split_bf16(hv[0][2], hv[0][3], hh[1], hl[1]);
          split_bf16(hv[1][0], hv[1][1], hh[2], hl[2]);
          split_bf16(hv[1][2], hv[1][3], hh[3], hl[3]);
          const uint32_t b0 = movmatrix_trans(bdh[0]), b1 = movmatrix_trans(bdh[1]);
          mma_m16n8k16_bf16(accp[i], hh, b0, b1);
          if (SPLIT) {
            mma_m16n8k16_bf16(accp[i], hl, b0, b1);
            mma_m16n8k16_bf16(accp[i], hh, movmatrix_trans(bdl[0]), movmatrix_trans(bdl[1]));
          }
        }
      }
    }
  }
  cp_async_wait_all();
#pragma unroll
  for (int i = 0; i < WPT; i++) {
    const KgTask<K>& t = tk[i].t;
    float b0 = db[i][0], b1 = db[i][1];
    if constexpr (DO_W) {
      b0 += __shfl_xor_sync(0xffffffff, b0, 1);
      b0 += __shfl_xor_sync(0xffffffff, b0, 2);
      b1 += __shfl_xor_sync(0xffffffff, b1, 1);
      b1 += __shfl_xor_sync(0xffffffff, b1, 2);
    }
    if (t.fb < 0) continue;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      if (!t.rv[h]) continue;
      const int c = t.c0 + g4 + 8 * h;
      if constexpr (DO_W) {
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
      if constexpr (DO_P) {
        float* pp = ar.dP + (t.fb * C + c) * F;
#pragma unroll
        for (int e = 0; e < 2; e++)
          if (f0 + e < F) pp[f0 + e] += accp[i][2 * h + e];
      }
    }
  }
}
