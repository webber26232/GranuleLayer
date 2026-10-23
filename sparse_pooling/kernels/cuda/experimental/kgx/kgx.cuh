// EXPERIMENTAL K = 64 kernels of the large-C / index-free / sparse study ("kgx" bundle;
// sparse_pooling/experimental/kgx_*.py, results/K64_LARGE_C_INDEXFREE_L1.md).
//
// Model (the kgen model, K = 64): logical block j of output g holds C neurons (C % 16 == 0, C <= 2048)
// sharing ONE 64-input record; every neuron has its own 64 weights and bias:
//     z = bias + sum_{k < 64} w[k] x[idx[k]],   a = ReLU(z),   y_g = mean over the n_g valid neurons.
// The 64 products accumulate into one FP32 accumulator (4 m16n8k16 tiles) before the single
// activation; backward recomputes the complete z (no sub-neurons).
//
// Variants (template flags, host names in experimental/kgx_layer.py):
//   IDF = false  "direct": the record of every logical block is gathered through its 64 indices and
//                the supergroup's feature map (as kgen), for any S in [64, 256];
//   IDF = true   "index-free" (S = K = 64 only): every record of supergroup q IS the set feat[q] in
//                local order k = 0..63, so no record indices or local lookups are read: the A operand
//                (tokens x 64 set members) is loaded once per warp and token tile and reused for every
//                record of the CTA;
//   ACC (IDF)    dX accumulates dA = dZ W^T for ALL records of the warp in registers (they share the 64
//                input rows) and stores it once into the warp's private slab copy (the copies are summed
//                in a fixed order at the end); without ACC the per-record slab read-modify-write of kgen
//                is kept (attribution ablation "idf_rmw").  (Shared FP32 atomicAdd is a CAS loop on
//                sm_80 / sm_86, so no kernel here uses it.)
//   TILE (IDF)   tile-sparse weights: a 16-neuron group keeps a 4-bit mask of its active 16-input
//                k16 tiles and the compact index of its first active tile; inactive tiles are never
//                loaded, multiplied or written (fwd z, dX z / dA, dW z^T / dW^T); db and the activation
//                of every valid neuron (also all-zero groups: z = bias) are kept.
//
// Large C: work items carry a neuron range [c0, c1) of their block(s) (physical neuron tile), so one
// logical C = 2048 record is processed by several warps / CTAs; the logical graph is unchanged.
//
// Storage (canonical): idx [G * NB, 64] int16; W dense [G * NB, C, 64] bf16 or tile-compact
// [n_tiles, 16 neurons, 16 inputs] bf16; bias [G * NB, C]; dW / dB FP32 (main_grad, same order).
// Work lists (experimental/kgx_graph.py):
//     items  [n, 4]  (g, first flat block, blocks, c0 | c1 << 16)  every valid neuron in exactly one item
//     chunks [m, 3]  (q, lo, hi)  a CTA's item range (fwd, dX) / task range (dW); one supergroup each
//     tasks  [n, 2]  (flat block, 16-neuron group)  dW ownership units, supergroup order
//     tmeta  [G * NB * C / 16]  (first compact tile << 4) | k16-tile mask   (TILE)
// Numerics: BF16 operands, FP32 accumulation, dZ as one BF16 or a hi + lo BF16 pair (SPLIT) in the dX / dW
// products, FP32 atomics across CTAs (fast reduction only; no deterministic mode).
#pragma once
#include "../../sparse_pool_sg.cuh"

struct KxArgs {
  const bf16* X;  // feature-major X^T: element (b, f) at X[f * sxf + b]
  long long sxf;
  const int16_t* idx;   // direct only
  const bf16* W;
  const bf16* bias;
  const int* counts;    // [G]
  const int16_t* feat;  // [Q, S]
  const int* items;     // [n_items, 4]
  const int* chunks;    // [n_chunks, 3]
  const int* tasks;     // [n_tasks, 2]
  const int* tmeta;     // TILE
  const int* rowptr;    // CSR: [G * NB * C + 1] entry offsets per neuron slot
  const unsigned char* kix;  // CSR: [nnz] local input (0..63) of every entry
  const int* wins;      // CSR dX: [n_win, 4] (g, first slot, neurons | entries << 8, first entry) static windows
  float* ysum;          // [G, B]
  const float* c1T;     // [G, B] Mean coefficients dL/dy / n_g
  float* dXT;           // [D, B]
  float* dW;
  float* dB;
  int B, G, D, S, NB, C;
  int dx_group_warps;
  int flags;
};

constexpr int KX_K = 64;
constexpr int KX_KT = 4;  // k16 tiles
constexpr int KX_KN = 8;  // k8 tiles

// compile-time tilings (reported by sp_kx_config)
struct KxCfg {
  static constexpr int FWD_MT = 2, FWD_MT_IDF = 4, FWD_NW = 8;
  static constexpr int DX_MT = 2, DX_NW = 8;
  static constexpr int DW_NW = 8, DW_TS = 2, DW_ST = 3;
  static constexpr int CS_TPL = 4, CS_NW = 8, CSX_NW = 8;  // CSR: fwd tokens per lane, warps; dX warps (1 token per lane)
  static constexpr int CSW_BT = 256, CSW_ST = 2;  // CSR dW: tokens per stage, stages
};

template <int MT>
__device__ __forceinline__ void kx_stage(const KxArgs& a, const int16_t* fq, int16_t* smap, bf16* slab, int t0,
                                         bool map) {
  constexpr int BM = 16 * MT, CH = BM / 8;
  const int nthr = blockDim.x;
  if (map)
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

template <int MT>
struct KxA {
  uint32_t r[MT][KX_KT][4];
};

// A fragments (rows = tokens, cols = the 64 record inputs) for all MT m16 tiles.  Lane L supplies the
// slab row of input 16 kt + 8 ((L >> 4) & 1) + (L & 7) at token chunk 2 mt + ((L >> 3) & 1); the slab
// row is smap[rec[k]] (direct) or k itself (index-free: the record is the set in local order).
template <int MT, bool IDF>
__device__ __forceinline__ void kx_gather(KxA<MT>& af, const int16_t* rec, const int16_t* smap, uint32_t slab_s,
                                          int lane) {
  const int kk = ((lane >> 4) & 1) * 8 + (lane & 7), co = (lane >> 3) & 1;
#pragma unroll
  for (int kt = 0; kt < KX_KT; kt++) {
    const int loc = IDF ? 16 * kt + kk : smap[rec[16 * kt + kk]];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) ldmatrix_x4_trans(af.r[mt][kt], slab_s + 2 * Swz<MT>::off(loc, 2 * mt + co));
  }
}

__device__ __forceinline__ uint32_t kx_ld2(const bf16* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }

// weight-panel addressing of one 8-neuron n-tile (neuron row c = 8 nt + lane / 4):
// dense: W[(fb C + c) 64 + k]; tile: the compact tile of k16 tile kt (rank among the group's active
// tiles), element (c % 16, k % 16).
struct KxPanel {
  const bf16* base;  // dense: neuron row of the lane; tile: first active tile of the 16-neuron group
  int mask;          // active k16 tiles (dense: 0xF)
  int nloc;          // tile: neuron row inside the 16-neuron group
};

template <bool TILE>
__device__ __forceinline__ KxPanel kx_panel(const KxArgs& a, long long fb, int nt, int lane) {
  KxPanel p;
  const int c = nt * 8 + (lane >> 2);
  if constexpr (TILE) {
    const int m = __ldg(a.tmeta + (fb * a.C + c) / 16);
    p.mask = m & 15;
    p.base = a.W + (long long)(m >> 4) * 256;
    p.nloc = c & 15;
  } else {
    p.mask = 0xF;
    p.base = a.W + (fb * a.C + c) * KX_K;
    p.nloc = 0;
  }
  return p;
}

// wf[k8] = W[c][8 k8 + 2 (lane % 4) .. + 1] as a BF16 pair (0 for inactive tiles)
template <bool TILE>
__device__ __forceinline__ void kx_wload(uint32_t* wf, const KxPanel& p, int t4, bool on) {
#pragma unroll
  for (int kt = 0; kt < KX_KT; kt++) {
    const bool act = on && ((p.mask >> kt) & 1);
    if constexpr (TILE) {
      const int rk = __popc(p.mask & ((1 << kt) - 1));
      const bf16* tp = p.base + rk * 256 + p.nloc * 16 + 2 * t4;
      wf[2 * kt] = act ? kx_ld2(tp) : 0u;
      wf[2 * kt + 1] = act ? kx_ld2(tp + 8) : 0u;
    } else {
      wf[2 * kt] = act ? kx_ld2(p.base + 16 * kt + 2 * t4) : 0u;
      wf[2 * kt + 1] = act ? kx_ld2(p.base + 16 * kt + 8 + 2 * t4) : 0u;
    }
  }
}

template <int MT>
__device__ __forceinline__ void kx_z(float* d, const KxA<MT>& af, int mt, const uint32_t* wf, int mask, float bx,
                                     float by) {
  d[0] = bx; d[1] = by; d[2] = bx; d[3] = by;
#pragma unroll
  for (int kt = 0; kt < KX_KT; kt++)
    if ((mask >> kt) & 1) mma_m16n8k16_bf16(d, af.r[mt][kt], wf[2 * kt], wf[2 * kt + 1]);
}

__device__ __forceinline__ void kx_item(const KxArgs& a, int it, int& g, int& fb0, int& nb, int& c0, int& c1) {
  const int4 v = __ldg(reinterpret_cast<const int4*>(a.items) + it);
  g = v.x; fb0 = v.y; nb = v.z; c0 = v.w & 0xFFFF; c1 = v.w >> 16;
}

// ============================================================================
// forward: Mean(ReLU)
// ============================================================================
template <int MT, int NW, bool IDF, bool TILE>
__global__ void __launch_bounds__(NW * 32) kx_fwd_kernel(KxArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * 16 * MT;
  kx_stage<MT>(a, a.feat + (long long)q * a.S, smap, slab, t0, !IDF);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  const int C = a.C;
  KxA<MT> af;
  if constexpr (IDF) kx_gather<MT, true>(af, nullptr, nullptr, slab_s, lane);  // once: every record is the set
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
      const int cend = min(c1, n_g - (j0 + bi) * C);  // valid neurons of the block inside the item's range
      if constexpr (!IDF) kx_gather<MT, false>(af, a.idx + fb * KX_K, smap, slab_s, lane);
      const bf16* bb = a.bias + fb * C;
      for (int nt = c0 / 8; nt * 8 < cend; nt++) {
        const KxPanel p = kx_panel<TILE>(a, fb, nt, lane);
        if (TILE && p.mask == 0) continue;  // all-zero group: bias-only neurons (handled outside, exactly)
        uint32_t wf[KX_KN];
        kx_wload<TILE>(wf, p, t4, true);
        const int cc = nt * 8 + 2 * t4;
        const float bx = __bfloat162float(bb[cc]), by = __bfloat162float(bb[cc + 1]);
        const bool v0 = cc < cend, v1 = cc + 1 < cend;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          float d[4];
          kx_z<MT>(d, af, mt, wf, p.mask, bx, by);
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
// dX
// ============================================================================
template <int MT, int NW, bool IDF, bool ACC, bool TILE, bool SPLIT>
__global__ void __launch_bounds__(NW * 32) kx_dx_kernel(KxArgs a) {
  static_assert(IDF || !ACC, "register accumulation over records needs the index-free layout");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int BM = 16 * MT;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  float* dxs = reinterpret_cast<float*>(smem_raw + sg_align16(a.D * 2) + a.S * BM * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * BM;
  const int WG = ACC ? 1 : a.dx_group_warps, ncopy = NW / WG;
  const bool priv = WG == 1;
  const int16_t* fq = a.feat + (long long)q * a.S;
  if constexpr (!ACC)  // ACC: every warp stores its whole copy exactly once
    for (int i = threadIdx.x; i < ncopy * a.S * BM; i += NW * 32) dxs[i] = 0.f;
  kx_stage<MT>(a, fq, smap, slab, t0, !IDF);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  float* mys = dxs + (warp / WG) * a.S * BM;
  const int C = a.C;
  KxA<MT> af;
  int f0[KX_KN], f1[KX_KN];  // local slab rows of this lane's two dA columns per k8 tile
  if constexpr (IDF) {
    kx_gather<MT, true>(af, nullptr, nullptr, slab_s, lane);
#pragma unroll
    for (int kn = 0; kn < KX_KN; kn++) {
      f0[kn] = 8 * kn + 2 * t4;
      f1[kn] = 8 * kn + 2 * t4 + 1;
    }
  }
  float da[MT][KX_KN][4];
  auto zero_da = [&]() {
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int kn = 0; kn < KX_KN; kn++) da[mt][kn][0] = da[mt][kn][1] = da[mt][kn][2] = da[mt][kn][3] = 0.f;
  };
  // add (store: ACC) the accumulated dA into the warp's slab copy (record inputs are distinct: no
  // intra-warp races)
  auto flush_da = [&](bool store) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int kn = 0; kn < KX_KN; kn++) {
        const float* d = da[mt][kn];
        float* p0 = mys + sg_dxs_off<MT>(f0[kn], 16 * mt + g4);
        float* p1 = mys + sg_dxs_off<MT>(f1[kn], 16 * mt + g4);
        if (store) {
          *reinterpret_cast<float2*>(p0) = make_float2(d[0], d[2]);
          *reinterpret_cast<float2*>(p1) = make_float2(d[1], d[3]);
        } else if (priv) {
          float2 v0 = *reinterpret_cast<float2*>(p0), v1 = *reinterpret_cast<float2*>(p1);
          v0.x += d[0]; v0.y += d[2]; v1.x += d[1]; v1.y += d[3];
          *reinterpret_cast<float2*>(p0) = v0;
          *reinterpret_cast<float2*>(p1) = v1;
        } else {
          atomic_add_f32x2(p0, d[0], d[2]);
          atomic_add_f32x2(p1, d[1], d[3]);
        }
      }
  };
  zero_da();
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
      if constexpr (!IDF) {
        const int16_t* rec = a.idx + fb * KX_K;
        kx_gather<MT, false>(af, rec, smap, slab_s, lane);
#pragma unroll
        for (int kn = 0; kn < KX_KN; kn++) {
          f0[kn] = smap[rec[8 * kn + 2 * t4]];
          f1[kn] = smap[rec[8 * kn + 2 * t4 + 1]];
        }
      }
      if constexpr (!ACC) zero_da();
      const bf16* bb = a.bias + fb * C;
      for (int c16 = c0 / 16; c16 * 16 < cend; c16++) {
        const bool two = c16 * 16 + 8 < cend;  // second n-tile holds valid neurons
        uint32_t wf[2][KX_KN], wt[2][KX_KN];
        float bx[2], by[2];
        int mask = 0;
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int nt = 2 * c16 + h;
          const bool on = h == 0 || two;
          const KxPanel p = kx_panel<TILE>(a, fb, nt, lane);
          mask = p.mask;  // one mask per 16-neuron group (both n-tiles)
          kx_wload<TILE>(wf[h], p, t4, on);
#pragma unroll
          for (int kn = 0; kn < KX_KN; kn++) wt[h][kn] = movmatrix_trans(wf[h][kn]);  // B[k = neuron][n = input]
          const int cc = nt * 8 + 2 * t4;
          bx[h] = on ? __bfloat162float(bb[cc]) : 0.f;
          by[h] = on ? __bfloat162float(bb[cc + 1]) : 0.f;
        }
        if (TILE && mask == 0) continue;  // all-zero group: no dA term (its db: bias-only path)
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
            kx_z<MT>(d, af, mt, wf[h], mask, bx[h], by[h]);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              const int r = i >> 1;
              float v = dz_f<P_MEAN, A_RELU, 0>(d[i], c1v[mt][r], 0.f, 0.f, 1.f, 0.f);
              if (cc + (i & 1) >= cend) v = 0.f;
              dz[h][i] = tv[mt][r] ? v : 0.f;  // select (not multiply): padded tokens may give inf
            }
          }
          uint32_t ah[4], al[4];
          split_bf16(dz[0][0], dz[0][1], ah[0], al[0]);
          split_bf16(dz[0][2], dz[0][3], ah[1], al[1]);
          split_bf16(dz[1][0], dz[1][1], ah[2], al[2]);
          split_bf16(dz[1][2], dz[1][3], ah[3], al[3]);
#pragma unroll
          for (int kn = 0; kn < KX_KN; kn++) {
            if (!((mask >> (kn >> 1)) & 1)) continue;
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
      if constexpr (!ACC) flush_da(false);
    }
  }
  if constexpr (ACC) flush_da(true);  // once per warp (all records of the CTA share the 64 inputs)
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
struct KxTask {
  uint32_t aw[KX_KT][4];  // W A fragments (rows = 16 neurons, cols = 64 inputs)
  float bz[2];            // bias of rows lane / 4, lane / 4 + 8
  bool rv[2];             // row valid
  int lt[KX_KT];          // z^T B operand (ldmatrix.trans): slab row per k16 tile
  int ln[KX_KT];          // dW B operand (ldmatrix): slab row per pair of k8 tiles
  long long fb;           // flat block (-1: no task)
  int c0, g, mask;
  long long wbase;        // TILE: first compact tile index of the group
};

template <int NW, int TS, int ST, bool IDF, bool TILE, bool SPLIT>
__global__ void __launch_bounds__(NW * 32) kx_dw_kernel(KxArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* ring = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g4 = lane >> 2, t4 = lane & 3;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int S = a.S, C = a.C;
  if constexpr (!IDF)
    for (int i = threadIdx.x; i < S; i += NW * 32) smap[fq[i]] = (int16_t)i;
  __syncthreads();

  KxTask t;
  float acc[KX_KN][4], db[2];
  {
    const int pos = lo + warp;
    t.fb = pos < hi ? a.tasks[2 * pos] : -1;
    t.c0 = pos < hi ? 16 * a.tasks[2 * pos + 1] : 0;
    t.g = t.fb >= 0 ? (int)(t.fb / a.NB) : 0;
    const int n_g = t.fb >= 0 ? a.counts[t.g] : 0;
    const int j = t.fb >= 0 ? (int)(t.fb - (long long)t.g * a.NB) : 0;
#pragma unroll
    for (int kn = 0; kn < KX_KN; kn++) acc[kn][0] = acc[kn][1] = acc[kn][2] = acc[kn][3] = 0.f;
    db[0] = db[1] = 0.f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int c = t.c0 + g4 + 8 * h;
      t.rv[h] = t.fb >= 0 && c < C && j * C + c < n_g;
      t.bz[h] = t.rv[h] ? __bfloat162float(a.bias[t.fb * C + c]) : 0.f;
    }
    const long long fbs = t.fb >= 0 ? t.fb : 0;
    t.mask = 0xF;
    t.wbase = 0;
    if constexpr (TILE) {
      const int m = t.fb >= 0 ? a.tmeta[(fbs * C + t.c0) / 16] : 0;
      t.mask = m & 15;
      t.wbase = m >> 4;
    }
    const int16_t* rec = IDF ? nullptr : a.idx + fbs * KX_K;
#pragma unroll
    for (int kt = 0; kt < KX_KT; kt++) {
      const bool act = (t.mask >> kt) & 1;
      if constexpr (TILE) {
        const bf16* tp = a.W + (t.wbase + __popc(t.mask & ((1 << kt) - 1))) * 256 + g4 * 16 + 2 * t4;
        t.aw[kt][0] = act && t.rv[0] ? kx_ld2(tp) : 0u;
        t.aw[kt][1] = act && t.rv[1] ? kx_ld2(tp + 8 * 16) : 0u;
        t.aw[kt][2] = act && t.rv[0] ? kx_ld2(tp + 8) : 0u;
        t.aw[kt][3] = act && t.rv[1] ? kx_ld2(tp + 8 * 16 + 8) : 0u;
      } else {
        const bf16* w0 = a.W + (fbs * C + t.c0 + g4) * KX_K;  // row c
        const bf16* w1 = w0 + 8 * KX_K;                         // row c + 8
        t.aw[kt][0] = t.rv[0] ? kx_ld2(w0 + 16 * kt + 2 * t4) : 0u;
        t.aw[kt][1] = t.rv[1] ? kx_ld2(w1 + 16 * kt + 2 * t4) : 0u;
        t.aw[kt][2] = t.rv[0] ? kx_ld2(w0 + 16 * kt + 8 + 2 * t4) : 0u;
        t.aw[kt][3] = t.rv[1] ? kx_ld2(w1 + 16 * kt + 8 + 2 * t4) : 0u;
      }
      const int m = lane >> 3, r = lane & 7;
      const int kz = 16 * kt + (m & 1) * 8 + r, kw = 8 * (2 * kt + (m >> 1)) + r;
      t.lt[kt] = IDF ? kz : smap[rec[kz]];  // matrices (k lo/hi) x (token chunk m >> 1)
      t.ln[kt] = IDF ? kw : smap[rec[kw]];  // matrices (k8 tile 2 kt + m >> 1) x (chunk m & 1)
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
      if (t.fb < 0) continue;
      const uint32_t slab_s = smem_u32(base + sub * S * 16);
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
      uint32_t bn[KX_KN][2];  // dW B operand: X[token][input] per k8 tile, token chunks 0 / 1
#pragma unroll
      for (int kt = 0; kt < KX_KT; kt++) {
        if (!((t.mask >> kt) & 1)) {
          bn[2 * kt][0] = bn[2 * kt][1] = bn[2 * kt + 1][0] = bn[2 * kt + 1][1] = 0u;
          continue;
        }
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
      float dz[2][4];
#pragma unroll
      for (int n = 0; n < 2; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = e >> 1, col = e & 1;
          float v = dz_f<P_MEAN, A_RELU, 0>(zt[n][e], cc[n][col], 0.f, 0.f, 1.f, 0.f);
          if (!t.rv[r]) v = 0.f;
          dz[n][e] = tv[n][col] ? v : 0.f;
        }
      db[0] += (dz[0][0] + dz[0][1]) + (dz[1][0] + dz[1][1]);
      db[1] += (dz[0][2] + dz[0][3]) + (dz[1][2] + dz[1][3]);
      uint32_t ah[4], al[4];  // A = dZ^T [16 neurons x 16 tokens]
      split_bf16(dz[0][0], dz[0][1], ah[0], al[0]);
      split_bf16(dz[0][2], dz[0][3], ah[1], al[1]);
      split_bf16(dz[1][0], dz[1][1], ah[2], al[2]);
      split_bf16(dz[1][2], dz[1][3], ah[3], al[3]);
#pragma unroll
      for (int kn = 0; kn < KX_KN; kn++) {
        if (!((t.mask >> (kn >> 1)) & 1)) continue;
        mma_m16n8k16_bf16(acc[kn], ah, bn[kn][0], bn[kn][1]);
        if (SPLIT) mma_m16n8k16_bf16(acc[kn], al, bn[kn][0], bn[kn][1]);
      }
    }
  }
  cp_async_wait_all();
  // write back (exclusive ownership): rows lane / 4 (acc[0..1]) and lane / 4 + 8 (acc[2..3])
  float b0 = db[0], b1 = db[1];
  b0 += __shfl_xor_sync(0xffffffff, b0, 1);
  b0 += __shfl_xor_sync(0xffffffff, b0, 2);
  b1 += __shfl_xor_sync(0xffffffff, b1, 1);
  b1 += __shfl_xor_sync(0xffffffff, b1, 2);
  if (t.fb < 0) return;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    if (!t.rv[h]) continue;
    const int c = t.c0 + g4 + 8 * h;
#pragma unroll
    for (int kn = 0; kn < KX_KN; kn++) {
      if (!((t.mask >> (kn >> 1)) & 1)) continue;
      float* wp;
      if constexpr (TILE)
        wp = a.dW + (t.wbase + __popc(t.mask & ((1 << (kn >> 1)) - 1))) * 256 + (g4 + 8 * h) * 16 + 8 * (kn & 1) +
             2 * t4;
      else
        wp = a.dW + (t.fb * C + c) * KX_K + 8 * kn + 2 * t4;
      float2 old = *reinterpret_cast<float2*>(wp);
      old.x += acc[kn][2 * h];
      old.y += acc[kn][2 * h + 1];
      *reinterpret_cast<float2*>(wp) = old;
    }
    if (t4 == 0) a.dB[t.fb * C + c] += h ? b1 : b0;
  }
}
