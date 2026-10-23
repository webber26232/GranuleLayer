// EXPERIMENTAL unstructured fixed-mask kernels (CSR per neuron, CUDA cores) of the K = 64 index-free
// layout ("kgx" bundle; experimental/kgx_sparse.py).  S = K = 64 only: every neuron of supergroup q
// draws its active inputs from the 64 set members, so an entry stores the LOCAL input k (0..63) and
// the value; inactive edges are absent (no load, no multiply, no gradient).
//
//     rowptr [G * NB * C + 1] int32   entries of neuron slot s: [rowptr[s], rowptr[s + 1])
//     vals   [nnz] bf16 (the trainable parameter; main_grad [nnz] FP32)   kix [nnz] uint8
// Padded neuron slots and all-zero neurons have no entries; every valid neuron keeps its bias, its
// activation ReLU(z) (z = bias for an empty row) and its db.
//
// kc_fwd  CTA = (32 TPL-token tile, chunk of one supergroup).  X rows of the set staged as BF16
//         [64][BT]; a warp owns a neuron at a time (lanes = tokens): the entries are loaded
//         lane-parallel, broadcast by shuffle; z in FP32; one atomicAdd per (token, item) into ysum.
// kc_dx   CTA = (32 NW tokens, chunk); warp w owns 32 token columns and runs every neuron of the chunk on
//         them: recomputes z, dz = c1 * (z > 0), adds w dz into its own columns of an FP32 [64][BT] slab
//         (plain read-modify-write, no atomics), then one global atomicAdd per (input, token).
// kc_dw   CTA = chunk of dW tasks (16 neurons per warp, exclusive), all tokens (BT-token stages).
//         Entries of the warp's 16 neurons stay in registers; per stage and neuron: z, dz, db, and per
//         entry the token sum of dz x (lane partials, butterfly reduce), accumulated by the entry's lane.
// Numerics: BF16 x / w, FP32 products and accumulation, dz in FP32 (no BF16 rounding of dz: finer than
// the split pair of the MMA kernels); FP32 atomics in fwd / dX (fast reduction only).
#pragma once
#include "kgx.cuh"

template <int TPL>
__device__ __forceinline__ void kc_xload(float* x, const bf16* row, int lane) {
  static_assert(TPL == 2 || TPL == 4 || TPL == 8, "TPL");
  if constexpr (TPL == 2) {
    const uint32_t v = *reinterpret_cast<const uint32_t*>(row + lane * 2);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
    x[0] = a.x; x[1] = a.y;
  } else if constexpr (TPL == 4) {
    const uint2 v = *reinterpret_cast<const uint2*>(row + lane * 4);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
    x[0] = a.x; x[1] = a.y; x[2] = b.x; x[3] = b.y;
  } else {
    const uint4 v = *reinterpret_cast<const uint4*>(row + lane * 8);
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[i]));
      x[2 * i] = f.x;
      x[2 * i + 1] = f.y;
    }
  }
}

// packed entry: (bf16 bits of w) << 16 | k
__device__ __forceinline__ uint32_t kc_entry(const KxArgs& a, long long e) {
  const uint32_t w = __ldg(reinterpret_cast<const unsigned short*>(a.W) + e);
  return (w << 16) | (uint32_t)__ldg(a.kix + e);
}
__device__ __forceinline__ float kc_w(uint32_t ent) { return __uint_as_float(ent & 0xFFFF0000u); }

// stage the 64 set rows of tokens [t0, t0 + BT) as BF16 [64][BT]
template <int BT>
__device__ __forceinline__ void kc_stage(const KxArgs& a, const int16_t* fq, bf16* xs, int t0) {
  constexpr int CH = BT / 8;
  for (int i = threadIdx.x; i < KX_K * CH; i += blockDim.x) {
    const int f = i / CH, c = i % CH;
    const int tok = t0 + c * 8;
    const int bytes = (tok + 8 <= a.B) ? 16 : (tok < a.B ? 2 * (a.B - tok) : 0);
    const bf16* src = bytes ? a.X + (long long)fq[f] * a.sxf + tok : a.X;
    cp_async16(xs + f * BT + c * 8, src, bytes);
  }
}

// z of one neuron for the lane's TPL tokens (entries [e0, e1), first chunk preloaded in `cur`); 4 entries per
// step into independent partial sums (instruction-level parallelism; the order of the FP32 sum differs from
// a serial one)
template <int TPL, int BT>
__device__ __forceinline__ void kc_z(float* z, const KxArgs& a, const bf16* xs, uint32_t cur, long long e0,
                                     long long e1, int lane) {
  float zp[3][TPL];
#pragma unroll
  for (int u = 0; u < 3; u++)
#pragma unroll
    for (int i = 0; i < TPL; i++) zp[u][i] = 0.f;
  for (long long e = e0; e < e1; e += 32) {
    const int n = (int)min(32LL, e1 - e);
    const uint32_t my = e == e0 ? cur : (lane < n ? kc_entry(a, e + lane) : 0u);
    for (int j = 0; j < n; j += 4) {
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const uint32_t ent = __shfl_sync(0xffffffff, my, (j + u) & 31);
        if (j + u < n) {
          float x[TPL];
          kc_xload<TPL>(x, xs + (ent & 0xFF) * BT, lane);
          const float w = kc_w(ent);
#pragma unroll
          for (int i = 0; i < TPL; i++) {
            if (u == 0) z[i] = fmaf(w, x[i], z[i]);
            else zp[u - 1][i] = fmaf(w, x[i], zp[u - 1][i]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < TPL; i++) z[i] += (zp[0][i] + zp[1][i]) + zp[2][i];
}

// ============================================================================
// neuron windows (fwd): up to 31 consecutive neuron slots whose entries (<= KC_WE) are staged once,
// lane-parallel and coalesced, into the warp's shared buffer as packed (w << 16 | k); per neuron the
// entries are then broadcast reads of shared memory (no global latency on the per-neuron path)
// ============================================================================
constexpr int KC_WE = 256;  // entries per window buffer (a neuron has <= 64)

struct KcWin {
  int nn;          // neurons in the window
  long long e0;    // first entry
  int rp;          // lane l (< 32): rowptr[slot0 + l] - e0 (lane nn: end of the window)
  float b;         // lane l < nn: bias of neuron l
};

__device__ __forceinline__ KcWin kc_window(const KxArgs& a, long long slot0, int avail, int lane) {
  KcWin w;
  const int lim = min(31, avail);
  const long long base = __ldg(a.rowptr + slot0);
  const int r = lane <= lim ? (int)(__ldg(a.rowptr + slot0 + lane) - base) : 0x7fffffff;
  // neurons whose entries end within the buffer: the largest n <= lim with rp[n] <= KC_WE (n >= 1 always)
  const unsigned fit = __ballot_sync(0xffffffff, lane <= lim && r <= KC_WE);
  w.nn = max(1, 31 - __clz(fit));  // highest lane l with rp[l] <= KC_WE (rp[0] = 0)
  w.e0 = base;
  w.rp = r;
  w.b = lane < w.nn ? __bfloat162float(a.bias[slot0 + lane]) : 0.f;
  return w;
}

__device__ __forceinline__ void kc_stage_entries(const KxArgs& a, const KcWin& w, uint32_t* buf, int lane) {
  const int ne = __shfl_sync(0xffffffff, w.rp, w.nn);
  for (int i = lane; i < ne; i += 32) buf[i] = kc_entry(a, w.e0 + i);
  __syncwarp();
}

// ============================================================================
// forward
// ============================================================================
template <int TPL, int NW>
__global__ void __launch_bounds__(NW * 32) kc_fwd_kernel(KxArgs a) {
  constexpr int BT = 32 * TPL;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* xs = reinterpret_cast<bf16*>(smem_raw);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t* ebuf = reinterpret_cast<uint32_t*>(smem_raw + KX_K * BT * 2) + warp * KC_WE;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * BT;
  kc_stage<BT>(a, a.feat + (long long)q * a.S, xs, t0);
  cp_async_wait_all();
  __syncthreads();
  const int C = a.C;
  for (int it = lo + warp; it < hi; it += NW) {
    int g, fb0, nb, c0, c1;
    kx_item(a, it, g, fb0, nb, c0, c1);
    const int n_g = a.counts[g];
    const int j0 = fb0 - g * a.NB;
    float s[TPL];
#pragma unroll
    for (int i = 0; i < TPL; i++) s[i] = 0.f;
    for (int bi = 0; bi < nb; bi++) {
      const long long fb = fb0 + bi;
      const int cend = min(c1, n_g - (j0 + bi) * C);
      for (int cw = c0; cw < cend;) {
        const KcWin w = kc_window(a, fb * C + cw, cend - cw, lane);
        __syncwarp();
        kc_stage_entries(a, w, ebuf, lane);
        for (int n = 0; n < w.nn; n++) {
          const int r0 = __shfl_sync(0xffffffff, w.rp, n), r1 = __shfl_sync(0xffffffff, w.rp, n + 1);
          float z[2][TPL];
          const float b = __shfl_sync(0xffffffff, w.b, n);
#pragma unroll
          for (int i = 0; i < TPL; i++) {
            z[0][i] = b;
            z[1][i] = 0.f;
          }
          int e = r0;
          for (; e + 1 < r1; e += 2) {  // two entries per step (independent accumulators)
#pragma unroll
            for (int u = 0; u < 2; u++) {
              const uint32_t ent = ebuf[e + u];
              float x[TPL];
              kc_xload<TPL>(x, xs + (ent & 0xFF) * BT, lane);
              const float wv = kc_w(ent);
#pragma unroll
              for (int i = 0; i < TPL; i++) z[u][i] = fmaf(wv, x[i], z[u][i]);
            }
          }
          if (e < r1) {
            const uint32_t ent = ebuf[e];
            float x[TPL];
            kc_xload<TPL>(x, xs + (ent & 0xFF) * BT, lane);
            const float wv = kc_w(ent);
#pragma unroll
            for (int i = 0; i < TPL; i++) z[0][i] = fmaf(wv, x[i], z[0][i]);
          }
#pragma unroll
          for (int i = 0; i < TPL; i++) s[i] += fmaxf(z[0][i] + z[1][i], 0.f);
        }
        cw += w.nn;
      }
    }
#pragma unroll
    for (int i = 0; i < TPL; i++) {
      const int tok = t0 + lane * TPL + i;
      if (tok < a.B) atomicAdd(a.ysum + (long long)g * a.B + tok, s[i]);
    }
  }
}

// ============================================================================
// dX
// ============================================================================
// CTA = (NW * 32 tokens, chunk of static windows of one supergroup); warp w owns tokens 32 w .. 32 w + 31
// of the tile (one per lane) and runs EVERY neuron of the chunk on them, so each FP32 slab element
// dxs[k][token] is read-modified-written by exactly one lane: no atomics (shared FP32 atomicAdd is a CAS
// loop on sm_86).  The neuron windows (<= 31 neurons of one item, <= KC_WE entries; built with the mask:
// kgx_sparse) are staged ONCE per CTA in shared memory -- packed entries, row offsets, biases -- and the
// next window is prefetched into registers while the current one is processed (one barrier per window).
constexpr int KC_DXT = 256;  // threads of kc_dx (= NW * 32): one prefetched entry per thread (KC_WE)

template <int NW>
__global__ void __launch_bounds__(NW * 32) kc_dx_kernel(KxArgs a) {
  static_assert(NW * 32 == KC_DXT && KC_WE <= KC_DXT, "one prefetched entry per thread");
  constexpr int BT = 32 * NW;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* xs = reinterpret_cast<bf16*>(smem_raw);
  float* dxs = reinterpret_cast<float*>(smem_raw + KX_K * BT * 2);
  uint32_t* ebuf = reinterpret_cast<uint32_t*>(smem_raw + KX_K * BT * 6);  // [2][KC_WE]
  int* rpb = reinterpret_cast<int*>(ebuf + 2 * KC_WE);                       // [2][32] row offsets
  float* bsb = reinterpret_cast<float*>(rpb + 64);                           // [2][32] biases
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
  const int q = a.chunks[3 * blockIdx.y], lo = a.chunks[3 * blockIdx.y + 1], hi = a.chunks[3 * blockIdx.y + 2];
  const int t0 = blockIdx.x * BT;
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int col = warp * 32 + lane;  // the thread's token column
  const int tok = t0 + col;
  for (int i = tid; i < KX_K * BT; i += NW * 32) dxs[i] = 0.f;
  kc_stage<BT>(a, fq, xs, t0);
  // window prefetch registers (thread tid: entry tid, row offset / bias of neuron tid < 32)
  uint32_t pe = 0u;
  int prp = 0;
  float pb = 0.f;
  auto prefetch = [&](int w) {
    const int4 m = __ldg(reinterpret_cast<const int4*>(a.wins) + w);
    const int nn = m.z & 0xFF, ne = m.z >> 8;
    pe = tid < ne ? kc_entry(a, (long long)m.w + tid) : 0u;
    prp = tid <= nn ? __ldg(a.rowptr + (long long)m.y + tid) - m.w : 0;
    pb = tid < nn ? __bfloat162float(a.bias[(long long)m.y + tid]) : 0.f;
  };
  auto publish = [&](int buf) {
    ebuf[buf * KC_WE + tid] = pe;  // (KC_WE == threads)
    if (tid < 32) {
      rpb[buf * 32 + tid] = prp;
      bsb[buf * 32 + tid] = pb;
    }
  };
  if (lo < hi) {
    prefetch(lo);
    publish(0);
  }
  cp_async_wait_all();
  __syncthreads();
  int g_prev = -1;
  float cf = 0.f;
  for (int w = lo; w < hi; w++) {
    const int buf = (w - lo) & 1;
    const int4 m = __ldg(reinterpret_cast<const int4*>(a.wins) + w);
    if (w + 1 < hi) prefetch(w + 1);  // global latency overlaps the processing of window w
    const int g = m.x, nn = m.z & 0xFF;
    if (g != g_prev) {
      cf = tok < a.B ? __ldg(a.c1T + (long long)g * a.B + tok) : 0.f;
      g_prev = g;
    }
    const uint32_t* eb = ebuf + buf * KC_WE;
    for (int n = 0; n < nn; n++) {
      const int r0 = rpb[buf * 32 + n], r1 = rpb[buf * 32 + n + 1];
      if (r1 == r0) continue;  // no inputs: no dX term
      float zz[2] = {bsb[buf * 32 + n], 0.f};
      int e = r0;
      for (; e + 1 < r1; e += 2) {
        const uint32_t u0 = eb[e], u1 = eb[e + 1];
        zz[0] = fmaf(kc_w(u0), __bfloat162float(xs[(u0 & 0xFF) * BT + col]), zz[0]);
        zz[1] = fmaf(kc_w(u1), __bfloat162float(xs[(u1 & 0xFF) * BT + col]), zz[1]);
      }
      if (e < r1) {
        const uint32_t u0 = eb[e];
        zz[0] = fmaf(kc_w(u0), __bfloat162float(xs[(u0 & 0xFF) * BT + col]), zz[0]);
      }
      const float dz = cf * ((zz[0] + zz[1]) > 0.f ? 1.f : 0.f);  // generic c1 * act'(z)
      if (!__any_sync(0xffffffff, dz != 0.f)) continue;
      for (e = r0; e + 1 < r1; e += 2) {  // a neuron's inputs are distinct: independent RMWs
        const uint32_t u0 = eb[e], u1 = eb[e + 1];
        float* p0 = dxs + (u0 & 0xFF) * BT + col;
        float* p1 = dxs + (u1 & 0xFF) * BT + col;
        const float v0 = *p0, v1 = *p1;
        *p0 = fmaf(kc_w(u0), dz, v0);
        *p1 = fmaf(kc_w(u1), dz, v1);
      }
      if (e < r1) {
        const uint32_t u0 = eb[e];
        float* p0 = dxs + (u0 & 0xFF) * BT + col;
        *p0 = fmaf(kc_w(u0), dz, *p0);
      }
    }
    if (w + 1 < hi) publish(buf ^ 1);
    __syncthreads();
  }
  for (int idx = tid; idx < KX_K * BT; idx += NW * 32) {
    const int k = idx / BT, tl = idx % BT;
    const float v = dxs[idx];
    if (t0 + tl < a.B && v != 0.f) atomicAdd(a.dXT + (long long)fq[k] * a.B + t0 + tl, v);
  }
}

// ============================================================================
// dW / db
// ============================================================================
template <int BT, int ST, int NW>
__global__ void __launch_bounds__(NW * 32) kc_dw_kernel(KxArgs a) {
  constexpr int TPL = BT / 32;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* ring = reinterpret_cast<bf16*>(smem_raw);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int C = a.C;
  // the warp's task: 16 neuron slots of one block
  const int pos = lo + warp;
  const long long fb = pos < hi ? a.tasks[2 * pos] : -1;
  const int c0 = pos < hi ? 16 * a.tasks[2 * pos + 1] : 0;
  const int g = fb >= 0 ? (int)(fb / a.NB) : 0;
  const int n_g = fb >= 0 ? a.counts[g] : 0;
  const int j = fb >= 0 ? (int)(fb - (long long)g * a.NB) : 0;
  const int nv = fb >= 0 ? max(0, min(16, n_g - j * C - c0)) : 0;  // valid neurons of the task
  const long long slot0 = (fb >= 0 ? fb : 0) * C + c0;
  const int myrp = lane <= nv ? __ldg(a.rowptr + slot0 + lane) : 0;
  const float myb = lane < nv ? __bfloat162float(a.bias[slot0 + lane]) : 0.f;
  uint32_t ent[16][2];  // entry lane, lane + 32 of every neuron (nnz <= 64)
  float acc[16][2], dbs[16];
#pragma unroll
  for (int n = 0; n < 16; n++) {
    const int e0 = __shfl_sync(0xffffffff, myrp, n), e1 = __shfl_sync(0xffffffff, myrp, min(n + 1, 31));
    const int cnt = n < nv ? e1 - e0 : 0;
    ent[n][0] = lane < cnt ? kc_entry(a, (long long)e0 + lane) : 0u;
    ent[n][1] = lane + 32 < cnt ? kc_entry(a, (long long)e0 + 32 + lane) : 0u;
    acc[n][0] = acc[n][1] = 0.f;
    dbs[n] = 0.f;
  }
  const int n_st = (a.B + BT - 1) / BT;
#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < n_st) kc_stage<BT>(a, fq, ring + s * KX_K * BT, s * BT);
    cp_async_commit();
  }
  for (int s = 0; s < n_st; s++) {
    cp_async_wait_group<ST - 2>();
    __syncthreads();
    if (s + ST - 1 < n_st) kc_stage<BT>(a, fq, ring + ((s + ST - 1) % ST) * KX_K * BT, (s + ST - 1) * BT);
    cp_async_commit();
    if (nv == 0) continue;
    const bf16* xs = ring + (s % ST) * KX_K * BT;
    float cf[TPL];
#pragma unroll
    for (int i = 0; i < TPL; i++) {
      const int tok = s * BT + lane * TPL + i;
      cf[i] = tok < a.B ? __ldg(a.c1T + (long long)g * a.B + tok) : 0.f;
    }
#pragma unroll
    for (int n = 0; n < 16; n++) {  // unrolled: ent / acc / dbs stay in registers
      if (n >= nv) break;
      const int e0 = __shfl_sync(0xffffffff, myrp, n), e1 = __shfl_sync(0xffffffff, myrp, n + 1);
      const int cnt = e1 - e0;
      float z[TPL];
      const float b = __shfl_sync(0xffffffff, myb, n);
#pragma unroll
      for (int i = 0; i < TPL; i++) z[i] = b;
      for (int j0 = 0; j0 < cnt; j0 += 2) {  // 2 entries per step into independent partial sums
        float zq[TPL];
#pragma unroll
        for (int i = 0; i < TPL; i++) zq[i] = 0.f;
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const int jj = j0 + u;
          const uint32_t e = __shfl_sync(0xffffffff, jj < 32 ? ent[n][0] : ent[n][1], jj & 31);
          if (jj < cnt) {
            float x[TPL];
            kc_xload<TPL>(x, xs + (e & 0xFF) * BT, lane);
            const float w = kc_w(e);
#pragma unroll
            for (int i = 0; i < TPL; i++) {
              if (u == 0) z[i] = fmaf(w, x[i], z[i]);
              else zq[i] = fmaf(w, x[i], zq[i]);
            }
          }
        }
#pragma unroll
        for (int i = 0; i < TPL; i++) z[i] += zq[i];
      }
      float dz[TPL], sdz = 0.f;
      bool any = false;
#pragma unroll
      for (int i = 0; i < TPL; i++) {
        dz[i] = cf[i] * (z[i] > 0.f ? 1.f : 0.f);
        sdz += dz[i];
        any |= dz[i] != 0.f;
      }
      dbs[n] += sdz;  // lane partial (reduced at the end)
      if (!__any_sync(0xffffffff, any)) continue;
      for (int j0 = 0; j0 < cnt; j0 += 4) {  // 4 entries at a time: independent partial sums / reductions
        float p[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const int jj = j0 + u;
          const uint32_t e = __shfl_sync(0xffffffff, jj < 32 ? ent[n][0] : ent[n][1], jj & 31);
          p[u] = 0.f;
          if (jj < cnt) {
            float x[TPL];
            kc_xload<TPL>(x, xs + (e & 0xFF) * BT, lane);
#pragma unroll
            for (int i = 0; i < TPL; i++) p[u] = fmaf(dz[i], x[i], p[u]);
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
          for (int u = 0; u < 4; u++) p[u] += __shfl_xor_sync(0xffffffff, p[u], o);
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const int jj = j0 + u;
          if (jj < cnt && lane == (jj & 31)) {
            if (jj < 32) acc[n][0] += p[u];
            else acc[n][1] += p[u];
          }
        }
      }
    }
  }
  cp_async_wait_all();
  // write back (exclusive ownership of the 16 neurons)
#pragma unroll
  for (int n = 0; n < 16; n++) {
    float d = dbs[n];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
    if (n >= nv) continue;
    const int e0 = __shfl_sync(0xffffffff, myrp, n), e1 = __shfl_sync(0xffffffff, myrp, n + 1);
    if (lane < e1 - e0) a.dW[e0 + lane] += acc[n][0];
    if (lane + 32 < e1 - e0) a.dW[e0 + 32 + lane] += acc[n][1];
    if (lane == 0) a.dB[slot0 + n] += d;
  }
}
