// Native Ampere forward: fused block-shared sparse projection (mma.m16n8k8,
// BF16 x BF16 -> FP32) + streaming pooling.  Mode B only.
//
// CTA  = (token tile of BM = 16*MT tokens, pooling unit u, physical split ps).
// The CTA stages the token tile of X as a feature-major slab [D][BM] (BF16) in
// shared memory, then every warp scans its share of the unit's index blocks:
//   ldmatrix.x4.trans  gathers A[BM x 8] = X[tokens, idx[0..7]] from the slab
//   ld.global          loads the block's weights (exactly the mma B fragment)
//   mma.m16n8k8        Z = A W + bias   (bias enters as the accumulator input)
//   epilogue           folds Z into per-lane pooling state (registers)
// Hidden activations never leave registers.  At the end the per-lane states are
// merged across the quad, then across warps via shared memory, and one compact
// state per (token, unit, split) is written.
#pragma once
#include "common.cuh"

// SP_ABLATE_FWD 1 (benchmarks only, WRONG results): every warp re-loads its first
// work unit, so parameter streaming becomes cache hits (attribution of param traffic)
// SP_ABLATE_FWD 2 (WRONG results): no bias, the MMA accumulator input is zero (cost of
// forming the bias C fragments); 3: as 2 and the epilogue folds one element per MMA
// into the pooling sum instead of four (cost of the ReLU + accumulate epilogue);
// 4: as 2 but the bias is still loaded (added once per n-tile to the sum)
#ifndef SP_ABLATE_FWD
#define SP_ABLATE_FWD 0
#endif

struct FwdArgs {
  const bf16* X;
  long long sxf, sxb;  // element (b, f) at X[f * sxf + b * sxb]
  const int16_t* idx;  // [G, NB, 8]
  const bf16* W;       // [G, NB, C, 8]
  const bf16* bias;    // [G, NB, C]
  const int* counts;   // [G]
  float* st_m;
  float* st_s;
  float* st_t;
  int* st_id;
  int B, U, S, NB, BPS, BPSP, D;
  float scale, p;
  int p_int;
  int packed;  // fragment-packed parameter storage
  int fepi;    // forward epilogue variant (host-side dispatch: FE template argument)
};

// ---------------------------------------------------------------- pooling ---
// Per-row streaming state.  P_INT: compile-time integer power-mean exponent
// (1, 2, 4, 8) or 0 for a generic runtime exponent.
template <int POOL, int P_INT>
struct PoolState {
  float m, s, t, inv;
  int id;
  __device__ __forceinline__ void init() {
    m = (POOL == P_MEAN || POOL == P_POW) ? 0.f : -INFINITY;
    s = 0.f;
    t = 0.f;
    inv = 0.f;
    id = INT_MAX;
  }
  __device__ __forceinline__ static float pw(float q, float p) {
    if (P_INT == 1) return q;
    if (P_INT == 2) return q * q;
    if (P_INT == 4) { float q2 = q * q; return q2 * q2; }
    if (P_INT == 8) { float q2 = q * q; float q4 = q2 * q2; return q4 * q4; }
    return q > 0.f ? exp2_approx(p * __log2f(q)) : 0.f;
  }

  // Fold N values (pre-activations z, slots increasing) of this row.
  // valid[k] == false marks padded neurons (only in the masked tail block).
  template <int ACT, bool TRACK, int N, bool MASKED>
  __device__ __forceinline__ void update(const float* z, const int* slot, const bool* valid, float scale, float p) {
    if (POOL == P_MAX) {
#pragma unroll
      for (int k = 0; k < N; k++) {
        float v = act_f<ACT>(z[k]);
        if (MASKED && !valid[k]) continue;
        if (TRACK) {
          if (v > m) { m = v; id = slot[k]; }  // strict: earlier (smaller) slot wins ties
        } else {
          m = fmaxf(m, v);
        }
      }
    } else if (POOL == P_MEAN) {
#pragma unroll
      for (int k = 0; k < N; k++)
        if (!MASKED || valid[k]) s += act_f<ACT>(z[k]);
    } else if (POOL == P_LME || POOL == P_SW) {
      float a[N], u[N];
      float um = -INFINITY;
#pragma unroll
      for (int k = 0; k < N; k++) {
        a[k] = act_f<ACT>(z[k]);
        u[k] = (MASKED && !valid[k]) ? -INFINITY : a[k] * scale;
        um = fmaxf(um, u[k]);
      }
      // a lane may see only padded neurons in the tail chunk (and m may still
      // be -inf): exp2(-inf - -inf) would be NaN
      if (MASKED && um == -INFINITY) return;
      // lazy max: rescale only when the reference is exceeded by > 2^8 (one
      // check per chunk); keeps every term <= 256 and the sum far from overflow
      if (um > m + 8.f) {
        float r = exp2_approx(m - um);
        s *= r;
        if (POOL == P_SW) t *= r;
        m = um;
      }
#pragma unroll
      for (int k = 0; k < N; k++) {
        float e = exp2_approx(u[k] - m);
        s += e;
        if (POOL == P_SW) t = fmaf(a[k], e, t);
      }
    } else if (POOL == P_POW) {
      float a[N];
      float am = 0.f;
#pragma unroll
      for (int k = 0; k < N; k++) {
        a[k] = (MASKED && !valid[k]) ? 0.f : fmaxf(z[k], 0.f);
        am = fmaxf(am, a[k]);
      }
      if (am > m) {  // exact running max, checked once per chunk
        float r = m / am;
        s *= pw(r, p);
        m = am;
        inv = 1.f / am;
      }
#pragma unroll
      for (int k = 0; k < N; k++) s += pw(a[k] * inv, p);
    }
  }

  // merge another state of the same row (other lanes / warps)
  __device__ __forceinline__ void merge(float m2, float s2, float t2, int id2, float p) {
    if (POOL == P_MAX) {
      if (m2 > m || (m2 == m && id2 < id)) { m = m2; id = id2; }
    } else if (POOL == P_MEAN) {
      s += s2;
    } else if (POOL == P_LME || POOL == P_SW) {
      float M = fmaxf(m, m2);
      if (M == -INFINITY) return;
      float r1 = exp2f(m - M), r2 = exp2f(m2 - M);
      s = s * r1 + s2 * r2;
      if (POOL == P_SW) t = t * r1 + t2 * r2;
      m = M;
    } else if (POOL == P_POW) {
      float M = fmaxf(m, m2);
      if (M <= 0.f) { m = 0.f; s = 0.f; return; }
      s = s * powf(m / M, p) + s2 * powf(m2 / M, p);
      m = M;
    }
  }
};

template <int MT>
struct Swz {
  static constexpr int BM = 16 * MT;
  static constexpr int CH = BM / 8;                    // 16-byte chunks per feature row
  static constexpr int SHIFT = (CH == 2) ? 2 : (CH == 4 ? 1 : 0);
  // element offset of (feature f, token chunk c) inside the slab
  __device__ __forceinline__ static int off(int f, int c) {
    return f * BM + ((c ^ ((f >> SHIFT) & (CH - 1))) << 3);
  }
};

// stage X[t0 : t0+BM, 0:D] as a swizzled feature-major slab
template <int MT>
__device__ __forceinline__ void stage_slab(const FwdArgs& a, bf16* slab, int t0) {
  constexpr int BM = 16 * MT, CH = BM / 8;
  const int nthr = blockDim.x;
  if (a.sxb == 1) {  // feature-major input: 16B rows chunks -> cp.async
    for (int i = threadIdx.x; i < a.D * CH; i += nthr) {
      int f = i / CH, c = i % CH;
      int tok = t0 + c * 8;
      const bf16* src = a.X + (long long)f * a.sxf + tok;
      int bytes = (tok + 8 <= a.B) ? 16 : (tok < a.B ? 2 * (a.B - tok) : 0);
      if (bytes == 0) src = a.X;
      cp_async16(slab + Swz<MT>::off(f, c), src, bytes);
    }
    cp_async_wait_all();
  } else {  // token-major input: 16B = 8 features of one token -> transpose
    for (int i = threadIdx.x; i < BM * (a.D / 8); i += nthr) {
      int tl = i / (a.D / 8), f8 = (i % (a.D / 8)) * 8;
      int tok = t0 + tl;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (tok < a.B) v = *reinterpret_cast<const uint4*>(a.X + (long long)tok * a.sxb + f8);
      const bf16* e = reinterpret_cast<const bf16*>(&v);
      int c = tl >> 3, o = tl & 7;
#pragma unroll
      for (int q = 0; q < 8; q++) slab[Swz<MT>::off(f8 + q, c) + o] = e[q];
    }
  }
}

template <int C>
struct BlockRegs {
  int fidx;               // this lane's ldmatrix row feature
  uint32_t w[C / 8];      // mma B fragments (one per n8 tile)
  uint32_t bb[C / 8];     // bias pairs (bf16x2) for this lane's two columns
};

template <int C>
__device__ __forceinline__ void load_block(BlockRegs<C>& r, const FwdArgs& a, long long blk, int lane) {
  r.fidx = __ldg(a.idx + blk * 8 + (lane & 7));
  const uint32_t* wp = reinterpret_cast<const uint32_t*>(a.W + blk * C * 8);
  const uint32_t* bp = reinterpret_cast<const uint32_t*>(a.bias + blk * C);
#pragma unroll
  for (int nt = 0; nt < C / 8; nt++) {
    r.w[nt] = __ldg(wp + nt * 32 + lane);
    r.bb[nt] = __ldg(bp + nt * 4 + (lane & 3));
  }
}

// ---- fragment-packed parameters (layout.py): a work unit is UB blocks whose
// MMA B fragments arrive as NQ4 16-byte loads per lane (+ NQ4 for bias, 1 idx).
template <int C>
struct Unit {
  static constexpr int NT = C / 8;
  static constexpr int UB = NT >= 4 ? 1 : 4 / NT;   // blocks per unit
  static constexpr int NQ4 = NT >= 4 ? NT / 4 : 1;  // uint4 per lane per unit
  uint4 w[NQ4];
  uint4 b[NQ4];
  uint32_t i0, i1;  // UB int16 indices for this lane's k = lane & 7
};

template <int C>
__device__ __forceinline__ void load_unit(Unit<C>& r, const FwdArgs& a, long long unit, int lane) {
  constexpr int UB = Unit<C>::UB, NQ4 = Unit<C>::NQ4;
  const uint4* wp = reinterpret_cast<const uint4*>(a.W) + unit * NQ4 * 32 + lane;
  const uint4* bp = reinterpret_cast<const uint4*>(a.bias) + unit * NQ4 * 4 + (lane & 3);
#pragma unroll
  for (int i = 0; i < NQ4; i++) {
    r.w[i] = __ldg(wp + i * 32);
    r.b[i] = __ldg(bp + i * 4);
  }
  const int16_t* ip = a.idx + (unit * 8 + (lane & 7)) * UB;
  if (UB == 4) {
    uint2 v = __ldg(reinterpret_cast<const uint2*>(ip));
    r.i0 = v.x; r.i1 = v.y;
  } else if (UB == 2) {
    r.i0 = __ldg(reinterpret_cast<const uint32_t*>(ip)); r.i1 = 0;
  } else {
    r.i0 = (uint16_t)__ldg(ip); r.i1 = 0;
  }
}

__device__ __forceinline__ uint32_t u4_word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// unpack block bu of a unit into the per-block register form
template <int C>
__device__ __forceinline__ void unit_block(BlockRegs<C>& cur, const Unit<C>& r, int bu) {
  constexpr int NT = C / 8;
  uint32_t iw = bu < 2 ? r.i0 : r.i1;
  cur.fidx = (int)((iw >> (16 * (bu & 1))) & 0xffffu);
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    const int f = bu * NT + nt;
    cur.w[nt] = u4_word(r.w[f >> 2], f & 3);
    cur.bb[nt] = u4_word(r.b[f >> 2], f & 3);
  }
}

// prefetch depth (blocks in flight per warp) - deeper for small blocks
template <int C>
struct Prefetch { static constexpr int PD = (C == 8) ? 4 : (C == 16 ? 3 : 2); };

// FE = 1 (Mean only): epilogue chunks of 4 n8 tiles instead of 2.  More MMAs in flight
// per chunk lets ptxas build the bias C fragments ahead of them; each row state still
// receives its values in the same order (n-tile, then column), so results are bitwise equal.
template <int POOL, int ACT, int C, int MT, bool TRACK, int P_INT, bool MASKED, int FE = 0>
__device__ __forceinline__ void process_block(PoolState<POOL, P_INT>* st, const BlockRegs<C>& cur, int j, int n_g,
                                              uint32_t slab_s, int lane, float scale, float p) {
  constexpr int NT = C / 8;
  constexpr int NTC = (FE == 1 && POOL == P_MEAN && NT >= 4) ? 4 : (NT < 2 ? NT : 2);  // n8 tiles per epilogue chunk
  const int mat = lane >> 3;
  const int col0 = 2 * (lane & 3);
  uint32_t af[4];
  {
    int c = (MT == 2) ? mat : (mat & 1);
    uint32_t addr = slab_s + 2 * Swz<MT>::off(cur.fidx, c);
    if (MT == 2) ldmatrix_x4_trans(af, addr);
    else ldmatrix_x2_trans(af, addr);
  }
#pragma unroll
  for (int n0 = 0; n0 < NT; n0 += NTC) {
    float d[NTC][MT][4];
#pragma unroll
    for (int q = 0; q < NTC; q++) {
      float2 bf = bf16x2_to_float2(cur.bb[n0 + q]);
      if (SP_ABLATE_FWD == 4) st[0].s += bf.x + bf.y;
      if (SP_ABLATE_FWD >= 2) bf = make_float2(0.f, 0.f);
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
        mma_m16n8k8_bf16(d[q][mt], af[2 * mt], af[2 * mt + 1], cur.w[n0 + q], bf.x, bf.y, bf.x, bf.y);
    }
    if (SP_ABLATE_FWD == 3) {
#pragma unroll
      for (int q = 0; q < NTC; q++)
#pragma unroll
        for (int mt = 0; mt < MT; mt++) st[2 * mt].s += d[q][mt][0];
      continue;
    }
    int slot[2 * NTC];
    bool valid[2 * NTC];
#pragma unroll
    for (int q = 0; q < NTC; q++) {
      slot[2 * q] = j * C + (n0 + q) * 8 + col0;
      slot[2 * q + 1] = slot[2 * q] + 1;
      valid[2 * q] = !MASKED || slot[2 * q] < n_g;
      valid[2 * q + 1] = !MASKED || slot[2 * q + 1] < n_g;
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
      for (int h = 0; h < 2; h++) {
        float v[2 * NTC];
#pragma unroll
        for (int q = 0; q < NTC; q++) {
          v[2 * q] = d[q][mt][2 * h];
          v[2 * q + 1] = d[q][mt][2 * h + 1];
        }
        st[2 * mt + h].template update<ACT, TRACK, 2 * NTC, MASKED>(v, slot, valid, scale, p);
      }
    }
  }
}

template <int POOL, int ACT, int C, int MT, int NW, bool TRACK, int P_INT, bool PACKED, int FE = 0>
__global__ void __launch_bounds__(NW * 32) fwd_mma_kernel(FwdArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* slab = reinterpret_cast<bf16*>(smem_raw);
  constexpr int BM = 16 * MT;
  constexpr int PD = Prefetch<C>::PD;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int t0 = blockIdx.x * BM, u = blockIdx.y, ps = blockIdx.z;
  const int g = u / a.S, sh = u % a.S;
  const int n_g = a.counts[g];
  const int b_lo = sh * a.BPS + ps * a.BPSP;
  const int b_hi = min(min(b_lo + a.BPSP, (sh + 1) * a.BPS), a.NB);
  const int b_full = min(max(n_g / C, b_lo), b_hi);
  const int b_any = min(max((n_g + C - 1) / C, b_lo), b_hi);
  const long long gbase = (long long)g * a.NB;

  stage_slab<MT>(a, slab, t0);
  __syncthreads();

  PoolState<POOL, P_INT> st[MT * 2];
#pragma unroll
  for (int i = 0; i < MT * 2; i++) st[i].init();
  const uint32_t slab_s = smem_u32(slab);

  if constexpr (PACKED) {
    // ---- main loop over fully valid units, ring prefetch QPD units deep
    constexpr int UB = Unit<C>::UB;
    constexpr int QPD = (C == 64) ? 1 : 2;
    const long long ubase = (gbase + b_lo) / UB;  // flat unit index of b_lo
    const int nu_full = (b_full - b_lo) / UB;
    Unit<C> ring[QPD];
#pragma unroll
    for (int q = 0; q < QPD; q++)
      if (warp + q * NW < nu_full) load_unit<C>(ring[q], a, ubase + warp + q * NW, lane);
    for (int u0 = warp; u0 < nu_full; u0 += QPD * NW) {
#pragma unroll
      for (int q = 0; q < QPD; q++) {
        const int uu = u0 + q * NW;
        if (uu < nu_full) {
          Unit<C> cur = ring[q];
          if (uu + QPD * NW < nu_full)
            load_unit<C>(ring[q], a, ubase + (SP_ABLATE_FWD == 1 ? warp + q * NW : uu + QPD * NW), lane);
#pragma unroll
          for (int bu = 0; bu < UB; bu++) {
            BlockRegs<C> blk;
            unit_block<C>(blk, cur, bu);
            process_block<POOL, ACT, C, MT, TRACK, P_INT, false, FE>(st, blk, b_lo + uu * UB + bu, n_g, slab_s,
                                                                     lane, a.scale, a.p);
          }
        }
      }
    }
    // ---- tail: the single unit containing the partially valid block(s)
    const int bt = b_lo + nu_full * UB;
    if (bt < b_any && warp == nu_full % NW) {
      Unit<C> cur;
      load_unit<C>(cur, a, ubase + nu_full, lane);
#pragma unroll
      for (int bu = 0; bu < UB; bu++) {
        if (bt + bu < b_any) {
          BlockRegs<C> blk;
          unit_block<C>(blk, cur, bu);
          process_block<POOL, ACT, C, MT, TRACK, P_INT, true, FE>(st, blk, bt + bu, n_g, slab_s, lane, a.scale, a.p);
        }
      }
    }
  } else {
    // ---- main loop over fully valid blocks: register ring prefetch PD deep
    BlockRegs<C> ring[PD];
    const int j0 = b_lo + warp;
#pragma unroll
    for (int q = 0; q < PD; q++)
      if (j0 + q * NW < b_full) load_block<C>(ring[q], a, gbase + j0 + q * NW, lane);
    for (int j = j0; j < b_full; j += PD * NW) {
#pragma unroll
      for (int q = 0; q < PD; q++) {
        const int jj = j + q * NW;
        if (jj < b_full) {
          BlockRegs<C> cur = ring[q];
          if (jj + PD * NW < b_full) load_block<C>(ring[q], a, gbase + jj + PD * NW, lane);
          process_block<POOL, ACT, C, MT, TRACK, P_INT, false, FE>(st, cur, jj, n_g, slab_s, lane, a.scale, a.p);
        }
      }
    }
    // ---- tail: at most one partially valid block per unit
    if (b_any > b_full && warp == (b_full - b_lo) % NW) {
      BlockRegs<C> cur;
      load_block<C>(cur, a, gbase + b_full, lane);
      process_block<POOL, ACT, C, MT, TRACK, P_INT, true, FE>(st, cur, b_full, n_g, slab_s, lane, a.scale, a.p);
    }

  }

  // ---- merge the 4 lanes of a quad (they hold the same rows)
#pragma unroll
  for (int i = 0; i < MT * 2; i++) {
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      float m2 = __shfl_xor_sync(0xffffffff, st[i].m, off);
      float s2 = __shfl_xor_sync(0xffffffff, st[i].s, off);
      float t2 = __shfl_xor_sync(0xffffffff, st[i].t, off);
      int id2 = __shfl_xor_sync(0xffffffff, st[i].id, off);
      st[i].merge(m2, s2, t2, id2, a.p);
    }
  }
  // ---- merge across warps through shared memory (slab no longer needed)
  __syncthreads();
  float* sm_m = reinterpret_cast<float*>(smem_raw);
  float* sm_s = sm_m + NW * BM;
  float* sm_t = sm_s + NW * BM;
  int* sm_id = reinterpret_cast<int*>(sm_t + NW * BM);
  if ((lane & 3) == 0) {
#pragma unroll
    for (int i = 0; i < MT * 2; i++) {
      int row = (i >> 1) * 16 + (i & 1) * 8 + (lane >> 2);
      sm_m[warp * BM + row] = st[i].m;
      sm_s[warp * BM + row] = st[i].s;
      sm_t[warp * BM + row] = st[i].t;
      sm_id[warp * BM + row] = st[i].id;
    }
  }
  __syncthreads();
  if (threadIdx.x < BM) {
    int row = threadIdx.x;
    PoolState<POOL, P_INT> r;
    r.m = sm_m[row]; r.s = sm_s[row]; r.t = sm_t[row]; r.id = sm_id[row];
    for (int w = 1; w < NW; w++)
      r.merge(sm_m[w * BM + row], sm_s[w * BM + row], sm_t[w * BM + row], sm_id[w * BM + row], a.p);
    int tok = t0 + row;
    if (tok < a.B) {
      long long o = ((long long)ps * a.B + tok) * a.U + u;
      a.st_m[o] = r.m;
      if (POOL != P_MAX) a.st_s[o] = r.s;
      if (POOL == P_SW) a.st_t[o] = r.t;
      if (TRACK) a.st_id[o] = r.id;
    }
  }
}

// ============================================================================
// Top-K mean forward (exact streaming selection, ranks raw pre-activations)
// ============================================================================
// Every lane keeps an exact unordered top-KP list per row it touches (TopK of a
// union = TopK of the union of TopKs).  A per-row threshold shared through
// shared memory (max over lanes of their KP-th best value, a valid lower bound
// of the CTA-wide KP-th best) filters candidates, so after warm-up almost no
// element reaches the insertion path.  Ties: smaller neuron id wins.  Within a
// lane slots arrive in increasing order, so a candidate must beat the lane's
// own worst entry strictly; it only needs to *equal* the shared threshold.
__device__ __forceinline__ int ord_enc(float f) {
  int i = __float_as_int(f);
  return i ^ ((i >> 31) & 0x7FFFFFFF);
}
__device__ __forceinline__ float ord_dec(int i) { return __int_as_float(i ^ ((i >> 31) & 0x7FFFFFFF)); }

template <int KP>
struct TopkList {
  float v[KP];
  int id[KP];
  float mn;        // worst kept value (-inf until the list is full)
  int wid;         // id of the worst entry
  unsigned wmask;  // one-hot position of the worst entry.  An integer index
                   // would let the compiler turn the unrolled select chain into
                   // a dynamic store v[pos] = x, forcing the list to local memory.
  __device__ __forceinline__ void init() {
#pragma unroll
    for (int k = 0; k < KP; k++) { v[k] = -INFINITY; id[k] = INT_MAX; }
    mn = -INFINITY; wid = INT_MAX; wmask = 1u;
  }
  __device__ __forceinline__ bool beats_worst(float x, int s) const { return x > mn || (x == mn && s < wid); }
  // Branch-free conditional insertion (predicated selects).  Data-dependent
  // branches around the update make every list element a merge point and the
  // compiler then demotes the whole list to local memory.
  __device__ __forceinline__ void replace_worst_if(bool ok, float x, int s) {
    ok = ok && beats_worst(x, s);
#pragma unroll
    for (int k = 0; k < KP; k++) {
      const bool hit = ok && ((wmask >> k) & 1u);
      v[k] = hit ? x : v[k];
      id[k] = hit ? s : id[k];
    }
    mn = v[0]; wid = id[0]; wmask = 1u;
#pragma unroll
    for (int k = 1; k < KP; k++) {
      const bool worse = v[k] < mn || (v[k] == mn && id[k] > wid);
      mn = worse ? v[k] : mn;
      wid = worse ? id[k] : wid;
      wmask = worse ? (1u << k) : wmask;
    }
  }
  __device__ __forceinline__ void replace_worst(float x, int s) {
#pragma unroll
    for (int k = 0; k < KP; k++) {
      const bool hit = (wmask >> k) & 1u;
      v[k] = hit ? x : v[k];
      id[k] = hit ? s : id[k];
    }
    mn = v[0]; wid = id[0]; wmask = 1u;
#pragma unroll
    for (int k = 1; k < KP; k++) {
      const bool worse = v[k] < mn || (v[k] == mn && id[k] > wid);
      mn = worse ? v[k] : mn;
      wid = worse ? id[k] : wid;
      wmask = worse ? (1u << k) : wmask;
    }
  }
};

struct TopkArgs {
  FwdArgs f;
  long long* st_keys;  // [PS, B, U, KP] when PS > 1
  float* st_pre;       // [B, U] when PS == 1
  int* st_slot;        // [B, U, KP] when PS == 1 and slots requested (else null)
  int act;             // activation for the finalized mean (0 id, 1 relu, 2 gelu)
};

template <int KP, int C, int MT, bool MASKED>
__device__ __forceinline__ void topk_block(TopkList<KP> (&st)[MT * 2], const float (&thr)[MT * 2],
                                           const BlockRegs<C>& cur, int j, int n_g, uint32_t slab_s, int lane,
                                           int* sh_thr) {
  constexpr int NT = C / 8;
  const int mat = lane >> 3;
  const int col0 = 2 * (lane & 3);
  uint32_t af[4];
  {
    int c = (MT == 2) ? mat : (mat & 1);
    uint32_t addr = slab_s + 2 * Swz<MT>::off(cur.fidx, c);
    if (MT == 2) ldmatrix_x4_trans(af, addr);
    else ldmatrix_x2_trans(af, addr);
  }
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    float2 bf = bf16x2_to_float2(cur.bb[nt]);
    const int slot0 = j * C + nt * 8 + col0;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float d[4];
      mma_m16n8k8_bf16(d, af[2 * mt], af[2 * mt + 1], cur.w[nt], bf.x, bf.y, bf.x, bf.y);
      // fast path: one compare per element against max(shared bound, own worst)
      bool cand[4];
      bool any = false;
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int r = 2 * mt + (e >> 1);
        cand[e] = d[e] >= fmaxf(thr[r], st[r].mn) && (!MASKED || slot0 + (e & 1) < n_g);
        any |= cand[e];
      }
      if (__any_sync(0xffffffff, any)) {  // rare after warm-up
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 2 * mt + (e >> 1);
          const int s = slot0 + (e & 1);
          // filter: shared lower bound (>=) and own worst (strict / id order)
          st[r].replace_worst_if(cand[e], d[e], s);
          if (cand[e] && st[r].mn > -INFINITY) {
            const int row = mt * 16 + (e >> 1) * 8 + (lane >> 2);
            atomicMax(sh_thr + row, ord_enc(st[r].mn));
          }
        }
      }
    }
  }
}

// pass 1 of Top-K: per-lane row maxima only (Max-kernel cost)
template <int C, int MT, bool MASKED>
__device__ __forceinline__ void topk_max_block(float (&lmax)[MT * 2], const BlockRegs<C>& cur, int j, int n_g,
                                               uint32_t slab_s, int lane) {
  constexpr int NT = C / 8;
  const int mat = lane >> 3;
  const int col0 = 2 * (lane & 3);
  uint32_t af[4];
  {
    int c = (MT == 2) ? mat : (mat & 1);
    uint32_t addr = slab_s + 2 * Swz<MT>::off(cur.fidx, c);
    if (MT == 2) ldmatrix_x4_trans(af, addr);
    else ldmatrix_x2_trans(af, addr);
  }
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    float2 bf = bf16x2_to_float2(cur.bb[nt]);
    const int slot0 = j * C + nt * 8 + col0;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float d[4];
      mma_m16n8k8_bf16(d, af[2 * mt], af[2 * mt + 1], cur.w[nt], bf.x, bf.y, bf.x, bf.y);
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const bool ok = !MASKED || slot0 + (e & 1) < n_g;
        const int r = 2 * mt + (e >> 1);
        lmax[r] = ok ? fmaxf(lmax[r], d[e]) : lmax[r];
      }
    }
  }
}

// CTA-wide merge of every lane's lists.  All threads must call it; threads
// t < BM return their row's merged list in r.  Lists are written to shared
// memory in warp groups that fit the allocated space.
// Fast path: lanes append only entries >= the row's shared lower bound (no
// entry below it can be in the top-KP) into compact per-row buffers; one thread
// per row merges the few survivors.  If any row overflows its buffer the CTA
// falls back to the exhaustive merge below (exactness is never traded).
template <int KP, int MT, int NW>
__device__ __forceinline__ bool topk_cta_merge_compact(TopkList<KP> (&st)[MT * 2], TopkList<KP>& r,
                                                       unsigned char* smem_raw, int D, int lane, const int* sh_thr,
                                                       int* cnt) {
  constexpr int BM = 16 * MT;
  const int smem_alloc = max(D * BM * 2, NW * BM * KP * 8);
  const int cap = min(4 * NW * KP, smem_alloc / (BM * 8));
  float* sv = reinterpret_cast<float*>(smem_raw);
  int* sid = reinterpret_cast<int*>(sv + BM * cap);
  __syncthreads();
  if (threadIdx.x < BM) cnt[threadIdx.x] = 0;
  __syncthreads();
  bool overflow = false;
#pragma unroll
  for (int i = 0; i < MT * 2; i++) {
    const int row = (i >> 1) * 16 + (i & 1) * 8 + (lane >> 2);
    const float lb = ord_dec(sh_thr[row]);
#pragma unroll
    for (int k = 0; k < KP; k++) {
      if (st[i].id[k] != INT_MAX && st[i].v[k] >= lb) {
        const int pos = atomicAdd(cnt + row, 1);
        if (pos < cap) {
          sv[row * cap + pos] = st[i].v[k];
          sid[row * cap + pos] = st[i].id[k];
        } else {
          overflow = true;
        }
      }
    }
  }
  if (__syncthreads_or(overflow)) return false;
  r.init();
  if (threadIdx.x < BM) {
    const int row = threadIdx.x;
    const int n = cnt[row];
#pragma unroll 1
    for (int e = 0; e < n; e++) {
      const float x = sv[row * cap + e];
      const int s = sid[row * cap + e];
      if (r.beats_worst(x, s)) r.replace_worst_if(true, x, s);
    }
  }
  __syncthreads();
  return true;
}

template <int KP, int MT, int NW>
__device__ __forceinline__ void topk_cta_merge(TopkList<KP> (&st)[MT * 2], TopkList<KP>& r, unsigned char* smem_raw,
                                               int D, int lane, int warp, const int* sh_thr) {
  constexpr int BM = 16 * MT;
  constexpr int ENT = 4 * KP;  // entries per (warp, row)
  const int per_warp = BM * ENT * 8;
  const int smem_alloc = max(D * BM * 2, NW * BM * KP * 8);  // see fwd_topk.cu
  const int wpp = max(1, min(NW, smem_alloc / per_warp));
  float* sv = reinterpret_cast<float*>(smem_raw);
  int* sid = reinterpret_cast<int*>(sv + wpp * BM * ENT);
  r.init();
  for (int w0 = 0; w0 < NW; w0 += wpp) {
    __syncthreads();
    if (warp >= w0 && warp < w0 + wpp) {
      const int wl = warp - w0;
#pragma unroll
      for (int i = 0; i < MT * 2; i++) {
        const int row = (i >> 1) * 16 + (i & 1) * 8 + (lane >> 2);
        const int base = ((wl * BM + row) * 4 + (lane & 3)) * KP;
#pragma unroll
        for (int k = 0; k < KP; k++) {
          sv[base + k] = st[i].v[k];
          sid[base + k] = st[i].id[k];
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < BM) {
      const int row = threadIdx.x;
      const int nw = min(wpp, NW - w0);
      // entries below the row's shared lower bound cannot be in the top-KP:
      // a cheap scan, the KP-wide insertion only for the few that pass
      const float lb = ord_dec(sh_thr[row]);
#pragma unroll 1
      for (int e = 0; e < nw * ENT; e++) {
        const int wl = e / ENT, q = e % ENT;
        const int o = (wl * BM + row) * ENT + q;
        const int s = sid[o];
        const float x = sv[o];
        if (s != INT_MAX && x >= lb && r.beats_worst(x, s)) r.replace_worst_if(true, x, s);
      }
    }
  }
  __syncthreads();
}

template <int KP, int C, int MT, int NW, bool PACKED>
__global__ void __launch_bounds__(NW * 32) fwd_topk_kernel(TopkArgs ta) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const FwdArgs& a = ta.f;
  bf16* slab = reinterpret_cast<bf16*>(smem_raw);
  constexpr int BM = 16 * MT;
  __shared__ int sh_thr[BM];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int t0 = blockIdx.x * BM, u = blockIdx.y, ps = blockIdx.z;
  const int PS = gridDim.z;
  const int g = u / a.S, sh = u % a.S;
  const int n_g = a.counts[g];
  const int b_lo = sh * a.BPS + ps * a.BPSP;
  const int b_hi = min(min(b_lo + a.BPSP, (sh + 1) * a.BPS), a.NB);
  const int b_full = min(max(n_g / C, b_lo), b_hi);
  const int b_any = min(max((n_g + C - 1) / C, b_lo), b_hi);
  const long long gbase = (long long)g * a.NB;

  if (threadIdx.x < BM) sh_thr[threadIdx.x] = ord_enc(-INFINITY);
  stage_slab<MT>(a, slab, t0);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);

  TopkList<KP> st[MT * 2];
  float thr[MT * 2];
#pragma unroll
  for (int i = 0; i < MT * 2; i++) { st[i].init(); thr[i] = -INFINITY; }
#define TOPK_REFRESH()                                                   \
  _Pragma("unroll") for (int i = 0; i < MT * 2; i++) {                   \
    const int row = (i >> 1) * 16 + (i & 1) * 8 + (lane >> 2);           \
    thr[i] = ord_dec(*(volatile int*)(sh_thr + row));                    \
  }

  if constexpr (PACKED) {
    constexpr int UB = Unit<C>::UB;
    const long long ubase = (gbase + b_lo) / UB;
    const int nu_full = (b_full - b_lo) / UB;
    const int bt = b_lo + nu_full * UB;
    // pass 1 (Max cost): per-lane row maxima.  Lanes cover disjoint neuron
    // sets, so the KP-th largest of the 4*NW lane maxima of a row is a valid
    // lower bound of that row's KP-th largest value -- and close to it.
    {
      float lmax[MT * 2];
#pragma unroll
      for (int i = 0; i < MT * 2; i++) lmax[i] = -INFINITY;
      Unit<C> nxt;
      if (warp < nu_full) load_unit<C>(nxt, a, ubase + warp, lane);
      for (int uu = warp; uu < nu_full; uu += NW) {
        Unit<C> cur = nxt;
        if (uu + NW < nu_full) load_unit<C>(nxt, a, ubase + uu + NW, lane);
#pragma unroll
        for (int bu = 0; bu < UB; bu++) {
          BlockRegs<C> blk;
          unit_block<C>(blk, cur, bu);
          topk_max_block<C, MT, false>(lmax, blk, b_lo + uu * UB + bu, n_g, slab_s, lane);
        }
      }
      if (bt < b_any && warp == nu_full % NW) {
        Unit<C> cur;
        load_unit<C>(cur, a, ubase + nu_full, lane);
#pragma unroll
        for (int bu = 0; bu < UB; bu++)
          if (bt + bu < b_any) {
            BlockRegs<C> blk;
            unit_block<C>(blk, cur, bu);
            topk_max_block<C, MT, true>(lmax, blk, bt + bu, n_g, slab_s, lane);
          }
      }
      // KP-th largest of the row's lane maxima (shared memory after the slab)
      float* smax = reinterpret_cast<float*>(smem_raw + (size_t)a.D * BM * 2);
#pragma unroll
      for (int i = 0; i < MT * 2; i++) {
        const int row = (i >> 1) * 16 + (i & 1) * 8 + (lane >> 2);
        smax[(row * NW + warp) * 4 + (lane & 3)] = lmax[i];
      }
      __syncthreads();
      if (threadIdx.x < BM) {
        TopkList<KP> m;
        m.init();
#pragma unroll 1
        for (int e = 0; e < NW * 4; e++) {
          const float x = smax[threadIdx.x * NW * 4 + e];
          if (x > -INFINITY && m.beats_worst(x, e)) m.replace_worst_if(true, x, e);
        }
        if (m.mn > -INFINITY) atomicMax(sh_thr + threadIdx.x, ord_enc(m.mn));
      }
      __syncthreads();
    }
    // pass 2: exact streaming top-KP, filtered by the seeded threshold
    {
      Unit<C> nxt;
      if (warp < nu_full) load_unit<C>(nxt, a, ubase + warp, lane);
      for (int uu = warp; uu < nu_full; uu += NW) {
        Unit<C> cur = nxt;
        if (uu + NW < nu_full) load_unit<C>(nxt, a, ubase + uu + NW, lane);
        TOPK_REFRESH();
#pragma unroll
        for (int bu = 0; bu < UB; bu++) {
          BlockRegs<C> blk;
          unit_block<C>(blk, cur, bu);
          topk_block<KP, C, MT, false>(st, thr, blk, b_lo + uu * UB + bu, n_g, slab_s, lane, sh_thr);
        }
      }
    }
    if (bt < b_any && warp == nu_full % NW) {
      Unit<C> cur;
      load_unit<C>(cur, a, ubase + nu_full, lane);
      TOPK_REFRESH();
#pragma unroll
      for (int bu = 0; bu < UB; bu++)
        if (bt + bu < b_any) {
          BlockRegs<C> blk;
          unit_block<C>(blk, cur, bu);
          topk_block<KP, C, MT, true>(st, thr, blk, bt + bu, n_g, slab_s, lane, sh_thr);
        }
    }
  } else {
    BlockRegs<C> nxt;
    const int j0 = b_lo + warp;
    if (j0 < b_full) load_block<C>(nxt, a, gbase + j0, lane);
    for (int j = j0; j < b_full; j += NW) {
      BlockRegs<C> cur = nxt;
      if (j + NW < b_full) load_block<C>(nxt, a, gbase + j + NW, lane);
      TOPK_REFRESH();
      topk_block<KP, C, MT, false>(st, thr, cur, j, n_g, slab_s, lane, sh_thr);
    }
    if (b_any > b_full && warp == (b_full - b_lo) % NW) {
      BlockRegs<C> cur;
      load_block<C>(cur, a, gbase + b_full, lane);
      TOPK_REFRESH();
      topk_block<KP, C, MT, true>(st, thr, cur, b_full, n_g, slab_s, lane, sh_thr);
    }
  }

#undef TOPK_REFRESH
  // ---- final CTA-wide merge
  __shared__ int sh_cnt[BM];
  TopkList<KP> r;
  if (!topk_cta_merge_compact<KP, MT, NW>(st, r, smem_raw, a.D, lane, sh_thr, sh_cnt))
    topk_cta_merge<KP, MT, NW>(st, r, smem_raw, a.D, lane, warp, sh_thr);
  if (threadIdx.x < BM) {
    const int row = threadIdx.x;
    const int tok = t0 + row;
    if (tok < a.B) {
      const long long o = (long long)tok * a.U + u;
      if (PS == 1) {
        const int n_u = min(max(n_g - sh * a.BPS * C, 0), a.BPS * C);
        const int kk = min(n_u, KP);
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < KP; k++)
          if (r.id[k] != INT_MAX) {
            float x = r.v[k];
            sum += ta.act == A_RELU ? fmaxf(x, 0.f) : (ta.act == A_GELU ? act_f<A_GELU>(x) : x);
          }
        ta.st_pre[o] = sum / (float)kk;
        if (ta.st_slot) {
#pragma unroll
          for (int k = 0; k < KP; k++) ta.st_slot[o * KP + k] = r.id[k] == INT_MAX ? -1 : r.id[k];
        }
      } else {
        const long long ok = (((long long)ps * a.B + tok) * a.U + u) * KP;
#pragma unroll
        for (int k = 0; k < KP; k++) {
          long long key;
          if (r.id[k] == INT_MAX) key = LLONG_MIN;
          else key = ((long long)ord_enc(r.v[k]) << 32) | (long long)(4294967295LL - (long long)r.id[k]);
          ta.st_keys[ok + k] = key;
        }
      }
    }
  }
}
