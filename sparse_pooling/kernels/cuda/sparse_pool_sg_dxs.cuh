// Experimental local-accumulation ("scatter") variants of the grouped C = 8 dX
// (results/C8_DX_SCATTER_OPTIMIZATION.md).  The baseline sg_dx_kernel (sparse_pool_sg.cuh) is
// left untouched as the control; these kernels share its staging, gather, coefficient and dA
// arithmetic (sg_stage, sg_gather, sg_rowcoef, sg_dx_tile).
//
// Per block and m16 token tile the MMA leaves dA in accumulator layout: lane (t = lane / 4,
// c = lane % 4) holds tokens t, t + 8 of fan-in positions 2c (f0) and 2c + 1 (f1).  A 64-bit
// shared access is served per half-warp (t in 0..3 or 4..7: 4 features x 4 token pairs) and a
// 128-bit access per quarter-warp (4 features x 2 token pairs), so the bank conflict degree of
// one phase is the largest number of its 4 features that the layout maps to the same 8-bank
// group.  All layouts below are bijections of the [S][BM] FP32 slab of one copy:
//
//   SC_BASE  sg_dxs_off: float2 (t, t + 8) pairs, 2 bank groups (feature parity)
//   SC_SWZ4  float2 pairs, 4 bank groups per (32-token block, half-warp): xor with f & 3
//   SC_BASEP / SC_SWZ4P  the same two layouts compiled for warp-private copies only (the
//            baseline kernel selects private / shared CAS at run time per tile: controls)
//   SC_VEC4  float4 records {tile 2i: t, t + 8, tile 2i + 1: t, t + 8} (one load/add/store for
//            two m16 tiles), 4 bank groups per quarter-warp: xor with f & 3
//   SC_VEC4P the float4 records with 2 bank groups (xor with 2 (f & 1)): layout control
//   SC_VEC4T the float4 records after an 8-shuffle transpose per two tiles: lane (u, k) of
//            quarter-warp k takes the records of fan-in position 2k (2k + 1) at token u from
//            lane 4u + k, so each 128-bit phase touches the 8 records of ONE feature: no bank
//            conflicts for any layout (records in token order)
//   SC_VEC4H / SC_VEC4TH  SC_VEC4 / SC_VEC4T with the record addresses computed once per block
//            (the offset is linear in the tile pair: base + 32 mp floats) instead of per record
//            (NVCC re-materialises the copy base from the thread id in the unrolled loop)
//   SC_OWNER no slab read-modify-write: the warp keeps dX[tokens, S features] in MMA
//            accumulators.  Per block the W^T fragment is routed by one shuffle per n8 tile of
//            local features (column n takes fan-in position p with feat[p] = n, or 0), so the
//            tensor core adds the block's contributions to all S features; the accumulators are
//            stored once per CTA into the warp's private copy.  S / 8 gradient MMAs per pass
//            (instead of 1): FP32 summation order differs from the baseline (accumulate inside
//            the MMA instead of a separate add).
// All but SC_OWNER keep every (feature, token)'s sequence of additions; all but SC_SWZ4 need
// warp-private copies (dx_group_warps == 1).
#pragma once
#include "sparse_pool_sg.cuh"

enum { SC_BASE = 0, SC_SWZ4 = 1, SC_VEC4 = 2, SC_VEC4P = 3, SC_OWNER = 4, SC_BASEP = 5, SC_SWZ4P = 6, SC_VEC4T = 7, SC_VEC4H = 8,
       SC_VEC4TH = 9 };
// slab layout of a variant (the hoisted variants share the vector layouts)
template <int SC>
struct SgxLayout { static constexpr int L = SC == SC_VEC4H ? SC_VEC4 : (SC == SC_VEC4TH ? SC_VEC4T : SC); };

// float offset of (local feature f, local token t) inside one FP32 slab copy (MT >= 2)
template <int SC_, int MT>
__device__ __forceinline__ int sgx_off(int f, int t) {
  constexpr int SC = SgxLayout<SC_>::L;
  if constexpr (SC == SC_BASE || SC == SC_BASEP || SC == SC_OWNER) {
    return sg_dxs_off<MT>(f, t);
  } else {
    const int seg = t >> 4, tp = t & 7, h = (t >> 3) & 1;
    if constexpr (SC == SC_SWZ4 || SC == SC_SWZ4P) {
      const int grp = (((seg & 1) << 1) | (tp >> 2)) ^ (f & 3);
      return f * 16 * MT + 32 * (seg >> 1) + 8 * grp + 2 * (tp & 3) + h;
    } else if constexpr (SC == SC_VEC4T) {
      return f * 16 * MT + 32 * (seg >> 1) + 4 * tp + 2 * (seg & 1) + h;
    } else {
      const int sw = (SC == SC_VEC4) ? (f & 3) : ((f & 1) << 1);
      return f * 16 * MT + 32 * (seg >> 1) + 8 * ((tp >> 1) ^ sw) + 4 * (tp & 1) + 2 * (seg & 1) + h;
    }
  }
}

// BF16 A operands (hi, lo) of the gradient MMA of one C = 8 block for one m16 tile: exactly
// the operands sg_dx_tile feeds its MMAs (mask epilogue: hi / lo of one[token] & (z > 0))
template <int ACT, bool MASKED, int EPI>
__device__ __forceinline__ void sgx_dz_c8(uint32_t (&ah)[2], uint32_t (&al)[2], const uint32_t* af,
                                          const BlockRegs<8>& blk, int j, int n_g, const RowCoef& rc, int lane) {
  constexpr int POOL = P_MEAN, P_INT = 0;
  const int col0 = 2 * (lane & 3);
  if constexpr (EPI == 2 && FastDz<POOL, ACT, P_INT>::mask) {
    uint32_t m0[4];
    tile_mask<ACT, MASKED>(m0, af, blk.w[0], blk.bb[0], j * 8 + col0, n_g);
    const uint32_t q00 = pair_mask(m0[0], m0[1]), q01 = pair_mask(m0[2], m0[3]);
    ah[0] = rc.hp[0] & q00; ah[1] = rc.hp[1] & q01;
    al[0] = rc.lp[0] & q00; al[1] = rc.lp[1] & q01;
  } else {
    float dz[4];
    tile_dz<POOL, ACT, P_INT, MASKED, EPI>(dz, af, blk.w[0], blk.bb[0], rc, j * 8 + col0, n_g, 1.f, 0.f);
    split_bf16(dz[0], dz[1], ah[0], al[0]);
    split_bf16(dz[2], dz[3], ah[1], al[1]);
  }
}

// owner routing: the fan-in position p of local feature n in this block (all lanes hold the
// block's 8 local rows packed: 4 bits each for S <= 16, 8 bits for S <= 32); -1 if absent
template <int NS>
struct SgxInv {
  uint32_t lo, hi;
  __device__ __forceinline__ void build(int fidx, int lane) {
    const int k = lane & 7;
    if constexpr (NS <= 2) {
      lo = (uint32_t)fidx << (4 * k);
      lo |= __shfl_xor_sync(0xffffffff, lo, 1);
      lo |= __shfl_xor_sync(0xffffffff, lo, 2);
      lo |= __shfl_xor_sync(0xffffffff, lo, 4);
      hi = 0;
    } else {
      const uint32_t b = (uint32_t)fidx << (8 * (k & 3));
      lo = (k < 4) ? b : 0u;
      hi = (k < 4) ? 0u : b;
#pragma unroll
      for (int s = 1; s <= 4; s <<= 1) {
        lo |= __shfl_xor_sync(0xffffffff, lo, s);
        hi |= __shfl_xor_sync(0xffffffff, hi, s);
      }
    }
  }
  __device__ __forceinline__ int find(int n) const {
    if constexpr (NS <= 2) {  // zero-nibble search; the lowest flagged nibble is exact
      const uint32_t x = lo ^ ((uint32_t)n * 0x11111111u);
      const uint32_t z = (x - 0x11111111u) & ~x & 0x88888888u;
      return z ? (__ffs(z) >> 2) - 1 : -1;
    } else {
      const uint32_t nn = (uint32_t)n * 0x01010101u;
      const uint32_t x0 = lo ^ nn, x1 = hi ^ nn;
      const uint32_t z0 = (x0 - 0x01010101u) & ~x0 & 0x80808080u;
      const uint32_t z1 = (x1 - 0x01010101u) & ~x1 & 0x80808080u;
      return z0 ? (__ffs(z0) >> 3) - 1 : (z1 ? (__ffs(z1) >> 3) + 3 : -1);
    }
  }
};

template <int ACT, int C, int MT, int NW, bool SPLIT, int EPI, int SC, int NS>
__device__ __forceinline__ void sgx_dx_body(const SgArgs& a) {
  static_assert(C == 8 && MT >= 2, "scatter variants: C = 8, >= 2 m16 tiles");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int BM = 16 * MT, UB = Unit<C>::UB;
  constexpr bool VEC = SC == SC_VEC4 || SC == SC_VEC4P || SC == SC_VEC4T || SC == SC_VEC4H || SC == SC_VEC4TH;
  constexpr bool TRANS = SgxLayout<SC>::L == SC_VEC4T;  // shuffle transpose before the record update
  constexpr bool HOIST = SC == SC_VEC4H || SC == SC_VEC4TH;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* slab = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  float* dxs = reinterpret_cast<float*>(smem_raw + sg_align16(a.D * 2) + a.S * BM * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int t0 = blockIdx.y * BM;
  constexpr bool CT_PRIV = SC != SC_SWZ4;  // compile-time warp-private (the launch checks dx_group_warps == 1)
  const int WG = CT_PRIV ? 1 : a.dx_group_warps;
  const int ncopy = NW / WG;
  const bool priv = WG == 1;
  const int16_t* fq = a.feat + (long long)q * a.S;
  FwdArgs fa;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;
  if (SC != SC_OWNER)  // the owner variant stores every element of its copy once
    for (int i = threadIdx.x; i < ncopy * a.S * BM; i += NW * 32) dxs[i] = 0.f;
  sg_stage<MT>(a, fq, smap, slab, t0);
  __syncthreads();
  const uint32_t slab_s = smem_u32(slab);
  float* mys = dxs + (warp / WG) * a.S * BM;
  const int t = lane >> 2;
  float acc[SC == SC_OWNER ? MT : 1][SC == SC_OWNER ? NS : 1][4];
  if constexpr (SC == SC_OWNER) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int ns = 0; ns < NS; ns++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[mt][ns][e] = 0.f;
  }

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
    const uint32_t wt = movmatrix_trans(blk.w[0]);  // W^T fragment: B[k = neuron][n = fan-in position]
    if constexpr (SC == SC_OWNER) {
      SgxInv<NS> inv;
      inv.build(blk.fidx, lane);
      uint32_t wx[NS];  // B[k = neuron][n = local feature 8 ns + t]
#pragma unroll
      for (int ns = 0; ns < NS; ns++) {
        const int p = inv.find(8 * ns + t);
        const uint32_t v = __shfl_sync(0xffffffff, wt, ((p & 7) << 2) | (lane & 3));
        wx[ns] = p >= 0 ? v : 0u;
      }
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        uint32_t ah[2], al[2];
        if (full) sgx_dz_c8<ACT, false, EPI>(ah, al, af + 2 * mt, blk, j, n_g, rc[mt], lane);
        else sgx_dz_c8<ACT, true, EPI>(ah, al, af + 2 * mt, blk, j, n_g, rc[mt], lane);
#pragma unroll
        for (int ns = 0; ns < NS; ns++) {
          float* d = acc[mt][ns];
          mma_m16n8k8_bf16(d, ah[0], ah[1], wx[ns], d[0], d[1], d[2], d[3]);
          if (SPLIT) mma_m16n8k8_bf16(d, al[0], al[1], wx[ns], d[0], d[1], d[2], d[3]);
        }
      }
    } else {
      const int col0 = TRANS ? 2 * (lane >> 3) : 2 * (lane & 3);  // (transposed) fan-in positions
      const int f0 = __shfl_sync(0xffffffff, blk.fidx, col0);
      const int f1 = __shfl_sync(0xffffffff, blk.fidx, col0 + 1);
      if constexpr (VEC) {
        float4* q0 = nullptr;
        float4* q1 = nullptr;
        if constexpr (HOIST) {  // record of tile pair mp = q + 8 mp (float4 units)
          q0 = reinterpret_cast<float4*>(mys + sgx_off<SC, MT>(f0, TRANS ? (lane & 7) : t));
          q1 = reinterpret_cast<float4*>(mys + sgx_off<SC, MT>(f1, TRANS ? (lane & 7) : t));
        }
#pragma unroll
        for (int mp = 0; mp < MT / 2; mp++) {
          float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
          if (full) {
            sg_dx_tile<ACT, C, SPLIT, false, EPI>(d0, af + 4 * mp, blk, &wt, j, n_g, rc[2 * mp], lane);
            sg_dx_tile<ACT, C, SPLIT, false, EPI>(d1, af + 4 * mp + 2, blk, &wt, j, n_g, rc[2 * mp + 1], lane);
          } else {
            sg_dx_tile<ACT, C, SPLIT, true, EPI>(d0, af + 4 * mp, blk, &wt, j, n_g, rc[2 * mp], lane);
            sg_dx_tile<ACT, C, SPLIT, true, EPI>(d1, af + 4 * mp + 2, blk, &wt, j, n_g, rc[2 * mp + 1], lane);
          }
          // 16-byte aligned records (token t of the even tile): this warp owns the copy and a
          // record's features are distinct, so no two lanes touch the same record
          const int tr = TRANS ? (lane & 7) : t;  // token of this lane's records
          if constexpr (TRANS) {
            const int src = 4 * (lane & 7) + (lane >> 3);
#pragma unroll
            for (int e = 0; e < 4; e++) {
              d0[e] = __shfl_sync(0xffffffff, d0[e], src);
              d1[e] = __shfl_sync(0xffffffff, d1[e], src);
            }
          }
          float4* p0 = HOIST ? q0 + 8 * mp : reinterpret_cast<float4*>(mys + sgx_off<SC, MT>(f0, 32 * mp + tr));
          float4* p1 = HOIST ? q1 + 8 * mp : reinterpret_cast<float4*>(mys + sgx_off<SC, MT>(f1, 32 * mp + tr));
          float4 v0 = *p0, v1 = *p1;
          v0.x += d0[0]; v0.y += d0[2]; v0.z += d1[0]; v0.w += d1[2];
          v1.x += d0[1]; v1.y += d0[3]; v1.z += d1[1]; v1.w += d1[3];
          *p0 = v0;
          *p1 = v1;
        }
      } else {
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          float da[4] = {0.f, 0.f, 0.f, 0.f};
          if (full) sg_dx_tile<ACT, C, SPLIT, false, EPI>(da, af + 2 * mt, blk, &wt, j, n_g, rc[mt], lane);
          else sg_dx_tile<ACT, C, SPLIT, true, EPI>(da, af + 2 * mt, blk, &wt, j, n_g, rc[mt], lane);
          float* p0 = mys + sgx_off<SC, MT>(f0, 16 * mt + t);
          float* p1 = mys + sgx_off<SC, MT>(f1, 16 * mt + t);
          if (priv) {
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
    }
  }
  SG_UNIT_LOOP_END
  SG_ITEM_LOOP_END
  if constexpr (SC == SC_OWNER) {  // accumulators -> this warp's copy (every element written once)
    const int col0 = 2 * (lane & 3);
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int ns = 0; ns < NS; ns++) {
        const int n0 = 8 * ns + col0;
        if (n0 < a.S)
          *reinterpret_cast<float2*>(mys + sg_dxs_off<MT>(n0, 16 * mt + t)) = make_float2(acc[mt][ns][0], acc[mt][ns][2]);
        if (n0 + 1 < a.S)
          *reinterpret_cast<float2*>(mys + sg_dxs_off<MT>(n0 + 1, 16 * mt + t)) =
              make_float2(acc[mt][ns][1], acc[mt][ns][3]);
      }
  }
  __syncthreads();
  // reduce the copies and flush: one global FP32 atomic per (feature, token) per CTA
  for (int i = threadIdx.x; i < a.S * BM; i += NW * 32) {
    const int f = i / BM, tl = i % BM;
    const int o = sgx_off<SC, MT>(f, tl);
    float v = dxs[o];
    for (int c = 1; c < ncopy; c++) v += dxs[c * a.S * BM + o];
    const int tok = t0 + tl;
    if (tok < a.B && v != 0.f) atomicAdd(a.dXT + (long long)fq[f] * a.B + tok, v);
  }
}

template <int ACT, int C, int MT, int NW, bool SPLIT, int EPI, int SC, int NS>
__global__ void __launch_bounds__(NW * 32) sgx_dx_kernel(SgArgs a) {
  if (EPI && !(a.nonfinite && *a.nonfinite)) sgx_dx_body<ACT, C, MT, NW, SPLIT, EPI, SC, NS>(a);
  else sgx_dx_body<ACT, C, MT, NW, SPLIT, 0, SC, NS>(a);
}
