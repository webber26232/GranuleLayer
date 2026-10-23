// Experimental fused shared-projection backward of the grouped C = 8 Mean(ReLU) layer, split
// gradients (schedule F1 of results/SCALE_DISPATCH_FUSED_BACKWARD.md section 8).
//
// CTA = one dW unit chunk (WPW consecutive units per warp of one supergroup, exactly the
// ownership of sg_dw), all tokens, through the same TS x 16-token cp.async ring.  Per block
// and 16-token tile the projection is evaluated ONCE, in the dX layout (rows = tokens):
//   z = mma(A = X tile (ldmatrix.trans, = dw_block's xa), B = packed weight fragment, C = bias)
//   m = (z > 0) masks (FSET.BF.GT.FTZ, as tile_mask), with the MASKED (ragged output) rule
//   dX:  dA = (hp & m, lp & m) W^T (2 HMMA.1688, the sg_dx_tile mask epilogue), scattered into
//        this warp's private FP32 window slab [S][16 TS] (sgx_off<SC_SWZ4P>, float2 RMW)
//   dW:  the 16-bit mask pairs are transposed into the dW A layout (rows = neurons) with one
//        movmatrix per 8-token half; the A operand (cc.hp / cc.lp & mask), the db sums
//        (one & mask, expanded from the 16-bit masks) and the HMMA.16816 are those of
//        dw_block's mask epilogue, which is bitwise equal to the fast epilogue of sg_dw: same
//        operands, same token order, same unit -> warp ownership -> dW / db BITWISE equal to
//        the separate grouped dW.
// SHARE = false is the recompute-preserving control of the same schedule: the dW masks come
// from a second, dW-layout projection (2 HMMA.1688 per block-tile, exactly dw_block's).
// After every ring stage (16 TS tokens) the warps' slabs are reduced in copy order and flushed
// with one FP32 atomicAdd per nonzero (feature, token) into dXT [D, B], then re-zeroed.  dX
// changes FP32 summation order vs the separate dX (one partial per 16-unit CTA instead of per
// 4096-unit chunk): characterized, never bitwise.
// Non-finite coefficients: the whole kernel exits (a.nonfinite) and the guarded fallback
// launches (generic sg_dx / sg_dw, BWD_FALLBACK) produce the gradients: exactly one path runs.
#pragma once
#include "sparse_pool_sg_dxs.cuh"

// Attribution ablations (benchmarks only, WRONG results; separate builds through
// SPARSE_POOL_NVCC_EXTRA=-DSP_ABLATE_SGF=n): 1 no global dX flush atomics, 2 no dX side
// (gradient MMAs, slab RMW, flush), 3 no dW side (masks transpose, db, gradient MMA)
#ifndef SP_ABLATE_SGF
#define SP_ABLATE_SGF 0
#endif

template <int NW, int WPW, int TS, int ST, bool SHARE>
__device__ __forceinline__ void sgf_body(const SgArgs& a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int C = 8, UB = Unit<C>::UB, BM = 16 * TS;
  static_assert(TS >= 2, "window of >= 2 m16 tiles (sgx_off)");
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* ring = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  float* dxs = reinterpret_cast<float*>(smem_raw + sg_align16(a.D * 2) + ST * TS * a.S * 16 * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int S = a.S;
  FwdArgs fa;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;
  for (int i = threadIdx.x; i < S; i += NW * 32) smap[fq[i]] = (int16_t)i;
  for (int i = threadIdx.x; i < NW * S * BM; i += NW * 32) dxs[i] = 0.f;

  Unit<C> un[WPW];
  DwAcc<C> acc[WPW];
  int unit[WPW], gi[WPW], ng[WPW], jb[WPW];
  float biasT[WPW][UB][2];
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
      for (int e = 0; e < 4; e++) acc[i].w[bu][0][e] = 0.f;
      acc[i].b[bu][0][0] = acc[i].b[bu][0][1] = 0.f;
      BlockRegs<C> blk;
      unit_block<C>(blk, un[i], bu);
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {  // as sg_dw_body (the second n8 tile does not exist at C = 8)
        const uint32_t word = (nt < 1) ? blk.bb[nt] : 0u;
        const int gq = lane >> 2;
        const uint32_t src = __shfl_sync(0xffffffff, word, gq >> 1);
        const float2 bf = bf16x2_to_float2(src);
        biasT[i][bu][nt] = (gq & 1) ? bf.y : bf.x;
      }
    }
  }
  __syncthreads();  // smap complete, slabs zeroed
  int lrow[WPW][UB];  // this lane's local slab row (fan-in k = lane & 7) of each block
  uint32_t f01[WPW][UB];  // local features of fan-in positions 2c, 2c + 1 (dX scatter), packed
#pragma unroll
  for (int i = 0; i < WPW; i++)
#pragma unroll
    for (int bu = 0; bu < UB; bu++) {
      BlockRegs<C> blk;
      unit_block<C>(blk, un[i], bu);
      lrow[i][bu] = unit[i] >= 0 ? smap[blk.fidx] : 0;
      const int f0 = __shfl_sync(0xffffffff, lrow[i][bu], 2 * (lane & 3));
      const int f1 = __shfl_sync(0xffffffff, lrow[i][bu], 2 * (lane & 3) + 1);
      f01[i][bu] = (uint32_t)f0 | ((uint32_t)f1 << 16);
    }

  const int n_tiles = (a.B + 15) / 16;
  const int n_st = (n_tiles + TS - 1) / TS;
  const int stage_elems = TS * S * 16;
  float* mys = dxs + warp * S * BM;
  const int t = lane >> 2;
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
  // reduce the NW private slabs of window s (copy order) and flush; re-zero for the next window
  auto flush = [&](int s) {
    const int t0 = s * BM;
    for (int i = threadIdx.x; i < S * BM; i += NW * 32) {
      const int f = i / BM, tl = i % BM;
      const int o = sgx_off<SC_SWZ4P, TS>(f, tl);
      float v = 0.f;
#pragma unroll 4
      for (int c = 0; c < NW; c++) {
        v += dxs[c * S * BM + o];
        dxs[c * S * BM + o] = 0.f;
      }
      const int tok = t0 + tl;
      if (SP_ABLATE_SGF == 1 || SP_ABLATE_SGF == 2) {
        if (v == 1.2345e-30f) a.dXT[0] = v;
      } else if (tok < a.B && v != 0.f) {
        atomicAdd(a.dXT + (long long)fq[f] * a.B + tok, v);
      }
    }
  };
#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < n_st) stage(s);
    cp_async_commit();
  }
  for (int s = 0; s < n_st; s++) {
    cp_async_wait_group<ST - 2>();
    __syncthreads();  // stage s visible; every warp is done with stage s - 1 (ring slot and dX slab)
    if (s > 0) {
      flush(s - 1);
      __syncthreads();  // slabs zeroed before any warp adds window s
    }
    if (s + ST - 1 < n_st) stage(s + ST - 1);
    cp_async_commit();
    const bf16* base = ring + (s % ST) * stage_elems;
#pragma unroll 1
    for (int sub = 0; sub < TS; sub++) {
      const int tt = s * TS + sub;
      if (tt >= n_tiles) break;
      const uint32_t slab_s = smem_u32(base + sub * S * 16);
      ColCoef cc;
      RowCoef rc;
#pragma unroll
      for (int i = 0; i < WPW; i++) {
        if (unit[i] < 0) continue;
        if (i == 0 || gi[i] != gi[i - 1] || unit[i - 1] < 0) {
          sg_colcoef(cc, a.c1T + (long long)gi[i] * a.B, a.B, tt * 16, lane);
          sg_rowcoef(rc, a.c1T + (long long)gi[i] * a.B, a.B, tt * 16, lane);
        }
#pragma unroll
        for (int bu = 0; bu < UB; bu++) {
          const int j = jb[i] + bu;
          if (j * C >= ng[i]) continue;
          const bool masked = (j + 1) * C > ng[i];
          BlockRegs<C> blk;
          unit_block<C>(blk, un[i], bu);
          uint32_t xa[2], xb[2];
          const uint32_t addr = slab_s + 2 * Swz<1>::off(lrow[i][bu], (lane >> 3) & 1);
          ldmatrix_x2_trans(xa, addr);  // = sg_gather<1>: A of the dX-layout projection, B of dw_block's
          ldmatrix_x2(xb, addr);        // B of dW^T
          // ---- the projection, once (dX layout: rows = tokens t, t + 8; cols = neurons 2c, 2c + 1)
          uint32_t m[4];
          if (masked) tile_mask<A_RELU, true>(m, xa, blk.w[0], blk.bb[0], j * C + 2 * (lane & 3), ng[i]);
          else tile_mask<A_RELU, false>(m, xa, blk.w[0], blk.bb[0], j * C + 2 * (lane & 3), ng[i]);
          const uint32_t q00 = pair_mask(m[0], m[1]), q01 = pair_mask(m[2], m[3]);
          // ---- dX: dA = dZ W^T (the sg_dx_tile mask epilogue), private window slab
          if (SP_ABLATE_SGF != 2) {
            const uint32_t wt = movmatrix_trans(blk.w[0]);
            float da[4] = {0.f, 0.f, 0.f, 0.f};
            mma_m16n8k8_bf16(da, rc.hp[0] & q00, rc.hp[1] & q01, wt, da[0], da[1], da[2], da[3]);
            mma_m16n8k8_bf16(da, rc.lp[0] & q00, rc.lp[1] & q01, wt, da[0], da[1], da[2], da[3]);
            const int f0 = (int)(f01[i][bu] & 0xffffu), f1 = (int)(f01[i][bu] >> 16);
            float2* p0 = reinterpret_cast<float2*>(mys + sgx_off<SC_SWZ4P, TS>(f0, 16 * sub + t));
            float2* p1 = reinterpret_cast<float2*>(mys + sgx_off<SC_SWZ4P, TS>(f1, 16 * sub + t));
            float2 v0 = *p0, v1 = *p1;  // this warp owns the slab; a record's features are distinct
            v0.x += da[0]; v0.y += da[2]; v1.x += da[1]; v1.y += da[3];
            *p0 = v0;
            *p1 = v1;
          }
          // ---- dW (dw_block's mask epilogue, C = 8): masks in the A layout (rows = neurons g)
          if (SP_ABLATE_SGF == 3) {
            if ((q00 ^ q01) == 0x12345u) acc[i].b[bu][0][0] += 1.f;
            continue;
          }
          uint32_t A[4];
#pragma unroll
          for (int th = 0; th < 2; th++) {
            uint32_t mq;  // 16-bit mask pair (token 2 (lane & 3) + e of half th, neuron lane / 4)
            if constexpr (SHARE) {
              mq = movmatrix_trans(th == 0 ? q00 : q01);
            } else {  // control: dw_block's own projection
              float d[4];
              mma_m16n8k8_bf16(d, blk.w[0], 0u, xa[th], biasT[i][bu][0], biasT[i][bu][0], biasT[i][bu][1],
                               biasT[i][bu][1]);
              const bool v0 = !masked || j * C + (lane >> 2) < ng[i];
              mq = pair_mask(v0 ? gt0_mask(d[0]) : 0u, v0 ? gt0_mask(d[1]) : 0u);
            }
            const uint32_t me0 = (uint32_t)((int32_t)(mq << 16) >> 31), me1 = (uint32_t)((int32_t)mq >> 31);
            A[2 * th] = cc.hp[th] & mq;
            A[2 * th + 1] = cc.lp[th] & mq;
            const uint32_t o0 = __float_as_uint(cc.one[2 * th]), o1 = __float_as_uint(cc.one[2 * th + 1]);
            acc[i].b[bu][0][0] += __uint_as_float(o0 & me0) + __uint_as_float(o1 & me1);
          }
          // rows 0-7 hi, rows 8-15 lo (the C = 8 packing of dw_block)
          mma_m16n8k16_bf16(acc[i].w[bu][0], A, xb[0], xb[1]);
        }
      }
    }
  }
  cp_async_wait_all();
  __syncthreads();
  if (n_st > 0) flush(n_st - 1);
  sg_dw_writeback<C, WPW>(a, acc, unit, lane);
}

// v2: the same ownership, projection sharing and dW arithmetic, with the loops of a 32-token
// window ordered block-outer / tile-inner (each dW / db accumulator still sees its tiles in token
// order: bitwise unchanged), so a block's decode, W^T transpose and dX record addresses are
// computed once per window, and dX uses the selected dX scatter (sgx SC_VEC4TH: the two tiles'
// dA transposed by 8 shuffles into conflict-free float4 records of one feature).
template <int NW, int WPW, int ST, bool SHARE>
__device__ __forceinline__ void sgf2_body(const SgArgs& a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int C = 8, UB = Unit<C>::UB, TS = 2, BM = 16 * TS;
  int16_t* smap = reinterpret_cast<int16_t*>(smem_raw);
  bf16* ring = reinterpret_cast<bf16*>(smem_raw + sg_align16(a.D * 2));
  float* dxs = reinterpret_cast<float*>(smem_raw + sg_align16(a.D * 2) + ST * TS * a.S * 16 * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = a.chunks[3 * blockIdx.x], lo = a.chunks[3 * blockIdx.x + 1], hi = a.chunks[3 * blockIdx.x + 2];
  const int16_t* fq = a.feat + (long long)q * a.S;
  const int S = a.S;
  FwdArgs fa;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;
  for (int i = threadIdx.x; i < S; i += NW * 32) smap[fq[i]] = (int16_t)i;
  for (int i = threadIdx.x; i < NW * S * BM; i += NW * 32) dxs[i] = 0.f;

  Unit<C> un[WPW];
  DwAcc<C> acc[WPW];
  int unit[WPW], gi[WPW], ng[WPW], jb[WPW];
  float biasT[WPW][SHARE ? 1 : UB][2];
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
      for (int e = 0; e < 4; e++) acc[i].w[bu][0][e] = 0.f;
      acc[i].b[bu][0][0] = acc[i].b[bu][0][1] = 0.f;
      if constexpr (!SHARE) {
        BlockRegs<C> blk;
        unit_block<C>(blk, un[i], bu);
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
          const uint32_t word = (nt < 1) ? blk.bb[nt] : 0u;
          const int gq = lane >> 2;
          const uint32_t src = __shfl_sync(0xffffffff, word, gq >> 1);
          const float2 bf = bf16x2_to_float2(src);
          biasT[i][bu][nt] = (gq & 1) ? bf.y : bf.x;
        }
      }
    }
  }
  __syncthreads();
  int lrow[WPW][UB];
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
  float* mys = dxs + warp * S * BM;
  const int src_lane = 4 * (lane & 7) + (lane >> 3);  // VEC4T transpose source
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
  auto flush = [&](int s) {
    const int t0 = s * BM;
    for (int i = threadIdx.x; i < S * BM; i += NW * 32) {
      const int f = i / BM, tl = i % BM;
      const int o = sgx_off<SC_VEC4TH, TS>(f, tl);
      float v = 0.f;
#pragma unroll 4
      for (int c = 0; c < NW; c++) {
        v += dxs[c * S * BM + o];
        dxs[c * S * BM + o] = 0.f;
      }
      const int tok = t0 + tl;
      if (SP_ABLATE_SGF == 1 || SP_ABLATE_SGF == 2) {
        if (v == 1.2345e-30f) a.dXT[0] = v;
      } else if (tok < a.B && v != 0.f) {
        atomicAdd(a.dXT + (long long)fq[f] * a.B + tok, v);
      }
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
    if (s > 0) {
      flush(s - 1);
      __syncthreads();
    }
    if (s + ST - 1 < n_st) stage(s + ST - 1);
    cp_async_commit();
    const bf16* base = ring + (s % ST) * stage_elems;
    const bool tile1 = s * TS + 1 < n_tiles;  // the window's second tile holds tokens
    RowCoef rc[TS];
    ColCoef cc[TS];
#pragma unroll
    for (int i = 0; i < WPW; i++) {
      if (unit[i] < 0) continue;
      if (i == 0 || gi[i] != gi[i - 1] || unit[i - 1] < 0) {
#pragma unroll
        for (int sub = 0; sub < TS; sub++) {
          sg_rowcoef(rc[sub], a.c1T + (long long)gi[i] * a.B, a.B, (s * TS + sub) * 16, lane);
          sg_colcoef(cc[sub], a.c1T + (long long)gi[i] * a.B, a.B, (s * TS + sub) * 16, lane);
        }
      }
#pragma unroll
      for (int bu = 0; bu < UB; bu++) {
        const int j = jb[i] + bu;
        if (j * C >= ng[i]) continue;
        const bool masked = (j + 1) * C > ng[i];
        BlockRegs<C> blk;
        unit_block<C>(blk, un[i], bu);
        const uint32_t wt = movmatrix_trans(blk.w[0]);
        float d[TS][4];
#pragma unroll
        for (int sub = 0; sub < TS; sub++) {
#pragma unroll
          for (int e = 0; e < 4; e++) d[sub][e] = 0.f;
          if (sub == 1 && !tile1) continue;  // no tokens: dX adds zeros, dW must not see the tile
          const uint32_t slab_s = smem_u32(base + sub * S * 16);
          uint32_t xa[2], xb[2];
          const uint32_t addr = slab_s + 2 * Swz<1>::off(lrow[i][bu], (lane >> 3) & 1);
          ldmatrix_x2_trans(xa, addr);
          ldmatrix_x2(xb, addr);
          uint32_t m[4];
          if (masked) tile_mask<A_RELU, true>(m, xa, blk.w[0], blk.bb[0], j * C + 2 * (lane & 3), ng[i]);
          else tile_mask<A_RELU, false>(m, xa, blk.w[0], blk.bb[0], j * C + 2 * (lane & 3), ng[i]);
          const uint32_t q00 = pair_mask(m[0], m[1]), q01 = pair_mask(m[2], m[3]);
          if (SP_ABLATE_SGF != 2) {
            mma_m16n8k8_bf16(d[sub], rc[sub].hp[0] & q00, rc[sub].hp[1] & q01, wt, d[sub][0], d[sub][1], d[sub][2],
                             d[sub][3]);
            mma_m16n8k8_bf16(d[sub], rc[sub].lp[0] & q00, rc[sub].lp[1] & q01, wt, d[sub][0], d[sub][1], d[sub][2],
                             d[sub][3]);
          }
          if (SP_ABLATE_SGF == 3) {
            if ((q00 ^ q01) == 0x12345u) acc[i].b[bu][0][0] += 1.f;
            continue;
          }
          uint32_t A[4];
#pragma unroll
          for (int th = 0; th < 2; th++) {
            uint32_t mq;
            if constexpr (SHARE) {
              mq = movmatrix_trans(th == 0 ? q00 : q01);
            } else {
              float dd[4];
              mma_m16n8k8_bf16(dd, blk.w[0], 0u, xa[th], biasT[i][bu][0], biasT[i][bu][0], biasT[i][bu][1],
                               biasT[i][bu][1]);
              const bool v0 = !masked || j * C + (lane >> 2) < ng[i];
              mq = pair_mask(v0 ? gt0_mask(dd[0]) : 0u, v0 ? gt0_mask(dd[1]) : 0u);
            }
            const uint32_t me0 = (uint32_t)((int32_t)(mq << 16) >> 31), me1 = (uint32_t)((int32_t)mq >> 31);
            A[2 * th] = cc[sub].hp[th] & mq;
            A[2 * th + 1] = cc[sub].lp[th] & mq;
            const uint32_t o0 = __float_as_uint(cc[sub].one[2 * th]), o1 = __float_as_uint(cc[sub].one[2 * th + 1]);
            acc[i].b[bu][0][0] += __uint_as_float(o0 & me0) + __uint_as_float(o1 & me1);
          }
          mma_m16n8k16_bf16(acc[i].w[bu][0], A, xb[0], xb[1]);
        }
        if (SP_ABLATE_SGF != 2) {  // dX: the two tiles' dA -> one float4 record per (feature, token) (SC_VEC4TH)
#pragma unroll
          for (int e = 0; e < 4; e++) {
            d[0][e] = __shfl_sync(0xffffffff, d[0][e], src_lane);
            d[1][e] = __shfl_sync(0xffffffff, d[1][e], src_lane);
          }
          const int f0 = __shfl_sync(0xffffffff, lrow[i][bu], 2 * (lane >> 3));
          const int f1 = __shfl_sync(0xffffffff, lrow[i][bu], 2 * (lane >> 3) + 1);
          float4* p0 = reinterpret_cast<float4*>(mys + sgx_off<SC_VEC4TH, TS>(f0, lane & 7));
          float4* p1 = reinterpret_cast<float4*>(mys + sgx_off<SC_VEC4TH, TS>(f1, lane & 7));
          float4 v0 = *p0, v1 = *p1;
          v0.x += d[0][0]; v0.y += d[0][2]; v0.z += d[1][0]; v0.w += d[1][2];
          v1.x += d[0][1]; v1.y += d[0][3]; v1.z += d[1][1]; v1.w += d[1][3];
          *p0 = v0;
          *p1 = v1;
        }
      }
    }
  }
  cp_async_wait_all();
  __syncthreads();
  if (n_st > 0) flush(n_st - 1);
  sg_dw_writeback<C, WPW>(a, acc, unit, lane);
}

template <int NW, int WPW, int ST, int MINB, bool SHARE>
__global__ void __launch_bounds__(NW * 32, MINB) sgf2_kernel(SgArgs a) {
  if (a.nonfinite && *a.nonfinite) return;
  sgf2_body<NW, WPW, ST, SHARE>(a);
}

template <int NW, int WPW, int TS, int ST, int MINB, bool SHARE>
__global__ void __launch_bounds__(NW * 32, MINB) sgf_kernel(SgArgs a) {
  if (a.nonfinite && *a.nonfinite) return;  // the guarded generic launches produce the gradients
  sgf_body<NW, WPW, TS, ST, SHARE>(a);
}
