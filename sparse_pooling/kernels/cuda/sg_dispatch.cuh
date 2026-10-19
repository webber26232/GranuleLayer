#pragma once
// launch templates of the input-supergroup kernels (sparse_pool_sg.cuh)
#include "sparse_pool_sg.cuh"

static inline size_t sg_map_bytes(const SgArgs& a) { return ((size_t)a.D * 2 + 15) & ~(size_t)15; }

template <typename K>
static int sg_launch(K k, dim3 grid, int threads, size_t smem, const SgArgs& a, cudaStream_t s, size_t& configured) {
  if (smem > 99 * 1024) return -3;  // above the sm_86 per-CTA opt-in limit
  if (smem > configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = smem;
  }
  k<<<grid, threads, smem, s>>>(a);
  return (int)cudaGetLastError();
}

template <int ACT, int C, int MT, int NW>
static int sg_fwd_launch(const SgArgs& a, int gx, int gy, cudaStream_t s) {
  static size_t configured = 0;
  size_t smem = sg_map_bytes(a) + (size_t)a.S * 16 * MT * 2;
  return sg_launch(sg_fwd_kernel<ACT, C, MT, NW>, dim3(gx, gy), NW * 32, smem, a, s, configured);
}

template <int ACT, int C, int MT, int NW, bool SPLIT, int EPI>
static int sg_dx_launch(const SgArgs& a, int gx, int gy, cudaStream_t s) {
  static size_t configured = 0;
  if (a.dx_group_warps < 1 || NW % a.dx_group_warps) return -4;
  const size_t BM = 16 * MT;
  size_t smem = sg_map_bytes(a) + (size_t)a.S * BM * 2 + (size_t)(NW / a.dx_group_warps) * a.S * BM * 4;
  return sg_launch(sg_dx_kernel<ACT, C, MT, NW, SPLIT, EPI>, dim3(gx, gy), NW * 32, smem, a, s, configured);
}

template <int ACT, int C, int NW, int WPW, int TS, int ST, bool SPLIT, int MINB, int EPI, int PROJ = 0>
static int sg_dw_launch(const SgArgs& a, int gx, cudaStream_t s) {
  static size_t configured = 0;
  size_t smem = sg_map_bytes(a) + (size_t)ST * TS * a.S * 16 * 2;
  return sg_launch(sg_dw_kernel<ACT, C, NW, WPW, TS, ST, SPLIT, MINB, EPI, PROJ>, dim3(gx), NW * 32, smem, a, s,
                   configured);
}

// dW tilings: cfg -> (warps, units per warp, 16-token tiles per stage, stages, min CTAs/SM)
#define SG_DW_CFGS(X)       \
  X(0, 8, 2, 1, 3, 2)       \
  X(1, 8, 4, 1, 3, 1)       \
  X(2, 8, 2, 4, 2, 2)       \
  X(3, 8, 4, 4, 2, 1)
