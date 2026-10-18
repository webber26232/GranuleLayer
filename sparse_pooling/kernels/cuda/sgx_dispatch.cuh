#pragma once
// launch template of the experimental grouped dX scatter variants (sparse_pool_sg_dxs.cuh)
#include "sg_dispatch.cuh"
#include "sparse_pool_sg_dxs.cuh"

// experimental grouped C = 8 dX scatter variants (sparse_pool_sg_dxs.cuh); vector layouts and
// the owner variant need warp-private copies
template <int ACT, int C, int MT, int NW, bool SPLIT, int EPI, int SC, int NS>
static int sgx_dx_launch(const SgArgs& a, int gx, int gy, cudaStream_t s) {
  static size_t configured = 0;
  if (a.dx_group_warps < 1 || NW % a.dx_group_warps) return -4;
  if (SC != SC_SWZ4 && a.dx_group_warps != 1) return -4;  // compile-time warp-private variants
  if (SC == SC_OWNER && a.S > 8 * NS) return -5;
  const size_t BM = 16 * MT;
  size_t smem = sg_map_bytes(a) + (size_t)a.S * BM * 2 + (size_t)(NW / a.dx_group_warps) * a.S * BM * 4;
  return sg_launch(sgx_dx_kernel<ACT, C, MT, NW, SPLIT, EPI, SC, NS>, dim3(gx, gy), NW * 32, smem, a, s, configured);
}
