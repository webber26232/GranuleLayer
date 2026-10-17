#pragma once
// backward launch templates (instantiated per pooling op in bwd_<op>.cu)
#include "sparse_pool_bwd.cuh"

template <int POOL, int ACT, int C, int NW, bool SPLIT, int PI>
static int launch_dx(const BwdArgs& a, int gx, int gy, cudaStream_t s) {
  auto k = k_dx<POOL, ACT, C, NW, SPLIT, PI>;
  size_t smem = (size_t)a.D * 16 * (2 + 4);
  static size_t configured = 0;  // per instantiation: grow the opt-in as needed
  if (smem > configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = smem;
  }
  k<<<dim3(gx, gy), NW * 32, smem, s>>>(a);
  return (int)cudaGetLastError();
}

template <int POOL, int ACT, int C, int NW, int WPW, bool SPLIT, int PI>
static int launch_dw(const BwdArgs& a, int gx, int gy, cudaStream_t s) {
  auto k = k_dw<POOL, ACT, C, NW, WPW, SPLIT, PI>;
  size_t smem = (size_t)a.D * 16 * 2 * 2;
  static size_t configured = 0;  // per instantiation: grow the opt-in as needed
  if (smem > configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = smem;
  }
  k<<<dim3(gx, gy), NW * 32, smem, s>>>(a);
  return (int)cudaGetLastError();
}

// kind: 0 = dx (cfg = warps 8|16), 1 = dw (cfg = 0: 8 warps x 2 wu, 1: 8 warps x 4 wu, 2: 16 warps x 1 wu)
template <int POOL, int ACT, int C, bool SPLIT, int PI>
static int bwd_cfg(const BwdArgs& a, int kind, int cfg, int gx, int gy, cudaStream_t s) {
  if (kind == 0) {
    if (cfg == 8) return launch_dx<POOL, ACT, C, 8, SPLIT, PI>(a, gx, gy, s);
    if (cfg == 16) return launch_dx<POOL, ACT, C, 16, SPLIT, PI>(a, gx, gy, s);
  } else {
    if (cfg == 0) return launch_dw<POOL, ACT, C, 8, 2, SPLIT, PI>(a, gx, gy, s);
    if (cfg == 1) return launch_dw<POOL, ACT, C, 8, 4, SPLIT, PI>(a, gx, gy, s);
    if (cfg == 2) return launch_dw<POOL, ACT, C, 16, 1, SPLIT, PI>(a, gx, gy, s);
  }
  return -1;
}

template <int POOL, int ACT, int PI>
static int bwd_c(const BwdArgs& a, int C, int split, int kind, int cfg, int gx, int gy, cudaStream_t s) {
#define CASE(CC)                                                                        \
  if (C == CC) {                                                                        \
    if (split) return bwd_cfg<POOL, ACT, CC, true, PI>(a, kind, cfg, gx, gy, s);       \
    return bwd_cfg<POOL, ACT, CC, false, PI>(a, kind, cfg, gx, gy, s);                 \
  }
  CASE(8) CASE(16) CASE(32) CASE(64)
#undef CASE
  return -1;
}

template <int POOL>
static int bwd_act(const BwdArgs& a, int act, int C, int split, int kind, int cfg, int gx, int gy, cudaStream_t s) {
  if (act == A_ID) return bwd_c<POOL, A_ID, 0>(a, C, split, kind, cfg, gx, gy, s);
  if (act == A_RELU) return bwd_c<POOL, A_RELU, 0>(a, C, split, kind, cfg, gx, gy, s);
  return -1;
}
