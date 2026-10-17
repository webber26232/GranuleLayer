#pragma once
// backward launch templates (instantiated per pooling op in bwd_<op>.cu)
#include "sparse_pool_bwd.cuh"

template <int POOL, int ACT, int C, int NW, bool SPLIT, int PI, int EPI = 0>
static int launch_dx_v(const BwdArgs& a, int gx, int gy, cudaStream_t s) {
  auto k = k_dx<POOL, ACT, C, NW, SPLIT, PI, EPI>;
  size_t smem = (size_t)a.D * 16 * (2 + 4);
  static size_t configured = 0;  // per instantiation: grow the opt-in as needed
  if (smem > configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = smem;
  }
  k<<<dim3(gx, gy), NW * 32, smem, s>>>(a);
  return (int)cudaGetLastError();
}

// fast-epilogue variants are instantiated only where they differ (FastDz<POOL, ACT>::on)
template <int POOL, int ACT, int C, int NW, bool SPLIT, int PI>
static int launch_dx(const BwdArgs& a, int gx, int gy, cudaStream_t s) {
  if constexpr (FastDz<POOL, ACT>::on) {
    if (a.flags & BWD_MASK_DX_EPI) return launch_dx_v<POOL, ACT, C, NW, SPLIT, PI, 2>(a, gx, gy, s);
    if (a.flags & BWD_FAST_DX_EPI) return launch_dx_v<POOL, ACT, C, NW, SPLIT, PI, 1>(a, gx, gy, s);
  }
  return launch_dx_v<POOL, ACT, C, NW, SPLIT, PI, 0>(a, gx, gy, s);
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

template <int POOL, int ACT, int C, int NW, int WPW, int ST, bool SPLIT, int PI, int MINB, int EPI>
static int launch_dw_idx_v(const BwdArgs& a, int gx, int gy, cudaStream_t s) {
  auto k = k_dw_idx<POOL, ACT, C, NW, WPW, ST, SPLIT, PI, MINB, EPI>;
  size_t smem = (size_t)NW * ST * (WPW * Unit<C>::UB * 8) * 16 * 2;  // per-warp ring of compact slabs
  static size_t configured = 0;
  if (smem > configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = smem;
  }
  k<<<dim3(gx, gy), NW * 32, smem, s>>>(a);
  return (int)cudaGetLastError();
}

template <int POOL, int ACT, int C, int NW, int WPW, int ST, bool SPLIT, int PI, int MINB = 1>
static int launch_dw_idx(const BwdArgs& a, int gx, int gy, cudaStream_t s) {
  if constexpr (FastDz<POOL, ACT>::on) {
    if (a.flags & BWD_MASK_DW_EPI) return launch_dw_idx_v<POOL, ACT, C, NW, WPW, ST, SPLIT, PI, MINB, 2>(a, gx, gy, s);
    if (a.flags & BWD_FAST_DW_EPI) return launch_dw_idx_v<POOL, ACT, C, NW, WPW, ST, SPLIT, PI, MINB, 1>(a, gx, gy, s);
  }
  return launch_dw_idx_v<POOL, ACT, C, NW, WPW, ST, SPLIT, PI, MINB, 0>(a, gx, gy, s);
}

// kind: 0 = dx (cfg = warps 8|16|32), 1 = dw (cfg = 0: 8 warps x 2 wu, 1: 8 warps x 4 wu, 2: 16 warps x 1 wu),
//       2 / 3 = dw with indexed staging, 2- / 3-stage ring (same cfg, plus cfg 3: 8 warps x 2 wu with
//       __launch_bounds__(256, 2), i.e. <= 128 registers for 2 CTAs/SM; C = 64 only, feature-major X)
template <int POOL, int ACT, int C, bool SPLIT, int PI>
static int bwd_cfg(const BwdArgs& a, int kind, int cfg, int gx, int gy, cudaStream_t s) {
  if (kind == 0) {
    if (cfg == 8) return launch_dx<POOL, ACT, C, 8, SPLIT, PI>(a, gx, gy, s);
    if (cfg == 16) return launch_dx<POOL, ACT, C, 16, SPLIT, PI>(a, gx, gy, s);
    if (cfg == 32) return launch_dx<POOL, ACT, C, 32, SPLIT, PI>(a, gx, gy, s);
  } else if (kind == 2 || kind == 3) {
    if constexpr (C == 64) {
      if (a.sxb != 1) return -2;
#define IDX_CFG(ST)                                                                    \
  if (cfg == 0) return launch_dw_idx<POOL, ACT, C, 8, 2, ST, SPLIT, PI>(a, gx, gy, s);  \
  if (cfg == 1) return launch_dw_idx<POOL, ACT, C, 8, 4, ST, SPLIT, PI>(a, gx, gy, s);  \
  if (cfg == 2) return launch_dw_idx<POOL, ACT, C, 16, 1, ST, SPLIT, PI>(a, gx, gy, s);  \
  if (cfg == 3) return launch_dw_idx<POOL, ACT, C, 8, 2, ST, SPLIT, PI, 2>(a, gx, gy, s);
      if (kind == 2) { IDX_CFG(2) }
      else { IDX_CFG(3) }
#undef IDX_CFG
    }
    return -2;
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
