#pragma once
// forward launch templates (instantiated per pooling op in fwd_<op>.cu)
#include "common.cuh"
#include "sparse_pool_fwd.cuh"

template <int POOL, int ACT, int C, int MT, int NW, bool TRACK, int PI, bool PK, int FE = 0>
static int launch_fwd(const FwdArgs& a, int n_tiles, int PS, cudaStream_t s) {
  auto k = fwd_mma_kernel<POOL, ACT, C, MT, NW, TRACK, PI, PK, FE>;
  int BM = 16 * MT;
  size_t smem = (size_t)a.D * BM * 2;
  size_t red = (size_t)4 * NW * BM * 4;
  if (red > smem) smem = red;
  static size_t configured = 0;  // per instantiation: grow the opt-in as needed
  if (smem > configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = smem;
  }
  dim3 grid(n_tiles, a.U, PS);
  k<<<grid, NW * 32, smem, s>>>(a);
  return (int)cudaGetLastError();
}

template <int POOL, int ACT, int C, int MT, bool TRACK, int PI>
static int fwd_nw(const FwdArgs& a, int n_tiles, int PS, int nw, cudaStream_t s) {
  if (a.packed) {
    if constexpr (POOL == P_MEAN && C >= 32) {  // chunk-4 Mean epilogue (process_block FE = 1)
      if (a.fepi == 1) {
        if (nw == 8) return launch_fwd<POOL, ACT, C, MT, 8, TRACK, PI, true, 1>(a, n_tiles, PS, s);
        if (nw == 16) return launch_fwd<POOL, ACT, C, MT, 16, TRACK, PI, true, 1>(a, n_tiles, PS, s);
        return -1;
      }
    }
    if (nw == 8) return launch_fwd<POOL, ACT, C, MT, 8, TRACK, PI, true>(a, n_tiles, PS, s);
    if (nw == 16) return launch_fwd<POOL, ACT, C, MT, 16, TRACK, PI, true>(a, n_tiles, PS, s);
  } else {
    if (nw == 8) return launch_fwd<POOL, ACT, C, MT, 8, TRACK, PI, false>(a, n_tiles, PS, s);
    if (nw == 16) return launch_fwd<POOL, ACT, C, MT, 16, TRACK, PI, false>(a, n_tiles, PS, s);
  }
  return -1;
}

template <int POOL, int ACT, bool TRACK, int PI = 0>
static int fwd_c(const FwdArgs& a, int C, int MT, int n_tiles, int PS, int nw, cudaStream_t s) {
#define CASE(CC, MM) \
  if (C == CC && MT == MM) return fwd_nw<POOL, ACT, CC, MM, TRACK, PI>(a, n_tiles, PS, nw, s);
  CASE(8, 1) CASE(8, 2) CASE(16, 1) CASE(16, 2) CASE(32, 1) CASE(32, 2) CASE(64, 1) CASE(64, 2)
#undef CASE
  return -1;
}

template <int POOL>
static int fwd_act(const FwdArgs& a, int act, int track, int C, int MT, int n_tiles, int PS, int nw, cudaStream_t s) {
  if constexpr (POOL == P_MAX) {
    if (track) {
      if (act == A_ID) return fwd_c<POOL, A_ID, true>(a, C, MT, n_tiles, PS, nw, s);
      if (act == A_RELU) return fwd_c<POOL, A_RELU, true>(a, C, MT, n_tiles, PS, nw, s);
      return -1;
    }
  }
  if (act == A_ID) return fwd_c<POOL, A_ID, false>(a, C, MT, n_tiles, PS, nw, s);
  if (act == A_RELU) return fwd_c<POOL, A_RELU, false>(a, C, MT, n_tiles, PS, nw, s);
  return -1;
}

