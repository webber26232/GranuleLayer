#pragma once
// launch templates of the experimental generalized-fan-in kernels (kgen.cuh); one translation unit per K
#include "../../launch.cuh"
#include "kgen.cuh"

static inline size_t kg_map_bytes(const KgArgs& a) { return ((size_t)a.D * 2 + 15) & ~(size_t)15; }

template <typename Kern>
static int kg_launch(Kern k, dim3 grid, int threads, size_t smem, const KgArgs& a, cudaStream_t s, LaunchSlot& slot) {
  if (int e = sp_prepare(k, smem, threads, slot)) return e;  // device limits (launch.cuh)
  k<<<grid, threads, smem, s>>>(a);
  return (int)cudaGetLastError();
}

template <int K, int ACT>
static int kg_fwd_launch(const KgArgs& a, int n_chunks, int n_tiles, cudaStream_t s) {
  using Cf = KgCfg<K>;
  static LaunchSlot slot;
  size_t smem = kg_map_bytes(a) + (size_t)a.S * 16 * Cf::FWD_MT * 2;
  return kg_launch(kg_fwd_kernel<K, Cf::FWD_MT, Cf::FWD_NW, ACT>, dim3(n_tiles, n_chunks), Cf::FWD_NW * 32, smem, a, s,
                   slot);
}

template <int K, int ACT, bool SPLIT>
static int kg_dx_launch(const KgArgs& a, int n_chunks, int n_tiles, cudaStream_t s) {
  using Cf = KgCfg<K>;
  static LaunchSlot slot;
  if (a.dx_group_warps < 1 || Cf::DX_NW % a.dx_group_warps) return -4;
  const size_t BM = 16 * Cf::DX_MT;
  size_t smem = kg_map_bytes(a) + (size_t)a.S * BM * 2 + (size_t)(Cf::DX_NW / a.dx_group_warps) * a.S * BM * 4;
  return kg_launch(kg_dx_kernel<K, Cf::DX_MT, Cf::DX_NW, ACT, SPLIT>, dim3(n_tiles, n_chunks), Cf::DX_NW * 32, smem, a,
                   s, slot);
}

template <int K, int ACT, bool SPLIT>
static int kg_dw_launch(const KgArgs& a, int n_chunks, cudaStream_t s) {
  using Cf = KgCfg<K>;
  static LaunchSlot slot;
  size_t smem = kg_map_bytes(a) + (size_t)Cf::DW_ST * Cf::DW_TS * a.S * 16 * 2;
  return kg_launch(kg_dw_kernel<K, Cf::DW_WPT, Cf::DW_NW, Cf::DW_TS, Cf::DW_ST, ACT, SPLIT>, dim3(n_chunks),
                   Cf::DW_NW * 32, smem, a, s, slot);
}

static inline bool kg_args_ok(const KgArgs* a) {
  return a->C >= 8 && a->C <= 256 && a->C % 8 == 0 && a->S >= 8 && a->S <= 256 && a->B % 8 == 0;
}

// one extern "C" family per K: kind 0 fwd, 1 dX, 2 dW
#define KG_ENTRY(K)                                                                                       \
  extern "C" int sp_kg_k##K(const KgArgs* a, int kind, int act, int split, int n_chunks, int n_tiles,       \
                            void* stream) {                                                               \
    cudaStream_t s = (cudaStream_t)stream;                                                                \
    if (!kg_args_ok(a)) return -2;                                                                        \
    if (act != A_ID && act != A_RELU) return -1;                                                          \
    if (n_chunks == 0) return 0;                                                                          \
    if (kind == 0) return act == A_RELU ? kg_fwd_launch<K, A_RELU>(*a, n_chunks, n_tiles, s)              \
                                        : kg_fwd_launch<K, A_ID>(*a, n_chunks, n_tiles, s);               \
    if (kind == 1) {                                                                                      \
      if (act == A_RELU) return split ? kg_dx_launch<K, A_RELU, true>(*a, n_chunks, n_tiles, s)           \
                                      : kg_dx_launch<K, A_RELU, false>(*a, n_chunks, n_tiles, s);         \
      return split ? kg_dx_launch<K, A_ID, true>(*a, n_chunks, n_tiles, s)                                \
                   : kg_dx_launch<K, A_ID, false>(*a, n_chunks, n_tiles, s);                              \
    }                                                                                                     \
    if (kind == 2) {                                                                                      \
      if (act == A_RELU) return split ? kg_dw_launch<K, A_RELU, true>(*a, n_chunks, s)                    \
                                      : kg_dw_launch<K, A_RELU, false>(*a, n_chunks, s);                  \
      return split ? kg_dw_launch<K, A_ID, true>(*a, n_chunks, s) : kg_dw_launch<K, A_ID, false>(*a, n_chunks, s); \
    }                                                                                                     \
    return -2;                                                                                            \
  }
