#include "fwd_dispatch.cuh"

template <int KP, int C, int MT, int NW, bool PK>
static int launch_topk(const TopkArgs& t, int n_tiles, int PS, cudaStream_t s) {
  auto k = fwd_topk_kernel<KP, C, MT, NW, PK>;
  size_t smem = (size_t)t.f.D * 16 * MT * 2 + (size_t)16 * MT * NW * 4 * 4;  // slab + pass-1 maxima
  size_t red = (size_t)NW * 16 * MT * KP * 8;
  if (red > smem) smem = red;
  static size_t configured = 0;
  if (smem > configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = smem;
  }
  k<<<dim3(n_tiles, t.f.U, PS), NW * 32, smem, s>>>(t);
  return (int)cudaGetLastError();
}

// packed storage only (canonical-layout Top-K runs on the Triton path).
// k >= 16 keeps 2 rows x k (value, id) per lane: 16-token tiles, 8 warps.
template <int KP, int C, int MT>
static int topk_nw(const TopkArgs& t, int n_tiles, int PS, int nw, cudaStream_t s) {
  if (!t.f.packed) return -2;
  // tile/warp pairs: 16 tokens -> 8 warps, 32 tokens -> 16 warps
  if constexpr (MT == 1) return launch_topk<KP, C, 1, 8, true>(t, n_tiles, PS, s);
  else return launch_topk<KP, C, 2, 16, true>(t, n_tiles, PS, s);
}

template <int KP>
static int topk_c(const TopkArgs& t, int C, int MT, int n_tiles, int PS, int nw, cudaStream_t s) {
#define CASE(CC)                                                              \
  if (C == CC) {                                                              \
    if (MT == 1) return topk_nw<KP, CC, 1>(t, n_tiles, PS, nw, s);            \
    if constexpr (KP < 16) return topk_nw<KP, CC, 2>(t, n_tiles, PS, nw, s);  \
  }
  CASE(8) CASE(16) CASE(32) CASE(64)
#undef CASE
  return -1;
}

extern "C" int sp_fwd_topk(const TopkArgs* t, int kp, int C, int MT, int n_tiles, int PS, int nw, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  switch (kp) {
    case 1: return topk_c<1>(*t, C, MT, n_tiles, PS, nw, s);
    case 2: return topk_c<2>(*t, C, MT, n_tiles, PS, nw, s);
    case 4: return topk_c<4>(*t, C, MT, n_tiles, PS, nw, s);
    case 8: return topk_c<8>(*t, C, MT, n_tiles, PS, nw, s);
    case 16: return topk_c<16>(*t, C, MT, n_tiles, PS, nw, s);
    case 32: return topk_c<32>(*t, C, MT, n_tiles, PS, nw, s);
  }
  return -1;
}
extern "C" int sp_topk_args_size() { return (int)sizeof(TopkArgs); }
