#include "sg_dispatch.cuh"
#include "sparse_pool_sg_bwdf.cuh"
// experimental fused shared-projection backward, C = 8, Mean(ReLU), split (sparse_pool_sg_bwdf.cuh)
// cfg -> (warps, units per warp, 16-token tiles per window / ring stage, stages, min CTAs / SM);
// must match SGF_CFGS in kernels/cuda_ops.py
#define SGF_CFGS(X)      \
  X(0, 8, 2, 4, 2, 1)    \
  X(1, 8, 2, 4, 2, 2)    \
  X(2, 8, 4, 4, 2, 1)    \
  X(3, 8, 2, 2, 2, 2)    \
  X(4, 16, 2, 2, 2, 1)   \
  X(5, 8, 4, 2, 2, 1)

// v2 (block-outer / tile-inner, 32-token windows, vector dX records): cfg -> (warps, units per warp, stages, min CTAs / SM)
#define SGF2_CFGS(X) \
  X(6, 8, 2, 2, 2)   \
  X(7, 16, 1, 2, 1)  \
  X(8, 8, 2, 2, 1)   \
  X(9, 8, 2, 3, 2)

template <int NW, int WPW, int ST, int MINB, bool SHARE>
static int sgf2_launch(const SgArgs& a, int gx, cudaStream_t s) {
  static size_t configured = 0;
  size_t smem = sg_map_bytes(a) + (size_t)ST * 2 * a.S * 16 * 2 + (size_t)NW * a.S * 32 * 4;
  return sg_launch(sgf2_kernel<NW, WPW, ST, MINB, SHARE>, dim3(gx), NW * 32, smem, a, s, configured);
}

template <int NW, int WPW, int TS, int ST, int MINB, bool SHARE>
static int sgf_launch(const SgArgs& a, int gx, cudaStream_t s) {
  static size_t configured = 0;
  size_t smem = sg_map_bytes(a) + (size_t)ST * TS * a.S * 16 * 2 + (size_t)NW * a.S * 16 * TS * 4;
  return sg_launch(sgf_kernel<NW, WPW, TS, ST, MINB, SHARE>, dim3(gx), NW * 32, smem, a, s, configured);
}

extern "C" int sp_sgf_c8(const SgArgs* a, int cfg, int share, int gx, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
#define X(I, NW, WPW, TS, ST, MINB) \
  if (cfg == I) return share ? sgf_launch<NW, WPW, TS, ST, MINB, true>(*a, gx, s) : sgf_launch<NW, WPW, TS, ST, MINB, false>(*a, gx, s);
  SGF_CFGS(X)
#undef X
#define X(I, NW, WPW, ST, MINB) \
  if (cfg == I) return share ? sgf2_launch<NW, WPW, ST, MINB, true>(*a, gx, s) : sgf2_launch<NW, WPW, ST, MINB, false>(*a, gx, s);
  SGF2_CFGS(X)
#undef X
  return -1;
}
