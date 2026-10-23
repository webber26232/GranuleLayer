// EXPERIMENTAL kgx dX kernels (kgx.cuh): direct, index-free (per-record slab adds / register accumulation),
// tile-sparse
#include "kgx_dispatch.cuh"

template <bool IDF, bool ACC, bool TILE, bool SPLIT>
static int kx_dx(const KxArgs& a, int variant, int n_chunks, int n_tiles, cudaStream_t s) {
  static LaunchSlot slot;
  if (!ACC && (a.dx_group_warps < 1 || KxCfg::DX_NW % a.dx_group_warps)) return -4;
  return kx_launch(kx_dx_kernel<KxCfg::DX_MT, KxCfg::DX_NW, IDF, ACC, TILE, SPLIT>, dim3(n_tiles, n_chunks),
                   KxCfg::DX_NW * 32, kx_smem(a, 1, variant), a, s, slot);
}

extern "C" int sp_kx_dx(const KxArgs* a, int variant, int split, int n_chunks, int n_tiles, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!kx_args_ok(a, variant)) return -2;
  if (n_chunks == 0) return 0;
#define KX_DX(I, A, T)                                                              \
  return split ? kx_dx<I, A, T, true>(*a, variant, n_chunks, n_tiles, s)              \
               : kx_dx<I, A, T, false>(*a, variant, n_chunks, n_tiles, s)
  switch (variant) {
    case KX_DIRECT: KX_DX(false, false, false);
    case KX_IDF_RMW: KX_DX(true, false, false);
    case KX_IDF: KX_DX(true, true, false);
    case KX_TILE: KX_DX(true, true, true);
    default: return -2;
  }
#undef KX_DX
}
