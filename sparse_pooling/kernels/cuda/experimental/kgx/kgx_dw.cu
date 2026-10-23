// EXPERIMENTAL kgx dW / db kernels (kgx.cuh): direct, index-free, tile-sparse
#include "kgx_dispatch.cuh"

template <bool IDF, bool TILE, bool SPLIT>
static int kx_dw(const KxArgs& a, int variant, int n_chunks, cudaStream_t s) {
  static LaunchSlot slot;
  using C = KxCfg;
  return kx_launch(kx_dw_kernel<C::DW_NW, C::DW_TS, C::DW_ST, IDF, TILE, SPLIT>, dim3(n_chunks), C::DW_NW * 32,
                   kx_smem(a, 2, variant), a, s, slot);
}

extern "C" int sp_kx_dw(const KxArgs* a, int variant, int split, int n_chunks, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!kx_args_ok(a, variant)) return -2;
  if (n_chunks == 0) return 0;
#define KX_DW(I, T) return split ? kx_dw<I, T, true>(*a, variant, n_chunks, s) : kx_dw<I, T, false>(*a, variant, n_chunks, s)
  switch (variant) {
    case KX_DIRECT: KX_DW(false, false);
    case KX_IDF_RMW:
    case KX_IDF: KX_DW(true, false);
    case KX_TILE: KX_DW(true, true);
    default: return -2;
  }
#undef KX_DW
}
