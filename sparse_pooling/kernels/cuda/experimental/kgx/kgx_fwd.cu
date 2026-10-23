// EXPERIMENTAL kgx forward kernels (kgx.cuh): direct, index-free, tile-sparse
#include "kgx_dispatch.cuh"

template <int MT, bool IDF, bool TILE>
static int kx_fwd(const KxArgs& a, int n_chunks, int n_tiles, cudaStream_t s) {
  static LaunchSlot slot;
  return kx_launch(kx_fwd_kernel<MT, KxCfg::FWD_NW, IDF, TILE>, dim3(n_tiles, n_chunks), KxCfg::FWD_NW * 32,
                   kx_smem(a, 0, IDF ? (TILE ? KX_TILE : KX_IDF) : KX_DIRECT), a, s, slot);
}

extern "C" int sp_kx_fwd(const KxArgs* a, int variant, int n_chunks, int n_tiles, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!kx_args_ok(a, variant)) return -2;
  if (n_chunks == 0) return 0;
  switch (variant) {
    case KX_DIRECT: return kx_fwd<KxCfg::FWD_MT, false, false>(*a, n_chunks, n_tiles, s);
    case KX_IDF_RMW:  // (the forward of both index-free dense variants is the same kernel)
    case KX_IDF: return kx_fwd<KxCfg::FWD_MT_IDF, true, false>(*a, n_chunks, n_tiles, s);
    case KX_TILE: return kx_fwd<KxCfg::FWD_MT_IDF, true, true>(*a, n_chunks, n_tiles, s);
    default: return -2;
  }
}
