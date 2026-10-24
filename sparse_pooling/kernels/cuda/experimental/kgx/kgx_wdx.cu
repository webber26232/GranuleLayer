// EXPERIMENTAL width-preserving kgx dX kernels (kgx_wide.cuh)
#include "kgx_wdispatch.cuh"

template <int TB, bool BB, bool SPLIT>
static int kw_dx(const KwArgs& w, int n_chunks, int n_tiles, cudaStream_t s) {
  static LaunchSlot slot;
  auto k = kw_dx_kernel<KxCfg::DX_MT, KxCfg::DX_NW, TB, BB, SPLIT>;
  if (int e = sp_prepare(k, kw_smem(w, 1), KxCfg::DX_NW * 32, slot)) return e;
  k<<<dim3(n_tiles, n_chunks), KxCfg::DX_NW * 32, kw_smem(w, 1), s>>>(w);
  return (int)cudaGetLastError();
}

extern "C" int sp_kw_dx(const KwArgs* w, int split, int n_chunks, int n_tiles, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!kw_args_ok(w)) return -2;
  if (n_chunks == 0) return 0;
#define KW_DX(T, B) \
  return split ? kw_dx<T, B, true>(*w, n_chunks, n_tiles, s) : kw_dx<T, B, false>(*w, n_chunks, n_tiles, s)
  if (w->tb == 8) KW_DX(8, false);
  if (w->bb) KW_DX(16, true);
  KW_DX(16, false);
#undef KW_DX
}
