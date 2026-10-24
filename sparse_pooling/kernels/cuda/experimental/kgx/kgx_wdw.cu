// EXPERIMENTAL width-preserving kgx dW / db / backbone-dv kernels (kgx_wide.cuh)
#include "kgx_wdispatch.cuh"

template <int TB, bool BB, bool SPLIT>
static int kw_dw(const KwArgs& w, int n_chunks, cudaStream_t s) {
  static LaunchSlot slot;
  using C = KxCfg;
  auto k = kw_dw_kernel<C::DW_NW, C::DW_TS, C::DW_ST, TB, BB, SPLIT>;
  if (int e = sp_prepare(k, kw_smem(w, 2), C::DW_NW * 32, slot)) return e;
  k<<<dim3(n_chunks), C::DW_NW * 32, kw_smem(w, 2), s>>>(w);
  return (int)cudaGetLastError();
}

extern "C" int sp_kw_dw(const KwArgs* w, int split, int n_chunks, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!kw_args_ok(w)) return -2;
  if (n_chunks == 0) return 0;
#define KW_DW(T, B) return split ? kw_dw<T, B, true>(*w, n_chunks, s) : kw_dw<T, B, false>(*w, n_chunks, s)
  if (w->tb == 8) KW_DW(8, false);
  if (w->bb) KW_DW(16, true);
  KW_DW(16, false);
#undef KW_DW
}
