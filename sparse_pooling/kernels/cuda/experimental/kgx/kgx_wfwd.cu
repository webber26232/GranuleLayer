// EXPERIMENTAL width-preserving kgx forward kernels (kgx_wide.cuh): 16 x 8 tiles, 16 x 16 tiles (+ backbone)
#include <cstddef>

#include "kgx_wdispatch.cuh"

template <int TB, bool BB>
static int kw_fwd(const KwArgs& w, int n_chunks, int n_tiles, cudaStream_t s) {
  static LaunchSlot slot;
  auto k = kw_fwd_kernel<KxCfg::FWD_MT_IDF, KxCfg::FWD_NW, TB, BB>;
  if (int e = sp_prepare(k, kw_smem(w, 0), KxCfg::FWD_NW * 32, slot)) return e;
  k<<<dim3(n_tiles, n_chunks), KxCfg::FWD_NW * 32, kw_smem(w, 0), s>>>(w);
  return (int)cudaGetLastError();
}

extern "C" int sp_kw_fwd(const KwArgs* w, int n_chunks, int n_tiles, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!kw_args_ok(w)) return -2;
  if (n_chunks == 0) return 0;
  if (w->tb == 8) return kw_fwd<8, false>(*w, n_chunks, n_tiles, s);
  return w->bb ? kw_fwd<16, true>(*w, n_chunks, n_tiles, s) : kw_fwd<16, false>(*w, n_chunks, n_tiles, s);
}

#define WOFF(F) (long long)offsetof(KwArgs, F)
extern "C" int sp_kw_struct_layout(long long* out, int n) {
  long long t[] = {(long long)sizeof(KwArgs), WOFF(a), WOFF(bbk), WOFF(bbw), WOFF(dBB), WOFF(tb), WOFF(bb)};
  const int k = sizeof(t) / sizeof(t[0]);
  if (k > n) return -1;
  for (int i = 0; i < k; i++) out[i] = t[i];
  return k;
}

extern "C" int sp_kw_smem(const KwArgs* w, int kind, long long* out) {
  *out = (long long)kw_smem(*w, kind);
  return 0;
}
