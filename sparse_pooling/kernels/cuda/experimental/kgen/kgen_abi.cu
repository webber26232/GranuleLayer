// identity of the kgen bundle's argument struct and compiled tilings (checked by experimental/kgen_runtime.py)
#include <cstddef>

#include "kgen.cuh"

#define KOFF(F) (long long)offsetof(KgArgs, F)
extern "C" int sp_kg_struct_layout(long long* out, int n) {
  long long t[] = {(long long)sizeof(KgArgs), KOFF(X), KOFF(sxf), KOFF(idx), KOFF(W), KOFF(bias), KOFF(counts),
                   KOFF(feat), KOFF(items), KOFF(chunks), KOFF(tasks), KOFF(ysum), KOFF(c1T), KOFF(dXT), KOFF(dW),
                   KOFF(dB), KOFF(B), KOFF(G), KOFF(D), KOFF(S), KOFF(NB), KOFF(C), KOFF(dx_group_warps),
                   KOFF(flags)};
  const int k = sizeof(t) / sizeof(t[0]);
  if (k > n) return -1;
  for (int i = 0; i < k; i++) out[i] = t[i];
  return k;
}

template <int K>
static void kg_cfg(int* o) {
  using C = KgCfg<K>;
  int v[] = {C::FWD_MT, C::FWD_NW, C::DX_MT, C::DX_NW, C::DW_WPT, C::DW_NW, C::DW_TS, C::DW_ST};
  for (int i = 0; i < 8; i++) o[i] = v[i];
}
// (fwd MT, fwd warps, dX MT, dX warps, dW tasks per warp, dW warps, dW tiles per stage, dW stages)
extern "C" int sp_kg_config(int K, int* out) {
  if (K == 8) kg_cfg<8>(out);
  else if (K == 16) kg_cfg<16>(out);
  else if (K == 32) kg_cfg<32>(out);
  else if (K == 64) kg_cfg<64>(out);
  else return -1;
  return 8;
}
