// identity of the kgx bundle's argument struct and compiled tilings (checked by experimental/kgx_runtime.py)
#include <cstddef>

#include "kgx_dispatch.cuh"

#define XOFF(F) (long long)offsetof(KxArgs, F)
extern "C" int sp_kx_struct_layout(long long* out, int n) {
  long long t[] = {(long long)sizeof(KxArgs), XOFF(X), XOFF(sxf), XOFF(idx), XOFF(W), XOFF(bias), XOFF(counts),
                   XOFF(feat), XOFF(items), XOFF(chunks), XOFF(tasks), XOFF(tmeta), XOFF(rowptr), XOFF(kix), XOFF(wins),
                   XOFF(ysum), XOFF(c1T), XOFF(dXT), XOFF(dW), XOFF(dB), XOFF(B), XOFF(G), XOFF(D), XOFF(S),
                   XOFF(NB), XOFF(C), XOFF(dx_group_warps), XOFF(flags)};
  const int k = sizeof(t) / sizeof(t[0]);
  if (k > n) return -1;
  for (int i = 0; i < k; i++) out[i] = t[i];
  return k;
}

// (fwd MT direct, fwd MT index-free, fwd warps, dX MT, dX warps, dW warps, dW tiles per stage, dW stages,
//  CSR fwd tokens per lane, CSR fwd warps, CSR dW tokens per stage, CSR dW stages, CSR dX warps)
extern "C" int sp_kx_config(int* out) {
  using C = KxCfg;
  int v[] = {C::FWD_MT, C::FWD_MT_IDF, C::FWD_NW, C::DX_MT, C::DX_NW, C::DW_NW, C::DW_TS, C::DW_ST,
             C::CS_TPL, C::CS_NW, C::CSW_BT, C::CSW_ST, C::CSX_NW};
  for (int i = 0; i < 13; i++) out[i] = v[i];
  return 13;
}

extern "C" int sp_kx_smem(const KxArgs* a, int kind, int variant, long long* out) {
  *out = (long long)kx_smem(*a, kind, variant);
  return 0;
}
