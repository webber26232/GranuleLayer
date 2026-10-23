// ABI identity of a native library (every bundle links this unit): the Python bridge
// (backends/cuda/runtime.py) checks the ABI version, the build kind / bundle it asked for, and the
// size and every field offset of the argument structs against its ctypes declarations before it
// binds any entry point.  SP_ABI_VERSION changes whenever an entry point or struct changes.
#include <cstddef>

#include "sparse_pool_sg.cuh"

#define SP_ABI_VERSION 3
#ifndef SP_BUILD_KIND_ID
#define SP_BUILD_KIND_ID 0  // 0 production, 1 development (extra flags), 2 ablation (SP_ABLATE_*: wrong results)
#endif
#ifndef SP_BUNDLE_ID
#define SP_BUNDLE_ID 0  // 0 production, 1 experimental, 2 diagnostics
#endif

extern "C" int sp_abi_version() { return SP_ABI_VERSION; }
extern "C" int sp_build_kind() { return SP_BUILD_KIND_ID; }
extern "C" int sp_bundle_id() { return SP_BUNDLE_ID; }
extern "C" int sp_cuda_runtime_compiled() { return CUDART_VERSION; }

#define OFF(S, F) (long long)offsetof(S, F)
// which: 0 FwdArgs, 1 TopkArgs, 2 BwdArgs, 3 SgArgs.  Writes sizeof then the offsets in declaration
// order into out[0 .. n), returns the number of values (or -1).
extern "C" int sp_struct_layout(int which, long long* out, int n) {
  long long v[64];
  int k = 0;
  if (which == 0) {
    long long t[] = {(long long)sizeof(FwdArgs), OFF(FwdArgs, X), OFF(FwdArgs, sxf), OFF(FwdArgs, sxb),
                     OFF(FwdArgs, idx), OFF(FwdArgs, W), OFF(FwdArgs, bias), OFF(FwdArgs, counts),
                     OFF(FwdArgs, st_m), OFF(FwdArgs, st_s), OFF(FwdArgs, st_t), OFF(FwdArgs, st_id), OFF(FwdArgs, B),
                     OFF(FwdArgs, U), OFF(FwdArgs, S), OFF(FwdArgs, NB), OFF(FwdArgs, BPS), OFF(FwdArgs, BPSP),
                     OFF(FwdArgs, D), OFF(FwdArgs, scale), OFF(FwdArgs, p), OFF(FwdArgs, p_int), OFF(FwdArgs, packed)};
    for (long long x : t) v[k++] = x;
  } else if (which == 1) {
    long long t[] = {(long long)sizeof(TopkArgs), OFF(TopkArgs, f), OFF(TopkArgs, st_keys), OFF(TopkArgs, st_pre),
                     OFF(TopkArgs, st_slot), OFF(TopkArgs, act)};
    for (long long x : t) v[k++] = x;
  } else if (which == 2) {
    long long t[] = {(long long)sizeof(BwdArgs), OFF(BwdArgs, X), OFF(BwdArgs, sxf), OFF(BwdArgs, sxb),
                     OFF(BwdArgs, idx), OFF(BwdArgs, W), OFF(BwdArgs, bias), OFF(BwdArgs, counts), OFF(BwdArgs, c1),
                     OFF(BwdArgs, c2), OFF(BwdArgs, c3), OFF(BwdArgs, dX), OFF(BwdArgs, sdf), OFF(BwdArgs, sdb),
                     OFF(BwdArgs, dW), OFF(BwdArgs, dB), OFF(BwdArgs, B), OFF(BwdArgs, U), OFF(BwdArgs, S),
                     OFF(BwdArgs, NB), OFF(BwdArgs, BPS), OFF(BwdArgs, D), OFF(BwdArgs, scale), OFF(BwdArgs, p),
                     OFF(BwdArgs, p_int), OFF(BwdArgs, units_per_cta), OFF(BwdArgs, wunits_per_cta),
                     OFF(BwdArgs, flags), OFF(BwdArgs, nonfinite)};
    for (long long x : t) v[k++] = x;
  } else if (which == 3) {
    long long t[] = {(long long)sizeof(SgArgs), OFF(SgArgs, X), OFF(SgArgs, sxf), OFF(SgArgs, idx), OFF(SgArgs, W),
                     OFF(SgArgs, bias), OFF(SgArgs, counts), OFF(SgArgs, feat), OFF(SgArgs, items),
                     OFF(SgArgs, chunks), OFF(SgArgs, units), OFF(SgArgs, ysum), OFF(SgArgs, c1T), OFF(SgArgs, dXT),
                     OFF(SgArgs, dW), OFF(SgArgs, dB), OFF(SgArgs, B), OFF(SgArgs, G), OFF(SgArgs, D), OFF(SgArgs, S),
                     OFF(SgArgs, NB), OFF(SgArgs, NU), OFF(SgArgs, dx_group_warps), OFF(SgArgs, flags),
                     OFF(SgArgs, nonfinite), OFF(SgArgs, part), OFF(SgArgs, part_t0), OFF(SgArgs, part_tw)};
    for (long long x : t) v[k++] = x;
  } else {
    return -1;
  }
  if (k > n) return -1;
  for (int i = 0; i < k; i++) out[i] = v[i];
  return k;
}
