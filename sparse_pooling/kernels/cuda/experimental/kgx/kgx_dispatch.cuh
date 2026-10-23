#pragma once
// launch templates of the EXPERIMENTAL K = 64 kgx kernels (kgx.cuh, kgx_csr.cuh)
#include "../../launch.cuh"
#include "kgx_csr.cuh"

// variant ids (experimental/kgx_runtime.py VARIANTS)
enum { KX_DIRECT = 0, KX_IDF_RMW = 1, KX_IDF = 2, KX_TILE = 3, KX_CSR = 4 };

static inline size_t kx_map_bytes(const KxArgs& a) { return ((size_t)a.D * 2 + 15) & ~(size_t)15; }

template <typename Kern>
static int kx_launch(Kern k, dim3 grid, int threads, size_t smem, const KxArgs& a, cudaStream_t s, LaunchSlot& slot) {
  if (int e = sp_prepare(k, smem, threads, slot)) return e;  // device limits (launch.cuh)
  k<<<grid, threads, smem, s>>>(a);
  return (int)cudaGetLastError();
}

static inline bool kx_args_ok(const KxArgs* a, int variant) {
  if (a->C < 16 || a->C > 2048 || a->C % 16 || a->B % 8 || a->S % 8) return false;
  if (variant == KX_DIRECT) return a->S >= 64 && a->S <= 256;
  return a->S == 64;  // index-free / tile / CSR: S = K
}

// smem of a launch (mirrored by backends/cuda/variants.kgx_smem)
static inline size_t kx_smem(const KxArgs& a, int kind, int variant) {
  using C = KxCfg;
  if (variant == KX_CSR) {
    const size_t bt = 32 * C::CS_TPL;
    if (kind == 0) return 64 * bt * 2 + (size_t)C::CS_NW * KC_WE * 4;
    if (kind == 1) return (size_t)64 * 32 * C::CSX_NW * 6 + 2 * KC_WE * 4 + 2 * 32 * 8;
    return (size_t)C::CSW_ST * 64 * C::CSW_BT * 2;
  }
  const bool idf = variant != KX_DIRECT;
  if (kind == 0) return kx_map_bytes(a) + (size_t)a.S * 16 * (idf ? C::FWD_MT_IDF : C::FWD_MT) * 2;
  if (kind == 1) {
    const size_t bm = 16 * C::DX_MT;
    const int copies = (variant == KX_IDF || variant == KX_TILE) ? C::DX_NW : C::DX_NW / a.dx_group_warps;
    return kx_map_bytes(a) + (size_t)a.S * bm * 2 + (size_t)copies * a.S * bm * 4;
  }
  return kx_map_bytes(a) + (size_t)C::DW_ST * C::DW_TS * a.S * 16 * 2;
}
