#pragma once
// launch templates of the EXPERIMENTAL width-preserving kgx kernels (kgx_wide.cuh)
#include "kgx_dispatch.cuh"
#include "kgx_wide.cuh"

// (tile width, backbone) combinations that are compiled
static inline bool kw_args_ok(const KwArgs* w) {
  if (!kx_args_ok(&w->a, KX_TILE)) return false;
  if (!((w->tb == 16 && (w->bb == 0 || w->bb == 1)) || (w->tb == 8 && w->bb == 0))) return false;
  if (w->bb && (w->bbk == nullptr || w->bbw == nullptr)) return false;
  return true;
}

// smem of a launch (same staging / slab copies as the tile variant; backends/cuda/variants.kgx_smem)
static inline size_t kw_smem(const KwArgs& w, int kind) { return kx_smem(w.a, kind, KX_TILE); }
