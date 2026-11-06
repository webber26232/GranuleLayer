// entry points of the learned sliding-window reduction kernels (K = 16): lrd.cuh
#include <cstddef>

#include "../../launch.cuh"
#include "lrd.cuh"

#define LOFF(F) (long long)offsetof(LrArgs, F)
extern "C" int sp_lr_struct_layout(long long* out, int n) {
  long long t[] = {(long long)sizeof(LrArgs), LOFF(k), LOFF(P), LOFF(U), LOFF(Dl), LOFF(dP), LOFF(F), LOFF(J), LOFF(PAD),
                   LOFF(pad0)};
  const int k = sizeof(t) / sizeof(t[0]);
  if (k > n) return -1;
  for (int i = 0; i < k; i++) out[i] = t[i];
  return k;
}

static inline size_t lr_map(const LrArgs& a) { return ((size_t)a.k.D * 2 + 15) & ~(size_t)15; }
static inline bool lr_ok(const LrArgs* a) {
  return a->k.C >= 8 && a->k.C <= 256 && a->k.C % 8 == 0 && a->k.S >= 8 && a->k.S <= 256 && a->k.B % 8 == 0 && a->F >= 1 &&
         a->F <= 8;
}

constexpr int LR_MT = 4, LR_NW = 8, LR_TS = 2, LR_ST = 3;

template <typename Kern>
static int lr_launch(Kern k, dim3 grid, int threads, size_t smem, const LrArgs& a, cudaStream_t s, LaunchSlot& slot) {
  if (int e = sp_prepare(k, smem, threads, slot)) return e;
  k<<<grid, threads, smem, s>>>(a);
  return (int)cudaGetLastError();
}

template <int EPI, int FX>
static int lr_fwd(const LrArgs& a, int n_chunks, int n_tiles, cudaStream_t s) {
  static LaunchSlot slot;
  size_t smem = lr_map(a) + (size_t)a.k.S * 16 * LR_MT * 2;
  return lr_launch(lr_fwd_kernel<16, LR_MT, LR_NW, EPI, FX>, dim3(n_tiles, n_chunks), LR_NW * 32, smem, a, s, slot);
}

template <bool SPLIT>
static int lr_dx(const LrArgs& a, int n_chunks, int n_tiles, cudaStream_t s) {
  static LaunchSlot slot;
  if (a.k.dx_group_warps < 1 || LR_NW % a.k.dx_group_warps) return -4;
  const size_t BM = 16 * LR_MT;
  size_t smem = lr_map(a) + (size_t)a.k.S * BM * 2 + (size_t)(LR_NW / a.k.dx_group_warps) * a.k.S * BM * 4;
  return lr_launch(lr_dx_kernel<16, LR_MT, LR_NW, SPLIT>, dim3(n_tiles, n_chunks), LR_NW * 32, smem, a, s, slot);
}

template <int WPT, bool SPLIT, int MODE>
static int lr_dw(const LrArgs& a, int n_chunks, cudaStream_t s) {
  static LaunchSlot slot;
  size_t smem = lr_map(a) + (size_t)LR_ST * LR_TS * a.k.S * 16 * 2;
  return lr_launch(lr_dw_kernel<16, WPT, LR_NW, LR_TS, LR_ST, SPLIT, MODE>, dim3(n_chunks), LR_NW * 32, smem, a, s, slot);
}

template <int WPT, int MODE>
static int lr_dw_s(const LrArgs& a, int split, int n_chunks, cudaStream_t s) {
  return split ? lr_dw<WPT, true, MODE>(a, n_chunks, s) : lr_dw<WPT, false, MODE>(a, n_chunks, s);
}

// fwd: variant 0 MMA BF16, 1 MMA split, 2 SIMT (FX = smallest compiled >= F)
extern "C" int sp_lr_fwd(const LrArgs* a, int variant, int n_chunks, int n_tiles, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!lr_ok(a)) return -2;
  if (n_chunks == 0) return 0;
  if (variant == 0) return lr_fwd<LR_MMA_BF16, 1>(*a, n_chunks, n_tiles, s);
  if (variant == 1) return lr_fwd<LR_MMA_SPLIT, 1>(*a, n_chunks, n_tiles, s);
  if (variant == 2) {
    if (a->F <= 1) return lr_fwd<LR_SIMT, 1>(*a, n_chunks, n_tiles, s);
    if (a->F <= 5) return lr_fwd<LR_SIMT, 5>(*a, n_chunks, n_tiles, s);
    return lr_fwd<LR_SIMT, 8>(*a, n_chunks, n_tiles, s);
  }
  return -5;
}

extern "C" int sp_lr_dx(const LrArgs* a, int split, int n_chunks, int n_tiles, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!lr_ok(a)) return -2;
  if (n_chunks == 0) return 0;
  return split ? lr_dx<true>(*a, n_chunks, n_tiles, s) : lr_dx<false>(*a, n_chunks, n_tiles, s);
}

// mode: LR_DO_DW | LR_DO_DP; wpt: dW tasks per warp (4 or 2)
extern "C" int sp_lr_dw(const LrArgs* a, int split, int mode, int wpt, int n_chunks, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!lr_ok(a)) return -2;
  if (n_chunks == 0) return 0;
  if (wpt == 4) {
    if (mode == 1) return lr_dw_s<4, 1>(*a, split, n_chunks, s);
    if (mode == 2) return lr_dw_s<4, 2>(*a, split, n_chunks, s);
    if (mode == 3) return lr_dw_s<4, 3>(*a, split, n_chunks, s);
  } else if (wpt == 2) {
    if (mode == 1) return lr_dw_s<2, 1>(*a, split, n_chunks, s);
    if (mode == 2) return lr_dw_s<2, 2>(*a, split, n_chunks, s);
    if (mode == 3) return lr_dw_s<2, 3>(*a, split, n_chunks, s);
  }
  return -5;
}

// Compiled attributes (register use, spills) of a kernel + theoretical active CTAs per SM.  out: numRegs,
// localSizeBytes, sharedSizeBytes (static), maxThreadsPerBlock, active CTAs per SM (-1 unknown), max dynamic smem.
// kind 0 fwd(variant, FX), 1 dx(split), 2 dw(split, mode, wpt)
template <typename Kern>
static int lr_attr(Kern k, int threads, long long smem, long long* out) {
  cudaFuncAttributes fa;
  if (cudaFuncGetAttributes(&fa, k) != cudaSuccess) {
    cudaGetLastError();
    return -8;
  }
  int blocks = -1;
  if ((smem > (long long)fa.maxDynamicSharedSizeBytes &&
       cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k, threads, (size_t)smem) != cudaSuccess) {
    cudaGetLastError();
    blocks = -1;
  }
  out[0] = fa.numRegs;
  out[1] = (long long)fa.localSizeBytes;
  out[2] = (long long)fa.sharedSizeBytes;
  out[3] = fa.maxThreadsPerBlock;
  out[4] = blocks;
  out[5] = fa.maxDynamicSharedSizeBytes;
  return 6;
}

extern "C" int sp_lr_attrs(int kind, int v0, int v1, int v2, long long smem, long long* out) {
  const int T = LR_NW * 32;
  if (kind == 0) {
    if (v0 == 0) return lr_attr(lr_fwd_kernel<16, LR_MT, LR_NW, LR_MMA_BF16, 1>, T, smem, out);
    if (v0 == 1) return lr_attr(lr_fwd_kernel<16, LR_MT, LR_NW, LR_MMA_SPLIT, 1>, T, smem, out);
    if (v1 <= 1) return lr_attr(lr_fwd_kernel<16, LR_MT, LR_NW, LR_SIMT, 1>, T, smem, out);
    if (v1 <= 5) return lr_attr(lr_fwd_kernel<16, LR_MT, LR_NW, LR_SIMT, 5>, T, smem, out);
    return lr_attr(lr_fwd_kernel<16, LR_MT, LR_NW, LR_SIMT, 8>, T, smem, out);
  }
  if (kind == 1) return v0 ? lr_attr(lr_dx_kernel<16, LR_MT, LR_NW, true>, T, smem, out)
                           : lr_attr(lr_dx_kernel<16, LR_MT, LR_NW, false>, T, smem, out);
  if (kind == 2) {
#define LR_A(W, M) \
  if (v2 == W && v1 == M) return v0 ? lr_attr(lr_dw_kernel<16, W, LR_NW, LR_TS, LR_ST, true, M>, T, smem, out) \
                                    : lr_attr(lr_dw_kernel<16, W, LR_NW, LR_TS, LR_ST, false, M>, T, smem, out);
    LR_A(4, 1) LR_A(4, 2) LR_A(4, 3) LR_A(2, 1) LR_A(2, 2) LR_A(2, 3)
#undef LR_A
  }
  return -5;
}
