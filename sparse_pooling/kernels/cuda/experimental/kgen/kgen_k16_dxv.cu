// EXPERIMENTAL K = 16 dX tiling variants of the kgen bundle (results/INTERACTION_COVERAGE_K16_DX.md section 5).
//
// The SAME kernel template as the legacy K16 dX (kgen.cuh kg_dx_kernel: S staged input rows, z recomputed, split
// dZ, per-warp / shared FP32 slab copies, cross-CTA FP32 atomics), instantiated at other compile-time token tiles MT
// and warp counts NW; the warps-per-copy WG stays a launch argument (KgArgs.dx_group_warps).  Forward and dW are not
// touched.  The legacy instantiation (MT 4, NW 8) keeps running through sp_kg_k16 (kgen_k16.cu); this unit only adds
// entry points:
//
//   sp_kg_k16_dxv(args, act, split, mt, nw, impl, n_chunks, n_tiles, stream)   launch one variant
//   sp_kg_k16_dxv_list(out, n)                                                (mt, nw, impl) of every compiled variant
//   sp_kg_k16_dxv_attrs(mt, nw, impl, act, split, smem, out, n)               compiled attributes + occupancy
//
// impl 0 = the generic template (any logical C % 8 == 0).
#include "kgen_dispatch.cuh"

// Launch preparation of the variants.  The (MT 4, NW 8) instantiation is the SAME kernel symbol as the legacy launch
// of kgen_k16.cu (one host stub for a template instantiation), whose LaunchSlot lives in that unit; a LaunchSlot here
// could LOWER the shared kernel's dynamic shared-memory attribute below what the legacy slot believes it configured.
// The variants therefore read the kernel's CURRENT attributes at every launch and only ever RAISE the attribute
// (one cudaFuncGetAttributes per dX launch: microseconds against a ~0.5 s kernel).
template <typename Kern>
static int kg16_ensure(Kern k, size_t smem, int threads) {
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) {
    cudaGetLastError();
    return SP_ERR_DEVICE;
  }
  cudaFuncAttributes fa;
  cudaError_t e = cudaFuncGetAttributes(&fa, k);
  if (e != cudaSuccess) {
    cudaGetLastError();
    return e == cudaErrorNoKernelImageForDevice || e == cudaErrorInvalidDeviceFunction ? SP_ERR_NO_IMAGE : SP_ERR_ATTR;
  }
  int optin = 0;
  if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
    cudaGetLastError();
    return SP_ERR_ATTR;
  }
  if (threads > fa.maxThreadsPerBlock) return SP_ERR_THREADS;
  if (smem + fa.sharedSizeBytes > (size_t)optin) return SP_ERR_SMEM;
  if ((long long)smem > (long long)fa.maxDynamicSharedSizeBytes &&
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) {
    cudaGetLastError();
    return SP_ERR_ATTR;
  }
  return 0;
}

template <int MT, int NW, int ACT, bool SPLIT>
static int kg16_dxv_launch(const KgArgs& a, int n_chunks, int n_tiles, cudaStream_t s) {
  if (a.dx_group_warps < 1 || NW % a.dx_group_warps) return -4;
  const size_t BM = 16 * MT;
  size_t smem = kg_map_bytes(a) + (size_t)a.S * BM * 2 + (size_t)(NW / a.dx_group_warps) * a.S * BM * 4;
  auto k = kg_dx_kernel<16, MT, NW, ACT, SPLIT>;
  if (int e = kg16_ensure(k, smem, NW * 32)) return e;
  k<<<dim3(n_tiles, n_chunks), NW * 32, smem, s>>>(a);
  return (int)cudaGetLastError();
}

template <int MT, int NW>
static int kg16_dxv_dispatch(const KgArgs& a, int act, int split, int n_chunks, int n_tiles, cudaStream_t s) {
  if (act == A_RELU) return split ? kg16_dxv_launch<MT, NW, A_RELU, true>(a, n_chunks, n_tiles, s)
                                  : kg16_dxv_launch<MT, NW, A_RELU, false>(a, n_chunks, n_tiles, s);
  return split ? kg16_dxv_launch<MT, NW, A_ID, true>(a, n_chunks, n_tiles, s)
               : kg16_dxv_launch<MT, NW, A_ID, false>(a, n_chunks, n_tiles, s);
}

template <int MT, int NW>
static const void* kg16_dxv_fn(int act, int split) {
  if (act == A_RELU) return split ? (const void*)kg_dx_kernel<16, MT, NW, A_RELU, true>
                                  : (const void*)kg_dx_kernel<16, MT, NW, A_RELU, false>;
  return split ? (const void*)kg_dx_kernel<16, MT, NW, A_ID, true> : (const void*)kg_dx_kernel<16, MT, NW, A_ID, false>;
}

// compiled (mt, nw, impl) variants; (4, 8, 0) is the legacy tiling compiled in this unit too (A/B identity checks)
static const int KG16_DXV[][3] = {{4, 8, 0}, {2, 8, 0}, {4, 4, 0}, {2, 4, 0}};
static const int KG16_DXV_N = sizeof(KG16_DXV) / sizeof(KG16_DXV[0]);

static const void* kg16_dxv_lookup(int mt, int nw, int impl, int act, int split) {
  if (impl != 0) return nullptr;
  if (mt == 4 && nw == 8) return kg16_dxv_fn<4, 8>(act, split);
  if (mt == 2 && nw == 8) return kg16_dxv_fn<2, 8>(act, split);
  if (mt == 4 && nw == 4) return kg16_dxv_fn<4, 4>(act, split);
  if (mt == 2 && nw == 4) return kg16_dxv_fn<2, 4>(act, split);
  return nullptr;
}

extern "C" int sp_kg_k16_dxv(const KgArgs* a, int act, int split, int mt, int nw, int impl, int n_chunks, int n_tiles,
                             void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!kg_args_ok(a)) return -2;
  if (act != A_ID && act != A_RELU) return -1;
  if (impl != 0) return -5;
  if (n_chunks == 0) return 0;
  if (mt == 4 && nw == 8) return kg16_dxv_dispatch<4, 8>(*a, act, split, n_chunks, n_tiles, s);
  if (mt == 2 && nw == 8) return kg16_dxv_dispatch<2, 8>(*a, act, split, n_chunks, n_tiles, s);
  if (mt == 4 && nw == 4) return kg16_dxv_dispatch<4, 4>(*a, act, split, n_chunks, n_tiles, s);
  if (mt == 2 && nw == 4) return kg16_dxv_dispatch<2, 4>(*a, act, split, n_chunks, n_tiles, s);
  return -5;  // not compiled
}

extern "C" int sp_kg_k16_dxv_list(int* out, int n) {
  if (n < 3 * KG16_DXV_N) return -1;
  for (int i = 0; i < KG16_DXV_N; i++)
    for (int j = 0; j < 3; j++) out[3 * i + j] = KG16_DXV[i][j];
  return KG16_DXV_N;
}

// out: numRegs, localSizeBytes (spill / stack), sharedSizeBytes (static), maxThreadsPerBlock, ptxVersion,
// binaryVersion, active CTAs per SM at the given dynamic shared memory (cudaOccupancyMaxActiveBlocksPerMultiprocessor;
// -1 if it fails), maxDynamicSharedSizeBytes currently configured
extern "C" int sp_kg_k16_dxv_attrs(int mt, int nw, int impl, int act, int split, long long smem, long long* out, int n) {
  if (n < 8) return -1;
  const void* f = kg16_dxv_lookup(mt, nw, impl, act, split);
  if (!f) return -5;
  cudaFuncAttributes fa;
  if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) {
    cudaGetLastError();
    return -8;
  }
  int blocks = -1;
  // never lower the attribute (see kg16_ensure); occupancy at the requested dynamic shared memory
  if ((smem > (long long)fa.maxDynamicSharedSizeBytes &&
       cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, f, nw * 32, (size_t)smem) != cudaSuccess) {
    cudaGetLastError();
    blocks = -1;
  }
  out[0] = fa.numRegs;
  out[1] = (long long)fa.localSizeBytes;
  out[2] = (long long)fa.sharedSizeBytes;
  out[3] = fa.maxThreadsPerBlock;
  out[4] = fa.ptxVersion;
  out[5] = fa.binaryVersion;
  out[6] = blocks;
  out[7] = fa.maxDynamicSharedSizeBytes;
  return 8;
}
