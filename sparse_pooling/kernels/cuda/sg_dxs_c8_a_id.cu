#include "sgx_dispatch.cuh"
// experimental scatter layouts of the grouped C = 8 dX, A_ID (sparse_pool_sg_dxs.cuh):
// sc = 1 (SC_SWZ4) / 2 (SC_VEC4) / 3 (SC_VEC4P) / 5 (SC_BASEP) / 6 (SC_SWZ4P) / 7 (SC_VEC4T); MT = 2, 4, 8; 8 / 16 warps; split: generic / mask, bf16: generic
extern "C" int sp_sgx_dx_c8_a_id(const SgArgs* a, int sc, int MT, int NW, int split, int epi, int gx, int gy,
                                   void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
#define X(SC, M, W)                                                                                \
  if (sc == SC && MT == M && NW == W) {                                                            \
    if (split && epi == 2) return sgx_dx_launch<A_ID, 8, M, W, true, 2, SC, 1>(*a, gx, gy, s);     \
    if (split && epi == 0) return sgx_dx_launch<A_ID, 8, M, W, true, 0, SC, 1>(*a, gx, gy, s);     \
    if (!split && epi == 0) return sgx_dx_launch<A_ID, 8, M, W, false, 0, SC, 1>(*a, gx, gy, s);   \
    return -2;                                                                                     \
  }
#define XS(SC) X(SC, 2, 8) X(SC, 2, 16) X(SC, 4, 8) X(SC, 4, 16) X(SC, 8, 8) X(SC, 8, 16)
  XS(SC_SWZ4) XS(SC_VEC4) XS(SC_VEC4P) XS(SC_BASEP) XS(SC_SWZ4P) XS(SC_VEC4T)
#undef XS
#undef X
  return -1;
}

// layout diagnostics (tests/test_dx_scatter.py): out[f * 16 MT + t] = sgx_off<SC, MT>(f, t)
template <int SC, int MT>
__global__ void sgx_layout_kernel(int S, int* out) {
  const int n = S * 16 * MT;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = sgx_off<SC, MT>(i / (16 * MT), i % (16 * MT));
}

extern "C" int sp_sgx_layout(int sc, int MT, int S, int* out, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
#define X(SC, M) \
  if (sc == SC && MT == M) { sgx_layout_kernel<SC, M><<<32, 256, 0, s>>>(S, out); return (int)cudaGetLastError(); }
#define XS(SC) X(SC, 2) X(SC, 4) X(SC, 8)
  XS(SC_BASE) XS(SC_SWZ4) XS(SC_VEC4) XS(SC_VEC4P) XS(SC_BASEP) XS(SC_SWZ4P) XS(SC_VEC4T)
#undef XS
#undef X
  return -1;
}
