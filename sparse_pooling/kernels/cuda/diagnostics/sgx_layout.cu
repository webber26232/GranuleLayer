#include "../sgx_dispatch.cuh"
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
  XS(SC_BASE) XS(SC_SWZ4) XS(SC_VEC4) XS(SC_VEC4P) XS(SC_BASEP) XS(SC_SWZ4P) XS(SC_VEC4T) XS(SC_VEC4H) XS(SC_VEC4TH)
#undef XS
#undef X
  return -1;
}
