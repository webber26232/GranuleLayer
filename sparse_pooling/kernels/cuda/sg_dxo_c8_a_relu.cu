#include "sgx_dispatch.cuh"
// owner (register-accumulator) variant of the grouped C = 8 dX, A_RELU (sparse_pool_sg_dxs.cuh):
// NS = ceil(S / 8) n8 tiles of local features (S <= 32); MT = 2, 4, 8; 8 / 16 warps
extern "C" int sp_sgo_dx_c8_a_relu(const SgArgs* a, int MT, int NW, int split, int epi, int gx, int gy, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  const int ns = a->S <= 8 ? 1 : (a->S <= 16 ? 2 : (a->S <= 32 ? 4 : 0));
#define X(N, M, W)                                                                                    \
  if (ns == N && MT == M && NW == W) {                                                                \
    if (split && epi == 2) return sgx_dx_launch<A_RELU, 8, M, W, true, 2, SC_OWNER, N>(*a, gx, gy, s);   \
    if (split && epi == 0) return sgx_dx_launch<A_RELU, 8, M, W, true, 0, SC_OWNER, N>(*a, gx, gy, s);   \
    if (!split && epi == 0) return sgx_dx_launch<A_RELU, 8, M, W, false, 0, SC_OWNER, N>(*a, gx, gy, s); \
    return -2;                                                                                        \
  }
#define XN(N) X(N, 2, 8) X(N, 2, 16) X(N, 4, 8) X(N, 4, 16) X(N, 8, 8) X(N, 8, 16)
  XN(1) XN(2) XN(4)
#undef XN
#undef X
  return ns ? -1 : -5;
}
