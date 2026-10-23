#include "../sgx_dispatch.cuh"
// experimental (explicit-only) scatter layouts of the grouped C = 8 dX, A_RELU (sparse_pool_sg_dxs.cuh):
// sc = 1 (SC_SWZ4) / 2 (SC_VEC4) / 3 (SC_VEC4P) / 5 (SC_BASEP) / 6 (SC_SWZ4P) / 7 (SC_VEC4T) /
// 8 (SC_VEC4H); MT = 2, 4, 8; 8 / 16 warps; split: generic / mask, bf16: generic
extern "C" int sp_sgxe_dx_c8_a_relu(const SgArgs* a, int sc, int MT, int NW, int split, int epi, int gx, int gy,
                                   void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
#define X(SC, M, W)                                                                                \
  if (sc == SC && MT == M && NW == W) {                                                            \
    if (split && epi == 2) return sgx_dx_launch<A_RELU, 8, M, W, true, 2, SC, 1>(*a, gx, gy, s);     \
    if (split && epi == 0) return sgx_dx_launch<A_RELU, 8, M, W, true, 0, SC, 1>(*a, gx, gy, s);     \
    if (!split && epi == 0) return sgx_dx_launch<A_RELU, 8, M, W, false, 0, SC, 1>(*a, gx, gy, s);   \
    return -2;                                                                                     \
  }
#define XS(SC) X(SC, 2, 8) X(SC, 2, 16) X(SC, 4, 8) X(SC, 4, 16) X(SC, 8, 8) X(SC, 8, 16)
  XS(SC_SWZ4) XS(SC_VEC4) XS(SC_VEC4P) XS(SC_BASEP) XS(SC_SWZ4P) XS(SC_VEC4T) XS(SC_VEC4H)
#undef XS
#undef X
  return -1;
}
