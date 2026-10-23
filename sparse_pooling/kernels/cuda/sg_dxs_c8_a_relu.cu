#include "sgx_dispatch.cuh"
// production scatter layouts of the grouped C = 8 dX, A_RELU (sparse_pool_sg_dxs.cuh), selected by the
// tuning profiles / named dX plans: sc = 9 (SC_VEC4TH: vectorized_transpose_hoisted; MT = 2, 4, 8;
// 8 / 16 warps) and 10 (SC_SWZ4M: swizzle4_capped; MT 4, 8, 8 warps).  The other layouts are
// experimental (experimental/sg_dxs_exp_c8_a_relu.cu, sp_sgxe_dx_c8_a_relu).  split: generic / mask, bf16: generic
extern "C" int sp_sgx_dx_c8_a_relu(const SgArgs* a, int sc, int MT, int NW, int split, int epi, int gx, int gy,
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
  XS(SC_VEC4TH) X(SC_SWZ4M, 4, 8) X(SC_SWZ4M, 8, 8)
#undef XS
#undef X
  return -1;
}
