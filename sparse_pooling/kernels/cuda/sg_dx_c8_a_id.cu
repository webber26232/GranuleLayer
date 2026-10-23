#include "sg_dispatch.cuh"
// dX instantiations for C = 8, A_ID: MT = 1, 2, 4, 8; 8 / 16 warps; split: generic / mask epilogue, bf16: generic
extern "C" int sp_sg_dx_c8_a_id(const SgArgs* a, int MT, int NW, int split, int epi, int gx, int gy, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
#define X(M, W)                                                                                 \
  if (MT == M && NW == W) {                                                                     \
    if (split && epi == 2) return sg_dx_launch<A_ID, 8, M, W, true, 2>(*a, gx, gy, s);         \
    if (split && epi == 0) return sg_dx_launch<A_ID, 8, M, W, true, 0>(*a, gx, gy, s);         \
    if (!split && epi == 0) return sg_dx_launch<A_ID, 8, M, W, false, 0>(*a, gx, gy, s);       \
    return -2;                                                                                  \
  }
  X(1, 8) X(1, 16) X(2, 8) X(2, 16) X(4, 8) X(4, 16) X(8, 8) X(8, 16)
#undef X
  return -1;
}
