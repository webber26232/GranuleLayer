#include "sg_dispatch.cuh"
// forward instantiations for C = 64: token tile 16 MT (MT = 1, 2, 4, 8), 4 / 8 / 16 warps
template <int ACT>
static int sg_fwd_act(const SgArgs& a, int MT, int NW, int gx, int gy, cudaStream_t s) {
#define F(M, W) if (MT == M && NW == W) return sg_fwd_launch<ACT, 64, M, W>(a, gx, gy, s);
  F(1, 4) F(1, 8) F(1, 16) F(2, 4) F(2, 8) F(2, 16) F(4, 4) F(4, 8) F(4, 16) F(8, 4) F(8, 8) F(8, 16)
#undef F
  return -1;
}
extern "C" int sp_sg_fwd_c64(const SgArgs* a, int act, int MT, int NW, int gx, int gy, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (act == A_ID) return sg_fwd_act<A_ID>(*a, MT, NW, gx, gy, s);
  if (act == A_RELU) return sg_fwd_act<A_RELU>(*a, MT, NW, gx, gy, s);
  return -1;
}
