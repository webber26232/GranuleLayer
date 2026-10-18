#include "sg_dispatch.cuh"
// dW instantiations for C = 64 (SG_DW_CFGS), generic / fast epilogue, split / bf16
template <int ACT, bool SPLIT, int EPI>
static int sg_dw_cfg(const SgArgs& a, int cfg, int gx, cudaStream_t s) {
#define X(I, NW, WPW, TS, ST, MINB) if (cfg == I) return sg_dw_launch<ACT, 64, NW, WPW, TS, ST, SPLIT, MINB, EPI>(a, gx, s);
  SG_DW_CFGS(X)
#undef X
  return -1;
}
template <int ACT>
static int sg_dw_act(const SgArgs& a, int cfg, int split, int epi, int gx, cudaStream_t s) {
  if (split) return epi ? sg_dw_cfg<ACT, true, 1>(a, cfg, gx, s) : sg_dw_cfg<ACT, true, 0>(a, cfg, gx, s);
  return epi ? sg_dw_cfg<ACT, false, 1>(a, cfg, gx, s) : sg_dw_cfg<ACT, false, 0>(a, cfg, gx, s);
}
extern "C" int sp_sg_dw_c64(const SgArgs* a, int act, int cfg, int split, int epi, int gx, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (act == A_ID) return sg_dw_act<A_ID>(*a, cfg, split, epi, gx, s);
  if (act == A_RELU) return sg_dw_act<A_RELU>(*a, cfg, split, epi, gx, s);
  return -1;
}
