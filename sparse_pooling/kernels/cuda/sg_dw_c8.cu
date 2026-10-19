#include "sg_dispatch.cuh"
// dW instantiations for C = 8 (SG_DW_CFGS), generic / fast epilogue, split / bf16; proj = 1: the
// compact projection (dw_block_c8_compact) of the fast Mean(ReLU) epilogue
template <int ACT, bool SPLIT, int EPI, int PROJ>
static int sg_dw_cfg(const SgArgs& a, int cfg, int gx, cudaStream_t s) {
#define X(I, NW, WPW, TS, ST, MINB) if (cfg == I) return sg_dw_launch<ACT, 8, NW, WPW, TS, ST, SPLIT, MINB, EPI, PROJ>(a, gx, s);
  SG_DW_CFGS(X)
#undef X
  return -1;
}
template <int ACT>
static int sg_dw_act(const SgArgs& a, int cfg, int split, int epi, int proj, int gx, cudaStream_t s) {
  if (proj) {  // compact projection: fast ReLU epilogue only
    if (ACT != A_RELU || !epi) return -2;
    return split ? sg_dw_cfg<ACT, true, 1, 1>(a, cfg, gx, s) : sg_dw_cfg<ACT, false, 1, 1>(a, cfg, gx, s);
  }
  if (split) return epi ? sg_dw_cfg<ACT, true, 1, 0>(a, cfg, gx, s) : sg_dw_cfg<ACT, true, 0, 0>(a, cfg, gx, s);
  return epi ? sg_dw_cfg<ACT, false, 1, 0>(a, cfg, gx, s) : sg_dw_cfg<ACT, false, 0, 0>(a, cfg, gx, s);
}
extern "C" int sp_sg_dw_c8(const SgArgs* a, int act, int cfg, int split, int epi, int proj, int gx, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (act == A_ID) return sg_dw_act<A_ID>(*a, cfg, split, epi, proj, gx, s);
  if (act == A_RELU) return sg_dw_act<A_RELU>(*a, cfg, split, epi, proj, gx, s);
  return -1;
}
