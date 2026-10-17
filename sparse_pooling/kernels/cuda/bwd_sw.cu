#include "bwd_dispatch.cuh"
int bwd_pool_sw(const BwdArgs& a, int act, int C, int split, int kind, int cfg, int gx, int gy, cudaStream_t s) {
  return bwd_act<P_SW>(a, act, C, split, kind, cfg, gx, gy, s);
}
