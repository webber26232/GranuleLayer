#include "bwd_dispatch.cuh"
int bwd_pool_mean(const BwdArgs& a, int act, int C, int split, int kind, int cfg, int gx, int gy, cudaStream_t s) {
  return bwd_act<P_MEAN>(a, act, C, split, kind, cfg, gx, gy, s);
}
