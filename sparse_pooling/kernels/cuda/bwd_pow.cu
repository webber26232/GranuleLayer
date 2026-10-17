#include "bwd_dispatch.cuh"
int bwd_pool_pow(const BwdArgs& a, int act, int C, int split, int kind, int cfg, int gx, int gy, cudaStream_t s) {
  switch (a.p_int) {
    case 1: return bwd_c<P_POW, A_RELU, 1>(a, C, split, kind, cfg, gx, gy, s);
    case 2: return bwd_c<P_POW, A_RELU, 2>(a, C, split, kind, cfg, gx, gy, s);
    case 4: return bwd_c<P_POW, A_RELU, 4>(a, C, split, kind, cfg, gx, gy, s);
    case 8: return bwd_c<P_POW, A_RELU, 8>(a, C, split, kind, cfg, gx, gy, s);
    default: return bwd_c<P_POW, A_RELU, 0>(a, C, split, kind, cfg, gx, gy, s);
  }
}
