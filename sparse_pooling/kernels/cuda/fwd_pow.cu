#include "fwd_dispatch.cuh"
int fwd_pool_pow(const FwdArgs& a, int act, int track, int C, int MT, int n_tiles, int PS, int nw, cudaStream_t s) {
  switch (a.p_int) {
    case 1: return fwd_c<P_POW, A_RELU, false, 1>(a, C, MT, n_tiles, PS, nw, s);
    case 2: return fwd_c<P_POW, A_RELU, false, 2>(a, C, MT, n_tiles, PS, nw, s);
    case 4: return fwd_c<P_POW, A_RELU, false, 4>(a, C, MT, n_tiles, PS, nw, s);
    case 8: return fwd_c<P_POW, A_RELU, false, 8>(a, C, MT, n_tiles, PS, nw, s);
    default: return fwd_c<P_POW, A_RELU, false, 0>(a, C, MT, n_tiles, PS, nw, s);
  }
}
