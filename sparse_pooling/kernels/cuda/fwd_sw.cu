#include "fwd_dispatch.cuh"
int fwd_pool_sw(const FwdArgs& a, int act, int track, int C, int MT, int n_tiles, int PS, int nw, cudaStream_t s) {
  return fwd_act<P_SW>(a, act, track, C, MT, n_tiles, PS, nw, s);
}
