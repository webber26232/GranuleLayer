// extern "C" entry points (loaded with ctypes; no PyTorch extension build).
#include "common.cuh"
#include "sparse_pool_fwd.cuh"
#include "sparse_pool_bwd.cuh"

int fwd_pool_max(const FwdArgs&, int, int, int, int, int, int, int, cudaStream_t);
int fwd_pool_mean(const FwdArgs&, int, int, int, int, int, int, int, cudaStream_t);
int fwd_pool_lme(const FwdArgs&, int, int, int, int, int, int, int, cudaStream_t);
int fwd_pool_pow(const FwdArgs&, int, int, int, int, int, int, int, cudaStream_t);
int fwd_pool_sw(const FwdArgs&, int, int, int, int, int, int, int, cudaStream_t);

extern "C" int sp_fwd(const FwdArgs* a, int pool, int act, int track, int C, int MT, int n_tiles, int PS,
                      int nw, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  switch (pool) {
    case P_MAX: return fwd_pool_max(*a, act, track, C, MT, n_tiles, PS, nw, s);
    case P_MEAN: return fwd_pool_mean(*a, act, 0, C, MT, n_tiles, PS, nw, s);
    case P_LME: return fwd_pool_lme(*a, act, 0, C, MT, n_tiles, PS, nw, s);
    case P_POW: return fwd_pool_pow(*a, act, 0, C, MT, n_tiles, PS, nw, s);
    case P_SW: return fwd_pool_sw(*a, act, 0, C, MT, n_tiles, PS, nw, s);
  }
  return -1;
}

extern "C" int sp_fwd_args_size() { return (int)sizeof(FwdArgs); }

int bwd_pool_mean(const BwdArgs&, int, int, int, int, int, int, int, cudaStream_t);
int bwd_pool_lme(const BwdArgs&, int, int, int, int, int, int, int, cudaStream_t);
int bwd_pool_sw(const BwdArgs&, int, int, int, int, int, int, int, cudaStream_t);
int bwd_pool_pow(const BwdArgs&, int, int, int, int, int, int, int, cudaStream_t);

// kind 0: k_dx (grid gx = token tiles, gy = unit ranges); kind 1: k_dw (gx = chunks, gy = units)
extern "C" int sp_bwd(const BwdArgs* a, int pool, int act, int C, int split, int kind, int cfg, int gx, int gy,
                      void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  switch (pool) {
    case P_MEAN: return bwd_pool_mean(*a, act, C, split, kind, cfg, gx, gy, s);
    case P_LME: return bwd_pool_lme(*a, act, C, split, kind, cfg, gx, gy, s);
    case P_POW: return bwd_pool_pow(*a, act, C, split, kind, cfg, gx, gy, s);
    case P_SW: return bwd_pool_sw(*a, act, C, split, kind, cfg, gx, gy, s);
  }
  return -1;
}
extern "C" int sp_bwd_args_size() { return (int)sizeof(BwdArgs); }
