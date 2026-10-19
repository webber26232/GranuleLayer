// Screening kernels for the fused-backward resource model (benchmarks/screen_fusion.py,
// results/SCALE_DISPATCH_FUSED_BACKWARD.md section 7).  Timing observations only: they do
// not compute gradients and are never part of a training path.
//
// screen_flush: the global dX flush pattern of a candidate schedule in isolation.  CTA
//   (blockIdx.x) belongs to supergroup cta_q[blockIdx.x]; it loops over `cta_windows`
//   consecutive token windows of Tw tokens starting at window blockIdx.y * cta_windows.
//   Per window it fills its local [S][Tw] FP32 slab with the known contribution `val`
//   (the stand-in for the local dX partials), synchronizes, and flushes the slab with one
//   FP32 atomicAdd per (feature, token) into dXT [D, B] (token-contiguous, i = f * Tw + t,
//   as sg_dx), then synchronizes again (slab reuse).  With val = 1 the result is exact:
//   dXT[f, t] = number of CTAs whose supergroup contains f and whose windows cover t.
// screen_rmw: one FP32 read-modify-write pass over n4 float4 (the dW / db partial
//   write-back + reload of a coarse-neuron / token-window schedule), buf += val.
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void screen_flush_kernel(const int16_t* feat, const int* cta_q, int S, int B, int Tw, int cta_windows,
                                    float* dXT, float val) {
  extern __shared__ float slab[];
  const int16_t* fq = feat + (long long)cta_q[blockIdx.x] * S;
  const int n = S * Tw;
  const int nwin = (B + Tw - 1) / Tw;
  for (int wi = 0; wi < cta_windows; wi++) {
    const int w = blockIdx.y * cta_windows + wi;
    if (w >= nwin) break;
    for (int i = threadIdx.x; i < n; i += blockDim.x) slab[i] = val;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int f = i / Tw, tok = w * Tw + i % Tw;
      const float v = slab[i];
      if (tok < B && v != 0.f) atomicAdd(dXT + (long long)fq[f] * B + tok, v);
    }
    __syncthreads();
  }
}

__global__ void screen_rmw_kernel(float4* buf, long long n4, float val) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
    float4 v = buf[i];
    v.x += val; v.y += val; v.z += val; v.w += val;
    buf[i] = v;
  }
}

extern "C" int sp_screen_flush(const int16_t* feat, const int* cta_q, int n_cta, int S, int B, int Tw,
                               int cta_windows, int threads, float* dXT, float val, void* stream) {
  const int nwin = (B + Tw - 1) / Tw;
  const dim3 grid(n_cta, (nwin + cta_windows - 1) / cta_windows);
  const size_t smem = (size_t)S * Tw * 4;
  if (smem > 99 * 1024) return -3;
  if (smem > 48 * 1024) cudaFuncSetAttribute(screen_flush_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  screen_flush_kernel<<<grid, threads, smem, (cudaStream_t)stream>>>(feat, cta_q, S, B, Tw, cta_windows, dXT, val);
  return (int)cudaGetLastError();
}

extern "C" int sp_screen_rmw(float* buf, long long n4, float val, void* stream) {
  screen_rmw_kernel<<<82 * 16, 256, 0, (cudaStream_t)stream>>>(reinterpret_cast<float4*>(buf), n4, val);
  return (int)cudaGetLastError();
}
