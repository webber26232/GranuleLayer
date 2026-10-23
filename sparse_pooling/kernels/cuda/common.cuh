#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <climits>
#include <cmath>

#include "isa_sm80.cuh"
#include "numerics.cuh"

typedef __nv_bfloat16 bf16;

enum { P_MAX = 0, P_TOPK = 1, P_MEAN = 2, P_LME = 3, P_POW = 4, P_SW = 5 };
enum { A_ID = 0, A_RELU = 1, A_GELU = 2 };

template <int ACT>
__device__ __forceinline__ float act_f(float z) {
  if (ACT == A_ID) return z;
  if (ACT == A_RELU) return fmaxf(z, 0.f);
  return 0.5f * z * (1.f + erff(z * 0.7071067811865476f));
}
template <int ACT>
__device__ __forceinline__ float dact_f(float z) {
  if (ACT == A_ID) return 1.f;
  if (ACT == A_RELU) return z > 0.f ? 1.f : 0.f;
  float cdf = 0.5f * (1.f + erff(z * 0.7071067811865476f));
  return cdf + z * 0.3989422804014327f * __expf(-0.5f * z * z);
}

// q^p for q >= 0 (integer fast paths; p_int == 0 -> generic)
__device__ __forceinline__ float powp(float q, float p, int p_int) {
  if (p_int == 1) return q;
  if (p_int == 2) return q * q;
  if (p_int == 4) { float q2 = q * q; return q2 * q2; }
  if (p_int == 8) { float q2 = q * q; float q4 = q2 * q2; return q4 * q4; }
  return q > 0.f ? exp2f(p * log2f(q)) : 0.f;
}

__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t v) {
  float2 r;
  r.x = __uint_as_float(v << 16);
  r.y = __uint_as_float(v & 0xffff0000u);
  return r;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
