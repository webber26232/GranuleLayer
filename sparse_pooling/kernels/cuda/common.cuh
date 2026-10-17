#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <climits>
#include <cmath>

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

__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// q^p for q >= 0 (integer fast paths; p_int == 0 -> generic)
__device__ __forceinline__ float powp(float q, float p, int p_int) {
  if (p_int == 1) return q;
  if (p_int == 2) return q * q;
  if (p_int == 4) { float q2 = q * q; return q2 * q2; }
  if (p_int == 8) { float q2 = q * q; float q4 = q2 * q2; return q4 * q4; }
  return q > 0.f ? exp2f(p * log2f(q)) : 0.f;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
// transpose an 8x8 b16 matrix held in mma fragment layout
__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}

// D = A(16x8, row) * B(8x8, col) + C, BF16 inputs, FP32 accumulate
__device__ __forceinline__ void mma_m16n8k8_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t b,
                                                 float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(b), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
// D += A(16x16, row) * B(16x8, col)
__device__ __forceinline__ void mma_m16n8k16_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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
