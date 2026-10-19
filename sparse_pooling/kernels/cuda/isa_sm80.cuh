#pragma once
// ISA-family primitives of the native kernels (inline PTX).  Requirements (PTX ISA Target ISA Notes,
// verified with ptxas 12.1 by tests/test_isa_requirements.py):
//   mma.sync.aligned.m16n8k8 / m16n8k16 .bf16 -> .f32   sm_80+   (PTX 7.0)
//   cp.async.cg.shared.global, commit_group, wait_group  sm_80+   (PTX 7.0)
//   ldmatrix.sync.aligned.m8n8.{x2,x4}[.trans].b16      sm_75+   (PTX 6.5)
//   movmatrix.sync.aligned.m8n8.trans.b16               sm_75+   (PTX 7.8)
// The fragment layouts assume 32-lane warps.  These are family requirements, not RTX 3090 facts:
// any device with these instructions may run the kernels (legality); which tiling is fast there is
// a tuning-profile question (sparse_pooling/backends/profiles).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#error "GranuleLayer native kernels need sm_80 or newer (BF16 mma.sync, cp.async)"
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

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

