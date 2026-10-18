// Phase-3 experiment A3: is mma(C = b) equal to mma(C = 0) + b?  Build: nvcc -arch=sm_86 -O2 mma_c_rounding.cu
#include <cstdio>
#include <cuda_bf16.h>
#include <stdint.h>
// one m16n8k8: A row 0 = a[0..7], B col 0 = b[0..7]; compare D(C=c) with D(C=0)+c, and fp64 exact
__global__ void k(const float* a, const float* b, float c, float* out) {
  int lane = threadIdx.x;
  // A fragment: a0 = rows g, cols 2t,2t+1 ; a1 = rows g+8. B: b = rows(k) 2t,2t+1, col g
  int g = lane >> 2, t = lane & 3;
  __nv_bfloat162 A0 = __floats2bfloat162_rn(g == 0 ? a[2*t] : 0.f, g == 0 ? a[2*t+1] : 0.f);
  __nv_bfloat162 A1 = __floats2bfloat162_rn(0.f, 0.f);
  __nv_bfloat162 B0 = __floats2bfloat162_rn(g == 0 ? b[2*t] : 0.f, g == 0 ? b[2*t+1] : 0.f);
  uint32_t a0 = *(uint32_t*)&A0, a1 = *(uint32_t*)&A1, bb = *(uint32_t*)&B0;
  float d[4], z[4];
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a0), "r"(a1), "r"(bb), "f"(c), "f"(c), "f"(c), "f"(c));
  float zc = 0.f;
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
    : "=f"(z[0]), "=f"(z[1]), "=f"(z[2]), "=f"(z[3]) : "r"(a0), "r"(a1), "r"(bb), "f"(zc), "f"(zc), "f"(zc), "f"(zc));
  if (lane == 0) { out[0] = d[0]; out[1] = z[0]; out[2] = __fadd_rn(z[0], c); }
}
int main() {
  struct Case { const char* name; float a[8], b[8], c; } cs[] = {
    {"1 + 2^-30, c=-1", {1, 0x1p-15f, 0,0,0,0,0,0}, {1, 0x1p-15f, 0,0,0,0,0,0}, -1.f},
    {"1 + 2^-24, c=-1", {1, 0x1p-12f, 0,0,0,0,0,0}, {1, 0x1p-12f, 0,0,0,0,0,0}, -1.f},
    {"1 + 3*2^-25, c=0.5", {1, 0x1p-12f, 0x1p-13f, 0,0,0,0,0}, {1, 0x1p-13f, 0x1p-12f,0,0,0,0,0}, 0.5f},
    {"2^20 + 1.5, c=-2^20", {1024, 1, 0.5f,0,0,0,0,0}, {1024, 1, 1,0,0,0,0,0}, -1048576.f},
    {"random-ish", {1.5f, -0.75f, 0.3f, 2.f, -1.f, 0.125f, 3.f, 0.01f}, {0.7f, 1.1f, -2.f, 0.4f, 0.9f, -3.f, 0.2f, 1.f}, 0.1f},
  };
  float *da, *db, *dout; cudaMalloc(&da, 32); cudaMalloc(&db, 32); cudaMalloc(&dout, 12);
  for (auto& s : cs) {
    cudaMemcpy(da, s.a, 32, cudaMemcpyHostToDevice); cudaMemcpy(db, s.b, 32, cudaMemcpyHostToDevice);
    k<<<1, 32>>>(da, db, s.c, dout); float o[3]; cudaMemcpy(o, dout, 12, cudaMemcpyDeviceToHost);
    double ex = s.c; for (int i = 0; i < 8; i++) ex += (double)__bfloat162float(__float2bfloat16(s.a[i])) * __bfloat162float(__float2bfloat16(s.b[i]));
    printf("%-22s mma(C=c)=%.10a  mma(C=0)+c=%.10a  exact=%.10a  %s\n", s.name, o[0], o[2], ex, o[0] == o[2] ? "SAME" : "DIFFERENT");
  }
}
