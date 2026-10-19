#include <cuda_bf16.h>
#include <stdint.h>
__global__ void k(float* o, const float* g, uint32_t a0, uint32_t a1, uint32_t b) {
  __shared__ __align__(16) unsigned short s[256];
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(s);
  uint32_t r[4]; float d[4];
#if defined(P_MMA8)
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(b));
#elif defined(P_MMA16)
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a0),"r"(a1),"r"(b),"r"(b));
#elif defined(P_LDM)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(addr)); d[0]=r[0];
#elif defined(P_MOVM)
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r[0]) : "r"(a0)); d[0]=r[0];
#elif defined(P_CPASYNC)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(addr), "l"(g), "r"(16));
  asm volatile("cp.async.commit_group;\n" ::); asm volatile("cp.async.wait_group 0;\n" ::); asm volatile("cp.async.wait_all;\n" ::);
  d[0]=s[3];
#endif
  o[threadIdx.x] = d[0];
}
