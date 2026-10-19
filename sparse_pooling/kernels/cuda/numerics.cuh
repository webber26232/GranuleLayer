#pragma once
// Compiler-sensitive numerical helpers.  The production build compiles with --use_fast_math
// (-ftz=true, approximate transcendental intrinsics); these helpers make the flush-to-zero /
// NaN semantics that the fast epilogues must share with the generic path explicit in the PTX,
// so they do not depend on what ptxas chooses for a C++ comparison.
//
//  * gt0_mask: (z > 0) as a 32-bit mask via set.gt.ftz.u32.f32: a positive subnormal z is inactive
//    (flushed), NaN is inactive.  The fast ReLU epilogues AND this mask into BF16 operands, which
//    ptxas 12.1 compiles to FSET.BF.GT.FTZ.  A mask that is only tested for non-zero was observed
//    to compile to FSETP.GT WITHOUT .FTZ (a positive subnormal z becomes active), i.e. a different
//    decision than the generic path's act'(z) (SCALE_DISPATCH_FUSED_BACKWARD.md section 6;
//    diagnostics/diag_proj_c8.cu).  Keep the mask consumed as data, never as a branch condition.
//  * mul_ftz: a * b with FTZ that the compiler cannot fold away (keeps the FTZ of c1 * 1 in the
//    fast epilogues, matching the generic path bit for bit).
//  * exp2_approx: ex2.approx.ftz.f32 (streaming LogMeanExp / softmax-weighted states).
// tests/test_codegen.py checks the FTZ forms in the SASS of representative production kernels.
#include <stdint.h>

__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// (z > 0) as a 32-bit mask with the generic path's compare semantics (FTZ, NaN -> false)
__device__ __forceinline__ uint32_t gt0_mask(float z) {
  uint32_t m;
  asm("set.gt.ftz.u32.f32 %0, %1, 0f00000000;" : "=r"(m) : "f"(z));
  return m;
}

__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));  // not foldable: keeps the FTZ of a * 1
  return r;
}
