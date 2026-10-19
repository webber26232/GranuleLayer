// Fixed-order second stage of the deterministic grouped reductions (PhysicalPlan.reduction =
// "deterministic"; cuda_ops.CudaBackend._sg_det_*).
//
// The first stage (sg_fwd / sg_dx with SgArgs::part set) stores every CTA's partial into rows
// of a workspace that only that CTA writes: part[row * tw + t] for the token window
// [w0, w0 + tw).  This kernel owns each destination (row r of out, token w0 + t) exclusively and
// sums its contributor rows rec[off[r] .. off[r + 1]) sequentially in list order, starting from
// +0.0f, then writes it once (no atomics, no read of out):
//
//   out[r * ostride + w0 + t] = (((0 + part[rec[off[r]]]) + part[rec[off[r] + 1]]) + ...)
//
// The contributor lists are static (built once from the work lists, supergroups.SgExec.det_*):
//   forward: r = output g, rec = the items (buckets) of g in item order (= supergroup order)
//   dX:      r = global input feature d, rec = chunk * S + local feature over every chunk whose
//            supergroup contains d, in chunk order (= supergroup, then item range order)
// The sum for a token depends only on the list order and the partial values, not on the
// window, the grid or the thread mapping, so any workspace cap gives the same bits.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void __launch_bounds__(256) sg_reduce_kernel(const float* __restrict__ part, long long tw,
                                                        const int* __restrict__ off, const int* __restrict__ rec,
                                                        float* __restrict__ out, long long ostride, int w0, int ntok) {
  const int r = blockIdx.x;
  const int t = 4 * (blockIdx.y * blockDim.x + threadIdx.x);  // 4 tokens per thread (ntok % 4 == 0)
  if (t >= ntok) return;
  const int k0 = off[r], k1 = off[r + 1];
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = k0; k < k1; k++) {
    const float4 p = *reinterpret_cast<const float4*>(part + (long long)rec[k] * tw + t);
    v.x += p.x;
    v.y += p.y;
    v.z += p.z;
    v.w += p.w;
  }
  *reinterpret_cast<float4*>(out + (long long)r * ostride + w0 + t) = v;
}

extern "C" int sp_sg_reduce(const float* part, long long tw, const int* off, const int* rec, int rows, float* out,
                            long long ostride, int w0, int ntok, void* stream) {
  if (ntok % 4 || tw % 4 || ostride % 4 || w0 % 4) return -2;  // float4 rows (B % 8 == 0, tiles of 16 tokens)
  if (rows == 0 || ntok == 0) return 0;
  dim3 grid(rows, (ntok / 4 + 255) / 256);
  sg_reduce_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(part, tw, off, rec, out, ostride, w0, ntok);
  return (int)cudaGetLastError();
}
