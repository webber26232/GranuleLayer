// Projection / ReLU-decision diagnostic of the grouped C = 8 backward kernels
// (benchmarks/proj_mask_diag.py, results/SCALE_DISPATCH_FUSED_BACKWARD.md section 6).
//
// For each work item (flat unit u, 16-token tile tt, valid neurons n_g of the unit's output)
// one warp recomputes the pre-activations of the unit's 4 blocks exactly as the two kernels do:
//   dX (sg_dx_tile / tile_mask):  z [16 tokens x 8 neurons] = mma(A = X tile (ldmatrix.trans),
//        B = packed weight fragment, C = bias pair of this lane's two neuron columns)
//   dW (dw_block, C = 8):         z^T [16 rows (8 neurons + 8 padding) x 8 tokens] per token half
//        = mma(A = {packed weight fragment, 0}, B = the same ldmatrix.trans registers,
//        C = bias of this lane's neuron row (biasT, built by shuffles as in sg_dw_body))
// and records z of both paths (canonical [item][block][token][neuron] FP32 bits) and the ReLU
// decisions of the dX mask epilogue (tile_mask), the dW fast epilogue (FastDz: z > 0 ? one : 0)
// and the dW mask epilogue (gt0_mask), with the MASKED (ragged-output) variants of each.
// The X slab of the block's 8 records is staged with the Swz<1> layout of both kernels.
// Decisions are stored as the raw words the kernels consume (32-bit masks, FP32 dz bits), never
// converted to booleans on the device: when a set.gt.ftz mask is only tested for non-zero,
// ptxas 12.1 emits FSETP.GT without .FTZ (positive subnormal z -> active), whereas the
// production kernels AND the mask into the BF16 operands (FSET.BF.GT.FTZ: inactive).
#include "../sparse_pool_sg.cuh"

__global__ void diag_proj_c8_kernel(const SgArgs a, const int* work, int n, float* zdx, float* zdw, uint32_t* mdx,
                                    uint32_t* mdxm, uint32_t* mdwf, uint32_t* mdwfm, uint32_t* mdwm, uint32_t* mdwmm) {
  __shared__ __align__(16) bf16 slabs[8][8 * 16];
  const int lane = threadIdx.x & 31, wl = threadIdx.x >> 5;
  const int item = blockIdx.x * 8 + wl;
  if (item >= n) return;  // whole warps exit together
  const int u = work[3 * item], tt = work[3 * item + 1], n_g = work[3 * item + 2];
  const int t0 = tt * 16;
  FwdArgs fa;
  fa.idx = a.idx; fa.W = a.W; fa.bias = a.bias;
  Unit<8> un;
  load_unit<8>(un, fa, u, lane);
  bf16* slab = slabs[wl];
  const uint32_t slab_s = smem_u32(slab);
  for (int bu = 0; bu < 4; bu++) {
    BlockRegs<8> blk;
    unit_block<8>(blk, un, bu);
    const int jb = (u % a.NU) * 4 + bu;  // block within the output (slot j * 8 + column)
    // stage the block's 8 records (local row k = fan-in position k) x 16 tokens, Swz<1>
    __syncwarp();
    for (int i = lane; i < 8 * 16; i += 32) {
      const int k = i >> 4, t = i & 15;
      const int f = __shfl_sync(0xffffffff, blk.fidx, k);  // all lanes participate (i loop is uniform)
      const int tok = t0 + t;
      slab[Swz<1>::off(k, t >> 3) + (t & 7)] = tok < a.B ? a.X[(long long)f * a.sxf + tok] : __float2bfloat16(0.f);
    }
    __syncwarp();
    const int krow = lane & 7;  // local row of this lane's ldmatrix address
    uint32_t af[2];
    ldmatrix_x2_trans(af, slab_s + 2 * Swz<1>::off(krow, (lane >> 3) & 1));  // sg_gather<1> == dw_block's xa
    // ---- dX path
    {
      float d[4];
      const float2 bf = bf16x2_to_float2(blk.bb[0]);
      mma_m16n8k8_bf16(d, af[0], af[1], blk.w[0], bf.x, bf.y, bf.x, bf.y);
      uint32_t m[4], mm[4];
      tile_mask<A_RELU, false>(m, af, blk.w[0], blk.bb[0], 0, 1 << 30);
      tile_mask<A_RELU, true>(mm, af, blk.w[0], blk.bb[0], jb * 8 + 2 * (lane & 3), n_g);
      for (int i = 0; i < 4; i++) {
        const int tok = (lane >> 2) + 8 * (i >> 1), col = 2 * (lane & 3) + (i & 1);
        const long long o = (((long long)item * 4 + bu) * 16 + tok) * 8 + col;
        zdx[o] = d[i];
        mdx[o] = m[i];
        mdxm[o] = mm[i];
      }
    }
    // ---- dW path (C = 8: the second n8 tile does not exist -> zero A rows 8-15, zero bias)
    {
      float biasT[2];
      for (int nt = 0; nt < 2; nt++) {  // as sg_dw_body
        const uint32_t word = (nt < 1) ? blk.bb[nt] : 0u;
        const int gq = lane >> 2;
        const uint32_t src = __shfl_sync(0xffffffff, word, gq >> 1);
        const float2 bf = bf16x2_to_float2(src);
        biasT[nt] = (gq & 1) ? bf.y : bf.x;
      }
      const float b0 = biasT[0], b1 = biasT[1];
      const int r0 = jb * 8 + (lane >> 2);
      for (int th = 0; th < 2; th++) {
        float d[4];
        mma_m16n8k8_bf16(d, blk.w[0], 0u, af[th], b0, b0, b1, b1);
        for (int i = 0; i < 2; i++) {  // rows g (neuron lane / 4); rows g + 8 are padding
          const int tok = 8 * th + 2 * (lane & 3) + i, nrn = lane >> 2;
          const long long o = (((long long)item * 4 + bu) * 16 + tok) * 8 + nrn;
          zdw[o] = d[i];
          const float c1 = 1.f, one = mul_ftz(c1, 1.f);
          const float vf = FastDz<P_MEAN, A_RELU, 0>::dz(d[i], c1, 0.f, one, 0.f, 1.f, 0.f);
          const float vf_m = (r0 < n_g) ? vf : 0.f;  // dw_block: dz = (MASKED && !v0) ? 0 : v
          const uint32_t gm = gt0_mask(d[i]);
          mdwf[o] = __float_as_uint(vf);
          mdwfm[o] = __float_as_uint(vf_m);
          mdwm[o] = gm;
          mdwmm[o] = (r0 < n_g) ? gm : 0u;
        }
      }
    }
  }
}

extern "C" int sp_diag_proj_c8(const SgArgs* a, const int* work, int n, float* zdx, float* zdw, uint32_t* m,
                               void* stream) {
  if (n <= 0) return 0;
  const long long e = (long long)n * 4 * 16 * 8;  // m: 6 consecutive [n, 4, 16, 8] uint32 planes
  diag_proj_c8_kernel<<<(n + 7) / 8, 256, 0, (cudaStream_t)stream>>>(*a, work, n, zdx, zdw, m, m + e, m + 2 * e,
                                                                     m + 3 * e, m + 4 * e, m + 5 * e);
  return (int)cudaGetLastError();
}
