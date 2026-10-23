// EXPERIMENTAL kgx unstructured fixed-mask (CSR) kernels (kgx_csr.cuh): kind 0 fwd, 1 dX, 2 dW
#include "kgx_dispatch.cuh"

extern "C" int sp_kc(const KxArgs* a, int kind, int n_chunks, int n_tiles, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  using C = KxCfg;
  if (!kx_args_ok(a, KX_CSR)) return -2;
  if (n_chunks == 0) return 0;
  if (kind == 0) {
    static LaunchSlot slot;
    return kx_launch(kc_fwd_kernel<C::CS_TPL, C::CS_NW>, dim3(n_tiles, n_chunks), C::CS_NW * 32, kx_smem(*a, 0, KX_CSR),
                     *a, s, slot);
  }
  if (kind == 1) {
    static LaunchSlot slot;
    return kx_launch(kc_dx_kernel<C::CSX_NW>, dim3(n_tiles, n_chunks), C::CSX_NW * 32, kx_smem(*a, 1, KX_CSR), *a, s,
                     slot);
  }
  if (kind == 2) {
    static LaunchSlot slot;
    return kx_launch(kc_dw_kernel<C::CSW_BT, C::CSW_ST, 8>, dim3(n_chunks), 8 * 32, kx_smem(*a, 2, KX_CSR), *a, s,
                     slot);
  }
  return -2;
}
