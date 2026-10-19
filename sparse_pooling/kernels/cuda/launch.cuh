#pragma once
// Launch guard of the native kernels: per-device, per-instantiation preparation of a launch.
//
// Before the first launch of a kernel instantiation on a device it reads the kernel's attributes
// (cudaFuncGetAttributes: static shared memory, maximum threads per block for its register use)
// and the device's opt-in shared-memory limit (cudaDevAttrMaxSharedMemoryPerBlockOptin).  Every
// launch is then checked against these facts of the CURRENT device (the Python side sets it to the
// tensors' device) and the dynamic shared-memory opt-in is raised per device as needed.  Failures
// return an error code before any launch; no limit is a compile-time constant of one GPU model.
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>

enum {
  SP_ERR_SMEM = -3,      // static + dynamic shared memory above the device's opt-in limit
  SP_ERR_THREADS = -6,   // block larger than the kernel / device allows
  SP_ERR_DEVICE = -7,    // no current device, or device ordinal >= SP_MAX_DEVICES
  SP_ERR_ATTR = -8,      // cudaFuncGetAttributes / cudaFuncSetAttribute failed
  SP_ERR_NO_IMAGE = -9,  // the library holds no code for this device (wrong build target)
};

#ifndef SP_MAX_DEVICES
#define SP_MAX_DEVICES 16
#endif

struct LaunchSlot {  // zero-initialised static storage, one per kernel instantiation
  std::atomic<int> ready[SP_MAX_DEVICES];
  int max_threads[SP_MAX_DEVICES];
  size_t static_smem[SP_MAX_DEVICES];
  size_t optin[SP_MAX_DEVICES];
  size_t configured[SP_MAX_DEVICES];
};

inline std::mutex& sp_launch_mutex() {
  static std::mutex m;
  return m;
}

template <typename K>
static int sp_prepare(K kernel, size_t smem, int threads, LaunchSlot& slot) {
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= SP_MAX_DEVICES) {
    cudaGetLastError();
    return SP_ERR_DEVICE;
  }
  if (!slot.ready[dev].load(std::memory_order_acquire) || smem > slot.configured[dev]) {
    std::lock_guard<std::mutex> lock(sp_launch_mutex());
    if (!slot.ready[dev].load(std::memory_order_relaxed)) {
      cudaFuncAttributes fa;
      cudaError_t e = cudaFuncGetAttributes(&fa, kernel);
      if (e != cudaSuccess) {
        cudaGetLastError();
        return e == cudaErrorNoKernelImageForDevice || e == cudaErrorInvalidDeviceFunction ? SP_ERR_NO_IMAGE
                                                                                            : SP_ERR_ATTR;
      }
      int optin = 0;
      if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
        cudaGetLastError();
        return SP_ERR_ATTR;
      }
      slot.max_threads[dev] = fa.maxThreadsPerBlock;
      slot.static_smem[dev] = fa.sharedSizeBytes;
      slot.optin[dev] = (size_t)optin;
      slot.configured[dev] = 0;
      slot.ready[dev].store(1, std::memory_order_release);
    }
    if (threads > slot.max_threads[dev]) return SP_ERR_THREADS;
    if (smem + slot.static_smem[dev] > slot.optin[dev]) return SP_ERR_SMEM;
    if (smem > slot.configured[dev]) {  // grow the opt-in as needed (per device)
      if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) {
        cudaGetLastError();
        return SP_ERR_ATTR;
      }
      slot.configured[dev] = smem;
    }
    return 0;
  }
  if (threads > slot.max_threads[dev]) return SP_ERR_THREADS;
  if (smem + slot.static_smem[dev] > slot.optin[dev]) return SP_ERR_SMEM;
  return 0;
}
