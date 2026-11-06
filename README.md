# GranuleLayer

A PyTorch layer that projects activations through ~100M sparse logical neurons (fan-in 8) and pools
them without ever materialising the hidden tensor, with native CUDA kernels (sm_80+, tuned and
validated on an RTX 3090) and a Triton fallback.

* Usage, design and measurements: [sparse_pooling/README.md](sparse_pooling/README.md)
* Cover-TopK/16 (opt-in layer: frozen 16-input supports selected from a K64 envelope, compact K16 / C16 execution,
  scoped RTX 3090 dX tuning): [docs/cover_topk16.md](docs/cover_topk16.md)
* GPU support and portability: [docs/gpu_portability.md](docs/gpu_portability.md)
* Architecture: [docs/architecture.md](docs/architecture.md); adding a GPU: [docs/adding_a_gpu.md](docs/adding_a_gpu.md)
* Evidence of the optimization campaign: [docs/evidence.md](docs/evidence.md)
