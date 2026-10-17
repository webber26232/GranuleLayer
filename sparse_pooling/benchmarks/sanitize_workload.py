"""Small forward+backward workload for NVIDIA Compute Sanitizer
(see benchmarks/run_sanitizer.sh).  Covers every kernel family."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402

CASES = [
    (8, "max", "save_winners", "cuda"),
    (8, "max", "minimal_stats", "cuda"),
    (16, {"type": "topk_mean", "k": 8}, "save_winners", "cuda"),
    (8, "mean", "minimal_stats", "cuda"),
    (64, {"type": "logmeanexp", "temperature": 0.5}, "minimal_stats", "cuda"),
    (32, {"type": "softmax_weighted", "beta": 2.0}, "minimal_stats", "cuda"),
    (16, {"type": "power_mean", "p": 3}, "minimal_stats", "cuda"),
    (8, {"type": "logmeanexp", "temperature": 0.5}, "minimal_stats", "triton"),
    (None, "mean", "minimal_stats", "triton"),
    (8, {"levels": [{"type": "max", "neurons_per_shard": 512, "transform": "affine"}, {"type": "mean"}]},
     "save_winners", "cuda"),
]
for C, pooling, pol, backend in CASES:
    lay = SparsePoolingLayer(1024 if backend == "cuda" else 64, 20011, 16, shared_neurons=C, pooling=pooling,
                             checkpoint_policy=pol, device="cuda", bias_std=0.1, backend=backend,
                             plan=PhysicalPlan(block_m=32))
    x = torch.randn(40, lay.layout.input_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lay(x).float().square().sum().backward()
    torch.cuda.synchronize()
    print("ok", C, lay.cfg.describe(), pol, lay.last_info.get("forward_backend"), lay.last_info.get("backward_backend"),
          flush=True)

# indexed / full-slab k_dw variants at C = 64 (partial blocks, ragged token tail)
from sparse_pooling.kernels import cuda_ops  # noqa: E402

for staging, cfg, stages, prec, B in [("indexed", 1, 2, "split", 40), ("indexed", 3, 2, "split", 40),
                                      ("indexed", 1, 3, "split", 24), ("indexed", 3, 3, "bf16", 40),
                                      ("indexed", 2, 2, "split", 8), ("full", 1, 2, "split", 40)]:
    cuda_ops.DW_INDEXED_STAGES = stages
    lay = SparsePoolingLayer(1024, 20011, 16, shared_neurons=64, pooling="mean", device="cuda", bias_std=0.1,
                             backend="cuda", plan=PhysicalPlan(block_m=32, cuda_dw_staging=staging, cuda_dw_cfg=cfg,
                                                               grad_precision=prec))
    x = torch.randn(B, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lay(x).float().square().sum().backward()
    torch.cuda.synchronize()
    print("ok k_dw", lay.last_info.get("dw_staging"), "cfg", lay.last_info.get("dw_cfg"), "stages", stages, prec,
          "B", B, flush=True)
