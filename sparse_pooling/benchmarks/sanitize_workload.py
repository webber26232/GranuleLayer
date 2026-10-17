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
