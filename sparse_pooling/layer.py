"""PyTorch-facing module."""
from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import nn

from .autograd import Level1Runtime, SemanticPoolFunction, SparsePoolFunction, resolve_policy
from .config import DEFAULT_WORKSPACE_BUDGET, PhysicalPlan, PoolingConfig
from .layout import SparseLayout, init_parameters


class SparsePoolingLayer(nn.Module):
    """Weight-sparse projection to ``hidden_dim`` logical neurons (fan-in 8),
    fused with configurable (optionally two-level) pooling to ``output_dim``.

    Args:
        shared_neurons: C in {8,16,32,64} -> Mode B (C neurons share one index
            set).  ``None`` -> Mode A (independent indices per neuron, SIMT).
        pooling: operator name, a level dict, or ``{"levels": [...]}``.
        checkpoint_policy: "minimal_stats" | "recompute_stats" |
            "save_winners" | "auto".
        plan: execution-only reduction plan (tiles, splits, input layout).
        backend: "triton" | "cuda" | "auto" (CUDA m16n8k8 kernels where supported).
        weight_layout: "canonical" (K-contiguous [G,NB,C,8]) | "packed"
            (mma-fragment-packed, see layout.py) | "auto" (packed for Mode B
            unless backend="triton").  Gradients (main_grad) use the same layout.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 100_000_000,
        output_dim: int = 1024,
        fan_in: int = 8,
        shared_neurons: Optional[int] = 8,
        pooling: Union[str, dict, PoolingConfig] = "max",
        checkpoint_policy: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        plan: Optional[PhysicalPlan] = None,
        workspace_budget: int = DEFAULT_WORKSPACE_BUDGET,
        device=None,
        seed: int = 0,
        bias_std: float = 0.0,
        backend: str = "auto",
        weight_layout: str = "auto",
    ):
        super().__init__()
        self.cfg = PoolingConfig.parse(pooling, checkpoint_policy)
        if isinstance(pooling, (str, PoolingConfig)) or "checkpoint_policy" not in (pooling or {}):
            self.cfg.checkpoint_policy = checkpoint_policy
        mode_b = shared_neurons is not None
        if weight_layout == "auto":
            weight_layout = "packed" if (mode_b and backend != "triton") else "canonical"
        if weight_layout not in ("packed", "canonical"):
            raise ValueError("weight_layout must be 'auto', 'packed' or 'canonical'")
        self.layout = SparseLayout(
            input_dim, hidden_dim, output_dim, fan_in,
            shared_neurons if mode_b else 8, mode_b, packed=(weight_layout == "packed"),
        )
        self.compute_dtype = dtype
        self.plan = plan or PhysicalPlan()
        self.backend_name = backend
        self.workspace_budget = workspace_budget
        idx, w, b = init_parameters(self.layout, device or "cpu", dtype=dtype, seed=seed, bias_std=bias_std)
        self.register_buffer("indices", idx)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)
        self._tnames = []
        if self.cfg.is_semantic_hierarchy:
            l1, l2 = self.cfg.levels
            S = math.ceil(self.layout.max_group / l1.neurons_per_shard)
            self.num_shards = S
            G = output_dim
            if l1.transform in ("affine", "gate"):
                self.t_alpha = nn.Parameter(torch.ones(G, S, device=device))
                self.t_beta = nn.Parameter(torch.zeros(G, S, device=device))
                self._tnames += ["alpha", "beta"]
            if l2.type == "weighted_sum":
                self.t_v = nn.Parameter(torch.full((G, S), 1.0 / S, device=device))
                self._tnames += ["v"]
        self._rt = None
        self._rt_key = None

    # ------------------------------------------------------------------
    def _runtime(self) -> Level1Runtime:
        key = (self.indices.device, self.indices.data_ptr(), id(self.plan))
        if self._rt is None or self._rt_key != key:
            backend = None
            if self.backend_name in ("cuda", "auto") and self.indices.is_cuda:
                try:
                    from .kernels.cuda_ops import CudaBackend

                    backend = CudaBackend.get(strict=(self.backend_name == "cuda"))
                except Exception:
                    if self.backend_name == "cuda":
                        raise
                    backend = None
            l1 = self.cfg.levels[0]
            self._rt = Level1Runtime(self.layout, self.indices, l1, self.plan,
                                     l1.neurons_per_shard if self.cfg.is_semantic_hierarchy else None,
                                     self.workspace_budget, backend)
            self._rt_key = key
        return self._rt

    @property
    def last_info(self):
        return {} if self._rt is None else dict(self._rt.last_info)

    @property
    def effective_policy(self) -> str:
        return resolve_policy(self.cfg.levels[0], self.cfg.checkpoint_policy)

    def enable_main_grad(self):
        """Accumulate W/b gradients into persistent FP32 ``main_grad`` buffers
        (autograd then returns None for them).  Saves the transient grad copy."""
        for p in (self.weight, self.bias):
            if getattr(p, "main_grad", None) is None:
                p.main_grad = torch.zeros(p.shape, device=p.device, dtype=torch.float32)

    def zero_main_grad(self):
        for p in (self.weight, self.bias):
            if getattr(p, "main_grad", None) is not None:
                p.main_grad.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        rt = self._runtime()
        policy = self.effective_policy
        if self.cfg.is_semantic_hierarchy:
            tparams = [getattr(self, "t_" + n) for n in self._tnames]
            y = SemanticPoolFunction.apply(x2, self.weight, self.bias, rt, self.cfg, policy,
                                           self.compute_dtype, self._tnames, *tparams)
        else:
            y = SparsePoolFunction.apply(x2, self.weight, self.bias, rt, policy, self.compute_dtype)
        return y.reshape(*lead, y.shape[-1])

    def extra_repr(self) -> str:
        L = self.layout
        mode = f"B(C={L.block})" if L.shared_indices else "A(independent)"
        return (f"in={L.input_dim}, hidden={L.hidden_dim:,}, out={L.groups}, fan_in={L.fan_in}, "
                f"mode={mode}, pooling={self.cfg.describe()}, policy={self.effective_policy}")
