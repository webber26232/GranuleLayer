"""Configuration objects for the sparse projection + fused pooling layer.

Two notions of hierarchy are kept strictly separate:

* **Physical reduction hierarchy** (``PhysicalPlan``): how the *same* pooling
  operator is split across CUDA programs and merged afterwards.  It never
  changes the function being computed (up to floating-point rounding for
  sum-type states; Max / Top-K merges are exact).

* **Semantic pooling hierarchy** (``PoolingConfig.levels``): a model
  architecture in which level 1 pools neuron *shards* into compact features
  ``[B, G, S]`` and level 2 (after an optional transform) pools those features
  into ``[B, G]``.  This changes the model.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

POOL_TYPES = (
    "max",
    "topk_mean",
    "mean",
    "logmeanexp",
    "power_mean",
    "softmax_weighted",
)
# level-2 only (operates on the small [B, G, S] tensor)
LEVEL2_ONLY_TYPES = ("weighted_sum",)

ACTIVATIONS = ("identity", "relu", "gelu")
TRANSFORMS = ("none", "activation", "affine", "gate")
CHECKPOINT_POLICIES = ("minimal_stats", "recompute_stats", "save_winners", "auto")

# integer ids used as tl.constexpr inside kernels
POOL_ID = {name: i for i, name in enumerate(POOL_TYPES)}
ACT_ID = {name: i for i, name in enumerate(ACTIVATIONS)}

# default neuron activation per operator (applied per neuron, before pooling)
_DEFAULT_ACT = {
    "max": "identity",
    "topk_mean": "identity",
    "mean": "relu",  # "mean" defaults to Mean(ReLU) - plain mean of a linear map collapses
    "logmeanexp": "identity",
    "power_mean": "relu",  # power mean is defined on nonnegative activations
    "softmax_weighted": "identity",
}
# default activation applied to the pooled value
_DEFAULT_OUT_ACT = {
    "max": "relu",  # y = max(0, max_j z_j)
    "topk_mean": "identity",
    "mean": "identity",
    "logmeanexp": "identity",
    "power_mean": "identity",
    "softmax_weighted": "identity",
}

_ALIASES = {
    "mean_relu": ("mean", {"activation": "relu"}),
    "lme": ("logmeanexp", {}),
    "softmax": ("softmax_weighted", {}),
    "power": ("power_mean", {}),
    "topk": ("topk_mean", {}),
}


@dataclass
class LevelSpec:
    """One pooling level.

    type:              pooling operator
    activation:        per-element activation applied *before* pooling
                       (to neurons for level 1, to transformed features for level 2)
    out_activation:    activation applied to the pooled value
    k:                 Top-K count (topk_mean); distinct from the sparse fan-in
    temperature:       LogMeanExp temperature tau > 0
    beta:              softmax-weighted inverse temperature > 0
    p:                 power-mean exponent >= 1
    neurons_per_shard: level-1 only, when a level 2 exists: neurons per shard
    transform:         applied to level-1 features before level 2
    """

    type: str
    activation: str = ""
    out_activation: str = ""
    k: int = 8
    temperature: float = 1.0
    beta: float = 1.0
    p: float = 2.0
    neurons_per_shard: Optional[int] = None
    transform: str = "none"
    transform_activation: str = "relu"

    def __post_init__(self):
        if self.type in _ALIASES:
            t, extra = _ALIASES[self.type]
            self.type = t
            for k_, v in extra.items():
                if not getattr(self, k_):
                    setattr(self, k_, v)
        if self.type not in POOL_TYPES + LEVEL2_ONLY_TYPES:
            raise ValueError(f"unknown pooling type {self.type!r}")
        if not self.activation:
            self.activation = _DEFAULT_ACT.get(self.type, "identity")
        if not self.out_activation:
            self.out_activation = _DEFAULT_OUT_ACT.get(self.type, "identity")
        for a in (self.activation, self.out_activation):
            if a not in ACTIVATIONS:
                raise ValueError(f"unknown activation {a!r}")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}")
        if self.type == "topk_mean":
            if self.k < 1 or (self.k & (self.k - 1)) != 0 or self.k > 64:
                raise ValueError("topk_mean k must be a power of two in [1, 64]")
        if self.type == "logmeanexp" and not self.temperature > 0:
            raise ValueError("temperature must be > 0")
        if self.type == "softmax_weighted" and not self.beta > 0:
            raise ValueError("beta must be > 0")
        if self.type == "power_mean" and not self.p >= 1:
            raise ValueError("power_mean requires p >= 1")
        if self.type == "power_mean" and self.activation != "relu":
            raise ValueError("power_mean is defined on relu activations")

    @property
    def pool_id(self) -> int:
        return POOL_ID[self.type]

    @property
    def act_id(self) -> int:
        return ACT_ID[self.activation]

    @property
    def out_act_id(self) -> int:
        return ACT_ID[self.out_activation]

    @property
    def scalar_param(self) -> float:
        """The single float hyper-parameter consumed by the kernels."""
        return {
            "logmeanexp": self.temperature,
            "softmax_weighted": self.beta,
            "power_mean": self.p,
        }.get(self.type, 0.0)

    def describe(self) -> str:
        s = self.type
        if self.type == "topk_mean":
            s += f"(k={self.k})"
        elif self.type == "logmeanexp":
            s += f"(tau={self.temperature:g})"
        elif self.type == "softmax_weighted":
            s += f"(beta={self.beta:g})"
        elif self.type == "power_mean":
            s += f"(p={self.p:g})"
        if self.activation != "identity":
            s = f"{s}[{self.activation}]"
        if self.out_activation != "identity":
            s = f"{self.out_activation}({s})"
        return s


@dataclass
class PoolingConfig:
    levels: List[LevelSpec]
    checkpoint_policy: str = "auto"
    # semantic level-1 features: keep them from forward ("save") or rebuild them
    # in backward by re-running the level-1 streaming reduction ("recompute")
    level1_checkpoint: str = "save"

    def __post_init__(self):
        if not 1 <= len(self.levels) <= 2:
            raise ValueError("1 or 2 pooling levels are supported")
        if self.levels[0].type in LEVEL2_ONLY_TYPES:
            raise ValueError(f"{self.levels[0].type} cannot be a level-1 operator")
        if len(self.levels) == 2:
            if self.levels[0].neurons_per_shard is None:
                raise ValueError("a two-level config needs level-1 neurons_per_shard")
        elif self.levels[0].neurons_per_shard is not None:
            raise ValueError(
                "neurons_per_shard on a single level would change the model; "
                "use PhysicalPlan for execution-only sharding"
            )
        if self.checkpoint_policy not in CHECKPOINT_POLICIES:
            raise ValueError(f"unknown checkpoint policy {self.checkpoint_policy!r}")
        if self.level1_checkpoint not in ("save", "recompute"):
            raise ValueError("level1_checkpoint must be 'save' or 'recompute'")
        if len(self.levels) == 2:
            l1, l2 = self.levels
            monotone = ("identity", "relu")  # non-decreasing: commute with max
            if (
                l1.type == l2.type == "max"
                and l1.transform == "none"
                and l1.out_activation in monotone
                and l2.activation in monotone
            ):
                import warnings

                warnings.warn(
                    "Max -> Max without a non-monotone transform between the levels is exactly "
                    "global Max (up to the monotone activations); it adds no expressiveness"
                )

    @property
    def is_semantic_hierarchy(self) -> bool:
        return len(self.levels) == 2

    @staticmethod
    def parse(spec: Union[str, Dict[str, Any], "PoolingConfig"], checkpoint_policy: str = "auto"):
        if isinstance(spec, PoolingConfig):
            return spec
        if isinstance(spec, str):
            return PoolingConfig([LevelSpec(type=spec)], checkpoint_policy=checkpoint_policy)
        if isinstance(spec, dict):
            if "levels" in spec:
                levels = [LevelSpec(**lv) for lv in spec["levels"]]
                return PoolingConfig(
                    levels,
                    checkpoint_policy=spec.get("checkpoint_policy", checkpoint_policy),
                    level1_checkpoint=spec.get("level1_checkpoint", "save"),
                )
            return PoolingConfig([LevelSpec(**spec)], checkpoint_policy=checkpoint_policy)
        raise TypeError(f"cannot parse pooling spec {spec!r}")

    def describe(self) -> str:
        return " -> ".join(lv.describe() for lv in self.levels)


@dataclass
class PhysicalPlan:
    """Execution-only reduction hierarchy (does not change the model).

    phys_splits:      number of programs that share one pooling unit's neurons;
                      their compact states are merged by a second-stage kernel.
    token_tiles_per_program: 1 = token-stationary (pooling state stays on chip
                      while scanning neurons); >1 loops token tiles inside a
                      program (weight-reuse oriented).
    block_m:          tokens per tile.
    """

    phys_splits: int = 0  # 0 = choose automatically (workspace-budget aware)
    block_m: int = 64
    num_warps: int = 4
    num_stages: int = 2
    token_tiles_per_program: int = 1
    input_layout: str = "feature_major"  # or "token_major"
    backend: str = "auto"  # "triton" | "cuda" | "auto"
    # native CUDA kernels: tokens per CTA (16 or 32 -> 32/64 KiB smem slab at D=1024)
    cuda_block_m: int = 32
    cuda_warps: int = 16
    # backward: dZ operand precision for tensor-core gradient products:
    #   "split" = hi + lo BF16 pair (~16-bit mantissa), "bf16" = single BF16
    grad_precision: str = "split"
    # dZ epilogue of k_dx / k_dw_idx: "generic", "fast" (per-element select) or "mask"
    # (per-token BF16 hi/lo + per-element bit masks); the specialised ones exist for
    # Mean(ReLU / identity) only, other operators always use the generic path
    cuda_dx_epilogue: str = "generic"
    cuda_dw_epilogue: str = "generic"
    cuda_dx_warps: int = 0  # 8 | 16 | 32; 0 = auto (32 at C = 64, else 16; results/MEMORY_TRAFFIC_OPTIMIZATION.md)
    cuda_dw_cfg: int = -1  # -1 = per-C default (see cuda_ops.DW_CFG)
    # k_dw input staging: "full" stages all D features per token tile; "indexed"
    # stages only the rows the CTA's index records use (sm_86, C = 64, packed,
    # feature-major input; anything else falls back to "full"); "auto" = indexed
    # where supported (results/MEMORY_TRAFFIC_OPTIMIZATION.md)
    cuda_dw_staging: str = "auto"

    def __post_init__(self):
        assert self.input_layout in ("feature_major", "token_major")
        assert self.backend in ("triton", "cuda", "auto")
        assert self.grad_precision in ("split", "bf16")
        assert self.cuda_dw_staging in ("auto", "full", "indexed")
        assert self.cuda_dx_epilogue in ("generic", "fast", "mask")
        assert self.cuda_dw_epilogue in ("generic", "fast", "mask")


def state_fields(level: LevelSpec) -> int:
    """Number of FP32 values per compact pooling state (excluding ids)."""
    return {
        "max": 1,
        "topk_mean": level.k,
        "mean": 1,  # sum (count is static metadata)
        "logmeanexp": 2,  # running max, sum of exp
        "power_mean": 2,  # running max, scaled power sum
        "softmax_weighted": 3,  # running max, sum exp, weighted sum
    }[level.type]


def state_id_fields(level: LevelSpec, track_ids: bool) -> int:
    if level.type == "topk_mean":
        return level.k
    if level.type == "max" and track_ids:
        return 1
    return 0


def state_bytes(level: LevelSpec, track_ids: bool) -> int:
    return 4 * (state_fields(level) + state_id_fields(level, track_ids))


DEFAULT_WORKSPACE_BUDGET = 512 * 2**20


def workspace_bytes(batch: int, units: int, splits: int, level: LevelSpec, track_ids: bool) -> int:
    return batch * units * splits * state_bytes(level, track_ids)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"
