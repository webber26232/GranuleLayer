"""Sparse fan-in-8 projection to a huge logical hidden layer, fused with
streaming (optionally hierarchical) pooling and recompute-based backward."""
from .config import LevelSpec, PhysicalPlan, PoolingConfig
from .layer import SparsePoolingLayer
from .layout import SparseLayout, init_parameters

__all__ = ["SparsePoolingLayer", "PoolingConfig", "LevelSpec", "PhysicalPlan", "SparseLayout", "init_parameters"]
