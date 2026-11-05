"""Sparse fan-in-8 projection to a huge logical hidden layer, fused with
streaming (optionally hierarchical) pooling and recompute-based backward.

``CoverTopK16Layer`` (sparse_pooling.cover_topk16) is the separate, opt-in Cover-TopK/16 path: a frozen 16-input
support per 16-neuron row group selected once from a K64 envelope, executed compactly at K16 / C16."""
from .config import LevelSpec, PhysicalPlan, PoolingConfig
from .layer import SparsePoolingLayer
from .layout import SparseLayout, init_parameters
from .cover_topk16 import CoverTopK16Layer

__all__ = ["SparsePoolingLayer", "PoolingConfig", "LevelSpec", "PhysicalPlan", "SparseLayout", "init_parameters",
           "CoverTopK16Layer"]
