"""EXPERIMENTAL extensions of the random K/C/S/R sweep (results/RANDOM_KCSR_COVERAGE_PERFORMANCE.md).

Nothing here is imported by ``sparse_pooling`` or selected by ``SparsePoolingLayer``; the production
layer still accepts only fan-in 8 and C in {8, 16, 32, 64}.  Explicit use only:

* ``kgen_layout`` / ``kgen_graph``: logical layout and generator for any fan-in K in {8, 16, 32, 64} and
  logical C (multiple of 8, <= 256): the documented two-level process of supergroups.py (R global
  partitions into S-feature sets, balanced quotas, local permutation passes cut into S / K disjoint
  K-records) with the assignment quantum as a parameter;
* ``kgen_layer``: ``KgenLayer``, input-supergroup Mean(ReLU / identity) layer on the experimental
  generalized-fan-in CUDA kernels (bundle "kgen", kernels/cuda/experimental/kgen);
* ``lowering``: a logical C = 128 / 256, K = 8 graph as a native C = 64 SparsePoolingLayer (each
  logical block = C / 64 physical blocks with the same record and independent weights);
* ``reference``: float64 materialising reference of the logical model (tests).
"""
