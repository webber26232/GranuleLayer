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
* ``reference``: float64 materialising reference of the logical model (tests);
* ``kgx_layer`` / ``kgx_graph`` / ``kgx_lowering``: the K = 64 study's ``KgxLayer`` (bundle "kgx"): logical C
  up to 2048 executed directly with physical neuron tiles, S = K = 64 index-free execution (records = sets,
  register-accumulated dX), and the existing kgen kernels as reference routes on the same tensors (C <= 256;
  C > 256 as physical C = 256 subblocks);
* ``kgx_sparse``: frozen-mask formats (tile 16 x 16, unstructured CSR), masks (learned / magnitude / tile /
  random), checkpoints; ``l1``: ``L1AdamW`` (coupled / proximal L1, frozen masks) and sparsity statistics;
* width-preserving sparsity study (results/WIDTH_PRESERVING_SPARSITY.md): ``width_masks`` (U / P / A / H2 /
  H2A supports with exact valid-slot budgets and feasibility), ``kgx_wide`` (``KgxWideLayer``: 16 x 8 tiles,
  16 x 16 tiles on new kernels, two-input backbone + 16 x 16 tiles), ``kgx_onetile`` (K16 / K8 C16 lowered
  control of one-tile supports), ``canonical`` (representation-independent parameter / state / mask identity).
"""
