"""Generalized-fan-in kgen modules and the Cover-TopK/16 dependency closure (integrated from the research branch
interaction-coverage-k16-dx, commit f4c538a; full research history and campaigns stay on that branch).

Nothing here is selected by ``SparsePoolingLayer`` (fan-in 8, C in {8, 16, 32, 64}).  The supported entry point that
uses these modules is ``sparse_pooling.cover_topk16.CoverTopK16Layer``; everything else is explicit, experimental use:

* ``kgen_layout`` / ``kgen_graph``: logical layout and generator for fan-in K in {8, 16, 32, 64} and logical C
  (multiple of 8; kernels: C <= 256): the documented two-level process of supergroups.py with the assignment quantum
  as a parameter (the K64 / C256 envelope graph of the Cover-TopK/16 model is generated here);
* ``kgen_layer`` / ``kgen_runtime``: ``KgenLayer`` on the generalized-fan-in CUDA kernels (bundle "kgen",
  kernels/cuda/experimental/kgen; the runtime binds the K8 / K16 / K32 / K64 entry points and the K16 dX tiling
  variants); ``kgen_dx_variants``: variant names / screen / legality wrappers over backends.dispatch.kgen_dx_resolve;
* ``kgen_lowered``: the K64 ``Envelope`` description and the compact K = k, C = 16 lowering of a local support;
* ``local_supports``: local-support policies of the research rounds.  Only ``Cover-TopK`` (k = 16) is promoted (via
  CoverTopK16Layer); FP-mag / FP-balanced / RP-* / BP-row / WRS / TopK / Balanced-TopK remain research comparisons,
  importable but not recommended defaults;
* ``row_normalize``: the one-time row-energy restoration (cap 8, relative epsilon 1e-6);
* ``width_masks``: geometry helpers (``layout_counts``) used by local_supports, plus the width-preserving study's
  tile supports (research comparisons);
* ``canonical``: representation-independent parameter / state / mask identity (envelope vs packed layer);
* ``reference``: float64 materialising reference of the logical model (tests).

Not integrated (kept on the research refs): the kgx bundle (K64 large-C / index-free / tile / CSR kernels), grouped
mixed-fan-in layers (Mix-*, Cover-Mix32), L1 treatments (``l1.L1AdamW``; with l1="none" it is the same update as
optim.FusedAdamW, docs/validation/cover_topk16_rtx3090_dx.md), support refresh, and every campaign script.
"""
