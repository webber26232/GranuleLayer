# K64: reducing weight density while preserving input-dependent width

Branch `k64-width-preserving-sparsity`, which starts from `k64-large-c-indexfree-l1` @ `982760d`. It is an
experiment only and is not merged into `main`.
* Production defaults, kernels and auto-dispatch are unchanged.
* Every new kernel is an experimental variant of the `kgx` bundle (`tilew16`, `tile8`, `hyb16`), and the
  dispatcher never selects one.
* Raw evidence is in `results/width_preserving_sparsity/`; the generated tables of every measured row are in
  its `tables.md`.

**Model.** D = G = 1024, K = 64, S = 64, R = 256, logical C = 256. H = 12.5M valid neurons, so
W0 = 800M valid weight slots; M = 8192.

**What was compared**, at exact valid-slot budgets, all from the same actual α = 0 checkpoint:
* free tile pruning: U16 and U8;
* protected tiles: P16 and P8;
* full-support coverage anchors: A16 and A8;
* a two-input thin backbone plus tiles, with and without anchors: H2 and H2A.

**What was measured for every support:**
* the complete forward / backward / optimizer step;
* exact and sampled tuple coverage, layer-wide and per output;
* tuple multiplicity;
* structural and numerical effective width;
* the input sensitivity of a fixed neuron sample;
* two synthetic learning diagnostics.

## Answers in brief

*(Times are complete training steps at C256 over graph seeds 0 / 1 / 2: mean ratio ± t95, 24 samples per seed.
The unpruned dense step is 1,568.7 ms.)*

PLACEHOLDER_ANSWERS
