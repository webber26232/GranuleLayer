Per-step SHA-1 of every recorded tensor (y, loss, dX, dW, db, params, moments, counters; det_training.py Writer), first process of each pair.

| Compact run | Padded run | Tensors recorded | Equal SHA-1 | Projection (compact / padded) |
|---|---|---:|---:|---|
| M_S16_R4_seed0_det_split_compact | M_S16_R4_seed0_det_split | 561 | 561 | compact / padded (default; field absent) |
| M_S32_R8_seed0_det_split_compact | M_S32_R8_seed0_det_split | 561 | 561 | compact / padded (default; field absent) |
| M_S64_R16_seed0_det_split_compact | M_S64_R16_seed0_det_split | 561 | 561 | compact / padded (default; field absent) |
| M_S64_R16_seed0_det_bf16_compact | M_S64_R16_seed0_det_bf16 | 561 | 561 | compact / padded (default; field absent) |
| P_S64_R16_seed0_det_split_compact | P_S64_R16_seed0_det_split | 165 | 165 | compact / padded |
