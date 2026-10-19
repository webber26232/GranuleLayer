
**S16_R4_ablate_movm** (C 8, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit 9ca57de)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| padded | tiles T64/WG1 | padded | 314.4 |  /  | 1060.3 |  /  | 1057.5 | 2118.8 | 21.4 | 2455.1 [2452.5–2502.5] | 8.84 / 10.38 |
| compact_nomovm | tiles T64/WG1 | compact | 315.5 |  /  | 1059.5 |  /  | 1037.3 | 2098.7 | 21.4 | 2433.8 [2431.8–2435.8] | 8.94 / 10.38 |
* paired compact_nomovm − padded step: -18.45 ms [-70.67, -17.16] (-0.75 % [-2.82, -0.70])
* paired compact_nomovm − padded dw_kernel: -19.41 ms [-42.88, -18.79] (-1.84 % [-3.97, -1.78])
* paired compact_nomovm − padded fwd_total: +1.35 ms [-1.24, +2.19] (+0.43 % [-0.39, +0.70])
* paired compact_nomovm − padded dx_kernel: -0.45 ms [-25.27, +1.71] (-0.04 % [-2.33, +0.16])

**S16_R4_epilogue** (C 8, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit 9ca57de)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| padded | tiles T64/WG1 | padded | 316.7 |  /  | 1074.7 |  /  | 1070.8 | 2145.2 | 21.4 | 2484.5 [2467.0–2502.8] | 8.84 / 10.38 |
| padded_mask | tiles T64/WG1 | padded | 315.7 |  /  | 1071.2 |  /  | 1087.2 | 2159.4 | 21.4 | 2498.0 [2489.3–2501.8] | 8.94 / 10.38 |
| compact | tiles T64/WG1 | compact | 313.8 |  /  | 1072.2 |  /  | 1104.5 | 2178.4 | 21.6 | 2510.2 [2504.3–2520.7] | 8.94 / 10.38 |
* paired compact − padded step: +29.98 ms [+16.90, +40.20] (+1.21 % [+0.68, +1.63])
* paired compact − padded dw_kernel: +32.85 ms [+24.32, +37.26] (+3.07 % [+2.25, +3.50])
* paired compact − padded fwd_total: -2.99 ms [-4.37, -2.21] (-0.94 % [-1.36, -0.70])
* paired compact − padded dx_kernel: -3.23 ms [-10.15, +2.49] (-0.30 % [-0.94, +0.23])
* paired padded_mask − padded step: +15.37 ms [-13.45, +24.23] (+0.62 % [-0.54, +0.98])
* paired padded_mask − padded dw_kernel: +15.59 ms [+4.65, +22.22] (+1.46 % [+0.43, +2.08])
* paired padded_mask − padded fwd_total: -1.13 ms [-4.12, -0.41] (-0.36 % [-1.29, -0.13])
* paired padded_mask − padded dx_kernel: -4.09 ms [-16.30, +3.28] (-0.38 % [-1.51, +0.31])
* paired compact − padded_mask step: +17.46 ms [+10.29, +30.35] (+0.70 % [+0.41, +1.22])
* paired compact − padded_mask dw_kernel: +17.26 ms [+15.04, +19.66] (+1.59 % [+1.38, +1.81])
* paired compact − padded_mask fwd_total: -1.12 ms [-3.29, -0.25] (-0.35 % [-1.04, -0.08])
* paired compact − padded_mask dx_kernel: +0.50 ms [-3.23, +9.34] (+0.05 % [-0.30, +0.88])

**S64_R16_ablate_movm** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 9ca57de)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| padded | private_t32 T32/WG1 | padded | 320.3 |  /  | 1598.8 |  /  | 1112.9 | 2712.9 | 21.4 | 3054.5 [3053.6–3057.3] | 8.84 / 10.38 |
| compact_nomovm | private_t32 T32/WG1 | compact | 319.1 |  /  | 1598.5 |  /  | 1098.7 | 2699.0 | 21.4 | 3039.7 [3038.6–3041.0] | 8.94 / 10.38 |
* paired compact_nomovm − padded step: -15.76 ms [-16.31, -14.44] (-0.52 % [-0.53, -0.47])
* paired compact_nomovm − padded dw_kernel: -14.12 ms [-14.46, -13.85] (-1.27 % [-1.30, -1.24])
* paired compact_nomovm − padded fwd_total: -0.69 ms [-2.84, +0.06] (-0.21 % [-0.89, +0.02])
* paired compact_nomovm − padded dx_kernel: +0.28 ms [-2.08, +1.04] (+0.02 % [-0.13, +0.07])

**S64_R16_bf16_ablate_movm** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, bf16; commit 9ca57de)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| padded | private_t32 T32/WG1 | padded | 321.8 |  /  | 1480.5 |  /  | 1030.5 | 2512.2 | 21.4 | 2855.2 [2855.1–2856.0] | 8.84 / 10.38 |
| compact_nomovm | private_t32 T32/WG1 | compact | 320.8 |  /  | 1481.4 |  /  | 1044.6 | 2527.4 | 21.4 | 2869.4 [2866.7–2871.7] | 8.94 / 10.38 |
* paired compact_nomovm − padded step: +13.91 ms [+11.61, +15.74] (+0.49 % [+0.41, +0.55])
* paired compact_nomovm − padded dw_kernel: +14.20 ms [+14.11, +15.32] (+1.38 % [+1.37, +1.49])
* paired compact_nomovm − padded fwd_total: -1.08 ms [-1.58, -0.57] (-0.33 % [-0.49, -0.18])
* paired compact_nomovm − padded dx_kernel: +0.69 ms [-0.73, +1.27] (+0.05 % [-0.05, +0.09])

**S64_R16_bf16_epilogue** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, bf16; commit 9ca57de)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| padded | private_t32 T32/WG1 | padded | 322.2 |  /  | 1481.1 |  /  | 1030.8 | 2512.9 | 21.4 | 2855.6 [2854.3–2856.9] | 8.84 / 10.38 |
| padded_mask | private_t32 T32/WG1 | padded | 320.1 |  /  | 1481.8 |  /  | 1093.0 | 2576.0 | 21.4 | 2918.5 [2916.3–2919.2] | 8.94 / 10.38 |
| compact | private_t32 T32/WG1 | compact | 320.1 |  /  | 1481.7 |  /  | 1115.3 | 2598.0 | 21.4 | 2939.3 [2938.2–2941.0] | 8.94 / 10.38 |
* paired compact − padded step: +83.44 ms [+82.00, +84.05] (+2.92 % [+2.87, +2.94])
* paired compact − padded dw_kernel: +84.91 ms [+84.13, +85.58] (+8.24 % [+8.17, +8.30])
* paired compact − padded fwd_total: -2.18 ms [-2.83, -1.61] (-0.68 % [-0.88, -0.50])
* paired compact − padded dx_kernel: +0.43 ms [-2.16, +1.44] (+0.03 % [-0.15, +0.10])
* paired padded_mask − padded step: +62.43 ms [+59.97, +63.27] (+2.19 % [+2.10, +2.22])
* paired padded_mask − padded dw_kernel: +62.16 ms [+61.91, +63.18] (+6.03 % [+6.01, +6.13])
* paired padded_mask − padded fwd_total: -2.55 ms [-3.41, -1.76] (-0.79 % [-1.05, -0.55])
* paired padded_mask − padded dx_kernel: +0.61 ms [-1.11, +1.69] (+0.04 % [-0.07, +0.11])
* paired compact − padded_mask step: +21.23 ms [+19.42, +22.95] (+0.73 % [+0.67, +0.79])
* paired compact − padded_mask dw_kernel: +22.42 ms [+21.67, +23.61] (+2.05 % [+1.98, +2.16])
* paired compact − padded_mask fwd_total: +0.37 ms [+0.14, +0.58] (+0.12 % [+0.04, +0.18])
* paired compact − padded_mask dx_kernel: -0.38 ms [-1.06, +0.14] (-0.03 % [-0.07, +0.01])

**S64_R16_epilogue** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 9ca57de)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| padded | private_t32 T32/WG1 | padded | 320.8 |  /  | 1611.3 |  /  | 1120.9 | 2733.9 | 21.4 | 3070.9 [3056.3–3088.1] | 8.84 / 10.38 |
| padded_mask | private_t32 T32/WG1 | padded | 319.7 |  /  | 1612.5 |  /  | 1135.0 | 2749.6 | 21.4 | 3086.3 [3070.0–3102.1] | 8.94 / 10.38 |
| compact | private_t32 T32/WG1 | compact | 320.3 |  /  | 1609.2 |  /  | 1165.8 | 2773.9 | 21.4 | 3112.0 [3102.3–3131.3] | 8.94 / 10.38 |
* paired compact − padded step: +44.62 ms [+43.01, +47.32] (+1.46 % [+1.40, +1.54])
* paired compact − padded dw_kernel: +46.34 ms [+42.04, +49.23] (+4.14 % [+3.77, +4.38])
* paired compact − padded fwd_total: -1.51 ms [-5.41, +1.00] (-0.47 % [-1.67, +0.31])
* paired compact − padded dx_kernel: +0.06 ms [-0.81, +0.80] (+0.00 % [-0.05, +0.05])
* paired padded_mask − padded step: +14.61 ms [+13.67, +17.39] (+0.48 % [+0.45, +0.56])
* paired padded_mask − padded dw_kernel: +16.71 ms [+14.14, +17.78] (+1.49 % [+1.27, +1.58])
* paired padded_mask − padded fwd_total: -1.31 ms [-2.36, -0.92] (-0.40 % [-0.74, -0.29])
* paired padded_mask − padded dx_kernel: +0.75 ms [-1.89, +1.80] (+0.05 % [-0.12, +0.11])
* paired compact − padded_mask step: +29.60 ms [+27.75, +32.33] (+0.95 % [+0.90, +1.05])
* paired compact − padded_mask dw_kernel: +29.64 ms [+27.90, +31.45] (+2.61 % [+2.47, +2.76])
* paired compact − padded_mask fwd_total: +0.13 ms [-4.08, +2.30] (+0.04 % [-1.26, +0.71])
* paired compact − padded_mask dx_kernel: -0.23 ms [-1.92, +1.07] (-0.01 % [-0.12, +0.07])
