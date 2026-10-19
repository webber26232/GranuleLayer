
**C64_S16_R4** (C 64, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_auto | tiles T64/WG8 | padded | 265.0 |  /  | 702.7 |  /  | 705.5 | 1409.4 | 21.4 | 1696.9 [1696.2–1697.7] | 8.69 / 9.89 |
| det | tiles T64/WG1 | padded | 275.1 | 264.5 / 10.0 | 647.9 | 647.6 / 0.2 | 704.8 | 1352.4 | 21.4 | 1650.0 [1649.5–1650.9] | 9.50 / 10.89 |

**C64_S64_R16** (C 64, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_auto | tiles T64/WG8 | None | 263.0 |  /  | 694.5 |  /  | 722.6 | 1418.6 | 21.4 | 1703.5 [1702.6–1705.1] | 8.68 / 9.88 |
| det | tiles T32/WG1 | None | 272.7 | 262.2 / 10.0 | 903.0 | 902.3 / 0.7 | 722.9 | 1627.4 | 21.4 | 1923.0 [1919.6–1924.7] | 9.49 / 10.89 |

**C8_S16_R4_seed0** (C 8, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 312.5 |  /  | 1047.9 |  /  | 1046.8 | 2096.4 | 21.4 | 2429.6 [2426.6–2485.0] | 8.84 / 10.38 |
| fast_compact | tiles T64/WG1 | compact | 309.7 |  /  | 1048.0 |  /  | 1078.4 | 2127.3 | 21.4 | 2458.3 [2457.3–2505.4] | 8.94 / 10.38 |
| det_padded | tiles T64/WG1 | padded | 321.3 | 310.6 / 10.0 | 1049.3 | 1048.8 / 0.5 | 1045.7 | 2096.1 | 21.4 | 2438.2 [2434.2–2486.3] | 9.66 / 10.38 |
| det_compact | tiles T64/WG1 | compact | 319.0 | 308.4 / 10.0 | 1057.5 | 1056.9 / 0.5 | 1088.4 | 2146.0 | 21.4 | 2482.4 [2464.9–2529.3] | 9.66 / 10.38 |
* paired det_padded − fast_padded step: +8.14 ms [-1.94, +57.47] (+0.34 % [-0.08, +2.37])
* paired det_padded − fast_padded dw_kernel: -1.28 ms [-7.67, +17.12] (-0.12 % [-0.71, +1.63])
* paired det_padded − fast_padded fwd_total: +9.09 ms [+8.47, +13.09] (+2.90 % [+2.71, +4.21])
* paired det_padded − fast_padded dx_kernel: +0.51 ms [-2.24, +20.25] (+0.05 % [-0.21, +1.93])
* paired det_compact − fast_compact step: +15.90 ms [+7.65, +63.55] (+0.64 % [+0.31, +2.59])
* paired det_compact − fast_compact dw_kernel: +1.61 ms [-0.02, +25.23] (+0.15 % [-0.00, +2.34])
* paired det_compact − fast_compact fwd_total: +9.56 ms [+8.16, +16.04] (+3.06 % [+2.64, +5.19])
* paired det_compact − fast_compact dx_kernel: +3.19 ms [-1.13, +22.89] (+0.30 % [-0.11, +2.19])
* paired fast_compact − fast_padded step: +29.66 ms [+20.34, +30.70] (+1.22 % [+0.82, +1.27])
* paired fast_compact − fast_padded dw_kernel: +31.46 ms [+26.67, +32.43] (+3.01 % [+2.48, +3.10])
* paired fast_compact − fast_padded fwd_total: -1.85 ms [-2.69, -1.80] (-0.59 % [-0.86, -0.58])
* paired fast_compact − fast_padded dx_kernel: +0.13 ms [-2.96, +0.62] (+0.01 % [-0.28, +0.06])
* paired det_compact − det_padded step: +32.90 ms [+29.51, +46.25] (+1.34 % [+1.21, +1.86])
* paired det_compact − det_padded dw_kernel: +35.49 ms [+33.35, +39.22] (+3.36 % [+3.19, +3.68])
* paired det_compact − det_padded fwd_total: -1.26 ms [-3.22, +1.15] (-0.39 % [-1.00, +0.36])
* paired det_compact − det_padded dx_kernel: +1.00 ms [-1.31, +6.35] (+0.09 % [-0.12, +0.59])

**C8_S16_R4_seed1** (C 8, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 312.0 |  /  | 1046.9 |  /  | 1045.2 | 2093.5 | 21.4 | 2425.4 [2424.5–2427.2] | 8.84 / 10.38 |
| fast_compact | tiles T64/WG1 | compact | 309.8 |  /  | 1046.8 |  /  | 1077.7 | 2126.5 | 21.4 | 2457.6 [2456.4–2458.3] | 8.94 / 10.38 |
* paired fast_compact − fast_padded step: +31.90 ms [+30.45, +33.81] (+1.32 % [+1.25, +1.39])
* paired fast_compact − fast_padded dw_kernel: +32.80 ms [+32.00, +33.29] (+3.14 % [+3.06, +3.19])
* paired fast_compact − fast_padded fwd_total: -2.35 ms [-2.37, -1.67] (-0.75 % [-0.76, -0.53])
* paired fast_compact − fast_padded dx_kernel: +0.17 ms [-0.99, +0.83] (+0.02 % [-0.09, +0.08])

**C8_S16_R4_seed2** (C 8, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 312.3 |  /  | 1046.8 |  /  | 1044.5 | 2093.4 | 21.4 | 2425.9 [2425.1–2427.9] | 8.84 / 10.38 |
| fast_compact | tiles T64/WG1 | compact | 309.8 |  /  | 1047.1 |  /  | 1077.2 | 2125.7 | 21.4 | 2456.4 [2453.4–2458.5] | 8.94 / 10.38 |
* paired fast_compact − fast_padded step: +30.65 ms [+28.27, +30.95] (+1.26 % [+1.17, +1.28])
* paired fast_compact − fast_padded dw_kernel: +32.69 ms [+32.35, +32.76] (+3.13 % [+3.10, +3.14])
* paired fast_compact − fast_padded fwd_total: -2.44 ms [-4.56, -1.59] (-0.78 % [-1.45, -0.51])
* paired fast_compact − fast_padded dx_kernel: +0.42 ms [-0.40, +0.89] (+0.04 % [-0.04, +0.08])

**C8_S32_R8_seed0** (C 8, S 32, R 8, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 316.4 |  /  | 1251.8 |  /  | 1081.8 | 2334.9 | 21.4 | 2673.0 [2669.5–2674.3] | 8.84 / 10.38 |
| fast_compact | tiles T32/WG1 | compact | 315.3 |  /  | 1251.2 |  /  | 1121.4 | 2374.3 | 21.4 | 2710.2 [2707.5–2712.3] | 8.94 / 10.38 |
| det_padded | tiles T32/WG1 | padded | 323.7 | 313.2 / 10.0 | 1251.9 | 1250.9 / 1.0 | 1081.4 | 2333.9 | 21.4 | 2678.5 [2678.0–2679.8] | 9.66 / 10.38 |
| det_compact | tiles T32/WG1 | compact | 322.3 | 311.7 / 10.0 | 1251.4 | 1250.4 / 1.0 | 1121.8 | 2374.9 | 21.4 | 2718.3 [2717.0–2719.4] | 9.66 / 10.38 |
* paired det_padded − fast_padded step: +5.64 ms [+4.38, +8.59] (+0.21 % [+0.16, +0.32])
* paired det_padded − fast_padded dw_kernel: -0.58 ms [-1.57, +0.16] (-0.05 % [-0.15, +0.01])
* paired det_padded − fast_padded fwd_total: +7.71 ms [+5.65, +8.86] (+2.44 % [+1.78, +2.80])
* paired det_padded − fast_padded dx_kernel: -0.02 ms [-0.50, +0.70] (-0.00 % [-0.04, +0.06])
* paired det_compact − fast_compact step: +8.10 ms [+6.62, +9.48] (+0.30 % [+0.24, +0.35])
* paired det_compact − fast_compact dw_kernel: +0.32 ms [-0.30, +1.28] (+0.03 % [-0.03, +0.11])
* paired det_compact − fast_compact fwd_total: +6.94 ms [+6.80, +7.23] (+2.20 % [+2.16, +2.29])
* paired det_compact − fast_compact dx_kernel: -0.28 ms [-0.71, +2.11] (-0.02 % [-0.06, +0.17])
* paired fast_compact − fast_padded step: +37.71 ms [+36.90, +38.74] (+1.41 % [+1.38, +1.45])
* paired fast_compact − fast_padded dw_kernel: +39.52 ms [+39.17, +39.86] (+3.65 % [+3.62, +3.68])
* paired fast_compact − fast_padded fwd_total: -1.43 ms [-1.98, -0.85] (-0.45 % [-0.63, -0.27])
* paired fast_compact − fast_padded dx_kernel: -0.72 ms [-1.21, +0.20] (-0.06 % [-0.10, +0.02])
* paired det_compact − det_padded step: +39.57 ms [+38.97, +41.02] (+1.48 % [+1.46, +1.53])
* paired det_compact − det_padded dw_kernel: +40.48 ms [+40.30, +40.99] (+3.75 % [+3.73, +3.79])
* paired det_compact − det_padded fwd_total: -1.69 ms [-4.04, -0.31] (-0.52 % [-1.24, -0.10])
* paired det_compact − det_padded dx_kernel: -0.71 ms [-1.74, +1.39] (-0.06 % [-0.14, +0.11])

**C8_S32_R8_seed1** (C 8, S 32, R 8, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 316.6 |  /  | 1251.5 |  /  | 1081.0 | 2334.7 | 21.4 | 2672.8 [2672.0–2672.7] | 8.84 / 10.38 |
| fast_compact | tiles T32/WG1 | compact | 315.4 |  /  | 1251.1 |  /  | 1121.1 | 2374.5 | 21.4 | 2710.7 [2710.6–2711.4] | 8.94 / 10.38 |
* paired fast_compact − fast_padded step: +38.69 ms [+38.51, +38.86] (+1.45 % [+1.44, +1.45])
* paired fast_compact − fast_padded dw_kernel: +40.68 ms [+39.31, +41.10] (+3.76 % [+3.63, +3.80])
* paired fast_compact − fast_padded fwd_total: -1.19 ms [-1.57, -0.82] (-0.38 % [-0.49, -0.26])
* paired fast_compact − fast_padded dx_kernel: -0.52 ms [-1.27, +2.00] (-0.04 % [-0.10, +0.16])

**C8_S32_R8_seed2** (C 8, S 32, R 8, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 316.4 |  /  | 1250.8 |  /  | 1081.5 | 2334.1 | 21.4 | 2671.0 [2670.4–2673.6] | 8.84 / 10.38 |
| fast_compact | tiles T32/WG1 | compact | 315.5 |  /  | 1252.4 |  /  | 1121.2 | 2374.7 | 21.4 | 2710.7 [2709.4–2713.5] | 8.94 / 10.38 |
* paired fast_compact − fast_padded step: +38.86 ms [+36.39, +43.06] (+1.46 % [+1.36, +1.61])
* paired fast_compact − fast_padded dw_kernel: +39.22 ms [+38.55, +39.31] (+3.62 % [+3.56, +3.63])
* paired fast_compact − fast_padded fwd_total: -1.24 ms [-1.81, -0.29] (-0.39 % [-0.57, -0.09])
* paired fast_compact − fast_padded dx_kernel: +1.33 ms [+0.48, +2.23] (+0.11 % [+0.04, +0.18])

**C8_S64_R16_bf16** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, bf16; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | private_t32 T32/WG1 | padded | 321.5 |  /  | 1478.9 |  /  | 1027.8 | 2508.6 | 21.4 | 2850.4 [2848.9–2851.4] | 8.84 / 10.38 |
| fast_compact | private_t32 T32/WG1 | compact | 319.2 |  /  | 1478.9 |  /  | 1114.1 | 2594.8 | 21.4 | 2935.5 [2934.0–2938.2] | 8.94 / 10.38 |
| det_padded | private_t32 T32/WG1 | padded | 328.8 | 318.2 / 10.0 | 1478.9 | 1476.9 / 1.9 | 1027.1 | 2507.5 | 21.4 | 2856.5 [2856.3–2860.1] | 9.84 / 10.38 |
| det_compact | private_t32 T32/WG1 | compact | 325.9 | 315.3 / 10.0 | 1478.6 | 1476.6 / 1.9 | 1114.3 | 2594.4 | 21.4 | 2943.6 [2940.1–2944.4] | 9.84 / 10.38 |
* paired det_padded − fast_padded step: +7.14 ms [+5.24, +8.77] (+0.25 % [+0.18, +0.31])
* paired det_padded − fast_padded dw_kernel: -0.66 ms [-2.46, +0.13] (-0.06 % [-0.24, +0.01])
* paired det_padded − fast_padded fwd_total: +7.52 ms [+6.49, +9.31] (+2.34 % [+2.02, +2.91])
* paired det_padded − fast_padded dx_kernel: +1.02 ms [-0.98, +2.35] (+0.07 % [-0.07, +0.16])
* paired det_compact − fast_compact step: +6.46 ms [+6.01, +9.34] (+0.22 % [+0.20, +0.32])
* paired det_compact − fast_compact dw_kernel: -0.59 ms [-1.21, -0.22] (-0.05 % [-0.11, -0.02])
* paired det_compact − fast_compact fwd_total: +6.63 ms [+5.95, +7.75] (+2.07 % [+1.86, +2.42])
* paired det_compact − fast_compact dx_kernel: +0.02 ms [-1.99, +0.63] (+0.00 % [-0.13, +0.04])
* paired fast_compact − fast_padded step: +85.52 ms [+83.01, +86.83] (+3.00 % [+2.91, +3.05])
* paired fast_compact − fast_padded dw_kernel: +87.18 ms [+84.17, +87.46] (+8.48 % [+8.18, +8.51])
* paired fast_compact − fast_padded fwd_total: -1.65 ms [-3.11, -0.26] (-0.51 % [-0.97, -0.08])
* paired fast_compact − fast_padded dx_kernel: +0.52 ms [-0.90, +2.22] (+0.04 % [-0.06, +0.15])
* paired det_compact − det_padded step: +84.86 ms [+83.78, +87.34] (+2.97 % [+2.93, +3.06])
* paired det_compact − det_padded dw_kernel: +86.76 ms [+86.02, +87.49] (+8.45 % [+8.37, +8.52])
* paired det_compact − det_padded fwd_total: -2.54 ms [-3.66, -1.82] (-0.77 % [-1.11, -0.55])
* paired det_compact − det_padded dx_kernel: -0.26 ms [-3.28, +1.43] (-0.02 % [-0.22, +0.10])

**C8_S64_R16_caps** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| cap256M | private_t32 T32/WG1 | padded | 330.7 | 320.1 / 10.0 | 1597.9 | 1595.8 / 1.9 | 1113.1 | 2713.0 | 21.4 | 3065.1 [3062.7–3065.8] | 9.09 / 10.38 |
| cap1G | private_t32 T32/WG1 | padded | 327.3 | 316.8 / 10.0 | 1597.3 | 1595.3 / 1.9 | 1113.1 | 2711.8 | 21.4 | 3061.2 [3057.8–3063.9] | 9.84 / 10.38 |
| cap4G | private_t32 T32/WG1 | padded | 327.1 | 316.8 / 9.8 | 1596.5 | 1594.5 / 1.9 | 1112.9 | 2711.2 | 21.4 | 3059.3 [3057.3–3062.3] | 12.66 / 14.38 |
* paired cap256M − cap1G step: +2.61 ms [+1.82, +4.93] (+0.09 % [+0.06, +0.16])
* paired cap256M − cap1G dw_kernel: +0.04 ms [-0.16, +0.36] (+0.00 % [-0.01, +0.03])
* paired cap256M − cap1G fwd_total: +3.03 ms [+2.62, +4.26] (+0.92 % [+0.80, +1.30])
* paired cap256M − cap1G dx_kernel: +0.28 ms [-0.93, +0.46] (+0.02 % [-0.06, +0.03])
* paired cap4G − cap1G step: -1.37 ms [-3.29, -0.46] (-0.04 % [-0.11, -0.02])
* paired cap4G − cap1G dw_kernel: -0.19 ms [-1.24, +0.35] (-0.02 % [-0.11, +0.03])
* paired cap4G − cap1G fwd_total: -0.03 ms [-0.37, +0.36] (-0.01 % [-0.11, +0.11])
* paired cap4G − cap1G dx_kernel: -1.09 ms [-2.70, -0.61] (-0.07 % [-0.17, -0.04])

**C8_S64_R16_composed** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast | private_t32 T32/WG1 | padded | 319.3 |  /  | 1600.0 |  /  | 1114.8 | 2717.2 | 21.5 | 3058.3 [3053.0–3060.6] | 8.90 / 10.39 |
| det | private_t32 T32/WG1 | padded | 327.7 | 316.8 / 10.0 | 1599.4 | 1597.4 / 1.9 | 1114.0 | 2716.2 | 21.5 | 3066.6 [3065.2–3068.0] | 9.89 / 10.39 |
* paired det − fast step: +7.95 ms [+7.22, +12.15] (+0.26 % [+0.24, +0.40])
* paired det − fast dw_kernel: -0.61 ms [-1.61, +0.66] (-0.05 % [-0.14, +0.06])
* paired det − fast fwd_total: +8.62 ms [+8.50, +8.86] (+2.69 % [+2.65, +2.78])
* paired det − fast dx_kernel: +0.01 ms [-2.13, +2.24] (+0.00 % [-0.13, +0.14])

**C8_S64_R16_seed0** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | private_t32 T32/WG1 | padded | 319.1 |  /  | 1596.7 |  /  | 1112.9 | 2711.1 | 21.4 | 3052.4 [3049.7–3054.3] | 8.84 / 10.38 |
| fast_compact | private_t32 T32/WG1 | compact | 319.5 |  /  | 1596.6 |  /  | 1157.4 | 2755.1 | 21.4 | 3095.3 [3094.5–3095.3] | 8.94 / 10.38 |
| det_padded | private_t32 T32/WG1 | padded | 326.5 | 316.0 / 10.0 | 1595.0 | 1593.1 / 1.9 | 1112.8 | 2709.9 | 21.4 | 3058.0 [3055.9–3059.9] | 9.84 / 10.38 |
| det_compact | private_t32 T32/WG1 | compact | 325.2 | 314.7 / 10.0 | 1596.4 | 1594.4 / 1.9 | 1157.8 | 2756.3 | 21.4 | 3102.8 [3101.6–3104.8] | 9.84 / 10.38 |
* paired det_padded − fast_padded step: +6.16 ms [+3.12, +7.55] (+0.20 % [+0.10, +0.25])
* paired det_padded − fast_padded dw_kernel: +0.10 ms [-0.16, +0.50] (+0.01 % [-0.01, +0.05])
* paired det_padded − fast_padded fwd_total: +8.56 ms [+5.85, +9.40] (+2.69 % [+1.82, +2.96])
* paired det_padded − fast_padded dx_kernel: -1.20 ms [-2.70, -0.37] (-0.08 % [-0.17, -0.02])
* paired det_compact − fast_compact step: +8.40 ms [+6.91, +9.88] (+0.27 % [+0.22, +0.32])
* paired det_compact − fast_compact dw_kernel: +0.85 ms [-0.07, +2.28] (+0.07 % [-0.01, +0.20])
* paired det_compact − fast_compact fwd_total: +6.53 ms [+3.51, +7.40] (+2.04 % [+1.10, +2.32])
* paired det_compact − fast_compact dx_kernel: -0.42 ms [-1.68, +2.93] (-0.03 % [-0.11, +0.18])
* paired fast_compact − fast_padded step: +42.58 ms [+40.66, +44.99] (+1.40 % [+1.33, +1.48])
* paired fast_compact − fast_padded dw_kernel: +44.85 ms [+43.59, +45.07] (+4.03 % [+3.92, +4.05])
* paired fast_compact − fast_padded fwd_total: +0.83 ms [-1.61, +2.06] (+0.26 % [-0.50, +0.65])
* paired fast_compact − fast_padded dx_kernel: -0.96 ms [-1.97, +1.84] (-0.06 % [-0.12, +0.12])
* paired det_compact − det_padded step: +45.15 ms [+44.35, +46.75] (+1.48 % [+1.45, +1.53])
* paired det_compact − det_padded dw_kernel: +45.45 ms [+44.60, +46.21] (+4.08 % [+4.01, +4.15])
* paired det_compact − det_padded fwd_total: -0.98 ms [-4.34, +0.03] (-0.30 % [-1.32, +0.01])
* paired det_compact − det_padded dx_kernel: +0.77 ms [+0.25, +2.06] (+0.05 % [+0.02, +0.13])

**C8_S64_R16_seed1** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | private_t32 T32/WG1 | padded | 319.2 |  /  | 1596.2 |  /  | 1112.7 | 2710.8 | 21.4 | 3051.3 [3050.8–3051.3] | 8.84 / 10.38 |
| fast_compact | private_t32 T32/WG1 | compact | 317.4 |  /  | 1596.0 |  /  | 1158.7 | 2756.3 | 21.4 | 3094.9 [3093.9–3096.0] | 8.94 / 10.38 |
* paired fast_compact − fast_padded step: +44.60 ms [+42.74, +45.19] (+1.46 % [+1.40, +1.48])
* paired fast_compact − fast_padded dw_kernel: +45.59 ms [+43.90, +47.62] (+4.10 % [+3.94, +4.28])
* paired fast_compact − fast_padded fwd_total: -0.88 ms [-2.74, +0.07] (-0.28 % [-0.86, +0.02])
* paired fast_compact − fast_padded dx_kernel: -0.42 ms [-2.51, +2.12] (-0.03 % [-0.16, +0.13])

**C8_S64_R16_seed2** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | private_t32 T32/WG1 | padded | 319.4 |  /  | 1596.4 |  /  | 1112.4 | 2710.1 | 21.4 | 3050.0 [3048.9–3052.4] | 8.84 / 10.38 |
| fast_compact | private_t32 T32/WG1 | compact | 319.1 |  /  | 1596.2 |  /  | 1156.9 | 2754.3 | 21.4 | 3095.2 [3090.4–3097.1] | 8.94 / 10.38 |
* paired fast_compact − fast_padded step: +44.63 ms [+39.40, +46.13] (+1.46 % [+1.29, +1.51])
* paired fast_compact − fast_padded dw_kernel: +44.77 ms [+43.23, +45.69] (+4.03 % [+3.89, +4.11])
* paired fast_compact − fast_padded fwd_total: +0.19 ms [-2.20, +1.38] (+0.06 % [-0.69, +0.43])
* paired fast_compact − fast_padded dx_kernel: -0.27 ms [-1.50, +2.13] (-0.02 % [-0.09, +0.13])

**mem_C8_S16_R4_det** (C 8, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| det | tiles T64/WG1 | padded | 309.5 | 298.9 / 10.0 | 1047.7 | 1047.1 / 0.5 | 1044.3 | 2093.2 | 21.4 | 2424.0 [2424.0–2424.0] | 9.66 / 10.38 |

**mem_C8_S16_R4_fast** (C 8, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast | tiles T64/WG1 | padded | 301.9 |  /  | 1046.0 |  /  | 1043.2 | 2090.4 | 21.4 | 2413.7 [2413.7–2413.7] | 8.84 / 10.38 |

**mem_C8_S32_R8_det** (C 8, S 32, R 8, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| det | tiles T32/WG1 | padded | 314.8 | 304.1 / 10.2 | 1259.8 | 1258.7 / 1.0 | 1091.7 | 2352.7 | 21.4 | 2688.9 [2688.9–2688.9] | 9.66 / 10.38 |

**mem_C8_S32_R8_fast** (C 8, S 32, R 8, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast | tiles T32/WG1 | padded | 305.2 |  /  | 1262.3 |  /  | 1089.9 | 2353.4 | 21.4 | 2680.0 [2680.0–2680.0] | 8.84 / 10.38 |

**mem_C8_S64_R16_det** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| det | private_t32 T32/WG1 | padded | 322.0 | 310.8 / 10.6 | 1606.2 | 1604.2 / 1.9 | 1120.4 | 2727.8 | 21.4 | 3071.1 [3071.1–3071.1] | 9.84 / 10.38 |

**mem_C8_S64_R16_fast** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast | private_t32 T32/WG1 | padded | 312.6 |  /  | 1608.5 |  /  | 1120.6 | 2730.4 | 21.4 | 3064.3 [3064.3–3064.3] | 8.84 / 10.38 |

**prev_C64_S16_R4** (C 64, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit c9a591e)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| prev_auto | tiles T64/WG8 | None | 262.9 |  /  | 700.8 |  /  | 704.2 | 1406.1 | 21.4 | 1690.9 [1689.6–1692.8] | 8.69 / 9.89 |

**prev_C64_S64_R16** (C 64, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit c9a591e)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| prev_auto | tiles T64/WG8 | None | 263.2 |  /  | 693.8 |  /  | 720.6 | 1414.6 | 21.4 | 1699.7 [1699.0–1703.3] | 8.68 / 9.88 |

**prev_C8_S16_R4** (C 8, S 16, R 4, H 100,000,000, M 8192, G 1024, split; commit c9a591e)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| prev_auto | tiles T64/WG1 | None | 312.7 |  /  | 1067.2 |  /  | 1065.8 | 2134.3 | 21.4 | 2471.1 [2425.2–2504.8] | 8.84 / 10.38 |

**prev_C8_S32_R8** (C 8, S 32, R 8, H 100,000,000, M 8192, G 1024, split; commit c9a591e)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| prev_auto | tiles T32/WG1 | None | 315.2 |  /  | 1252.8 |  /  | 1081.8 | 2336.4 | 21.4 | 2672.1 [2671.5–2672.8] | 8.84 / 10.38 |

**prev_C8_S64_R16** (C 8, S 64, R 16, H 100,000,000, M 8192, G 1024, split; commit c9a591e)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| prev_auto | private_t32 T32/WG1 | None | 315.9 |  /  | 1599.4 |  /  | 1112.2 | 2713.2 | 21.4 | 3049.5 [3048.1–3052.3] | 8.84 / 10.38 |
