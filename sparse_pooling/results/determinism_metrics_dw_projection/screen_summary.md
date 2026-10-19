
**H10000000_M1024_S16_R4** (C 8, S 16, R 4, H 10,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 4.7 |  /  | 13.3 |  /  | 15.9 | 29.5 | 2.2 | 36.3 [36.3–36.5] | 0.91 / 1.33 |
| fast_compact | tiles T64/WG1 | compact | 4.8 |  /  | 13.5 |  /  | 16.8 | 30.4 | 2.2 | 37.4 [37.2–37.6] | 0.93 / 1.33 |
| det_padded | tiles T64/WG1 | padded | 5.9 | 4.4 / 1.2 | 13.4 | 13.4 / 0.0 | 16.0 | 29.7 | 2.2 | 37.7 [37.5–38.4] | 1.88 / 2.33 |
| det_compact | tiles T64/WG1 | compact | 6.0 | 4.5 / 1.2 | 13.5 | 13.4 / 0.0 | 16.8 | 30.5 | 2.2 | 38.6 [38.6–39.0] | 1.88 / 2.33 |
* paired det_padded − fast_padded step: +1.56 ms [+0.99, +2.16] (+4.27 % [+2.73, +5.95])
* paired det_padded − fast_padded dw_kernel: +0.07 ms [-0.19, +0.17] (+0.43 % [-1.21, +1.05])
* paired det_padded − fast_padded fwd_total: +1.20 ms [+1.17, +1.22] (+25.43 % [+24.49, +25.83])
* paired det_padded − fast_padded dx_kernel: +0.08 ms [-0.13, +0.23] (+0.63 % [-0.96, +1.72])
* paired det_compact − fast_compact step: +1.30 ms [+1.15, +1.72] (+3.48 % [+3.08, +4.62])
* paired det_compact − fast_compact dw_kernel: +0.05 ms [-0.05, +0.12] (+0.27 % [-0.30, +0.69])
* paired det_compact − fast_compact fwd_total: +1.21 ms [+1.16, +1.27] (+25.50 % [+23.92, +26.81])
* paired det_compact − fast_compact dx_kernel: +0.08 ms [-0.01, +0.15] (+0.58 % [-0.11, +1.15])
* paired fast_compact − fast_padded step: +1.02 ms [+0.93, +1.10] (+2.80 % [+2.55, +3.03])
* paired fast_compact − fast_padded dw_kernel: +0.84 ms [+0.81, +0.88] (+5.28 % [+5.06, +5.50])
* paired fast_compact − fast_padded fwd_total: +0.03 ms [-0.01, +0.08] (+0.58 % [-0.20, +1.70])
* paired fast_compact − fast_padded dx_kernel: +0.16 ms [-0.00, +0.20] (+1.19 % [-0.02, +1.54])
* paired det_compact − det_padded step: +0.99 ms [+0.18, +1.12] (+2.63 % [+0.47, +2.95])
* paired det_compact − det_padded dw_kernel: +0.85 ms [+0.69, +1.02] (+5.31 % [+4.29, +6.46])
* paired det_compact − det_padded fwd_total: +0.05 ms [+0.01, +0.07] (+0.87 % [+0.11, +1.23])
* paired det_compact − det_padded dx_kernel: +0.03 ms [+0.01, +0.47] (+0.24 % [+0.11, +3.54])

**H10000000_M1024_S32_R8** (C 8, S 32, R 8, H 10,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 5.0 |  /  | 16.8 |  /  | 16.5 | 33.3 | 2.2 | 40.5 [40.4–40.9] | 0.91 / 1.33 |
| fast_compact | tiles T32/WG1 | compact | 5.0 |  /  | 16.9 |  /  | 17.3 | 34.4 | 2.2 | 41.6 [41.6–41.9] | 0.93 / 1.33 |
| det_padded | tiles T32/WG1 | padded | 6.1 | 4.6 / 1.2 | 16.8 | 16.7 / 0.0 | 16.5 | 33.4 | 2.2 | 41.6 [41.4–41.7] | 1.88 / 2.33 |
| det_compact | tiles T32/WG1 | compact | 6.1 | 4.6 / 1.2 | 16.9 | 16.8 / 0.0 | 17.3 | 34.4 | 2.2 | 42.6 [42.5–42.9] | 1.88 / 2.33 |
* paired det_padded − fast_padded step: +0.94 ms [+0.78, +1.07] (+2.33 % [+1.90, +2.64])
* paired det_padded − fast_padded dw_kernel: -0.05 ms [-0.30, +0.12] (-0.30 % [-1.79, +0.73])
* paired det_padded − fast_padded fwd_total: +1.05 ms [+1.04, +1.05] (+20.77 % [+20.57, +21.02])
* paired det_padded − fast_padded dx_kernel: -0.03 ms [-0.17, +0.04] (-0.18 % [-1.01, +0.25])
* paired det_compact − fast_compact step: +1.02 ms [+0.66, +1.27] (+2.45 % [+1.58, +3.06])
* paired det_compact − fast_compact dw_kernel: +0.02 ms [-0.15, +0.22] (+0.12 % [-0.88, +1.26])
* paired det_compact − fast_compact fwd_total: +1.05 ms [+0.99, +1.10] (+20.73 % [+19.32, +21.71])
* paired det_compact − fast_compact dx_kernel: -0.05 ms [-0.17, +0.09] (-0.32 % [-1.03, +0.51])
* paired fast_compact − fast_padded step: +1.03 ms [+0.82, +1.19] (+2.53 % [+2.01, +2.94])
* paired fast_compact − fast_padded dw_kernel: +0.83 ms [+0.67, +0.99] (+5.01 % [+4.00, +6.06])
* paired fast_compact − fast_padded fwd_total: +0.07 ms [+0.03, +0.09] (+1.34 % [+0.52, +1.76])
* paired fast_compact − fast_padded dx_kernel: +0.11 ms [-0.02, +0.14] (+0.68 % [-0.09, +0.86])
* paired det_compact − det_padded step: +1.09 ms [+0.82, +1.32] (+2.61 % [+1.97, +3.18])
* paired det_compact − det_padded dw_kernel: +0.93 ms [+0.88, +0.96] (+5.66 % [+5.35, +5.82])
* paired det_compact − det_padded fwd_total: +0.07 ms [+0.04, +0.09] (+1.11 % [+0.63, +1.55])
* paired det_compact − det_padded dx_kernel: +0.08 ms [-0.07, +0.26] (+0.49 % [-0.43, +1.54])

**H10000000_M1024_S64_R16** (C 8, S 64, R 16, H 10,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG8 | padded | 5.1 |  /  | 21.4 |  /  | 16.7 | 38.3 | 2.2 | 45.5 [45.4–46.1] | 0.91 / 1.33 |
| fast_compact | tiles T64/WG8 | compact | 5.1 |  /  | 21.4 |  /  | 17.5 | 39.2 | 2.2 | 46.4 [46.1–46.8] | 0.93 / 1.33 |
| det_padded | private_t32 T32/WG1 | padded | 6.1 | 4.6 / 1.2 | 26.6 | 26.5 / 0.1 | 16.0 | 42.7 | 2.2 | 51.0 [50.8–51.1] | 1.88 / 2.33 |
| det_compact | private_t32 T32/WG1 | compact | 6.1 | 4.6 / 1.2 | 26.7 | 26.6 / 0.1 | 16.7 | 43.6 | 2.2 | 51.9 [51.6–52.0] | 1.88 / 2.33 |
* paired det_padded − fast_padded step: +5.42 ms [+5.01, +5.50] (+11.92 % [+10.87, +12.11])
* paired det_padded − fast_padded dw_kernel: -0.80 ms [-0.95, -0.71] (-4.78 % [-5.60, -4.24])
* paired det_padded − fast_padded fwd_total: +1.02 ms [+0.92, +1.05] (+20.33 % [+17.90, +20.79])
* paired det_padded − fast_padded dx_kernel: +5.18 ms [+5.14, +5.27] (+24.24 % [+24.07, +24.47])
* paired det_compact − fast_compact step: +5.41 ms [+5.16, +5.73] (+11.64 % [+11.02, +12.43])
* paired det_compact − fast_compact dw_kernel: -0.83 ms [-1.08, -0.73] (-4.75 % [-6.08, -4.16])
* paired det_compact − fast_compact fwd_total: +1.01 ms [+0.92, +1.06] (+19.77 % [+17.81, +21.05])
* paired det_compact − fast_compact dx_kernel: +5.31 ms [+5.13, +5.43] (+24.70 % [+23.98, +25.37])
* paired fast_compact − fast_padded step: +0.68 ms [+0.54, +1.41] (+1.49 % [+1.17, +3.09])
* paired fast_compact − fast_padded dw_kernel: +0.78 ms [+0.71, +0.96] (+4.68 % [+4.23, +5.75])
* paired fast_compact − fast_padded fwd_total: -0.01 ms [-0.04, +0.14] (-0.15 % [-0.74, +2.75])
* paired fast_compact − fast_padded dx_kernel: +0.00 ms [-0.11, +0.25] (+0.00 % [-0.50, +1.18])
* paired det_compact − det_padded step: +0.93 ms [+0.60, +1.20] (+1.82 % [+1.17, +2.36])
* paired det_compact − det_padded dw_kernel: +0.76 ms [+0.68, +0.82] (+4.76 % [+4.28, +5.15])
* paired det_compact − det_padded fwd_total: +0.01 ms [-0.02, +0.04] (+0.22 % [-0.28, +0.70])
* paired det_compact − det_padded dx_kernel: +0.09 ms [+0.06, +0.32] (+0.34 % [+0.21, +1.19])

**H10000000_M8192_S16_R4** (C 8, S 16, R 4, H 10,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 36.6 |  /  | 117.0 |  /  | 124.2 | 242.7 | 2.2 | 281.4 [280.2–281.5] | 1.12 / 1.33 |
| fast_compact | tiles T64/WG1 | compact | 36.8 |  /  | 116.8 |  /  | 133.8 | 251.9 | 2.2 | 290.5 [289.3–291.1] | 1.15 / 1.33 |
| det_padded | tiles T64/WG1 | padded | 46.6 | 36.1 / 10.0 | 114.8 | 114.5 / 0.2 | 123.7 | 239.7 | 2.2 | 288.3 [287.9–289.0] | 1.94 / 2.33 |
| det_compact | tiles T64/WG1 | compact | 46.6 | 36.1 / 10.0 | 115.7 | 115.4 / 0.2 | 130.9 | 248.8 | 2.2 | 297.8 [296.7–299.8] | 1.94 / 2.33 |
* paired det_padded − fast_padded step: +7.39 ms [+7.00, +7.78] (+2.63 % [+2.49, +2.78])
* paired det_padded − fast_padded dw_kernel: -0.60 ms [-0.72, -0.47] (-0.48 % [-0.58, -0.38])
* paired det_padded − fast_padded fwd_total: +9.81 ms [+9.72, +9.95] (+26.66 % [+26.32, +27.26])
* paired det_padded − fast_padded dx_kernel: -2.04 ms [-2.52, -1.18] (-1.74 % [-2.15, -1.02])
* paired det_compact − fast_compact step: +7.31 ms [+6.43, +10.15] (+2.52 % [+2.21, +3.50])
* paired det_compact − fast_compact dw_kernel: -0.85 ms [-2.73, +1.10] (-0.63 % [-2.02, +0.83])
* paired det_compact − fast_compact fwd_total: +9.92 ms [+9.84, +10.11] (+26.94 % [+26.67, +27.74])
* paired det_compact − fast_compact dx_kernel: -1.48 ms [-1.66, -0.94] (-1.26 % [-1.42, -0.81])
* paired fast_compact − fast_padded step: +9.17 ms [+8.37, +9.79] (+3.26 % [+2.98, +3.48])
* paired fast_compact − fast_padded dw_kernel: +8.57 ms [+6.67, +10.56] (+6.90 % [+5.39, +8.49])
* paired fast_compact − fast_padded fwd_total: -0.04 ms [-0.30, +0.39] (-0.10 % [-0.81, +1.08])
* paired fast_compact − fast_padded dx_kernel: +0.15 ms [-0.80, +0.67] (+0.13 % [-0.68, +0.58])
* paired det_compact − det_padded step: +9.08 ms [+8.48, +11.48] (+3.15 % [+2.93, +3.98])
* paired det_compact − det_padded dw_kernel: +8.12 ms [+7.61, +9.57] (+6.56 % [+6.17, +7.74])
* paired det_compact − det_padded fwd_total: +0.12 ms [-0.01, +0.29] (+0.27 % [-0.02, +0.63])
* paired det_compact − det_padded dx_kernel: +0.61 ms [+0.00, +1.17] (+0.53 % [+0.00, +1.02])

**H10000000_M8192_S32_R8** (C 8, S 32, R 8, H 10,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 38.3 |  /  | 143.6 |  /  | 133.3 | 278.1 | 2.2 | 318.8 [318.3–320.0] | 1.12 / 1.33 |
| fast_compact | tiles T32/WG1 | compact | 38.4 |  /  | 144.4 |  /  | 140.2 | 285.9 | 2.2 | 326.5 [326.2–326.8] | 1.15 / 1.33 |
| det_padded | tiles T32/WG1 | padded | 47.7 | 37.2 / 10.0 | 142.3 | 141.9 / 0.4 | 133.3 | 277.4 | 2.2 | 327.2 [326.8–327.7] | 1.94 / 2.33 |
| det_compact | tiles T32/WG1 | compact | 47.8 | 37.3 / 10.0 | 143.1 | 142.7 / 0.4 | 139.3 | 283.9 | 2.2 | 334.4 [333.3–335.4] | 1.94 / 2.33 |
* paired det_padded − fast_padded step: +8.33 ms [+7.71, +8.54] (+2.61 % [+2.41, +2.68])
* paired det_padded − fast_padded dw_kernel: -0.12 ms [-0.16, +0.56] (-0.09 % [-0.12, +0.42])
* paired det_padded − fast_padded fwd_total: +9.24 ms [+9.19, +9.63] (+24.00 % [+23.85, +25.23])
* paired det_padded − fast_padded dx_kernel: -1.35 ms [-1.93, -1.13] (-0.94 % [-1.33, -0.78])
* paired det_compact − fast_compact step: +8.30 ms [+6.50, +8.95] (+2.54 % [+1.99, +2.74])
* paired det_compact − fast_compact dw_kernel: -0.77 ms [-1.12, -0.57] (-0.55 % [-0.80, -0.40])
* paired det_compact − fast_compact fwd_total: +9.42 ms [+9.36, +9.49] (+24.53 % [+24.34, +24.81])
* paired det_compact − fast_compact dx_kernel: -1.14 ms [-1.85, -0.48] (-0.79 % [-1.28, -0.33])
* paired fast_compact − fast_padded step: +7.73 ms [+6.49, +7.92] (+2.42 % [+2.03, +2.49])
* paired fast_compact − fast_padded dw_kernel: +6.82 ms [+6.58, +6.98] (+5.12 % [+4.93, +5.24])
* paired fast_compact − fast_padded fwd_total: -0.11 ms [-0.19, +0.28] (-0.28 % [-0.49, +0.73])
* paired fast_compact − fast_padded dx_kernel: +0.43 ms [-0.10, +0.66] (+0.30 % [-0.07, +0.46])
* paired det_compact − det_padded step: +7.38 ms [+5.87, +8.37] (+2.25 % [+1.79, +2.56])
* paired det_compact − det_padded dw_kernel: +5.92 ms [+5.67, +6.30] (+4.44 % [+4.24, +4.74])
* paired det_compact − det_padded fwd_total: +0.07 ms [+0.01, +0.12] (+0.16 % [+0.02, +0.26])
* paired det_compact − det_padded dx_kernel: +0.61 ms [+0.11, +1.25] (+0.43 % [+0.07, +0.88])

**H10000000_M8192_S64_R16** (C 8, S 64, R 16, H 10,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG8 | padded | 39.4 |  /  | 180.6 |  /  | 135.1 | 317.2 | 2.2 | 358.9 [358.0–359.2] | 1.12 / 1.33 |
| fast_compact | tiles T64/WG8 | compact | 39.4 |  /  | 181.0 |  /  | 141.8 | 324.6 | 2.2 | 366.6 [365.5–367.5] | 1.15 / 1.33 |
| det_padded | private_t32 T32/WG1 | padded | 48.4 | 37.9 / 10.0 | 216.8 | 216.1 / 0.7 | 129.8 | 347.9 | 2.2 | 398.5 [397.9–398.8] | 1.94 / 2.33 |
| det_compact | private_t32 T32/WG1 | compact | 48.5 | 38.0 / 10.0 | 216.0 | 215.2 / 0.7 | 136.7 | 354.1 | 2.2 | 405.0 [404.4–405.3] | 1.94 / 2.33 |
* paired det_padded − fast_padded step: +39.82 ms [+38.90, +40.52] (+11.10 % [+10.84, +11.32])
* paired det_padded − fast_padded dw_kernel: -5.19 ms [-5.43, -5.09] (-3.84 % [-4.01, -3.77])
* paired det_padded − fast_padded fwd_total: +9.10 ms [+9.02, +9.27] (+23.09 % [+22.83, +23.66])
* paired det_padded − fast_padded dx_kernel: +35.90 ms [+35.62, +36.51] (+19.88 % [+19.65, +20.25])
* paired det_compact − fast_compact step: +38.79 ms [+36.88, +39.53] (+10.59 % [+10.03, +10.81])
* paired det_compact − fast_compact dw_kernel: -5.06 ms [-5.18, -4.95] (-3.56 % [-3.65, -3.50])
* paired det_compact − fast_compact fwd_total: +9.00 ms [+8.85, +9.68] (+22.73 % [+22.35, +24.92])
* paired det_compact − fast_compact dx_kernel: +35.00 ms [+33.52, +35.59] (+19.33 % [+18.38, +19.72])
* paired fast_compact − fast_padded step: +8.12 ms [+6.32, +8.57] (+2.27 % [+1.76, +2.39])
* paired fast_compact − fast_padded dw_kernel: +6.70 ms [+6.62, +6.97] (+4.96 % [+4.90, +5.16])
* paired fast_compact − fast_padded fwd_total: +0.12 ms [-0.36, +0.18] (+0.31 % [-0.91, +0.45])
* paired fast_compact − fast_padded dx_kernel: +0.30 ms [-0.31, +1.99] (+0.17 % [-0.17, +1.10])
* paired det_compact − det_padded step: +6.60 ms [+6.02, +6.86] (+1.66 % [+1.51, +1.72])
* paired det_compact − det_padded dw_kernel: +6.90 ms [+6.77, +7.05] (+5.31 % [+5.21, +5.43])
* paired det_compact − det_padded fwd_total: -0.00 ms [-0.09, +0.20] (-0.01 % [-0.18, +0.40])
* paired det_compact − det_padded dx_kernel: -0.76 ms [-0.86, -0.16] (-0.35 % [-0.40, -0.07])

**H1000000_M1024_S16_R4** (C 8, S 16, R 4, H 1,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 0.7 |  /  | 1.5 |  /  | 1.5 | 3.1 | 0.2 | 4.1 [4.0–4.4] | 0.12 / 0.14 |
| fast_compact | tiles T64/WG1 | compact | 0.7 |  /  | 1.5 |  /  | 1.6 | 3.2 | 0.2 | 4.2 [4.1–4.3] | 0.13 / 0.14 |
| det_padded | tiles T64/WG1 | padded | 0.9 | 0.5 / 0.2 | 1.4 | 1.4 / 0.0 | 1.5 | 3.1 | 0.2 | 4.2 [4.1–4.4] | 0.22 / 0.26 |
| det_compact | tiles T64/WG1 | compact | 0.9 | 0.5 / 0.2 | 1.4 | 1.4 / 0.0 | 1.5 | 3.1 | 0.2 | 4.2 [4.2–4.3] | 0.22 / 0.26 |
* paired det_padded − fast_padded step: +0.13 ms [-0.05, +0.34] (+3.21 % [-1.02, +8.41])
* paired det_padded − fast_padded dw_kernel: -0.00 ms [-0.18, +0.00] (-0.19 % [-9.97, +0.20])
* paired det_padded − fast_padded fwd_total: +0.18 ms [+0.17, +0.22] (+25.93 % [+24.34, +31.47])
* paired det_padded − fast_padded dx_kernel: -0.03 ms [-0.06, +0.15] (-2.03 % [-3.91, +10.14])
* paired det_compact − fast_compact step: +0.06 ms [-0.04, +0.16] (+1.45 % [-1.02, +3.88])
* paired det_compact − fast_compact dw_kernel: -0.07 ms [-0.16, -0.00] (-4.33 % [-9.38, -0.20])
* paired det_compact − fast_compact fwd_total: +0.18 ms [+0.17, +0.19] (+26.04 % [+23.94, +28.08])
* paired det_compact − fast_compact dx_kernel: -0.04 ms [-0.06, -0.03] (-2.92 % [-4.01, -1.74])
* paired fast_compact − fast_padded step: +0.12 ms [-0.18, +0.19] (+3.03 % [-4.07, +4.76])
* paired fast_compact − fast_padded dw_kernel: +0.10 ms [-0.09, +0.26] (+6.60 % [-5.13, +17.65])
* paired fast_compact − fast_padded fwd_total: +0.00 ms [-0.01, +0.02] (+0.26 % [-1.72, +3.51])
* paired fast_compact − fast_padded dx_kernel: +0.01 ms [-0.07, +0.02] (+0.45 % [-4.57, +1.45])
* paired det_compact − det_padded step: +0.00 ms [-0.18, +0.12] (+0.11 % [-4.08, +2.78])
* paired det_compact − det_padded dw_kernel: +0.09 ms [-0.05, +0.10] (+6.14 % [-3.24, +6.98])
* paired det_compact − det_padded fwd_total: +0.00 ms [-0.02, +0.01] (+0.46 % [-2.42, +1.02])
* paired det_compact − det_padded dx_kernel: -0.04 ms [-0.17, +0.01] (-2.53 % [-10.29, +0.39])

**H1000000_M1024_S32_R8** (C 8, S 32, R 8, H 1,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 0.7 |  /  | 1.9 |  /  | 1.5 | 3.6 | 0.2 | 4.5 [4.5–4.8] | 0.12 / 0.14 |
| fast_compact | tiles T32/WG1 | compact | 0.7 |  /  | 2.0 |  /  | 1.7 | 3.8 | 0.2 | 4.8 [4.6–5.3] | 0.13 / 0.14 |
| det_padded | tiles T32/WG1 | padded | 0.9 | 0.5 / 0.2 | 1.9 | 1.8 / 0.0 | 1.6 | 3.6 | 0.2 | 4.7 [4.7–5.4] | 0.22 / 0.26 |
| det_compact | tiles T32/WG1 | compact | 0.9 | 0.5 / 0.2 | 1.9 | 1.8 / 0.0 | 1.6 | 3.7 | 0.2 | 4.8 [4.8–4.8] | 0.22 / 0.26 |
* paired det_padded − fast_padded step: +0.31 ms [-0.02, +0.56] (+6.91 % [-0.46, +11.61])
* paired det_padded − fast_padded dw_kernel: +0.09 ms [-0.00, +0.18] (+5.66 % [-0.03, +11.02])
* paired det_padded − fast_padded fwd_total: +0.18 ms [+0.15, +0.21] (+24.91 % [+19.89, +29.12])
* paired det_padded − fast_padded dx_kernel: +0.01 ms [-0.11, +0.04] (+0.57 % [-5.28, +1.90])
* paired det_compact − fast_compact step: +0.06 ms [-0.44, +0.20] (+1.29 % [-8.43, +4.42])
* paired det_compact − fast_compact dw_kernel: -0.05 ms [-0.32, +0.02] (-2.60 % [-16.46, +1.41])
* paired det_compact − fast_compact fwd_total: +0.16 ms [+0.14, +0.19] (+22.51 % [+19.01, +26.14])
* paired det_compact − fast_compact dx_kernel: -0.06 ms [-0.30, +0.01] (-3.25 % [-13.62, +0.60])
* paired fast_compact − fast_padded step: +0.13 ms [+0.03, +0.55] (+2.92 % [+0.56, +11.59])
* paired fast_compact − fast_padded dw_kernel: +0.08 ms [+0.08, +0.41] (+5.12 % [+4.86, +26.89])
* paired fast_compact − fast_padded fwd_total: +0.00 ms [-0.00, +0.02] (+0.37 % [-0.31, +2.42])
* paired fast_compact − fast_padded dx_kernel: +0.05 ms [-0.03, +0.19] (+2.60 % [-1.45, +9.76])
* paired det_compact − det_padded step: -0.00 ms [-0.56, +0.12] (-0.05 % [-10.41, +2.66])
* paired det_compact − det_padded dw_kernel: +0.01 ms [-0.21, +0.09] (+0.76 % [-11.15, +6.04])
* paired det_compact − det_padded fwd_total: -0.01 ms [-0.03, +0.00] (-0.74 % [-2.90, +0.28])
* paired det_compact − det_padded dx_kernel: +0.00 ms [-0.11, +0.00] (+0.01 % [-5.32, +0.16])

**H1000000_M1024_S64_R16** (C 8, S 64, R 16, H 1,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG8 | padded | 0.7 |  /  | 2.1 |  /  | 1.6 | 3.9 | 0.2 | 4.9 [4.8–5.1] | 0.12 / 0.14 |
| fast_compact | tiles T64/WG8 | compact | 0.7 |  /  | 2.1 |  /  | 1.7 | 4.0 | 0.2 | 5.0 [4.9–5.2] | 0.13 / 0.14 |
| det_padded | private_t32 T32/WG1 | padded | 0.9 | 0.5 / 0.2 | 3.1 | 3.0 / 0.1 | 1.6 | 4.9 | 0.2 | 6.0 [6.0–6.1] | 0.22 / 0.26 |
| det_compact | private_t32 T32/WG1 | compact | 0.9 | 0.5 / 0.2 | 3.1 | 3.0 / 0.1 | 1.7 | 5.0 | 0.2 | 6.1 [6.1–6.2] | 0.22 / 0.26 |
* paired det_padded − fast_padded step: +1.16 ms [+0.93, +1.21] (+23.78 % [+18.21, +24.90])
* paired det_padded − fast_padded dw_kernel: -0.02 ms [-0.12, +0.01] (-1.05 % [-6.99, +0.47])
* paired det_padded − fast_padded fwd_total: +0.18 ms [+0.15, +0.18] (+24.92 % [+19.24, +25.27])
* paired det_padded − fast_padded dx_kernel: +1.00 ms [+0.93, +1.03] (+47.31 % [+41.67, +48.48])
* paired det_compact − fast_compact step: +1.19 ms [+0.87, +1.27] (+24.19 % [+16.82, +25.69])
* paired det_compact − fast_compact dw_kernel: -0.03 ms [-0.13, +0.02] (-1.55 % [-7.31, +1.09])
* paired det_compact − fast_compact fwd_total: +0.16 ms [+0.14, +0.18] (+21.97 % [+19.29, +24.93])
* paired det_compact − fast_compact dx_kernel: +1.03 ms [+0.87, +1.15] (+48.89 % [+39.24, +54.52])
* paired fast_compact − fast_padded step: +0.06 ms [+0.06, +0.10] (+1.33 % [+1.11, +2.02])
* paired fast_compact − fast_padded dw_kernel: +0.07 ms [+0.07, +0.09] (+4.56 % [+3.76, +5.39])
* paired fast_compact − fast_padded fwd_total: -0.00 ms [-0.01, +0.01] (-0.60 % [-1.20, +0.76])
* paired fast_compact − fast_padded dx_kernel: -0.00 ms [-0.01, +0.01] (-0.12 % [-0.27, +0.28])
* paired det_compact − det_padded step: +0.06 ms [-0.00, +0.23] (+1.07 % [-0.07, +3.88])
* paired det_compact − det_padded dw_kernel: +0.07 ms [+0.06, +0.09] (+4.37 % [+3.41, +5.35])
* paired det_compact − det_padded fwd_total: -0.01 ms [-0.03, -0.01] (-1.06 % [-3.55, -0.94])
* paired det_compact − det_padded dx_kernel: +0.00 ms [-0.06, +0.18] (+0.13 % [-1.79, +5.86])

**H1000000_M8192_S16_R4** (C 8, S 16, R 4, H 1,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 4.5 |  /  | 12.1 |  /  | 12.6 | 25.9 | 0.2 | 30.7 [30.3–31.1] | 0.34 / 0.36 |
| fast_compact | tiles T64/WG1 | compact | 4.5 |  /  | 12.0 |  /  | 13.3 | 26.5 | 0.2 | 31.2 [31.1–31.3] | 0.34 / 0.36 |
| det_padded | tiles T64/WG1 | padded | 5.6 | 3.9 / 1.2 | 12.1 | 11.9 / 0.2 | 12.6 | 25.9 | 0.2 | 31.7 [31.7–31.8] | 1.12 / 1.33 |
| det_compact | tiles T64/WG1 | compact | 5.6 | 3.9 / 1.2 | 12.1 | 11.9 / 0.2 | 13.4 | 26.7 | 0.2 | 32.5 [32.3–32.5] | 1.12 / 1.33 |
* paired det_padded − fast_padded step: +0.97 ms [+0.62, +1.51] (+3.17 % [+2.00, +4.99])
* paired det_padded − fast_padded dw_kernel: -0.02 ms [-0.12, +0.11] (-0.13 % [-0.93, +0.88])
* paired det_padded − fast_padded fwd_total: +1.08 ms [+1.06, +1.16] (+23.80 % [+23.11, +25.90])
* paired det_padded − fast_padded dx_kernel: +0.07 ms [-0.25, +0.16] (+0.56 % [-2.05, +1.36])
* paired det_compact − fast_compact step: +1.20 ms [+1.17, +1.41] (+3.85 % [+3.73, +4.54])
* paired det_compact − fast_compact dw_kernel: +0.02 ms [-0.09, +0.08] (+0.16 % [-0.70, +0.62])
* paired det_compact − fast_compact fwd_total: +1.15 ms [+1.14, +1.19] (+25.86 % [+25.52, +26.64])
* paired det_compact − fast_compact dx_kernel: +0.13 ms [+0.08, +0.21] (+1.11 % [+0.65, +1.73])
* paired fast_compact − fast_padded step: +0.37 ms [+0.18, +0.96] (+1.19 % [+0.59, +3.17])
* paired fast_compact − fast_padded dw_kernel: +0.72 ms [+0.62, +0.94] (+5.74 % [+4.89, +7.52])
* paired fast_compact − fast_padded fwd_total: -0.06 ms [-0.11, -0.02] (-1.43 % [-2.51, -0.37])
* paired fast_compact − fast_padded dx_kernel: -0.12 ms [-0.43, -0.04] (-0.98 % [-3.47, -0.33])
* paired det_compact − det_padded step: +0.69 ms [+0.60, +0.78] (+2.18 % [+1.90, +2.47])
* paired det_compact − det_padded dw_kernel: +0.77 ms [+0.71, +0.84] (+6.09 % [+5.61, +6.68])
* paired det_compact − det_padded fwd_total: +0.01 ms [-0.03, +0.02] (+0.09 % [-0.45, +0.28])
* paired det_compact − det_padded dx_kernel: -0.06 ms [-0.10, +0.01] (-0.46 % [-0.79, +0.08])

**H1000000_M8192_S32_R8** (C 8, S 32, R 8, H 1,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 4.7 |  /  | 15.9 |  /  | 13.1 | 30.3 | 0.2 | 35.2 [34.9–35.3] | 0.34 / 0.36 |
| fast_compact | tiles T32/WG1 | compact | 4.7 |  /  | 16.0 |  /  | 13.9 | 31.1 | 0.2 | 36.0 [35.9–36.2] | 0.34 / 0.36 |
| det_padded | tiles T32/WG1 | padded | 5.7 | 4.1 / 1.2 | 16.1 | 15.7 / 0.4 | 13.2 | 30.5 | 0.2 | 36.5 [36.3–36.7] | 1.12 / 1.33 |
| det_compact | tiles T32/WG1 | compact | 5.7 | 4.0 / 1.2 | 16.0 | 15.6 / 0.4 | 13.9 | 31.1 | 0.2 | 37.0 [36.8–37.1] | 1.12 / 1.33 |
* paired det_padded − fast_padded step: +1.43 ms [+1.26, +1.57] (+4.08 % [+3.59, +4.47])
* paired det_padded − fast_padded dw_kernel: +0.06 ms [-0.03, +0.16] (+0.48 % [-0.23, +1.19])
* paired det_padded − fast_padded fwd_total: +1.11 ms [+1.06, +1.13] (+23.93 % [+22.62, +24.62])
* paired det_padded − fast_padded dx_kernel: +0.27 ms [+0.21, +0.29] (+1.67 % [+1.34, +1.84])
* paired det_compact − fast_compact step: +0.96 ms [+0.79, +1.07] (+2.66 % [+2.20, +2.97])
* paired det_compact − fast_compact dw_kernel: +0.03 ms [-0.06, +0.09] (+0.20 % [-0.44, +0.63])
* paired det_compact − fast_compact fwd_total: +1.03 ms [+0.98, +1.07] (+21.97 % [+20.75, +22.95])
* paired det_compact − fast_compact dx_kernel: -0.03 ms [-0.16, +0.09] (-0.18 % [-1.03, +0.54])
* paired fast_compact − fast_padded step: +0.94 ms [+0.76, +1.07] (+2.69 % [+2.14, +3.06])
* paired fast_compact − fast_padded dw_kernel: +0.77 ms [+0.73, +0.79] (+5.84 % [+5.52, +6.05])
* paired fast_compact − fast_padded fwd_total: +0.03 ms [+0.01, +0.10] (+0.74 % [+0.32, +2.07])
* paired fast_compact − fast_padded dx_kernel: +0.12 ms [-0.03, +0.18] (+0.75 % [-0.16, +1.17])
* paired det_compact − det_padded step: +0.40 ms [+0.35, +0.66] (+1.09 % [+0.94, +1.82])
* paired det_compact − det_padded dw_kernel: +0.71 ms [+0.62, +0.84] (+5.37 % [+4.66, +6.43])
* paired det_compact − det_padded fwd_total: -0.04 ms [-0.06, +0.01] (-0.72 % [-1.03, +0.25])
* paired det_compact − det_padded dx_kernel: -0.18 ms [-0.27, -0.14] (-1.12 % [-1.69, -0.89])

**H1000000_M8192_S64_R16** (C 8, S 64, R 16, H 1,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG8 | padded | 4.8 |  /  | 17.9 |  /  | 14.1 | 33.1 | 0.2 | 38.1 [38.0–38.3] | 0.34 / 0.36 |
| fast_compact | tiles T64/WG8 | compact | 4.8 |  /  | 17.8 |  /  | 14.8 | 33.8 | 0.2 | 38.8 [38.7–39.0] | 0.34 / 0.36 |
| det_padded | private_t32 T32/WG1 | padded | 5.7 | 4.1 / 1.2 | 26.1 | 25.3 / 0.7 | 13.3 | 40.6 | 0.2 | 46.5 [46.5–46.6] | 1.12 / 1.33 |
| det_compact | private_t32 T32/WG1 | compact | 5.7 | 4.1 / 1.2 | 26.2 | 25.4 / 0.7 | 14.0 | 41.3 | 0.2 | 47.3 [47.2–47.6] | 1.12 / 1.33 |
* paired det_padded − fast_padded step: +8.41 ms [+8.18, +8.52] (+22.04 % [+21.34, +22.40])
* paired det_padded − fast_padded dw_kernel: -0.79 ms [-0.86, -0.74] (-5.60 % [-6.05, -5.28])
* paired det_padded − fast_padded fwd_total: +0.95 ms [+0.92, +0.96] (+19.90 % [+19.02, +20.10])
* paired det_padded − fast_padded dx_kernel: +8.26 ms [+8.10, +8.33] (+46.41 % [+45.15, +46.86])
* paired det_compact − fast_compact step: +8.47 ms [+8.33, +8.63] (+21.75 % [+21.41, +22.33])
* paired det_compact − fast_compact dw_kernel: -0.83 ms [-0.88, -0.71] (-5.57 % [-5.90, -4.84])
* paired det_compact − fast_compact fwd_total: +0.95 ms [+0.92, +0.96] (+19.76 % [+19.03, +20.04])
* paired det_compact − fast_compact dx_kernel: +8.35 ms [+8.28, +8.49] (+46.93 % [+46.42, +47.48])
* paired fast_compact − fast_padded step: +0.71 ms [+0.45, +0.95] (+1.87 % [+1.18, +2.48])
* paired fast_compact − fast_padded dw_kernel: +0.70 ms [+0.62, +0.78] (+4.95 % [+4.36, +5.53])
* paired fast_compact − fast_padded fwd_total: +0.01 ms [-0.00, +0.02] (+0.15 % [-0.07, +0.33])
* paired fast_compact − fast_padded dx_kernel: -0.01 ms [-0.14, +0.14] (-0.06 % [-0.78, +0.81])
* paired det_compact − det_padded step: +0.72 ms [+0.70, +1.08] (+1.54 % [+1.51, +2.33])
* paired det_compact − det_padded dw_kernel: +0.69 ms [+0.64, +0.72] (+5.16 % [+4.76, +5.42])
* paired det_compact − det_padded fwd_total: +0.00 ms [-0.00, +0.01] (+0.03 % [-0.01, +0.23])
* paired det_compact − det_padded dx_kernel: +0.07 ms [+0.04, +0.32] (+0.26 % [+0.14, +1.24])

**H23456789_M8192_S32_R8_ragged** (C 8, S 32, R 8, H 23,456,789, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 78.0 |  /  | 313.8 |  /  | 275.1 | 590.6 | 5.0 | 673.9 [673.4–674.3] | 2.27 / 2.68 |
| fast_compact | tiles T32/WG1 | compact | 78.0 |  /  | 314.2 |  /  | 286.1 | 601.9 | 5.0 | 684.9 [684.6–685.0] | 2.31 / 2.68 |
| det_padded | tiles T32/WG1 | padded | 87.2 | 76.6 / 10.0 | 313.5 | 313.1 / 0.4 | 275.1 | 590.0 | 5.0 | 682.5 [681.8–682.8] | 3.09 / 3.68 |
| det_compact | tiles T32/WG1 | compact | 87.2 | 76.6 / 10.0 | 313.5 | 313.1 / 0.4 | 286.0 | 601.1 | 5.0 | 694.0 [693.1–694.7] | 3.09 / 3.68 |
* paired det_padded − fast_padded step: +8.43 ms [+8.11, +9.12] (+1.25 % [+1.20, +1.35])
* paired det_padded − fast_padded dw_kernel: +0.09 ms [-0.35, +0.19] (+0.03 % [-0.13, +0.07])
* paired det_padded − fast_padded fwd_total: +9.18 ms [+8.94, +9.30] (+11.77 % [+11.43, +11.94])
* paired det_padded − fast_padded dx_kernel: -0.55 ms [-1.63, +0.56] (-0.18 % [-0.52, +0.18])
* paired det_compact − fast_compact step: +8.57 ms [+8.22, +9.79] (+1.25 % [+1.20, +1.43])
* paired det_compact − fast_compact dw_kernel: +0.11 ms [-0.44, +0.39] (+0.04 % [-0.15, +0.14])
* paired det_compact − fast_compact fwd_total: +9.37 ms [+8.93, +9.49] (+12.03 % [+11.42, +12.19])
* paired det_compact − fast_compact dx_kernel: -0.87 ms [-2.00, -0.29] (-0.28 % [-0.64, -0.09])
* paired fast_compact − fast_padded step: +10.92 ms [+10.38, +11.55] (+1.62 % [+1.54, +1.71])
* paired fast_compact − fast_padded dw_kernel: +11.11 ms [+10.34, +11.24] (+4.04 % [+3.75, +4.09])
* paired fast_compact − fast_padded fwd_total: -0.07 ms [-0.16, +0.07] (-0.08 % [-0.21, +0.09])
* paired fast_compact − fast_padded dx_kernel: +0.55 ms [+0.11, +1.51] (+0.17 % [+0.03, +0.48])
* paired det_compact − det_padded step: +10.79 ms [+10.37, +12.86] (+1.58 % [+1.52, +1.89])
* paired det_compact − det_padded dw_kernel: +11.03 ms [+10.47, +11.43] (+4.01 % [+3.80, +4.16])
* paired det_compact − det_padded fwd_total: +0.11 ms [-0.01, +0.15] (+0.12 % [-0.01, +0.17])
* paired det_compact − det_padded dx_kernel: +0.04 ms [-0.29, +1.05] (+0.01 % [-0.09, +0.34])

**H30000000_M1024_S16_R4** (C 8, S 16, R 4, H 30,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 11.3 |  /  | 39.1 |  /  | 45.2 | 84.5 | 6.4 | 102.2 [102.0–102.5] | 2.62 / 3.42 |
| fast_compact | tiles T64/WG1 | compact | 11.3 |  /  | 39.4 |  /  | 46.7 | 86.3 | 6.4 | 104.0 [103.8–104.1] | 2.66 / 3.42 |
| det_padded | tiles T64/WG1 | padded | 12.5 | 11.0 / 1.2 | 39.4 | 39.3 / 0.0 | 45.2 | 84.8 | 6.4 | 103.7 [103.5–103.7] | 3.60 / 4.42 |
| det_compact | tiles T64/WG1 | compact | 12.5 | 11.0 / 1.2 | 39.3 | 39.3 / 0.0 | 46.7 | 86.2 | 6.4 | 105.2 [105.1–105.3] | 3.60 / 4.42 |
* paired det_padded − fast_padded step: +1.40 ms [+1.19, +1.52] (+1.37 % [+1.16, +1.49])
* paired det_padded − fast_padded dw_kernel: +0.04 ms [-0.20, +0.09] (+0.10 % [-0.44, +0.20])
* paired det_padded − fast_padded fwd_total: +1.20 ms [+1.10, +1.23] (+10.66 % [+9.67, +10.94])
* paired det_padded − fast_padded dx_kernel: +0.19 ms [+0.09, +0.24] (+0.49 % [+0.23, +0.61])
* paired det_compact − fast_compact step: +1.22 ms [+1.00, +1.45] (+1.17 % [+0.96, +1.39])
* paired det_compact − fast_compact dw_kernel: -0.08 ms [-0.14, +0.12] (-0.18 % [-0.31, +0.25])
* paired det_compact − fast_compact fwd_total: +1.18 ms [+1.15, +1.22] (+10.45 % [+10.13, +10.82])
* paired det_compact − fast_compact dx_kernel: -0.11 ms [-0.18, +0.10] (-0.27 % [-0.45, +0.25])
* paired fast_compact − fast_padded step: +1.73 ms [+1.44, +1.95] (+1.69 % [+1.41, +1.91])
* paired fast_compact − fast_padded dw_kernel: +1.47 ms [+1.30, +1.72] (+3.26 % [+2.87, +3.81])
* paired fast_compact − fast_padded fwd_total: +0.02 ms [-0.02, +0.09] (+0.20 % [-0.18, +0.80])
* paired fast_compact − fast_padded dx_kernel: +0.23 ms [+0.12, +0.33] (+0.58 % [+0.31, +0.84])
* paired det_compact − det_padded step: +1.54 ms [+1.45, +1.73] (+1.48 % [+1.40, +1.67])
* paired det_compact − det_padded dw_kernel: +1.45 ms [+1.37, +1.52] (+3.21 % [+3.03, +3.38])
* paired det_compact − det_padded fwd_total: +0.04 ms [-0.01, +0.05] (+0.29 % [-0.04, +0.41])
* paired det_compact − det_padded dx_kernel: -0.06 ms [-0.12, +0.14] (-0.15 % [-0.31, +0.36])

**H30000000_M1024_S32_R8** (C 8, S 32, R 8, H 30,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 11.6 |  /  | 46.8 |  /  | 47.2 | 94.2 | 6.4 | 112.2 [112.0–112.9] | 2.62 / 3.42 |
| fast_compact | tiles T32/WG1 | compact | 11.7 |  /  | 46.9 |  /  | 48.4 | 95.4 | 6.4 | 113.6 [113.4–114.3] | 2.66 / 3.42 |
| det_padded | tiles T32/WG1 | padded | 12.8 | 11.3 / 1.2 | 46.7 | 46.6 / 0.0 | 47.2 | 93.9 | 6.4 | 113.1 [113.1–113.6] | 3.60 / 4.42 |
| det_compact | tiles T32/WG1 | compact | 12.8 | 11.3 / 1.2 | 46.9 | 46.8 / 0.0 | 48.5 | 95.1 | 6.4 | 114.5 [114.0–114.7] | 3.60 / 4.42 |
* paired det_padded − fast_padded step: +0.98 ms [+0.68, +1.18] (+0.87 % [+0.60, +1.05])
* paired det_padded − fast_padded dw_kernel: -0.08 ms [-0.10, +0.05] (-0.17 % [-0.22, +0.10])
* paired det_padded − fast_padded fwd_total: +1.13 ms [+1.08, +1.19] (+9.78 % [+9.21, +10.24])
* paired det_padded − fast_padded dx_kernel: -0.07 ms [-0.29, +0.13] (-0.16 % [-0.62, +0.27])
* paired det_compact − fast_compact step: +0.83 ms [+0.16, +1.07] (+0.73 % [+0.14, +0.95])
* paired det_compact − fast_compact dw_kernel: -0.01 ms [-0.30, +0.21] (-0.02 % [-0.61, +0.44])
* paired det_compact − fast_compact fwd_total: +1.06 ms [+1.06, +1.14] (+9.04 % [+9.01, +9.80])
* paired det_compact − fast_compact dx_kernel: -0.12 ms [-0.25, +0.00] (-0.25 % [-0.54, +0.00])
* paired fast_compact − fast_padded step: +1.39 ms [+1.36, +1.46] (+1.23 % [+1.21, +1.30])
* paired fast_compact − fast_padded dw_kernel: +1.24 ms [+1.21, +1.27] (+2.62 % [+2.57, +2.69])
* paired fast_compact − fast_padded fwd_total: +0.09 ms [-0.00, +0.11] (+0.79 % [-0.03, +0.98])
* paired fast_compact − fast_padded dx_kernel: +0.10 ms [+0.05, +0.36] (+0.22 % [+0.10, +0.78])
* paired det_compact − det_padded step: +1.10 ms [+0.85, +1.61] (+0.97 % [+0.75, +1.43])
* paired det_compact − det_padded dw_kernel: +1.32 ms [+0.90, +1.52] (+2.81 % [+1.91, +3.22])
* paired det_compact − det_padded fwd_total: +0.01 ms [-0.02, +0.06] (+0.11 % [-0.15, +0.51])
* paired det_compact − det_padded dx_kernel: +0.16 ms [-0.14, +0.27] (+0.34 % [-0.29, +0.58])

**H30000000_M1024_S64_R16** (C 8, S 64, R 16, H 30,000,000, M 1024, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG8 | padded | 11.8 |  /  | 64.9 |  /  | 46.8 | 111.9 | 6.4 | 130.1 [130.0–130.3] | 2.62 / 3.42 |
| fast_compact | tiles T64/WG8 | compact | 11.8 |  /  | 64.9 |  /  | 48.1 | 113.1 | 6.4 | 131.4 [131.4–132.2] | 2.66 / 3.42 |
| det_padded | private_t32 T32/WG1 | padded | 12.8 | 11.3 / 1.2 | 65.7 | 65.6 / 0.1 | 45.6 | 111.5 | 6.4 | 130.7 [130.2–131.4] | 3.60 / 4.42 |
| det_compact | private_t32 T32/WG1 | compact | 12.8 | 11.3 / 1.2 | 65.5 | 65.4 / 0.1 | 46.9 | 112.6 | 6.4 | 131.8 [131.6–132.4] | 3.60 / 4.42 |
* paired det_padded − fast_padded step: +0.80 ms [+0.14, +1.14] (+0.61 % [+0.11, +0.88])
* paired det_padded − fast_padded dw_kernel: -1.21 ms [-1.39, -0.97] (-2.58 % [-2.98, -2.09])
* paired det_padded − fast_padded fwd_total: +1.00 ms [+0.91, +1.08] (+8.49 % [+7.59, +9.15])
* paired det_padded − fast_padded dx_kernel: +1.01 ms [+0.42, +1.08] (+1.55 % [+0.65, +1.67])
* paired det_compact − fast_compact step: +0.22 ms [+0.18, +0.51] (+0.17 % [+0.14, +0.38])
* paired det_compact − fast_compact dw_kernel: -1.19 ms [-1.32, -1.02] (-2.47 % [-2.74, -2.12])
* paired det_compact − fast_compact fwd_total: +1.01 ms [+0.93, +1.05] (+8.50 % [+7.85, +8.89])
* paired det_compact − fast_compact dx_kernel: +0.48 ms [+0.34, +0.75] (+0.74 % [+0.52, +1.15])
* paired fast_compact − fast_padded step: +1.56 ms [+1.33, +2.17] (+1.20 % [+1.02, +1.67])
* paired fast_compact − fast_padded dw_kernel: +1.35 ms [+1.27, +1.55] (+2.89 % [+2.71, +3.30])
* paired fast_compact − fast_padded fwd_total: +0.00 ms [-0.11, +0.15] (+0.00 % [-0.89, +1.24])
* paired fast_compact − fast_padded dx_kernel: +0.17 ms [+0.04, +0.35] (+0.26 % [+0.06, +0.54])
* paired det_compact − det_padded step: +1.24 ms [+0.37, +2.01] (+0.95 % [+0.28, +1.54])
* paired det_compact − det_padded dw_kernel: +1.45 ms [+1.00, +1.70] (+3.17 % [+2.18, +3.73])
* paired det_compact − det_padded fwd_total: +0.01 ms [-0.08, +0.11] (+0.06 % [-0.65, +0.89])
* paired det_compact − det_padded dx_kernel: -0.20 ms [-0.60, +0.23] (-0.30 % [-0.91, +0.36])

**H30000000_M8192_G512_S64_R8** (C 8, S 64, R 8, H 30,000,000, M 8192, G 512, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | private_t32 T32/WG1 | padded | 90.9 |  /  | 492.0 |  /  | 337.8 | 830.2 | 6.4 | 927.5 [926.8–927.9] | 2.74 / 3.42 |
| fast_compact | private_t32 T32/WG1 | compact | 91.4 |  /  | 493.5 |  /  | 349.3 | 843.6 | 6.4 | 941.3 [941.0–941.9] | 2.76 / 3.42 |
| det_padded | private_t32 T32/WG1 | padded | 93.0 | 90.0 / 2.5 | 491.5 | 490.8 / 0.7 | 336.3 | 829.0 | 6.4 | 928.3 [927.4–929.0] | 3.63 / 4.42 |
| det_compact | private_t32 T32/WG1 | compact | 93.2 | 90.3 / 2.5 | 491.2 | 490.5 / 0.7 | 351.9 | 843.7 | 6.4 | 943.6 [941.5–945.4] | 3.63 / 4.42 |
* paired det_padded − fast_padded step: +1.00 ms [-0.06, +1.14] (+0.11 % [-0.01, +0.12])
* paired det_padded − fast_padded dw_kernel: -0.66 ms [-1.50, -0.40] (-0.20 % [-0.45, -0.12])
* paired det_padded − fast_padded fwd_total: +2.20 ms [+1.22, +2.25] (+2.43 % [+1.33, +2.48])
* paired det_padded − fast_padded dx_kernel: -0.92 ms [-2.40, +2.51] (-0.19 % [-0.49, +0.51])
* paired det_compact − fast_compact step: +1.58 ms [-0.44, +3.91] (+0.17 % [-0.05, +0.42])
* paired det_compact − fast_compact dw_kernel: +2.51 ms [+1.94, +2.89] (+0.72 % [+0.56, +0.83])
* paired det_compact − fast_compact fwd_total: +1.86 ms [+1.49, +2.02] (+2.04 % [+1.63, +2.21])
* paired det_compact − fast_compact dx_kernel: -2.17 ms [-3.73, -0.72] (-0.44 % [-0.76, -0.15])
* paired fast_compact − fast_padded step: +14.36 ms [+13.14, +14.66] (+1.55 % [+1.42, +1.58])
* paired fast_compact − fast_padded dw_kernel: +11.80 ms [+11.41, +12.30] (+3.50 % [+3.38, +3.65])
* paired fast_compact − fast_padded fwd_total: +0.49 ms [-0.18, +0.80] (+0.54 % [-0.19, +0.89])
* paired fast_compact − fast_padded dx_kernel: +1.60 ms [+0.82, +3.28] (+0.33 % [+0.17, +0.67])
* paired det_compact − det_padded step: +14.31 ms [+13.88, +17.58] (+1.54 % [+1.49, +1.89])
* paired det_compact − det_padded dw_kernel: +15.04 ms [+14.33, +15.99] (+4.47 % [+4.25, +4.75])
* paired det_compact − det_padded fwd_total: +0.29 ms [-0.12, +0.50] (+0.31 % [-0.13, +0.53])
* paired det_compact − det_padded dx_kernel: -0.14 ms [-0.66, +1.17] (-0.03 % [-0.13, +0.24])

**H30000000_M8192_S16_R4** (C 8, S 16, R 4, H 30,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG1 | padded | 94.7 |  /  | 327.2 |  /  | 331.1 | 659.7 | 6.4 | 761.9 [755.9–763.5] | 2.84 / 3.42 |
| fast_compact | tiles T64/WG1 | compact | 94.7 |  /  | 326.8 |  /  | 344.3 | 672.8 | 6.4 | 774.9 [770.4–776.1] | 2.88 / 3.42 |
| det_padded | tiles T64/WG1 | padded | 104.0 | 93.4 / 10.0 | 327.1 | 326.8 / 0.2 | 331.0 | 659.9 | 6.4 | 770.4 [766.5–772.6] | 3.65 / 4.42 |
| det_compact | tiles T64/WG1 | compact | 103.9 | 93.4 / 10.0 | 326.7 | 326.5 / 0.2 | 344.1 | 673.1 | 6.4 | 783.5 [782.3–784.1] | 3.65 / 4.42 |
* paired det_padded − fast_padded step: +8.55 ms [+7.68, +10.61] (+1.12 % [+1.01, +1.40])
* paired det_padded − fast_padded dw_kernel: +0.12 ms [-1.65, +0.59] (+0.04 % [-0.50, +0.18])
* paired det_padded − fast_padded fwd_total: +9.49 ms [+9.14, +9.82] (+10.06 % [+9.63, +10.36])
* paired det_padded − fast_padded dx_kernel: -0.17 ms [-0.86, +0.73] (-0.05 % [-0.26, +0.22])
* paired det_compact − fast_compact step: +8.55 ms [+6.17, +12.75] (+1.10 % [+0.79, +1.66])
* paired det_compact − fast_compact dw_kernel: -0.31 ms [-1.02, +2.82] (-0.09 % [-0.30, +0.83])
* paired det_compact − fast_compact fwd_total: +9.27 ms [+8.96, +9.62] (+9.79 % [+9.44, +10.21])
* paired det_compact − fast_compact dx_kernel: +0.33 ms [+0.11, +0.51] (+0.10 % [+0.03, +0.16])
* paired fast_compact − fast_padded step: +13.31 ms [+11.91, +14.51] (+1.75 % [+1.56, +1.92])
* paired fast_compact − fast_padded dw_kernel: +13.45 ms [+12.05, +15.04] (+4.08 % [+3.62, +4.54])
* paired fast_compact − fast_padded fwd_total: -0.12 ms [-0.52, +1.08] (-0.13 % [-0.55, +1.15])
* paired fast_compact − fast_padded dx_kernel: -0.49 ms [-0.62, +0.43] (-0.15 % [-0.19, +0.13])
* paired det_compact − det_padded step: +13.04 ms [+10.96, +16.65] (+1.69 % [+1.42, +2.17])
* paired det_compact − det_padded dw_kernel: +14.02 ms [+12.94, +15.03] (+4.24 % [+3.91, +4.56])
* paired det_compact − det_padded fwd_total: -0.20 ms [-0.65, +0.54] (-0.19 % [-0.62, +0.52])
* paired det_compact − det_padded dx_kernel: -0.01 ms [-0.32, +0.90] (-0.00 % [-0.10, +0.27])

**H30000000_M8192_S32_R8** (C 8, S 32, R 8, H 30,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T32/WG1 | padded | 96.3 |  /  | 387.8 |  /  | 341.3 | 730.8 | 6.4 | 833.6 [833.1–834.7] | 2.84 / 3.42 |
| fast_compact | tiles T32/WG1 | compact | 96.5 |  /  | 387.7 |  /  | 351.4 | 740.9 | 6.4 | 844.7 [843.7–845.2] | 2.88 / 3.42 |
| det_padded | tiles T32/WG1 | padded | 104.9 | 94.4 / 10.0 | 386.9 | 386.5 / 0.4 | 341.2 | 731.1 | 6.4 | 842.1 [840.8–844.1] | 3.65 / 4.42 |
| det_compact | tiles T32/WG1 | compact | 105.2 | 94.7 / 10.0 | 387.5 | 387.1 / 0.4 | 351.4 | 741.3 | 6.4 | 853.1 [852.5–854.3] | 3.65 / 4.42 |
* paired det_padded − fast_padded step: +7.62 ms [+6.17, +10.86] (+0.91 % [+0.74, +1.30])
* paired det_padded − fast_padded dw_kernel: -0.20 ms [-0.96, +0.55] (-0.06 % [-0.28, +0.16])
* paired det_padded − fast_padded fwd_total: +8.57 ms [+8.44, +9.11] (+8.88 % [+8.74, +9.47])
* paired det_padded − fast_padded dx_kernel: -0.45 ms [-2.03, +0.85] (-0.12 % [-0.52, +0.22])
* paired det_compact − fast_compact step: +9.03 ms [+7.25, +10.23] (+1.07 % [+0.86, +1.21])
* paired det_compact − fast_compact dw_kernel: +0.55 ms [+0.09, +1.00] (+0.16 % [+0.03, +0.28])
* paired det_compact − fast_compact fwd_total: +8.84 ms [+8.64, +9.18] (+9.16 % [+8.96, +9.48])
* paired det_compact − fast_compact dx_kernel: +0.31 ms [-0.24, +0.35] (+0.08 % [-0.06, +0.09])
* paired fast_compact − fast_padded step: +10.12 ms [+9.37, +12.00] (+1.21 % [+1.12, +1.44])
* paired fast_compact − fast_padded dw_kernel: +10.14 ms [+9.05, +11.00] (+2.97 % [+2.64, +3.22])
* paired fast_compact − fast_padded fwd_total: +0.11 ms [-0.22, +0.58] (+0.12 % [-0.23, +0.61])
* paired fast_compact − fast_padded dx_kernel: -0.25 ms [-1.87, +0.45] (-0.06 % [-0.48, +0.12])
* paired det_compact − det_padded step: +12.00 ms [+8.39, +12.48] (+1.43 % [+0.99, +1.48])
* paired det_compact − det_padded dw_kernel: +10.98 ms [+10.09, +11.27] (+3.22 % [+2.96, +3.30])
* paired det_compact − det_padded fwd_total: +0.22 ms [-0.14, +1.12] (+0.21 % [-0.13, +1.06])
* paired det_compact − det_padded dx_kernel: +0.36 ms [-0.62, +0.81] (+0.09 % [-0.16, +0.21])

**H30000000_M8192_S64_R16_bf16** (C 8, S 64, R 16, H 30,000,000, M 8192, G 1024, bf16; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | private_t32 T32/WG1 | padded | 99.3 |  /  | 471.8 |  /  | 324.9 | 798.5 | 6.4 | 904.5 [904.2–904.7] | 2.84 / 3.42 |
| fast_compact | private_t32 T32/WG1 | compact | 100.5 |  /  | 471.9 |  /  | 347.4 | 822.4 | 6.4 | 929.3 [928.0–930.4] | 2.88 / 3.42 |
| det_padded | private_t32 T32/WG1 | padded | 108.1 | 97.6 / 10.0 | 471.2 | 470.5 / 0.7 | 325.3 | 798.1 | 6.4 | 912.5 [912.0–914.3] | 3.65 / 4.42 |
| det_compact | private_t32 T32/WG1 | compact | 108.9 | 98.4 / 10.0 | 472.5 | 471.7 / 0.7 | 346.7 | 820.8 | 6.4 | 937.5 [936.1–937.7] | 3.65 / 4.42 |
* paired det_padded − fast_padded step: +8.43 ms [+7.40, +10.09] (+0.93 % [+0.82, +1.12])
* paired det_padded − fast_padded dw_kernel: +0.56 ms [+0.25, +1.86] (+0.17 % [+0.08, +0.57])
* paired det_padded − fast_padded fwd_total: +8.69 ms [+8.52, +9.14] (+8.74 % [+8.58, +9.21])
* paired det_padded − fast_padded dx_kernel: -0.68 ms [-1.82, +0.72] (-0.14 % [-0.39, +0.15])
* paired det_compact − fast_compact step: +8.47 ms [+5.67, +9.67] (+0.91 % [+0.61, +1.04])
* paired det_compact − fast_compact dw_kernel: -1.52 ms [-3.57, +1.72] (-0.44 % [-1.02, +0.50])
* paired det_compact − fast_compact fwd_total: +8.57 ms [+7.94, +9.56] (+8.54 % [+7.87, +9.52])
* paired det_compact − fast_compact dx_kernel: +1.07 ms [-3.94, +2.73] (+0.23 % [-0.83, +0.58])
* paired fast_compact − fast_padded step: +24.38 ms [+23.38, +26.25] (+2.70 % [+2.58, +2.90])
* paired fast_compact − fast_padded dw_kernel: +23.21 ms [+20.85, +25.04] (+7.15 % [+6.41, +7.70])
* paired fast_compact − fast_padded fwd_total: +1.07 ms [+0.94, +1.71] (+1.08 % [+0.94, +1.72])
* paired fast_compact − fast_padded dx_kernel: +0.26 ms [-0.81, +4.81] (+0.05 % [-0.17, +1.02])
* paired det_compact − det_padded step: +24.11 ms [+22.85, +25.26] (+2.64 % [+2.50, +2.77])
* paired det_compact − det_padded dw_kernel: +21.22 ms [+19.93, +21.83] (+6.52 % [+6.10, +6.69])
* paired det_compact − det_padded fwd_total: +1.01 ms [+0.51, +1.86] (+0.94 % [+0.47, +1.72])
* paired det_compact − det_padded dx_kernel: +1.88 ms [+1.10, +3.05] (+0.40 % [+0.23, +0.65])

**H30000000_M8192_S64_R16_caps** (C 8, S 64, R 16, H 30,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| cap256M | private_t32 T32/WG1 | padded | 113.3 | 102.7 / 10.0 | 514.9 | 514.1 / 0.7 | 348.4 | 865.5 | 6.4 | 985.4 [983.8–986.9] | 3.09 / 3.67 |
| cap1G | private_t32 T32/WG1 | padded | 109.5 | 98.9 / 10.0 | 515.4 | 514.7 / 0.7 | 348.4 | 866.0 | 6.4 | 981.8 [980.5–982.7] | 3.65 / 4.67 |
| cap4G | private_t32 T32/WG1 | padded | 109.7 | 99.3 / 9.8 | 514.7 | 514.0 / 0.7 | 349.1 | 865.4 | 6.4 | 981.8 [981.2–983.4] | 6.65 / 8.67 |
| fast | tiles T64/WG8 | padded | 99.9 |  /  | 521.7 |  /  | 353.3 | 877.0 | 6.4 | 983.9 [982.0–985.1] | 2.88 / 8.67 |
* paired cap256M − cap1G step: +3.92 ms [+1.14, +4.53] (+0.40 % [+0.12, +0.46])
* paired cap256M − cap1G dw_kernel: -0.09 ms [-0.55, +0.01] (-0.03 % [-0.16, +0.00])
* paired cap256M − cap1G fwd_total: +3.71 ms [+2.76, +4.31] (+3.39 % [+2.49, +3.93])
* paired cap256M − cap1G dx_kernel: -0.67 ms [-1.58, +2.05] (-0.13 % [-0.31, +0.40])
* paired cap4G − cap1G step: -0.44 ms [-1.18, +2.90] (-0.05 % [-0.12, +0.30])
* paired cap4G − cap1G dw_kernel: +0.31 ms [-2.11, +2.03] (+0.09 % [-0.60, +0.58])
* paired cap4G − cap1G fwd_total: +0.31 ms [-1.22, +0.82] (+0.28 % [-1.09, +0.75])
* paired cap4G − cap1G dx_kernel: -0.74 ms [-1.42, -0.02] (-0.14 % [-0.28, -0.00])
* paired cap1G − fast step: -1.39 ms [-4.56, +0.10] (-0.14 % [-0.46, +0.01])
* paired cap1G − fast dw_kernel: -5.20 ms [-7.07, -1.23] (-1.47 % [-1.99, -0.35])
* paired cap1G − fast fwd_total: +8.64 ms [+7.44, +11.89] (+8.59 % [+7.28, +11.99])
* paired cap1G − fast dx_kernel: -5.83 ms [-6.97, -5.36] (-1.12 % [-1.34, -1.03])

**H30000000_M8192_S64_R16** (C 8, S 64, R 16, H 30,000,000, M 8192, G 1024, split; commit 09b5bc4)

| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | Step [round range] | Peak alloc / res GiB (warm-up) |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| fast_padded | tiles T64/WG8 | padded | 98.7 |  /  | 515.3 |  /  | 347.7 | 864.7 | 6.4 | 971.3 [970.7–971.5] | 2.84 / 3.42 |
| fast_compact | tiles T64/WG8 | compact | 99.5 |  /  | 515.6 |  /  | 360.7 | 878.2 | 6.4 | 984.4 [983.5–985.1] | 2.88 / 3.42 |
| det_padded | private_t32 T32/WG1 | padded | 108.4 | 97.6 / 10.0 | 508.8 | 508.0 / 0.7 | 344.2 | 855.2 | 6.4 | 969.6 [968.2–971.3] | 3.65 / 4.42 |
| det_compact | private_t32 T32/WG1 | compact | 109.1 | 98.5 / 10.0 | 509.9 | 509.2 / 0.7 | 357.7 | 869.2 | 6.4 | 984.2 [981.4–985.1] | 3.65 / 4.42 |
* paired det_padded − fast_padded step: -1.61 ms [-2.91, -0.23] (-0.17 % [-0.30, -0.02])
* paired det_padded − fast_padded dw_kernel: -2.68 ms [-4.51, -0.92] (-0.77 % [-1.30, -0.26])
* paired det_padded − fast_padded fwd_total: +9.08 ms [+6.84, +10.32] (+9.12 % [+6.77, +10.49])
* paired det_padded − fast_padded dx_kernel: -6.47 ms [-7.52, -5.78] (-1.25 % [-1.46, -1.12])
* paired det_compact − fast_compact step: +0.57 ms [-3.68, +1.58] (+0.06 % [-0.37, +0.16])
* paired det_compact − fast_compact dw_kernel: -3.15 ms [-5.84, -1.50] (-0.87 % [-1.62, -0.41])
* paired det_compact − fast_compact fwd_total: +9.61 ms [+8.14, +10.25] (+9.64 % [+8.05, +10.37])
* paired det_compact − fast_compact dx_kernel: -6.79 ms [-8.09, -2.81] (-1.31 % [-1.56, -0.55])
* paired fast_compact − fast_padded step: +12.70 ms [+12.39, +13.79] (+1.31 % [+1.28, +1.42])
* paired fast_compact − fast_padded dw_kernel: +13.15 ms [+12.90, +13.39] (+3.78 % [+3.71, +3.85])
* paired fast_compact − fast_padded fwd_total: +0.37 ms [-1.65, +1.55] (+0.37 % [-1.64, +1.57])
* paired fast_compact − fast_padded dx_kernel: +0.60 ms [-0.50, +0.86] (+0.12 % [-0.10, +0.17])
* paired det_compact − det_padded step: +14.24 ms [+11.57, +16.93] (+1.47 % [+1.19, +1.75])
* paired det_compact − det_padded dw_kernel: +12.50 ms [+11.59, +13.13] (+3.62 % [+3.38, +3.79])
* paired det_compact − det_padded fwd_total: +0.84 ms [+0.04, +1.20] (+0.78 % [+0.03, +1.11])
* paired det_compact − det_padded dx_kernel: +0.72 ms [-0.57, +2.48] (+0.14 % [-0.11, +0.49])
