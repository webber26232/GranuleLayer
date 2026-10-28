
## Erratum: global-support sharing (graph 0, parent K16 supports)

| support | scope | parent (local key) | legacy key (recomputed) | labelled (q, local) | global distinct | groups / support | neurons / support | mask sha equal | exact pair / triple equal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP-mag | record | 194512 | 194512 | 194512 | 194512 | 4.017 | 64.26 | yes | yes |
| FP-mag | set | 16384 | 16384 | 16384 | 16384 | 47.688 | 762.94 | yes | yes |
| FP-mag | output | 4096 | 4096 | 194512 | 194512 | 4.017 | 64.26 | yes | yes |
| FP-mag | layer | 4 | 4 | 16384 | 16384 | 47.688 | 762.94 | yes | yes |
| BP-row#0 | record | 781312 | 781312 | 781312 | 781312 | 1.000 | 16.00 | yes | yes |
| BP-row#0 | set | 781312 | 781312 | 781312 | 781312 | 1.000 | 16.00 | yes | yes |
| BP-row#0 | output | 781312 | 781312 | 781312 | 781312 | 1.000 | 16.00 | yes | yes |
| BP-row#0 | layer | 781312 | 781312 | 781312 | 781312 | 1.000 | 16.00 | yes | yes |
| U16@0.25 | record | 376518 | 376518 | 376518 | 376518 | 2.075 | 33.20 | yes | yes |
| U16@0.25 | set | 61902 | 61902 | 61902 | 61902 | 12.622 | 201.93 | yes | yes |
| U16@0.25 | output | 16325 | 16325 | 376518 | 328520 | 2.378 | 38.05 | yes | yes |
| U16@0.25 | layer | 16 | 16 | 61902 | 57807 | 13.516 | 216.24 | yes | yes |
| A8@0.25 | record | 609658 | 609658 | 609658 | 609658 | 1.282 | 20.50 | yes | yes |
| A8@0.25 | set | 222148 | 222148 | 222148 | 222148 | 3.517 | 56.27 | yes | yes |
| A8@0.25 | output | 98726 | 98726 | 609658 | 609658 | 1.282 | 20.50 | yes | yes |
| A8@0.25 | layer | 247 | 247 | 222148 | 222148 | 3.517 | 56.27 | yes | yes |


## Erratum: total Q and occupied sets

| model | support | parent Q | total Q | occupied | empty | sets in buckets | support fingerprint unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|
| H 262,144 (g0) | BP-row#0 | 4084 | 4096 | 1024 | 3072 | 1024 | True |
| H 262,144 (g0) | BP-row/8#0 | None | 4096 | 1024 | 3072 | 1024 | n/a |
| H 262,144 (g0) | FP-mag | None | 4096 | 1024 | 3072 | 1024 | n/a |
| H 12.5M (g0) | BP-row#0 | 4096 | 4096 | 4096 | 0 | 4096 | True |
| H 12.5M (g0) | BP-row#1 | 4096 | 4096 | 4096 | 0 | 4096 | True |
| H 12.5M (g0) | BP-row/8#0 | 4096 | 4096 | 4096 | 0 | 4096 | True |
| H 12.5M (g0) | BP-row/8#1 | 4096 | 4096 | 4096 | 0 | 4096 | True |
| H 12.5M (g1) | BP-row#0 | 4096 | 4096 | 4096 | 0 | 4096 | True |
| H 12.5M (g1) | BP-row#1 | 4096 | 4096 | 4096 | 0 | 4096 | True |
| H 12.5M (g1) | BP-row/8#0 | None | 4096 | 4096 | 0 | 4096 | n/a |
| H 12.5M (g1) | BP-row/8#1 | None | 4096 | 4096 | 0 | 4096 | n/a |
| H 12.5M (g2) | BP-row#0 | 4096 | 4096 | 4096 | 0 | 4096 | True |
| H 12.5M (g2) | BP-row#1 | 4096 | 4096 | 4096 | 0 | 4096 | True |
| H 12.5M (g2) | BP-row/8#0 | None | 4096 | 4096 | 0 | 4096 | n/a |
| H 12.5M (g2) | BP-row/8#1 | None | 4096 | 4096 | 0 | 4096 | n/a |


## K8 coverage on the timed supports

| support | graph | layer pairs % | layer triples % (exact) | refined prediction % | layer quads % [95 % CI] | per-output triples % (median of 38) | per-output triples % pooled [95 % CI] | per-output features | distinct supports (output scope, sum) | distinct supports (layer) | structural bias-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8#0 | 0 | 100.000 | 19.540 | 19.540 | 0.1187 [0.1166, 0.1208] | 0.0240 | 0.0240 [0.0237, 0.0244] | 978 | 781312 | 781312 | 0.00 % |
| BP-row/8#1 | 0 | 100.000 | 19.538 | 19.540 | 0.1192 [0.1171, 0.1214] | 0.0237 | 0.0236 [0.0233, 0.0240] | 978 | 781312 | 781312 | 0.00 % |
| BP-row/8#0 | 1 | 100.000 | 19.541 | 19.540 | 0.1172 [0.1151, 0.1194] | 0.0239 | 0.0239 [0.0235, 0.0242] | 976 | 781312 | 781312 | 0.00 % |
| BP-row/8#1 | 1 | 100.000 | 19.537 | 19.540 | 0.1183 [0.1162, 0.1205] | 0.0240 | 0.0240 [0.0236, 0.0243] | 976 | 781312 | 781312 | 0.00 % |
| BP-row/8#0 | 2 | 100.000 | 19.541 | 19.540 | 0.1190 [0.1169, 0.1211] | 0.0240 | 0.0240 [0.0236, 0.0243] | 976 | 781312 | 781312 | 0.00 % |
| BP-row/8#1 | 2 | 100.000 | 19.541 | 19.540 | 0.1190 [0.1169, 0.1211] | 0.0241 | 0.0239 [0.0236, 0.0243] | 976 | 781312 | 781312 | 0.00 % |
| U8@0.125 | 0 | 100.000 | 37.623 |  | 1.3624 [1.3552, 1.3696] | 0.1065 | 0.1049 [0.1042, 0.1057] | 956 | 415335 | 166039 | 34.39 % |
| FP-mag/8 | 0 | 27.077 | 0.881 | 1.023 | 0.0052 [0.0047, 0.0056] | 0.0108 | 0.0107 [0.0104, 0.0109] | 958 | 345835 | 32768 | 0.00 % |
| U8@0.125 | 1 | 100.000 | 37.580 |  | 1.3625 [1.3553, 1.3697] | 0.1061 | 0.1062 [0.1054, 0.1069] | 957 | 415246 | 166087 | 34.39 % |
| FP-mag/8 | 1 | 27.112 | 0.881 | 1.023 | 0.0052 [0.0048, 0.0057] | 0.0105 | 0.0107 [0.0104, 0.0109] | 955 | 345914 | 32768 | 0.00 % |
| U8@0.125 | 2 | 100.000 | 37.580 |  | 1.3592 [1.3520, 1.3664] | 0.1071 | 0.1070 [0.1063, 0.1078] | 955 | 415405 | 166068 | 34.39 % |
| FP-mag/8 | 2 | 27.057 | 0.881 | 1.023 | 0.0052 [0.0047, 0.0056] | 0.0106 | 0.0106 [0.0104, 0.0108] | 954 | 345866 | 32768 | 0.00 % |


## K8 confirmation timing (24 samples per process)

| support | format | step ms per graph (layout-averaged) | graph mean ± 95 % CI | max layout spread | mean within-run range | ratio to U8@0.125 (same session) | fwd | dX | dW / db | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A8@0.25 | wide | 990.5 / 988.1 / 977.2 | 985.3 ± 17.6 | 0.0 | 10.0 | 1.414 ± 0.010 | 124.4 | 453.1 | 401.9 | 2.18 |
| BP-row | indexed | 878.0 / 876.1 / 869.5 | 874.5 ± 11.1 | 0.0 | 10.0 | 1.255 ± 0.008 | 69.7 | 610.1 | 188.4 | 2.19 |
| BP-row/8 | indexed | 634.1 / 634.3 / 627.7 | 632.1 ± 9.4 | 1.0 | 6.9 | 0.907 ± 0.003 | 49.1 | 379.4 | 199.5 | 1.24 |
| FP-mag/8 | indexed | 649.0 / 649.0 / 641.5 | 646.5 ± 10.7 | 0.0 | 6.7 | 0.928 ± 0.004 | 48.9 | 395.4 | 198.1 | 1.25 |
| FP-mag/8 | wide | 824.2 / 821.9 / 824.2 | 823.5 ± 3.3 | 0.0 | 7.2 | 1.182 ± 0.021 | 112.7 | 396.7 | 310.4 | 1.25 |
| U16@0.25 | wide | 825.4 / 823.6 / 814.9 | 821.3 ± 14.0 | 0.0 | 8.8 | 1.179 ± 0.007 | 118.0 | 429.9 | 267.3 | 2.27 |
| U8@0.125 | wide | 698.1 / 700.0 / 691.9 | 696.7 ± 10.5 | 0.0 | 8.3 | 1.000 ± 0.000 | 99.7 | 361.5 | 231.3 | 1.36 |
| dense | dense | 1570.2 / 1570.9 / 1564.1 | 1568.4 ± 9.2 | 17.3 | 18.5 | 2.251 ± 0.021 | 210.4 | 766.3 | 571.6 | 7.80 |


## K8 quality, 32 updates (development, seed 0)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8 | none | 1 | 0.2629 | 0.2629 |  | 0.116 | 162.8 |
| BP-row/8 | energy | 1 | 0.2607 | 0.2607 | 0.001 | 1.000 | 162.6 |
| U8@0.125 | none | 1 | 0.2627 | 0.2627 |  | 0.196 | 151.6 |
| U8@0.125 | energy | 1 | 0.2608 | 0.2608 | 0.000 | 1.000 | 151.6 |
| FP-mag/8 | none | 1 | 0.2626 | 0.2626 |  | 0.138 | 159.3 |
| FP-mag/8 | energy | 1 | 0.2604 | 0.2604 | 0.000 | 1.000 | 159.1 |


## K8 quality, 256 updates (development seeds 0/1/2)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8 | none | 3 | 0.2329 / 0.2332 / 0.2324 | 0.2328 ± 0.0009 |  | 0.116 | 161.2 |
| BP-row/8 | energy | 3 | 0.2240 / 0.2260 / 0.2228 | 0.2243 ± 0.0040 | 0.001 | 1.000 | 162.3 |
| U8@0.125 | none | 3 | 0.2221 / 0.2219 / 0.2212 | 0.2217 ± 0.0011 |  | 0.196 | 149.9 |
| U8@0.125 | energy | 3 | 0.2121 / 0.2138 / 0.2114 | 0.2124 ± 0.0031 | 0.000 | 1.000 | 151.0 |
| FP-mag/8 | none | 3 | 0.2301 / 0.2300 / 0.2297 | 0.2300 ± 0.0006 |  | 0.138 | 157.9 |
| FP-mag/8 | energy | 3 | 0.2198 / 0.2216 / 0.2191 | 0.2201 ± 0.0033 | 0.000 | 1.000 | 158.9 |


## K8 graph / layout checks (selected treatment, data / SR seed 0, 256 updates)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8#0 | energy | 1 | 0.2235 | 0.2235 | 0.001 | 1.002 | 156.4 |
| BP-row/8#1 | energy | 1 | 0.2236 | 0.2236 | 0.001 | 1.002 | 159.1 |
| U8@0.125 | energy | 1 | 0.2112 | 0.2112 | 0.000 | 1.003 | 151.4 |


## 

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8#0 | energy | 1 | 0.2240 | 0.2240 | 0.001 | 1.002 | 160.3 |
| BP-row/8#1 | energy | 1 | 0.2242 | 0.2242 | 0.001 | 1.002 | 157.4 |
| U8@0.125 | energy | 1 | 0.2118 | 0.2118 | 0.000 | 1.000 | 154.8 |


## Weighted K8 screen: coverage (graph 0)

| support | graph | layer pairs % | layer triples % (exact) | refined prediction % | layer quads % [95 % CI] | per-output triples % (median of 38) | per-output triples % pooled [95 % CI] | per-output features | distinct supports (output scope, sum) | distinct supports (layer) | structural bias-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WRS/8:4#0 | 0 | 100.000 | 19.470 |  | 0.1174 [0.1153, 0.1196] | 0.0237 | 0.0236 [0.0232, 0.0239] | 958 | 781312 | 781311 | 0.00 % |
| WRS/8:16#0 | 0 | 100.000 | 19.469 |  | 0.1171 [0.1150, 0.1193] | 0.0235 | 0.0237 [0.0234, 0.0241] | 954 | 781312 | 781312 | 0.00 % |
| TopK/8 | 0 | 100.000 | 19.465 |  | 0.1191 [0.1170, 0.1212] | 0.0239 | 0.0238 [0.0235, 0.0242] | 954 | 781312 | 781312 | 0.00 % |
| BP-row/8#0 | 0 | 100.000 | 19.540 | 19.540 | 0.1187 [0.1166, 0.1208] | 0.0240 | 0.0240 [0.0237, 0.0244] | 978 | 781312 | 781312 | 0.00 % |
| FP-mag/8 | 0 | 27.077 | 0.881 | 1.023 | 0.0052 [0.0047, 0.0056] | 0.0108 | 0.0107 [0.0104, 0.0109] | 958 | 345835 | 32768 | 0.00 % |


## Weighted K8 screen: timing (8 samples, same session)

| support | format | step ms per graph (layout-averaged) | graph mean ± 95 % CI | max layout spread | mean within-run range | ratio to BP-row/8 (same session) | fwd | dX | dW / db | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8 | indexed | 625.3 | 625.3 | 0.0 | 5.8 | 1.000 | 48.7 | 375.8 | 197.0 | 1.24 |
| FP-mag/8 | indexed | 640.1 | 640.1 | 0.0 | 3.5 | 1.024 | 48.4 | 392.1 | 196.0 | 1.25 |
| TopK/8 | indexed | 624.9 | 624.9 | 0.0 | 5.9 | 0.999 | 48.6 | 375.5 | 196.8 | 1.24 |
| WRS/8:16 | indexed | 626.9 | 626.9 | 0.0 | 2.5 | 1.002 | 48.9 | 376.6 | 197.5 | 1.24 |
| WRS/8:4 | indexed | 626.3 | 626.3 | 0.0 | 4.9 | 1.002 | 48.8 | 375.9 | 197.0 | 1.24 |
| dense | dense | 1543.3 | 1543.3 | 2.1 | 9.0 | 2.468 | 204.8 | 754.3 | 564.1 | 7.80 |


## Weighted K8 screen: 32 updates

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| WRS/8:4#0 | none | 1 | 0.2627 | 0.2627 |  | 0.131 | 160.4 |
| WRS/8:4#0 | energy | 1 | 0.2604 | 0.2604 | 0.001 | 1.001 | 160.2 |
| WRS/8:16#0 | none | 1 | 0.2625 | 0.2625 |  | 0.143 | 161.8 |
| WRS/8:16#0 | energy | 1 | 0.2603 | 0.2603 | 0.001 | 1.001 | 161.6 |
| TopK/8 | none | 1 | 0.2619 | 0.2619 |  | 0.195 | 159.5 |
| TopK/8 | energy | 1 | 0.2598 | 0.2598 | 0.000 | 0.999 | 159.3 |


## Weighted K8 development: 256 updates

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TopK/8 | none | 3 | 0.2230 / 0.2236 / 0.2225 | 0.2231 ± 0.0014 |  | 0.195 | 158.6 |
| TopK/8 | energy | 3 | 0.2132 / 0.2139 / 0.2117 | 0.2130 ± 0.0027 | 0.000 | 0.999 | 159.1 |


## 200M-slot 2 x 2: coverage (graph 0)

| support | graph | layer pairs % | layer triples % (exact) | refined prediction % | layer quads % [95 % CI] | per-output triples % (median of 38) | per-output triples % pooled [95 % CI] | per-output features | distinct supports (output scope, sum) | distinct supports (layer) | structural bias-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TopK/16 | 0 | 100.000 | 58.745 |  | 2.3870 [2.3775, 2.3964] | 0.2169 | 0.2169 [0.2158, 0.2179] | 976 | 781312 | 781312 | 0.00 % |
| Mix-BP#0 | 0 | 100.000 | 61.636 |  | 5.5579 [5.5437, 5.5721] | 1.1163 | 1.1147 [1.1124, 1.1171] | 978 | 781312 | 736256 | 0.00 % |
| Mix-TopK#0 | 0 | 100.000 | 61.636 |  | 5.5579 [5.5437, 5.5721] | 1.1163 | 1.1147 [1.1124, 1.1171] | 978 | 781312 | 736256 | 0.00 % |
| BP-row#0 | 0 | 100.000 | 58.900 | 58.908 | 2.3912 [2.3818, 2.4007] | 0.2203 | 0.2208 [0.2197, 0.2219] | 978 | 781312 | 781312 | 0.00 % |
| A8@0.25 | 0 | 100.000 | 61.636 |  | 5.5579 [5.5437, 5.5721] | 1.1163 | 1.1147 [1.1124, 1.1171] | 978 | 609658 | 222148 | 0.00 % |
| U16@0.25 | 0 | 100.000 | 60.903 |  | 5.1395 [5.1258, 5.1532] | 0.5440 | 0.5487 [0.5471, 0.5504] | 976 | 328520 | 57807 | 31.69 % |


## 200M-slot 2 x 2: timing (8 samples, same session)

| support | format | step ms per graph (layout-averaged) | graph mean ± 95 % CI | max layout spread | mean within-run range | ratio to BP-row (same session) | fwd | dX | dW / db | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A8@0.25 | wide | 974.8 | 974.8 | 0.0 | 4.2 | 1.128 | 122.8 | 447.7 | 397.9 | 2.18 |
| BP-row | indexed | 864.1 | 864.1 | 0.0 | 10.3 | 1.000 | 69.0 | 602.8 | 186.2 | 2.19 |
| Mix-BP | grouped | 1076.2 | 1076.2 | 0.0 | 5.9 | 1.245 | 91.8 | 680.9 | 296.5 | 2.21 |
| Mix-TopK | grouped | 1074.6 | 1074.6 | 0.0 | 3.3 | 1.244 | 91.6 | 680.1 | 296.6 | 2.21 |
| TopK/16 | indexed | 862.2 | 862.2 | 0.0 | 8.9 | 0.998 | 68.3 | 602.3 | 186.1 | 2.19 |
| U16@0.25 | wide | 812.9 | 812.9 | 0.0 | 7.5 | 0.941 | 116.2 | 425.8 | 265.0 | 2.27 |
| dense | dense | 1544.0 | 1544.0 | 5.4 | 10.5 | 1.787 | 205.1 | 754.0 | 564.1 | 7.80 |


## 200M-slot 2 x 2: 32 updates

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TopK/16 | none | 1 | 0.2599 | 0.2599 |  | 0.361 | 160.6 |
| TopK/16 | energy | 1 | 0.2580 | 0.2580 | 0.000 | 1.000 | 160.6 |
| Mix-BP#0 | none | 1 | 0.2611 | 0.2611 |  | 0.202 | 158.7 |
| Mix-BP#0 | energy | 1 | 0.2592 | 0.2592 | 0.001 | 1.000 | 158.6 |
| Mix-TopK#0 | none | 1 | 0.2601 | 0.2601 |  | 0.309 | 161.2 |
| Mix-TopK#0 | energy | 1 | 0.2583 | 0.2583 | 0.000 | 1.000 | 161.1 |
| BP-row#0 | none | 1 | 0.2611 | 0.2611 |  | 0.244 | 162.7 |
| BP-row#0 | energy | 1 | 0.2590 | 0.2590 | 0.000 | 0.999 | 162.6 |
| A8@0.25 | none | 1 | 0.2608 | 0.2608 |  | 0.210 | 157.3 |
| A8@0.25 | energy | 1 | 0.2589 | 0.2589 | 0.000 | 1.000 | 157.2 |
| U16@0.25 | none | 1 | 0.2608 | 0.2608 |  | 0.351 | 156.7 |
| U16@0.25 | energy | 1 | 0.2592 | 0.2592 | 0.000 | 0.999 | 156.6 |
| dense | none | 1 | 0.2502 | 0.2502 |  | 1.000 | 161.3 |


## 200M-slot development: 256 updates

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TopK/16 | none | 3 | 0.1733 / 0.1748 / 0.1728 | 0.1737 ± 0.0025 |  | 0.361 | 159.0 |
| TopK/16 | energy | 3 | 0.1729 / 0.1715 / 0.1695 | 0.1713 ± 0.0042 | 0.000 | 1.000 | 159.9 |
| Mix-TopK#0 | none | 3 | 0.1741 / 0.1755 / 0.1741 | 0.1746 ± 0.0020 |  | 0.309 | 159.8 |
| Mix-TopK#0 | energy | 3 | 0.1752 / 0.1741 / 0.1723 | 0.1739 ± 0.0037 | 0.000 | 1.000 | 160.4 |


## 200M-slot controls: 256 updates (parent records, reused)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP-row | none | 3 | 0.1819 / 0.1830 / 0.1821 | 0.1823 ± 0.0015 |  | 0.244 | 160.7 |
| BP-row | energy | 3 | 0.1796 / 0.1797 / 0.1782 | 0.1791 ± 0.0021 | 0.000 | 0.999 | 162.1 |
| A8@0.25 | none | 3 | 0.1774 / 0.1786 / 0.1773 | 0.1778 ± 0.0018 |  | 0.210 | 155.3 |
| A8@0.25 | energy | 3 | 0.1792 / 0.1788 / 0.1776 | 0.1785 ± 0.0020 | 0.000 | 1.000 | 156.4 |
| U16@0.25 | none | 3 | 0.1745 / 0.1749 / 0.1743 | 0.1746 ± 0.0007 |  | 0.351 | 154.4 |
| U16@0.25 | energy | 3 | 0.1686 / 0.1697 / 0.1677 | 0.1687 ± 0.0025 | 0.000 | 0.999 | 155.6 |
| dense | none | 3 | 0.1144 / 0.1140 / 0.1125 | 0.1136 ± 0.0024 |  | 1.000 | 159.5 |


## 200M-slot confirmation timing, graph 0 (24 samples)

| support | format | step ms per graph (layout-averaged) | graph mean ± 95 % CI | max layout spread | mean within-run range | ratio to BP-row (same session) | fwd | dX | dW / db | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A8@0.25 | wide | 969.6 | 969.6 | 0.0 | 14.0 | 1.122 | 122.5 | 445.5 | 395.0 | 2.18 |
| BP-row | indexed | 864.5 | 864.5 | 0.0 | 4.1 | 1.000 | 69.3 | 603.0 | 185.8 | 2.19 |
| Mix-TopK | grouped | 1063.9 | 1063.9 | 0.0 | 12.8 | 1.231 | 91.2 | 672.7 | 294.0 | 2.21 |
| TopK/16 | indexed | 864.8 | 864.8 | 0.0 | 9.6 | 1.000 | 69.2 | 603.4 | 185.9 | 2.19 |
| U16@0.25 | wide | 817.0 | 817.0 | 0.0 | 9.6 | 0.945 | 117.6 | 427.1 | 265.2 | 2.27 |
| dense | dense | 1550.2 | 1550.2 | 15.2 | 15.9 | 1.793 | 208.1 | 757.3 | 564.6 | 7.80 |


## TopK/8 graph / layout timing confirmation (24 samples)

| support | format | step ms per graph (layout-averaged) | graph mean ± 95 % CI | max layout spread | mean within-run range | ratio to U8@0.125 (same session) | fwd | dX | dW / db | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8 | indexed | 627.7 / 626.4 / 624.0 | 626.0 ± 4.7 | 3.9 | 5.2 | 0.906 ± 0.007 | 48.8 | 376.0 | 197.3 | 1.24 |
| TopK/8 | indexed | 628.1 / 631.9 / 627.3 | 629.1 ± 6.1 | 0.0 | 4.9 | 0.910 ± 0.011 | 48.9 | 377.9 | 198.4 | 1.24 |
| U8@0.125 | wide | 691.7 / 690.3 / 691.4 | 691.1 ± 1.9 | 0.0 | 6.0 | 1.000 ± 0.000 | 99.1 | 358.8 | 229.4 | 1.36 |
| dense | dense | 1556.6 / 1550.8 / 1551.6 | 1553.0 ± 7.8 | 12.5 | 16.4 | 2.247 ± 0.008 | 208.3 | 758.7 | 566.3 | 7.80 |


## Mix-TopK graph / layout timing confirmation (24 samples)

| support | format | step ms per graph (layout-averaged) | graph mean ± 95 % CI | max layout spread | mean within-run range | ratio to BP-row (same session) | fwd | dX | dW / db | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A8@0.25 | wide | 977.1 / 979.7 / 976.1 | 977.6 ± 4.6 | 0.0 | 7.1 | 1.129 ± 0.021 | 123.8 | 449.5 | 398.1 | 2.18 |
| BP-row | indexed | 858.4 / 868.8 / 870.0 | 865.7 ± 15.8 | 0.0 | 7.4 | 1.000 ± 0.000 | 69.2 | 604.0 | 186.5 | 2.19 |
| Mix-TopK | grouped | 1073.8 / 1076.6 / 1075.6 | 1075.3 ± 3.5 | 0.9 | 8.4 | 1.242 ± 0.019 | 91.8 | 680.2 | 297.1 | 2.21 |
| U16@0.25 | wide | 815.0 / 808.7 / 817.4 | 813.7 ± 11.2 | 0.0 | 9.1 | 0.940 ± 0.023 | 117.1 | 425.8 | 264.4 | 2.27 |
| dense | dense | 1553.1 / 1555.2 / 1550.4 | 1552.9 ± 6.0 | 12.4 | 15.2 | 1.794 ± 0.035 | 208.5 | 758.5 | 566.0 | 7.80 |


## TopK/8 coverage, graphs 1 / 2

| support | graph | layer pairs % | layer triples % (exact) | refined prediction % | layer quads % [95 % CI] | per-output triples % (median of 38) | per-output triples % pooled [95 % CI] | per-output features | distinct supports (output scope, sum) | distinct supports (layer) | structural bias-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TopK/8 | 1 | 100.000 | 19.470 |  | 0.1187 [0.1166, 0.1209] | 0.0236 | 0.0235 [0.0231, 0.0238] | 955 | 781312 | 781312 | 0.00 % |
| TopK/8 | 2 | 100.000 | 19.465 |  | 0.1190 [0.1168, 0.1211] | 0.0238 | 0.0237 [0.0234, 0.0241] | 954 | 781312 | 781312 | 0.00 % |


## Mix-TopK coverage, graph / layout checks

| support | graph | layer pairs % | layer triples % (exact) | refined prediction % | layer quads % [95 % CI] | per-output triples % (median of 38) | per-output triples % pooled [95 % CI] | per-output features | distinct supports (output scope, sum) | distinct supports (layer) | structural bias-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mix-TopK#1 | 0 | 100.000 | 61.636 |  | 5.5579 [5.5437, 5.5721] | 1.1163 | 1.1147 [1.1124, 1.1171] | 978 | 781312 | 736256 | 0.00 % |
| Mix-TopK#0 | 1 | 100.000 | 61.650 |  | 5.5475 [5.5334, 5.5617] | 1.1142 | 1.1148 [1.1125, 1.1172] | 976 | 781312 | 736256 | 0.00 % |
| Mix-TopK#1 | 1 | 100.000 | 61.650 |  | 5.5475 [5.5334, 5.5617] | 1.1142 | 1.1148 [1.1125, 1.1172] | 976 | 781312 | 736256 | 0.00 % |
| Mix-TopK#0 | 2 | 100.000 | 61.641 |  | 5.5532 [5.5390, 5.5674] | 1.1162 | 1.1161 [1.1137, 1.1184] | 976 | 781312 | 736256 | 0.00 % |
| Mix-TopK#1 | 2 | 100.000 | 61.641 |  | 5.5532 [5.5390, 5.5674] | 1.1162 | 1.1161 [1.1137, 1.1184] | 976 | 781312 | 736256 | 0.00 % |


## TopK/8 graph 1 (data / SR seed 0)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TopK/8 | energy | 1 | 0.2124 | 0.2124 | 0.000 | 1.001 | 158.9 |


## TopK/8 graph 2 (data / SR seed 0)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TopK/8 | energy | 1 | 0.2134 | 0.2134 | 0.000 | 1.000 | 162.3 |


## Mix-TopK graph 0 layouts (data / SR seed 0)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mix-TopK#1 | energy | 1 | 0.1750 | 0.1750 | 0.000 | 1.000 | 160.3 |


## Mix-TopK graph 1 layouts (data / SR seed 0)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mix-TopK#0 | energy | 1 | 0.1745 | 0.1745 | 0.000 | 1.000 | 158.7 |
| Mix-TopK#1 | energy | 1 | 0.1746 | 0.1746 | 0.000 | 1.000 | 157.3 |


## Mix-TopK graph 2 layouts (data / SR seed 0)

| support | treatment | seeds | held-out per seed | mean ± 95 % CI | clipped (connected) | rel. variance (start) | PR (after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mix-TopK#0 | energy | 1 | 0.1757 | 0.1757 | 0.000 | 1.000 | 161.5 |
| Mix-TopK#1 | energy | 1 | 0.1754 | 0.1754 | 0.000 | 1.000 | 159.8 |


## Locked final test (batches 19,000-19,007; data / SR seeds 3 / 4 / 5; graph 0 / layout 0)

| method | treatment | final test per seed (3 / 4 / 5) | final test mean ± 95 % CI | dev validation (same runs) mean ± CI |
|---|---:|---:|---:|---:|
| TopK/8 | energy | 0.2111 / 0.2125 / 0.2105 | 0.2114 ± 0.0025 | 0.2125 ± 0.0029 |
| BP-row/8#0 | energy | 0.2226 / 0.2243 / 0.2237 | 0.2235 ± 0.0022 | 0.2247 ± 0.0020 |
| U8@0.125 | energy | 0.2115 / 0.2117 / 0.2123 | 0.2118 ± 0.0010 | 0.2130 ± 0.0014 |
| Mix-TopK#0 | energy | 0.1722 / 0.1729 / 0.1721 | 0.1724 ± 0.0011 | 0.1731 ± 0.0014 |
| BP-row#0 | energy | 0.1775 / 0.1787 / 0.1772 | 0.1778 ± 0.0020 | 0.1784 ± 0.0022 |
| A8@0.25 | none | 0.1769 / 0.1770 / 0.1777 | 0.1772 ± 0.0011 | 0.1778 ± 0.0016 |
| U16@0.25 | energy | 0.1678 / 0.1679 / 0.1681 | 0.1679 ± 0.0004 | 0.1683 ± 0.0003 |
| dense | none | 0.1132 / 0.1129 / 0.1129 | 0.1130 ± 0.0005 | 0.1131 ± 0.0005 |

| comparison | per seed | mean | paired 95 % CI | verdict |
|---|---:|---:|---:|---:|
| TopK/8 - BP-row/8#0 | -0.0114 / -0.0118 / -0.0131 | -0.0121 | [-0.0143, -0.0099] | improvement |
| TopK/8 - U8@0.125 | -0.0003 / +0.0008 / -0.0017 | -0.0004 | [-0.0035, +0.0027] | noninferior (margin 0.003) |
| Mix-TopK#0 - A8@0.25 | -0.0046 / -0.0041 / -0.0056 | -0.0048 | [-0.0067, -0.0029] | improvement |
| Mix-TopK#0 - U16@0.25 | +0.0044 / +0.0050 / +0.0040 | +0.0045 | [+0.0032, +0.0058] | worse (whole interval above 0; not noninferior) |
| Mix-TopK#0 - BP-row#0 | -0.0052 / -0.0058 / -0.0051 | -0.0054 | [-0.0063, -0.0045] | improvement |


## Retained pre-pruning weight energy and selection cost (graph 0)

| support | enabled slots | layer energy fraction | selection s (in fine-tuning) | chosen candidate histogram | duplicate candidates | support fingerprint |
|---|---:|---:|---:|---:|---:|---:|
| BP-row/8 | 100000000 |  | 2.2 |  |  | f7e81dc3fcca02c5 |
| WRS/8:4#0 | 100000000 | 0.1402 | 5.7 | [195196, 195664, 195153, 195299] | {'total': 0, 'groups_with_any': 0} | d855e4e3b14e8617 |
| WRS/8:16#0 | 100000000 | 0.1517 | 17.5 | [48826, 48863, 48581, 48426, 48994, 48901, 49408, 48808, 48592, 48627, 49122, 48928, 48932, 48887, 48611, 48806] | {'total': 1, 'groups_with_any': 1} | ae28601b39bbcb49 |
| TopK/8 | 100000000 | 0.2049 | 3.0 |  |  | 4126edb5230e8e10 |
| BP-row#0 | 200000000 |  | 2.4 |  |  | 56bf2444b7886e55 |
| TopK/16 | 200000000 | 0.3677 | 3.3 |  |  | d402a11e790eb0eb |
| Mix-BP#0 | 200000000 |  | 2.3 |  |  | 2a38f7291b3790b1 |
| Mix-TopK#0 | 200000000 | 0.3460 | 3.3 |  |  | 0eafd24b293a8f95 |


## Phase breakdown (median ms, 24 samples, graph 0)

| run | fwd layout | fwd zero | fwd kernels | fwd finalize | bwd coef | bwd layout | dX init | dX | dW / db | bwd post | optimizer | step | host enqueue | fwd K64 | fwd K16 | fwd K8 | dX K64 | dX K16 | dX K8 | dW K64 | dW K16 | dW K8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mix-TopK_ls0_g0_grouped | 0.1 | 0.0 | 90.7 | 0.2 | 0.3 | 0.1 | 0.0 | 672.7 | 294.0 | 0.1 | 5.0 | 1063.9 | 1.5 | 29.1 | 40.0 | 21.5 | 162.9 | 351.3 | 158.4 | 44.8 | 115.0 | 134.0 |
| TopK_k16_g0_indexed | 0.2 | 0.0 | 68.7 | 0.2 | 0.3 | 0.1 | 0.0 | 603.4 | 185.9 | 0.1 | 5.1 | 864.8 | 1.3 |  |  |  |  |  |  |  |  |  |
| BP-row_ls0_g0_indexed | 0.2 | 0.0 | 68.8 | 0.2 | 0.3 | 0.1 | 0.0 | 603.0 | 185.8 | 0.1 | 5.1 | 864.5 | 1.3 |  |  |  |  |  |  |  |  |  |
| A8_r0.25_g0_wide | 0.2 | 0.0 | 121.9 | 0.3 | 0.3 | 0.1 | 0.0 | 445.5 | 395.0 | 0.2 | 5.1 | 969.6 | 1.5 |  |  |  |  |  |  |  |  |  |
| U16_r0.25_g0_wide | 0.2 | 0.0 | 116.5 | 0.8 | 0.3 | 0.1 | 0.0 | 427.1 | 265.2 | 0.6 | 5.0 | 817.0 | 1.6 |  |  |  |  |  |  |  |  |  |


## Interaction screen (reduced H 262,144 / C256, T16)

| support | init | init seed | layer lr | updates | explained 2 / 3 / 4 (dev held-out) | explained 2 / 3 / 4 (reserved final batches) | terms without eligible neuron (2 / 3 / 4) | oracle nMSE vs BF16-input targets (order 2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8#0 | ordinary | 17 | 0.003 | 1000 | 0.118 / 0.119 / 0.109 |  | 3219 / 4001 / 4088 |  |
| BP-row/8#0 | ordinary | 17 | 0.01 | 1000 | 0.175 / 0.148 / 0.131 |  | 3219 / 4001 / 4088 |  |
| TopK/8 | ordinary | 17 | 0.003 | 1000 | -0.420 / -0.335 / -0.342 |  | 3267 / 4010 / 4087 |  |
| TopK/8 | ordinary | 17 | 0.01 | 1000 | -0.351 / -0.299 / -0.314 |  | 3267 / 4010 / 4087 |  |
| Mix-TopK#0 | ordinary | 17 | 0.003 | 1000 | 0.544 / 0.538 / 0.529 |  | 0 / 0 / 0 |  |
| Mix-TopK#0 | ordinary | 17 | 0.01 | 1000 | 0.775 / 0.722 / 0.710 |  | 0 / 0 / 0 |  |
| A8@0.25 | ordinary | 17 | 0.003 | 1000 | 0.306 / 0.421 / 0.452 |  | 0 / 0 / 0 |  |
| A8@0.25 | ordinary | 17 | 0.01 | 1000 | 0.656 / 0.678 / 0.682 |  | 0 / 0 / 0 |  |
| dense | ordinary | 17 | 0.003 | 1000 | 0.782 / 0.852 / 0.876 |  | 0 / 0 / 0 |  |
| dense | ordinary | 17 | 0.01 | 1000 | 0.987 / 0.990 / 0.991 |  | 0 / 0 / 0 |  |
| Mix-TopK#0 | anchor | 17 | 0.003 | 1000 | 0.999 / 0.999 / 1.000 |  | 0 / 0 / 0 | 2.75e-14 |
| Mix-TopK#0 | anchor | 17 | 0.003 | 1000 | 1.000 / 1.000 / 1.000 |  | 0 / 0 / 0 | 2.80e+00 |


## Interaction: extra seeds / continuations (beyond the predeclared rule)

| support | init | init seed | layer lr | updates | explained 2 / 3 / 4 (dev held-out) | explained 2 / 3 / 4 (reserved final batches) | terms without eligible neuron (2 / 3 / 4) | oracle nMSE vs BF16-input targets (order 2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| int_TopK_k8_T16_none_i17_s4000_lr0.01_final | failed |  |  |  |  |  |  |  |
| int_BP-row_k8_ls0_T16_none_i17_s4000_lr0.01_final | failed |  |  |  |  |  |  |  |
| BP-row/8#0 | ordinary | 18 | 0.01 | 1000 | 0.174 / 0.148 / 0.131 | 0.179 / 0.146 / 0.129 | 3219 / 4001 / 4088 |  |
| BP-row/8#0 | ordinary | 19 | 0.01 | 1000 | 0.174 / 0.148 / 0.131 | 0.179 / 0.147 / 0.129 | 3219 / 4001 / 4088 |  |
| TopK/8 | ordinary | 18 | 0.01 | 1000 | -0.378 / -0.324 / -0.266 | -0.374 / -0.328 / -0.266 | 3320 / 4005 / 4087 |  |
| TopK/8 | ordinary | 19 | 0.01 | 1000 | -0.342 / -0.269 / -0.301 | -0.332 / -0.271 / -0.306 | 3257 / 4001 / 4091 |  |
| Mix-TopK#0 | ordinary | 18 | 0.01 | 1000 | 0.768 / 0.723 / 0.710 | 0.770 / 0.722 / 0.709 | 0 / 0 / 0 |  |
| Mix-TopK#0 | ordinary | 19 | 0.01 | 1000 | 0.774 / 0.718 / 0.710 | 0.777 / 0.718 / 0.708 | 0 / 0 / 0 |  |
| A8@0.25 | ordinary | 18 | 0.01 | 1000 | 0.675 / 0.678 / 0.681 | 0.678 / 0.677 / 0.679 | 0 / 0 / 0 |  |
| A8@0.25 | ordinary | 19 | 0.01 | 1000 | 0.664 / 0.673 / 0.685 | 0.665 / 0.672 / 0.684 | 0 / 0 / 0 |  |


## Interaction: 4,000-update continuations after the --resume fix

| support | init | init seed | layer lr | updates | explained 2 / 3 / 4 (dev held-out) | explained 2 / 3 / 4 (reserved final batches) | terms without eligible neuron (2 / 3 / 4) | oracle nMSE vs BF16-input targets (order 2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TopK/8 | ordinary | 17 | 0.01 | 4000 | -0.320 / -0.270 / -0.285 | -0.309 / -0.273 / -0.290 | 3267 / 4010 / 4087 |  |
| BP-row/8#0 | ordinary | 17 | 0.01 | 4000 | 0.201 / 0.174 / 0.157 | 0.205 / 0.172 / 0.155 | 3219 / 4001 / 4088 |  |

