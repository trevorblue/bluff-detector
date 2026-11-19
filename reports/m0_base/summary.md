# Results: m0_base

Served detector: `probe_linear`  
tau_flag 0.953 · t_abstain 0.250

## Detection (AUROC, higher is better)

| Detector | trivia_test | kenya | kenya_unanswerable | western | OOD drop |
|---|---|---|---|---|---|
| logreg | 0.782 | 0.760 | 0.394 | 0.826 | 0.022 |
| xgboost | 0.753 | 0.722 | 0.294 | 0.885 | 0.031 |
| probe_linear | 0.773 | 0.731 | 0.300 | 0.856 | 0.042 |
| probe_mlp | 0.815 | 0.749 | 0.419 | 0.881 | 0.066 |

## Knowing vs saying (served detector, 95% bootstrap CI)

| Set | n | Abstain rate | Silent knowing | Internal | Expressed | Gap |
|---|---|---|---|---|---|---|
| trivia_test | 218 | 0.353 | 0.338 [0.237, 0.438] | 0.773 [0.707, 0.837] | 0.598 [0.535, 0.661] | 0.175 [0.104, 0.245] |
| kenya | 211 | 0.280 | 0.471 [0.370, 0.564] | 0.731 [0.651, 0.803] | 0.607 [0.551, 0.663] | 0.124 [0.033, 0.209] |
| kenya_unanswerable | 82 | 0.573 | 0.343 [0.187, 0.514] | 0.300 [0.194, 0.407] | 0.281 [0.229, 0.333] | 0.019 [-0.096, 0.129] |
| western | 33 | 0.061 | 0.615 [0.333, 0.889] | 0.856 [0.714, 0.970] | 0.567 [0.500, 0.667] | 0.289 [0.143, 0.426] |

## Kenya vs UK twins (paired, 33 pairs)

| | Kenya | UK twin |
|---|---|---|
| Wrong rate | 0.636 | 0.455 |
| Abstain rate | 0.061 | 0.061 |
| Mean bluff score | 0.644 | 0.623 |

Wrong-rate difference, Kenya minus UK: 0.182 [-0.030, 0.394] · McNemar p 0.210

Generation latency ms: p50 187.8, p95 443.0, p99 716.7
