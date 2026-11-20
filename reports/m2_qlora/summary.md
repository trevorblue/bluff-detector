# Results: m2_qlora

Served detector: `probe_linear`  
tau_flag 0.919 · t_abstain 0.250

## Detection (AUROC, higher is better)

| Detector | trivia_test | kenya | kenya_unanswerable | western | OOD drop |
|---|---|---|---|---|---|
| logreg | 0.813 | 0.802 | 0.222 | 0.873 | 0.010 |
| xgboost | 0.785 | 0.787 | 0.407 | 0.827 | -0.001 |
| probe_linear | 0.819 | 0.702 | 0.284 | 0.800 | 0.118 |
| probe_mlp | 0.774 | 0.696 | 0.519 | 0.904 | 0.077 |

## Knowing vs saying (served detector, 95% bootstrap CI)

| Set | n | Abstain rate | Silent knowing | Internal | Expressed | Gap |
|---|---|---|---|---|---|---|
| trivia_test | 218 | 0.202 | 0.476 [0.385, 0.569] | 0.819 [0.759, 0.877] | 0.593 [0.544, 0.640] | 0.227 [0.157, 0.296] |
| kenya | 211 | 0.133 | 0.437 [0.353, 0.523] | 0.702 [0.623, 0.776] | 0.537 [0.492, 0.579] | 0.165 [0.078, 0.252] |
| kenya_unanswerable | 82 | 0.354 | 0.491 [0.362, 0.625] | 0.284 [0.188, 0.392] | 0.173 [0.123, 0.228] | 0.111 [0.012, 0.210] |
| western | 33 | 0.030 | 0.500 [0.200, 0.778] | 0.800 [0.632, 0.934] | 0.538 [0.500, 0.625] | 0.262 [0.058, 0.426] |

## Kenya vs UK twins (paired, 33 pairs)

| | Kenya | UK twin |
|---|---|---|
| Wrong rate | 0.667 | 0.394 |
| Abstain rate | 0.030 | 0.030 |
| Mean bluff score | 0.577 | 0.480 |

Wrong-rate difference, Kenya minus UK: 0.273 [0.091, 0.455] · McNemar p 0.022

Generation latency ms: p50 403.9, p95 872.7, p99 1273.8
