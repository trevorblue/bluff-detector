# Results: m1_rag

Served detector: `probe_linear`  
tau_flag 0.988 · t_abstain 0.250

## Detection (AUROC, higher is better)

| Detector | trivia_test | kenya | kenya_unanswerable | western | OOD drop |
|---|---|---|---|---|---|
| logreg | 0.760 | 0.727 | 0.235 | 0.808 | 0.033 |
| xgboost | 0.750 | 0.716 | 0.469 | 0.723 | 0.034 |
| probe_linear | 0.806 | 0.683 | 0.210 | 0.765 | 0.122 |
| probe_mlp | 0.823 | 0.670 | 0.173 | 0.838 | 0.153 |

## Knowing vs saying (served detector, 95% bootstrap CI)

| Set | n | Abstain rate | Silent knowing | Internal | Expressed | Gap |
|---|---|---|---|---|---|---|
| trivia_test | 218 | 0.876 | 0.154 [0.000, 0.375] | 0.806 [0.740, 0.868] | 0.560 [0.509, 0.612] | 0.246 [0.169, 0.318] |
| kenya | 211 | 0.275 | 0.416 [0.301, 0.521] | 0.683 [0.606, 0.755] | 0.681 [0.633, 0.729] | 0.002 [-0.087, 0.089] |
| kenya_unanswerable | 82 | 0.573 | 0.600 [0.429, 0.759] | 0.210 [0.123, 0.296] | 0.284 [0.231, 0.338] | -0.074 [-0.179, 0.025] |
| western | 33 | 0.788 | 0.000 [0.000, 0.000] | 0.765 [0.565, 0.915] | 0.515 [0.375, 0.674] | 0.250 [0.065, 0.414] |

## Kenya vs UK twins (paired, 33 pairs)

| | Kenya | UK twin |
|---|---|---|
| Wrong rate | 0.485 | 0.606 |
| Abstain rate | 0.212 | 0.788 |
| Mean bluff score | 0.819 | 0.700 |

Wrong-rate difference, Kenya minus UK: -0.121 [-0.273, 0.030] · McNemar p 0.289

Generation latency ms: p50 1523.9, p95 2053.8, p99 2458.3
