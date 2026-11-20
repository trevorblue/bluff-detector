# Results: m3_rag_qlora

Served detector: `probe_linear`  
tau_flag 0.979 · t_abstain 0.250

## Detection (AUROC, higher is better)

| Detector | trivia_test | kenya | kenya_unanswerable | western | OOD drop |
|---|---|---|---|---|---|
| logreg | 0.807 | 0.791 | 0.309 | 0.617 | 0.016 |
| xgboost | 0.789 | 0.760 | 0.407 | 0.613 | 0.030 |
| probe_linear | 0.823 | 0.637 | 0.370 | 0.805 | 0.186 |
| probe_mlp | 0.833 | 0.643 | 0.457 | 0.853 | 0.190 |

## Knowing vs saying (served detector, 95% bootstrap CI)

| Set | n | Abstain rate | Silent knowing | Internal | Expressed | Gap |
|---|---|---|---|---|---|---|
| trivia_test | 218 | 0.633 | 0.465 [0.314, 0.617] | 0.823 [0.758, 0.881] | 0.634 [0.563, 0.703] | 0.189 [0.102, 0.279] |
| kenya | 211 | 0.275 | 0.521 [0.408, 0.635] | 0.637 [0.559, 0.712] | 0.697 [0.649, 0.742] | -0.060 [-0.152, 0.028] |
| kenya_unanswerable | 82 | 0.561 | 0.556 [0.389, 0.719] | 0.370 [0.266, 0.481] | 0.278 [0.222, 0.331] | 0.093 [-0.025, 0.216] |
| western | 33 | 0.636 | 0.500 [0.000, 1.000] | 0.805 [0.633, 0.940] | 0.568 [0.395, 0.730] | 0.237 [0.044, 0.433] |

## Kenya vs UK twins (paired, 33 pairs)

| | Kenya | UK twin |
|---|---|---|
| Wrong rate | 0.455 | 0.424 |
| Abstain rate | 0.242 | 0.636 |
| Mean bluff score | 0.803 | 0.691 |

Wrong-rate difference, Kenya minus UK: 0.030 [-0.152, 0.212] · McNemar p 1.000

Generation latency ms: p50 2060.1, p95 2867.9, p99 3455.5
