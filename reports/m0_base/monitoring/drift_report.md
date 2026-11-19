# Monitoring report: m0_base

Generated 2026-11-16T20:46:54 from `logs/requests.jsonl`. Serving `probe_linear`, version `0.1.0+cf9e27e`.

**Overall: TOO EARLY** · 5 live requests (2026-11-16T17:43:20.293936+00:00 to 2026-11-16T17:46:33.185690+00:00)

> Fewer than 30 requests: the checks below are shown, but it is too early to call drift.

## Behaviour

| | Validation baseline | Live |
|---|---|---|
| Block rate (P(wrong) > 0.25) | 0.793 | 0.800 |
| Latency p50 / p95 / p99 (ms) | - | 26218 / 39382 / 41235 |

## Drift checks

PSI: under 0.1 = stable, 0.1–0.25 = watch, above 0.25 = drift. KS p under 0.01 = the two samples differ. *Data* checks compare the 8 confidence numbers of live answers with the training answers; the *prediction* check compares live P(wrong) with validation.

| Check | Kind | Reference mean | Live mean | PSI | KS p | Status |
|---|---|---|---|---|---|---|
| mean_logprob | data | -0.520 | -0.025 | 6.826 | 0.0000 | DRIFT |
| min_logprob | data | -1.027 | -0.059 | 6.826 | 0.0001 | DRIFT |
| mean_entropy | data | 1.305 | 0.124 | 6.826 | 0.0000 | DRIFT |
| max_entropy | data | 2.377 | 0.247 | 6.826 | 0.0000 | DRIFT |
| last_entropy | data | 0.777 | 0.008 | 7.044 | 0.0045 | DRIFT |
| n_tokens | data | 3.609 | 2.400 | 3.691 | 0.3948 | DRIFT |
| frac_low_prob | data | 0.300 | 0.000 | 4.886 | 0.0286 | DRIFT |
| bluff_score (P(wrong)) | prediction | 0.658 | 0.654 | 5.742 | 0.5172 | DRIFT |

## What to do

- **Data drift** (the questions changed): check what users are asking; if it is a new kind of question, add similar questions with known answers and retrain.
- **Prediction drift** (P(wrong) or the block rate moved) without data drift: the model or its hardware changed (e.g. a laptop CPU in bfloat16 instead of the T4 in float16). Re-measure the detector on that setup before trusting its threshold.
