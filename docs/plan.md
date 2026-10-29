# Build plan: scaffold to submission

Everything is built and rehearsed end to end on stand-in data **before** the students' questions
arrive. On **Tuesday 10 November** their data is plugged in with one checklist (section 6), about
three hours, mostly waiting for GPUs. Submission is **Wednesday 18 November 2026**.

**Who's who:** *Trevor* (project owner), *Students* (the four contributors who write and review
the Kenya questions), *Claude* (AI assistant, disclosed in [AI_USE.md](../AI_USE.md)). How each
piece works: [concepts.md](concepts.md). GPU setup: [gpu-setup.md](gpu-setup.md).

## 1. The principle: never wait for the data

| Data | Stand-in until it arrives | Real data arrives | What changes when it does |
|---|---|---|---|
| TriviaQA and TruthfulQA | None needed: real, downloaded automatically | Now | Nothing |
| Kenya questions (300 real, 80 fake, 60 UK twins) | Trevor's 9 starter questions, already in `data/kenya/questions.jsonl` | Tue 10 Nov | Import → ingest → generation resumes on the new questions only |
| Students' trial questions | None | Sat 31 Oct | They test the sheet importer on real rows |
| Document links | A handful of core public documents (the Constitution, a few Acts) | Mon 2 Nov | Collections rebuilt and re-audited |
| Hand-checked labels | Trevor checks a few rows as a test | Thu 12 Nov | Label-noise rate computed |

The 9 starter questions sit in every run from the first one, so the whole Kenya path (labels,
metrics, twins table) is exercised early. **By Monday 9 November every command in the plug-in
checklist has already run for M0, M1 and M2.** Plug-in day repeats commands that worked.

## 2. Scope

| | Contents |
|---|---|
| **Must ship** | Kenya set (300 real + 80 fake, verified) and 60 UK twins · variants M0, M1 and M2 · the four standard ML detectors (logistic regression, XGBoost, linear probe, MLP probe) · the full metric suite, the H3 test and the twin comparison · split GPU runs · monitoring · API, Docker, CI and CD · MLflow tracking switched on |
| **Only if ahead** | Probe transfer (M0's probe applied unchanged to M1 and M2) · MLP layer sweep · semantic entropy on the Kenya set |
| **Dropped, and said so in the report** | M3 and H4 · the LSTM detector · the LLM judge (hand-checked samples instead) · the LoRA-vs-QLoRA cost footnote · retraining on a self-hosted GPU runner |

## 3. The plan by CRISP-DM phase

| Phase | Work | Steps | When |
|---|---|---|---|
| **1. Business understanding** | Question, hypotheses, costs and gates are set in the [proposal](proposal.md) and `configs/base.yaml`. Left: the novelty check. | Trevor | Fri 30 Oct |
| **2. Data understanding** | Real-model smoke test; trial questions; label noise | B4, B9 | Weeks 1–3 |
| **3. Data preparation** | Split GPU runs; sheet importer; document collections; generation for every variant | B1–B3, B6, B7 | Weeks 1–2 |
| **4. Modeling** | Layer pre-registration; the standard ML detectors per variant; the QLoRA adapter (M2) and retrieval (M1) | B5, B8 | Weeks 1–2 |
| **5. Evaluation** | All metrics with confidence intervals; M0 vs M2 (H3), M0 vs M1, Kenya vs UK twins (H2) | B12 if ahead | Week 3 |
| **6. Deployment** | Serving the champion, latency, skew check, monitoring, the real-model image | B10, B11 | Weeks 2–3 |

## 4. The build steps

Built by Claude unless marked; none of them needs the students' data. New commands are marked
*(new)*.

| Step | CRISP-DM | What gets built | Done when | Day |
|---|---|---|---|---|
| **B1. Resumable, splittable generation** | Data preparation | `generate` saves after every question and skips finished ones (`--resume`); `--shard i/N` runs one slice; `bluff merge` *(new)* joins slices; `bluff push-questions` *(new)* and slice uploads sync with the shared drive | On the fake model, 3 slices stopped mid-way, resumed and merged give exactly the same results as one run (a CI test) | Fri 30 Oct |
| **B2. Hardware settings** | Data preparation | `dtype: auto` (bfloat16 where supported, float16 on T4) for generation and QLoRA; pinned model revision; each slice records its GPU and library versions, and `merge` checks they match | Unit tests pass; a T4 run reports float16 | Fri 30 Oct |
| **B3. GPU runner notebook** | Data preparation | `notebooks/gpu_runner.ipynb`, one notebook for Colab in VS Code and Kaggle in the background (two GPUs): settings cell, T4 check, install, token prompt or Kaggle Secrets, run, upload | Trevor runs B4 with it from VS Code | Sat 31 Oct |
| **B4. Real-model smoke test** | Data understanding | `configs/variants/m0_smoke.yaml`: the real model on 30 questions | Answers, labels and timings look sane; under 15 minutes on a T4 | Sat 31 Oct |
| **B5. Layer choice (pre-registration)** | Modeling | A pilot saving 5 layers on 500 trivia questions; `bluff sweep-layers` *(new)* reports validation AUROC per layer | Trevor picks the layer; it is committed to `configs/base.yaml` before any Kenya question is scored | Sun 1 Nov |
| **B6. Sheet importer** | Data preparation | `bluff import-sheet --csv` *(new)*: assigns ids by domain, maps Type to pool, links each twin to its Kenya question by text, runs the data rules and lists every problem by sheet row | The students' trial rows import cleanly, or the problems go back to them | Sat 31 Oct – Sun 1 Nov |
| **B7. Document-collection builder** | Data preparation | `bluff build-corpus --docs documents.csv` *(new)*: downloads each page or PDF, extracts the text, cuts 300–800-token chunks, assigns each document to retrieval or fine-tuning (seeded, by document), runs the leakage audit | Built from stand-in documents (Sat 31 Oct), then from the students' links (Tue 3 Nov) | Sat 31 Oct, Tue 3 Nov |
| **B8. QLoRA fixes** | Modeling | Documents split into 1,024-token windows instead of cut off; `dtype: auto`; adapter uploaded to the shared drive | A 50-step pilot runs on a T4 | Tue 3 Nov |
| **B9. Label-noise rate** | Data understanding | `bluff label-noise` *(new)*: how often automatic marking disagrees with the hand checks, with a confidence interval; added to `summary.md` | Runs on Trevor's test rows | Thu 5 Nov |
| **B10. Monitoring** | Deployment | The API appends every audit record and its 8 features to a log; `bluff monitor` *(new)* compares the log with the training data (PSI per feature and on the bluff score, abstain rate) and writes `reports/monitor.md` | A burst of unusual questions shows up as drift in the report | Thu 5 – Fri 6 Nov |
| **B11. Serving the real model** | Deployment | The API loads `@champion` from the registry; laptop serving in bfloat16; a latency script (p50/p99); a skew check (API vs pipeline scores on 100 questions); more memory and a model cache for the real-model Docker image | The real model answers on the laptop; latency and skew are measured | Fri 6 – Sat 7 Nov |
| **B12. Probe transfer** *(if ahead)* | Evaluation | `bluff transfer` *(new)*: M0's probe scored unchanged on M1 and M2 answers; the accuracy drop | Numbers in the report | Fri 13 Nov |

## 5. Day by day

### Week 1 · Fri 30 Oct – Tue 3 Nov: build, and the first real results

| Day | Trevor | Students | Claude | GPU runs |
|---|---|---|---|---|
| **Fri 30** | GPU setup, [gpu-setup.md](gpu-setup.md) section 2 (~45 min); novelty check | Briefing and sheet, if not done yet | B1, B2 | — |
| **Sat 31** | Run the smoke test from VS Code on Colab | 10 trial questions | B3, B4; B6 on the trial rows; B7 on stand-in documents | Smoke test (Colab, ~15 min); layer pilot (Kaggle, ~30 min) |
| **Sun 1 Nov** | Feedback on trial questions; choose the probe layer and commit it | Fix trial questions | B5 report | Full M0 on trivia (Kaggle ×2 + Colab, ~1.5 h) |
| **Mon 2** | Read the first real M0 results | Document links (40–60 each) | Merge M0; label, featurize, train, evaluate | — |
| **Tue 3** | — | Writing | B7 with the students' links; B8 | QLoRA pilot (~15 min) |

**Checkpoint, Tue 3 Nov:** real M0 results on trivia; the layer pre-registered; both document
collections built and audited.

### Week 2 · Wed 4 – Tue 10 Nov: M2 and M1, rehearsal, plug-in

| Day | Trevor | Students | Claude | GPU runs |
|---|---|---|---|---|
| **Wed 4** | Start the Colab slice | Writing | Laptop stages for M2 | QLoRA training (~1 h), then M2 (~1.5 h) |
| **Thu 5** | Start the Colab slice | Writing | Laptop stages for M1; B9, B10 | M1 (~2.5 h) |
| **Fri 6** | Try the real model on the laptop | **Finish writing** | B11 | — |
| **Sat 7 – Mon 9** | Settle review disagreements as they come in | Review; fake-question test | **Dress rehearsal:** the section 6 checklist end to end for M0, M1, M2; fix anything that breaks | Rehearsal runs (short) |
| **Tue 10** | **Freeze the Kenya set. Plug-in day:** section 6 | — | Runs the checklist with Trevor | Kenya questions for M0, M1, M2 (~30–45 min each, in parallel) |

**Checkpoint, Tue 10 Nov:** the Kenya set frozen; all three variants generated on every question.

### Week 3 · Wed 11 – Wed 18 Nov: results, ship, report

| Day | Trevor | Students | Claude |
|---|---|---|---|
| **Wed 11** | Look at the first Kenya results | Label checks (~40 rows each) | Laptop stages for M0, M1, M2 with Kenya |
| **Thu 12** | Interpret the results | Finish label checks | Label-noise rate; final evaluation (10,000 bootstrap resamples); compare M0 vs M2 (H3) and M0 vs M1; twin results (H2) |
| **Fri 13** | Start the report | — | Register the champion; serve it; latency, skew check, monitoring demo; real-model Docker image; B12 if ahead |
| **Sat 14 – Tue 17** | Write the report | — | Real results into the README (Sun 15) |
| **Sun 15** | Update AI_USE.md; make the repo public | — | — |
| **Wed 18** | **Submit** | — | — |

## 6. Plug-in day checklist (Tue 10 Nov)

Every command here has already run in the dress rehearsal. Times are estimates.

1. **Export the sheet** as CSV (one file per tab), then import it *(5 min)*:
   `bluff import-sheet --csv sheet/*.csv`. Fix any listed problem with the student and run it
   again.
2. **Check, assemble and audit** *(10 min)*:
   ```bash
   bluff validate --config configs/variants/m0_base.yaml
   bluff ingest   --config configs/variants/m0_base.yaml
   bluff audit    --config configs/variants/m0_base.yaml
   bluff push-questions
   ```
   If the audit flags a question, rewrite or drop **the question**; never retrain the adapter.
3. **Generate the new questions on GPUs** *(30–45 min, in parallel)*: the runner notebook with
   `--resume` for `m0_base`, `m2_qlora` and `m1_rag`, one variant per GPU (Kaggle GPU 0, Kaggle
   GPU 1, Colab). Only the ~450 new questions run.
4. **Laptop, per variant** *(about 10 min each)*:
   ```bash
   bluff merge     --config configs/variants/m0_base.yaml
   bluff label     --config configs/variants/m0_base.yaml
   bluff featurize --config configs/variants/m0_base.yaml
   bluff train     --config configs/variants/m0_base.yaml
   bluff evaluate  --config configs/variants/m0_base.yaml
   bluff register  --config configs/variants/m0_base.yaml
   ```
   Then the same for `m2_qlora.yaml` and `m1_rag.yaml`.
5. **Compare** *(2 min)*:
   ```bash
   bluff compare --config configs/variants/m0_base.yaml --other configs/variants/m2_qlora.yaml
   bluff compare --config configs/variants/m0_base.yaml --other configs/variants/m1_rag.yaml
   ```
6. **Send the label-check rows** (`data/interim/<variant>/review_sample.jsonl`) to the students.

**Done when** each `reports/<variant>/summary.md` shows rows for kenya, kenya_unanswerable and
western, plus the Kenya-vs-UK twins table.

## 7. If the students' data is late or short

| Situation | What to do |
|---|---|
| Fewer than 300 verified real questions on Tue 10 Nov | Freeze what is verified. Trevor adds questions to reach 300 if time allows; otherwise report with wider confidence intervals. |
| Twins incomplete | Use the pairs that exist; 30 or more pairs still show a large gap |
| Document links late | Keep the stand-in collections from Sat 31 Oct; QLoRA trains on those |
| Label checks not done by Thu 12 Nov | Trevor checks the ~150 rows (about 3 hours) |
| A student drops out | Their domain has fewer questions; Trevor takes over their reviews |

## 8. Other risks

| Risk | Response |
|---|---|
| A GPU session dies mid-run | Resume continues from the last upload |
| The probe is no better than chance in the pilot | Check labels and the layer choice before the full run; week 1 has slack |
| Kaggle's GPU quota runs out | Colab, the students' Kaggle accounts, or a rented GPU (a few dollars) |
| Fine-tuning makes the model worse | Report it: the proposal predicts this, and it is a result, not a failure |
| The laptop is too slow to serve | Report the CPU latency honestly; for the live demo, serve from a Colab GPU |

## 9. The students' share

| | Student A | Student B | Student C | Student D |
|---|---|---|---|---|
| **Domains** | civics_law | geography_society, language | institutions_services, money_livelihoods | western_default_traps |
| **Questions** | 75 real + 20 fake | 75 + 20 | 75 + 20 | 75 + 20 |
| **UK twins** | 15 | 15 | 15 | 15 |
| **Reviews** | B's 110 | C's 110 | D's 110 | A's 110 |
| **Documents** | 40–60 links | 40–60 | 40–60 | 40–60 |
| **Label checks** | ~40 rows | ~40 | ~40 | ~40 |

About 15–17 hours each. The full rules, examples and dates are in the student brief.
