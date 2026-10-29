# Three-week build plan

From the current scaffold to a submitted project, organised by CRISP-DM phase. Dates assume
submission on **Wednesday 18 November 2026**; shift them if the deadline differs.

**Who's who:** *Trevor* (project owner), *Students* (the four contributors who write and review
the Kenya questions), *Claude* (AI assistant, disclosed in [AI_USE.md](../AI_USE.md)).

## Scope for three weeks

| | Contents |
|---|---|
| **Must ship** | Kenya set (300 real + 80 fake questions, verified) and 60 UK twins · variants M0, M1 and M2 · the four standard ML detectors (logistic regression, XGBoost, linear probe, MLP probe) · the full metric suite and the H3 test · API, Docker and CI (already working) · MLflow tracking switched on |
| **Only if ahead** | Probe transfer (the M0 probe applied unchanged to M1 and M2) · MLP layer sweep · semantic entropy on the Kenya set |
| **Dropped, and said so in the report** | M3 and H4 · the LSTM detector · the LLM judge (hand-checked samples instead) · the LoRA-vs-QLoRA cost footnote · retraining on a self-hosted GPU runner |

## The plan by CRISP-DM phase

| Phase | In this project | When | Done when |
|---|---|---|---|
| **1. Business understanding** | Question, hypotheses, cost matrix and success criteria are set in the [proposal](proposal.md) and `configs/base.yaml`. One task left: the novelty check. | Fri 30 Oct | Nobody has published the knowing-vs-saying gap in hidden states |
| **2. Data understanding** | The Kenya question rules, trial questions, the pilot run's accuracy per pool, how noisy the labels are | Weeks 1–3 | Feedback given on trial questions; label noise measured on hand-checked samples |
| **3. Data preparation** | Kenya set written and reviewed; document collections built; the language model run on every question to create the dataset; labels and features | Weeks 1–2 | `bluff validate` and `bluff audit` pass; every variant generated on every question |
| **4. Modeling** | Train the standard ML detectors per variant; build the M2 adapter (QLoRA) and M1 retrieval | Weeks 1–2 | Detectors trained for M0, M1, M2 |
| **5. Evaluation** | All metrics with confidence intervals; M0 vs M2 (H3) and M0 vs M1 | Week 3 | `summary.md` per variant and both comparisons written |
| **6. Deployment** | Register the best detector, serve it, measure latency, ship the image | Week 3 | The API answers with the real model; p50/p99 latency measured |

## Week 1 · Wed 28 Oct – Tue 3 Nov: set up, and M0 on trivia

| When | Task | Who | Phase |
|---|---|---|---|
| Wed 28 – Thu 29 | Accounts: Kaggle (verify the account to unlock GPUs), Hugging Face with a private dataset repo (the shared drive), the Colab extension in VS Code | Trevor | Setup |
| Wed 28 – Thu 29 | Make the code ready for split GPU runs: resume after a timeout, slices, merge; automatic float16/bfloat16; pinned model and library versions; one runner notebook for Kaggle and Colab | Claude | Data preparation |
| Thu 29 | Student pack: a question sheet matching the format, a one-page brief, a sheet-to-`questions.jsonl` import script | Claude | Data understanding |
| Thu 29 | Brief the students and assign domains | Trevor | Business understanding |
| Fri 30 | Novelty check | Trevor (Claude helps search) | Business understanding |
| Fri 30 – Sat 31 | Ingest TriviaQA and TruthfulQA. Pilot M0 on 200 questions, saving five layers: measure the time per question, check the model's accuracy, check the probe beats chance. | Claude runs it; Trevor starts Colab slices | Data understanding |
| Sat 31 | Each student writes 10 trial questions | Students | Data preparation |
| Sun 1 Nov | Feedback on the trial questions | Trevor | Data understanding |
| Sun 1 | Choose the probe layer on the trivia validation split and commit it. Choosing before any Kenya question is scored is the pre-registration. | Trevor decides | Modeling |
| Sun 1 – Mon 2 | Full M0 trivia run split across GPUs; merge; label, featurize, train, evaluate | Claude + GPUs | Preparation → modeling → evaluation |
| Mon 2 | Students send document URLs for their domain (40–60 each) | Students | Data preparation |
| Tue 3 | Build both document collections, split by document; run `bluff audit` | Claude | Data preparation |

**Checkpoint, Tue 3 Nov:** M0 detectors trained on trivia; half the Kenya questions drafted;
both document collections built.

## Week 2 · Wed 4 – Tue 10 Nov: M2 and M1, and finish the Kenya set

| When | Task | Who | Phase |
|---|---|---|---|
| Wed 4 | QLoRA fine-tuning on the fine-tuning collection (one GPU session, about an hour); adapter saved to the shared drive | Claude runs it; Trevor starts it if on Colab | Modeling |
| Wed 4 – Thu 5 | M2 trivia run (T4s, same settings as M0); label, featurize, train, evaluate | Claude + GPUs | Preparation → modeling |
| Thu 5 – Sat 7 | M1 trivia run with retrieval; label, featurize, train, evaluate | Claude + GPUs | Preparation → modeling |
| Fri 6 | Students finish writing: 75 real + 20 fake questions and 15 UK twins each | Students | Data preparation |
| Sat 7 – Mon 9 | Cross-review (A→B→C→D→A) and the blind fake-question test | Students | Data understanding |
| Mon 9 – Tue 10 | Settle disagreements; `bluff validate` and `bluff audit` pass; freeze the Kenya set | Trevor | Data understanding |
| Tue 10 | Kenya runs for M0, M1 and M2 (resume runs only the new questions: 30–45 min each) | Claude + GPUs | Data preparation |

**Checkpoint, Tue 10 Nov:** Kenya set frozen (at least 300 real + 80 fake); all three variants
generated on every question.

## Week 3 · Wed 11 – Tue 17 Nov: results, deployment, report

| When | Task | Who | Phase |
|---|---|---|---|
| Wed 11 | Label, featurize, train and evaluate M0, M1 and M2 with the Kenya set | Claude | Preparation → evaluation |
| Wed 11 – Thu 12 | Hand-check the label samples (50 rows per variant, about 40 rows each) | Students | Data understanding |
| Thu 12 | Final evaluation with 10,000 bootstrap resamples; `bluff compare` M0 vs M2 (H3) and M0 vs M1 | Claude runs; Trevor interprets | Evaluation |
| Fri 13 | Register the best detector; serve it with the real model; measure p50/p99 latency; check the API's scores match the pipeline's on 100 questions | Claude | Deployment |
| Fri 13 – Tue 17 | Write the report: results, interpretation, limitations, deliberate omissions | Trevor | All phases |
| Sun 15 | Real results in the README; AI_USE.md updated; repo made public | Trevor, with Claude | Deployment |
| Wed 18 | Submit | Trevor | |

If the schedule is ahead by Thu 12, add probe transfer first (it reuses existing runs, so no GPU
time), then the MLP layer sweep, then semantic entropy on the Kenya set.

## The students' share

| | Student A | Student B | Student C | Student D |
|---|---|---|---|---|
| **Domains** | civics_law | geography_society, language | institutions_services, money_livelihoods | western_default_traps |
| **Questions** | 75 real + 20 fake | 75 + 20 | 75 + 20 | 75 + 20 |
| **UK twins** | 15 | 15 | 15 | 15 |
| **Reviews** | B's 110 | C's 110 | D's 110 | A's 110 |
| **Documents** | 40–60 URLs | 40–60 | 40–60 | 40–60 |
| **Label checks** | ~40 rows | ~40 | ~40 | ~40 |

About 15–17 hours each over the three weeks. The rules, in short:

- **Write each question in your own words**; never copy a sentence from the source. The leakage
  check rejects any question whose wording appears in the document collections.
- One clear answer per question, with every accepted spelling listed. Avoid facts that change
  (office holders, fees), or record when the answer stops being true.
- Official sources only (kenyalaw.org, .go.ke sites, KNBS, CBK), with the link and the date
  checked.
- Fake questions must look real: a plausible county, Act or agency that does not exist.
- Reviewers answer from the source *before* looking at the author's answer. A question is
  verified only if both agree.
- A UK twin asks one of your real questions about the United Kingdom, with the same shape and
  the same kind of answer, so Kenya and UK can be compared pair by pair.

Full rules and the question format: [data/README.md](../data/README.md).

## GPU plan

- **Every generation run uses T4 GPUs in float16** with the same settings, so numbers stay
  comparable across slices and variants. Kaggle ("GPU T4 x2", runs in the background) and Colab
  both provide T4s. QLoRA training can use any GPU.
- Each slice saves its progress to the shared drive every ~100 questions, so a session that
  times out resumes where it stopped, on any machine.
- Budget: about 15 GPU-hours for M0, M1 and M2 together, which fits in Kaggle's weekly free
  quota. The students' own Kaggle accounts are spare capacity (one account per person).

## Risks

| Risk | Response |
|---|---|
| The Kenya set runs late | Freeze whatever is verified on Tue 10 Nov if it reaches 300; below that, report with wider confidence intervals |
| A free GPU session dies mid-run | Resume continues from the last save; nothing is lost |
| The probe does not beat chance in the pilot | Check labels and layer choice before the full run; week 1 has slack for this |
| A Kenya question's wording matches a document | Rewrite or drop the question; never retrain the adapter |
| Fine-tuning makes the model worse | Report it: the proposal predicts this outcome, and it is a result, not a failure |
