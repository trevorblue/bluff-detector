# Concepts: how and why everything works

Every idea in the project explained in plain words: the models, the metrics, serving, Docker,
CI/CD/CT, MLOps, monitoring and the GPU setup. Read it before a presentation or viva. For which
file does what, see [architecture.md](architecture.md); for the schedule, see [plan.md](plan.md).

> **Built vs not built.** Every section says plainly what exists today and what does not.
> Section 13 lists it all in one table. Examiners probe exactly these gaps, so know them.

## 1. The big picture

```
 OFFLINE (the pipeline, `bluff all`)                       ONLINE (the API)
 ┌──────────────────────────────────────────────┐          ┌──────────────────────────────┐
 │ questions → run the LLM → label → features → │ trained  │ question → LLM answers →     │
 │ train detectors → evaluate → register        │ detector │ detector scores → answer or  │
 └──────────────────────────────────────────────┘ ───────► │ abstain → audit record       │
                    ▲                                       └──────────────────────────────┘
 OPS around both: git + GitHub Actions (CI, CD, CT) · Docker · MLflow · registry · DVC · tests
```

Where the code came from: the scaffold (pipeline, API, Docker, CI and tests) was generated with
Claude on 16 October 2026 and is disclosed in [AI_USE.md](../AI_USE.md). Later work published
the repository, fixed the CD workflow, wrote these docs and added the UK-twin comparison.

## 2. The models

There are two kinds of model, and keeping them apart is the key to the whole project.

### 2a. The language model under study: Qwen2.5-1.5B-Instruct

- A 1.5-billion-parameter **open-weights** model. Open weights matter because the project reads
  the model's insides; hosted APIs such as ChatGPT do not expose them.
- It builds an answer **one token at a time** (a token is a word or part of a word). At each
  step it gives a probability to every possible next token and picks the most likely one
  (*greedy* decoding, at most 32 tokens).
- Two signals come from each step:
  - **Token probability:** how much probability it gave the token it chose. Saying "52" with
    0.31 means it was far from sure.
  - **Entropy:** how spread out its probability was over all options. High entropy means it was
    torn between many possibilities.
- **Hidden states:** the model has 28 layers. Each layer turns every token into a vector of
  1,536 numbers, its internal "thought" at that point. The code saves this vector at two layers
  (8th and 4th from the top) and two positions (the last answer token, and the average over all
  answer tokens).

### 2b. The variants: the interventions under test

| Variant | What changes | How |
|---|---|---|
| **M0** | Nothing | The model as downloaded |
| **M1 + RAG** | It gets reading material | Retrieval-augmented generation (section 3) |
| **M2 + QLoRA** | It is fine-tuned on Kenyan documents | Section 4 |
| **M3** | Both | Cut from the three-week scope |

Every variant runs the same pipeline with the same detectors and metrics; only its config file
differs. That is what makes the comparison fair.

### 2c. The detectors: the models you actually train

Standard supervised ML. **Input:** signals recorded from one answer. **Output:** P(wrong).
**Label:** was the answer wrong (1) or right (0).

| Detector | Model | Input |
|---|---|---|
| `logreg` | Logistic regression (scikit-learn), inputs standardised | 8 summary numbers: average and minimum log-probability; average, maximum and last entropy; number of tokens; share of tokens under 0.5 probability; top retrieval score |
| `xgboost` | Gradient-boosted trees: 300 trees, depth 3, learning rate 0.05 | The same 8 numbers |
| `probe_linear` (served) | Logistic regression on the hidden state, strong L2 regularisation (C = 0.05) | The 1,536-number hidden state, 8th layer from the top, last answer token |
| `probe_mlp` | Neural network, one hidden layer of 128 units, early stopping | The same hidden state |

They are trained on **trivia questions only** (70% train, 15% validation, 15% test). Kenya
questions and their UK twins are never used in training, which makes them true
out-of-distribution tests. *Probe* just means a simple classifier that reads a model's hidden
states.

## 3. RAG, step by step (variant M1)

**Retrieval-Augmented Generation:** look things up, then answer. M0 sits a closed-book exam; M1
an open-book one. The question is whether the open book helps a small model on Kenyan facts, or
only makes it sound more sure of itself (H4).

1. **The knowledge base** (`data/corpus/retrieval_chunks.jsonl`): Kenyan documents (the
   Constitution, Acts, county pages, Wikipedia) cut into **chunks** of 300–800 tokens, roughly
   one section each. Chunks, because a whole Act is far too long to paste into a prompt. Each
   line: `{"id": "constitution-art-6", "text": "...", "source": "Constitution of Kenya 2010",
   "doc_id": "constitution"}`. It should hold the *answers* to Kenya questions but never their
   *wording*; the leakage audit checks that.
2. **The search engine** (`src/bluff/retrieval.py`, **BM25**), a classic keyword ranking:
   - Split text into lowercase words: "How many counties does Kenya have?" →
     `how, many, counties, does, kenya, have`.
   - At startup, count how many chunks contain each word, and each chunk's length.
   - Score each chunk. For each question word it contains, add points: **rare words count more**
     ("counties" a lot, "does" almost nothing; this is *IDF*, inverse document frequency);
     **repeats help with diminishing returns** (`k1 = 1.5`); **short, focused chunks win**
     (`b = 0.75`). Per word *q* the score is
     `IDF(q) × f·(k1+1) / (f + k1·(1 − b + b·length/average_length))`, summed over the words,
     where *f* is how often *q* appears in the chunk.
   - Keep the **top 3** chunks (`top_k: 3`).
3. **The generator** (`src/bluff/prompts.py`): the 3 chunks go in front of the question, under
   "Use the context if it is relevant. Context: [1] … [2] … [3] …".

How it plugs in: `m1_rag.yaml` sets `retrieval.enabled: true`; `generate` searches before every
question and records which chunks it used and their scores; the **top retrieval score is the 8th
detector feature**; in the API, a gated answer that used passages comes back as
`answer_with_citation` with the chunk ids as `sources`.

Limits: BM25 matches exact words, so "devolved units" will not find "counties", and Swahili will
not match English. The upgrade is *dense retrieval* (meaning-based embeddings) plus a reranker.
BM25 came first because it is fast, needs no GPU, is a strong baseline and is easy to explain.

**Status:** code built and unit-tested; the document collections are not built yet; M1 has never
run on real data.

## 4. Quantisation, LoRA and QLoRA (variant M2)

### Quantisation

A model is numbers: 1.5 billion **weights**. Memory depends on the bits per weight:

| Format | Bytes per weight | Qwen 1.5B in memory |
|---|---|---|
| float32 | 4 | ~6 GB |
| float16 / bfloat16 | 2 | ~3 GB |
| 8-bit | 1 | ~1.5 GB |
| **4-bit** | 0.5 | **~0.8 GB** |

**Quantisation stores each weight with fewer bits by rounding it to a smaller set of allowed
values**, like recording heights to the nearest 5 cm instead of the nearest millimetre. With 4
bits each weight can take only 16 values. Three tricks make that enough, all switched on in the
code:

1. **Blocks with their own scale:** weights are grouped in blocks of 64; each block stores one
   scale, and each weight is stored as "level 0–15 × my block's scale".
2. **NF4 (NormalFloat-4):** weights follow a bell curve bunched near zero, so NF4 puts its 16
   levels where the weights are: dense near zero, sparse in the tails.
3. **Double quantisation:** the block scales are themselves stored in 8-bit.

Maths cannot be done in 4-bit: each layer's weights are expanded to 16-bit for the moment of the
calculation. **Stored in 4-bit, computed in 16-bit.**

**float16 vs bfloat16:** both 16 bits. bfloat16 keeps float32's *range* (no overflow) with less
precision; float16 is more precise but can overflow. Free T4 GPUs have no native bfloat16, so the
code must use float16 there.

### LoRA

- One Qwen attention matrix (`q_proj`) is 1,536 × 1,536, about **2.36 million** numbers. Full
  fine-tuning changes all of them, in every layer, and needs a lot of GPU memory.
- **LoRA** freezes that matrix and learns a small correction: two thin matrices, B (1,536 × 16)
  times A (16 × 1,536), **49,152** numbers, about 2% of the original. The layer's output becomes
  *original output + 2 × (B·A applied to the input)*: 16 is the **rank** (`r`), 2 is
  `alpha / r` = 32 / 16. B starts at zero, so training starts from exactly the original model.
- It works because the change fine-tuning needs tends to be *low-rank*: a few directions carry it.
- Applied to all 7 projections (q, k, v, o, gate, up, down) in all 28 layers: about **18.5
  million trainable numbers, ~1.2% of the model**. The result is an **adapter**, a folder of
  tens of MB that sits on top of the unchanged base model.

### QLoRA = quantised base + LoRA

The frozen base is stored in **4-bit NF4** (~0.8 GB instead of 3 GB); only the small LoRA
matrices are trained, in 16-bit. **Gradient checkpointing** saves more memory by recomputing
intermediate results instead of storing them. LoRA vs QLoRA in one line: same adapter idea;
QLoRA only changes how the frozen base is *stored*.

### The training settings (`src/bluff/pipeline/finetune_qlora.py`)

| Setting | Meaning |
|---|---|
| Learning rate 2e-4 | How big each correction step is |
| 3% warmup, then cosine | Ramp the step size up gently, then lower it smoothly towards zero |
| Batch 4 × gradient accumulation 4 | Process 4 texts at a time, add up 4 rounds of corrections, update once: a batch of 16 without the memory cost |
| 1 epoch | One pass over the Kenyan documents |
| Objective: next-token prediction | Read Kenyan text and learn to predict each next word. It absorbs facts and phrasing; it is **not** taught to answer questions or to say "I don't know". |

That last row is why H3 is interesting: the model may gain knowledge *and* confidence, and lose
its willingness to abstain. M2 loads the base model in 16-bit and attaches the adapter on top.

**Status:** a template that has never run. Known problem: each document is **cut off at 1,024
tokens**, which throws away most of a long Act; it must be split into 1,024-token windows before
training (plan step B8).

## 5. Labels: how "right" and "wrong" are decided

- **Correct:** the open answer, lower-cased with punctuation and "a/an/the" removed, contains one
  of the accepted answers as whole words ("47", "forty-seven").
- **Abstained:** each question is also asked with an "I don't know" option in 3 wordings. If the
  model refuses in at least 2 of the 3 ("I don't know", "not sure", "no such", "sijui"…), it
  counts as abstained. Three wordings stop one prompt's phrasing from skewing the result.
- **Label noise:** automatic marking makes mistakes; "Kenya shillings" is marked wrong because the
  plural is not an accepted spelling. A random 50 answers per variant are checked by hand and the
  disagreement rate is reported as the *noise floor*. The code that computes that rate is not
  written yet (plan step B9).

## 6. What is measured, and what each number means

**Is the detector good at spotting wrong answers?**

| Metric | Meaning | Good |
|---|---|---|
| **AUROC** | Pick one wrong and one right answer at random: the chance the detector gives the wrong one the higher P(wrong). Needs no threshold. | 0.5 = coin flip, 1.0 = perfect |
| **PR-AUC** | How well it finds wrong answers without false alarms; more sensitive when wrong answers are rare | Higher; baseline = share of wrong answers |

**Can its probabilities be trusted?**

| Metric | Meaning | Good |
|---|---|---|
| **ECE** (Expected Calibration Error) | Sort answers into 10 bins by P(wrong); in each bin compare the average prediction with the share actually wrong; take the weighted average gap. If it says 30%, are 30% really wrong? | 0 = honest probabilities |
| **Reliability curve** | The per-bin table behind ECE, for plotting | Close to the diagonal |

Calibration matters because the abstain threshold assumes P(wrong) is honest.

**How good is answer-or-abstain?**

| Metric | Meaning | Good |
|---|---|---|
| **Risk–coverage curve** | Answer only the top k% most confident questions. Coverage = share answered; risk = error rate among them. | Risk stays low as coverage grows |
| **Risk at 70/80/90% coverage** | E.g. "answering only the 80% most confident, 12% of answers are wrong" | Lower |
| **AURC** | Average risk across all coverages: one number for answer-or-abstain quality | Lower |

**Knowing vs saying: the core of the research**

| Metric | Meaning |
|---|---|
| **Four cells** | Every answer placed by *probe flags uncertain?* × *took the "I don't know" option?*: honest uncertainty, **silent knowing**, abstention inflation, earned confidence |
| **Silent-knowing rate** (H1) | Among wrong answers the model delivered without abstaining, the share the probe flagged. High = it "knew" and answered anyway. |
| **Inflation rate** | Among abstentions, the share it answered correctly when given no option. High = it refuses things it knows. |
| **Internal detection** | AUROC of the probe score: how much the hidden states know about errors |
| **Expressed detection** | AUROC of abstention (yes/no) as an error predictor: how much the behaviour shows. 0.5 = its refusals are unrelated to its errors. |
| **Verbalisation gap** | Internal − expressed. Positive = **the model knows more than it says**. H3 predicts fine-tuning makes it grow. |
| **Hallucination rate** | Share of all answers that were wrong and stated rather than refused |

**Kenya against general knowledge (H2)**

| Comparison | Meaning |
|---|---|
| **Trivia vs Kenya** | Every metric reported separately for TriviaQA/TruthfulQA test questions and for Kenya, side by side |
| **Kenya vs UK twins** | 60 Kenya questions paired with the same question about the UK. Paired wrong and abstain rates, a bootstrap interval on the difference (Kenya minus UK), and McNemar's test. Pairing removes question style as an explanation. |

**Does it hold up out of distribution?**

| Metric | Meaning |
|---|---|
| **OOD drop** (H5) | AUROC on trivia test − AUROC on Kenya. Big = the detector does not travel well. Also a promotion gate: must be ≤ 0.25. |
| **Transfer gap** | The M0 probe applied unchanged to M1 and M2: the AUROC drop. Not built yet (plan step B12, if time allows). |

**Statistics: how sure are we?**

| Method | What it does |
|---|---|
| **Bootstrap CI** | Resample the questions with replacement 2,000 times (10,000 for the final report), recompute the metric each time; the middle 95% of results is the 95% confidence interval |
| **Paired bootstrap** | For two variants (or Kenya vs twin) on the same questions: resample the same questions for both and take the interval of the difference |
| **McNemar's test** | Did something change a yes/no outcome? It counts only the questions where the outcome changed; if nothing really changed, the two directions should be about equal |
| **Holm correction** | Several tests raise the chance of a false alarm; Holm makes the p-values stricter to compensate |

**Cost:** latency **p50/p95/p99** = median, 95th and 99th percentile response time. "p99 = 4 s"
means 1 request in 100 is slower than 4 seconds. Percentiles, not averages, because slow outliers
are what users feel.

## 7. The two thresholds

- **t_abstain = 0.25** (the API's gate). From costs: a confident wrong answer costs 3, a needless
  refusal costs 1. Answering costs p × 3 on average and refusing (1 − p) × 1, so refuse when
  p > 1/(1+3) = 0.25. A business decision written as maths (CRISP-DM business understanding).
- **tau_flag** (when the probe "says uncertain", for the four cells). Set on validation so that
  10% of *correct* answers are flagged, then fixed before any Kenya question is scored.

## 8. Serving, start to end

1. **Train:** each detector is saved as a folder: `detector.joblib` (the fitted model) plus
   `metadata.json` (thresholds, layer, git SHA, data hash, seed). The served one is copied to
   `artefacts/<variant>/serving/`.
2. **Register:** stored as version N. The first becomes `@champion`; later ones become
   `@challenger` only if they pass the gates.
3. **Package:** the Dockerfile copies that folder into the image (or `make serve` runs locally).
4. **Start:** uvicorn starts FastAPI; a *lifespan* handler loads the config, the detector and the
   language model **once**. `/ready` returns 503 until loading finishes.
5. **Request:** `POST /v1/ask {"question": "..."}`. Pydantic checks it (3–500 characters, no
   unknown fields, else error 422). For M1 the top 3 passages are retrieved. The model answers
   while signals are recorded.
6. **Score:** `signals.py` builds the same features as in training and `Detector.inputs()` picks
   the same layer and position, so there is no **training–serving skew**.
7. **Decide:** above 0.25 the answer is hidden (`abstain`); otherwise `answer`, or
   `answer_with_citation` when passages were used.
8. **Reply:** an audit record: request id, verdict, answer, bluff score, threshold, sources,
   variant, detector, version, latency.

The endpoint is a plain `def`, not `async`, so FastAPI runs the heavy work in a thread pool and
`/health` stays responsive.

**Gaps:** the API loads the `serving/` folder, not "whatever is `@champion`" (plan step B11).
On a laptop with 7.7 GB RAM and no NVIDIA GPU, the real model runs in bfloat16 (~3 GB) at several
seconds per answer: fine for a one-question-at-a-time demo, and by design (the proposal targets
CPU serving). Never shrink or quantise the model just to serve it faster: the probe was trained on
this exact model's hidden states.

### Why not serve with llamafile or Ollama?

[llamafile](https://github.com/mozilla-ai/llamafile) (Mozilla.ai) packs a quantised model and the
llama.cpp engine into one executable that runs fast on an ordinary CPU, with no installs. Ollama
is similar. Neither fits the served system:

- **No hidden states.** Their interface returns the answer text (and token probabilities), not the
  internal vectors at chosen layers that the probe reads. No hidden states means no probe, no bluff
  score and no gate.
- **A different model.** They run quantised weights (e.g. 4-bit), whose numbers differ from the
  model the detectors were trained on: training–serving skew.

The principle: **a guardrail that reads a model's internals ties you to a serving stack that
exposes them.** The project trades some speed for being able to see inside the model. llamafile
is still useful for a **live demo of the raw model bluffing**: it answers instantly on a laptop,
with no GPU and no bluff detector, which is exactly the contrast the system is built to show.

## 9. Docker

- **Image vs container:** an image is a frozen package of code, Python, libraries and the trained
  detector; a container is that image running. It behaves the same on any machine.
- **Multi-stage build:** stage 1 installs everything (with compilers); stage 2 copies only the
  finished environment. Smaller, fewer security holes.
- **Security:** runs as a non-root user (uid 10001); read-only filesystem in compose;
  dependencies installed from the hash-checked lock, so a tampered package fails to install.
- **HEALTHCHECK:** Docker calls `/health` every 30 seconds and marks the container unhealthy if it
  stops answering.
- **The detector is baked into the image**, so the image's unique ID *is* the release.
- **`docker-compose.yml`** runs the API (limited to 2 CPUs and 2 GB) plus an optional MLflow
  server. **`scripts/smoke.sh`** starts the container, waits for `/ready`, asks a question, checks
  the reply and checks that an empty question is rejected with 422.
- **Gap:** the image does not yet hold or cache the real model's weights, and 2 GB is too small for
  them (plan step B11).

## 10. Git, GitHub Actions, and CI / CD / CT

- **Git** records snapshots (commits) of the code; GitHub hosts them; `main` is the official
  branch.
- **GitHub Actions** is GitHub's automation: each YAML file in `.github/workflows/` says "when
  event X happens, start a fresh Linux machine and run these steps".

| | Trigger | Steps | Purpose |
|---|---|---|---|
| **CI** (Continuous Integration), `ci.yml` | Every push or pull request to `main` | Install locked deps → lint (ruff) → check the Kenya data → 44 tests → full pipeline on the fake model → Docker build → smoke test | The **merge gate**: red means don't merge |
| **CD** (Continuous Delivery), `cd.yml` | Every push to `main` | Install → train a detector (fake for now) → build the image tagged version + commit → **Trivy** scans it and fails on critical vulnerabilities → push to **GHCR** (GitHub's image registry) | Every change produces a scanned, versioned image. It publishes the image but runs it nowhere: there is no live server. |
| **CT** (Continuous Training), `ct.yml` | Manual button, pick a variant | On a GPU machine: install → pull data (DVC) → check data → full pipeline → upload reports | Retrain on demand. Never run: it needs a GPU machine registered with GitHub (a *self-hosted runner*). |

Trivy is pinned to a **commit SHA**, not a version tag, because its tags were hijacked in a 2026
supply-chain attack and a SHA cannot be swapped.

## 11. The rest of the MLOps

- **MLflow (experiment tracking):** when switched on (`tracking.mlflow: true`), every training run
  logs its settings, metrics, artefact, git SHA and data hash; registrations are mirrored to
  MLflow's registry. Off by default; needs an MLflow server running.
- **Registry and promotion gates:** numbered versions with movable labels. A challenger needs AUROC
  ≥ the champion's, ECE no worse, and an OOD drop ≤ 0.25. `bluff promote` moves the champion
  label (the human sign-off); **rollback** is promoting the old version again, nothing rebuilt.
- **DVC (Data Version Control):** like git for big data files; stores them by content hash, and
  `dvc repro` reruns only the stages whose inputs changed. Only the M0 stages are defined; no
  remote storage is set up.
- **Reproducibility:** each result is a function of code + data + config + environment + seed.
  Every saved detector records the git SHA, the data's sha256, the config path and the seed (42);
  `requirements.lock` pins the environment.
- **Testing pyramid:** unit (small functions) → data (the Kenya file passes its rules) → API (the
  service's behaviour) → integration (the whole pipeline on the fake model) → promotion gates.
- **Data contract and leakage audit:** `validate` enforces the question rules (unique ids, a
  source, a different reviewer, twins linked to real Kenya questions); `audit` fails the run if a
  question's wording appears in a document collection, so RAG or fine-tuning cannot "cheat".
- **Pre-registration:** the probe's layer and threshold are fixed on trivia validation data
  *before* Kenya is scored, so they cannot be tuned to flatter the results.

## 12. Monitoring and drift

**Exists today:** `/health` (liveness), `/ready` (readiness), `/metadata` (which detector is live
and where it came from), latency in every response, Docker's healthcheck.

**Does not exist yet:** request logging, anything that watches live traffic, drift detection.
Plan step B10 builds a minimal version: the API appends each audit record to a log, and
`bluff monitor` compares live traffic with the training data.

| Kind of drift | What it is here | How to monitor it |
|---|---|---|
| **Data drift** | The questions change, e.g. users start asking about Kenyan taxes when the detector was trained on trivia | Compare the 8 summary features of live answers with the training data: PSI (Population Stability Index) or a KS test |
| **Prediction drift** | The share of answers flagged or refused changes | Track the bluff-score distribution and the abstain rate per day; alert when they move far from the validation baseline |
| **Concept drift / model change** | The language model is swapped or fine-tuned, so the probe reads a different "mind" | The transfer experiment: how far the probe's accuracy falls on a new variant. The system's answer is to retrain and pass the gates. |

The proposal also describes **shadow mode**: a new detector scores live traffic beside the
champion without making decisions, until it is trusted.

## 13. The GPU setup

Only `generate` (and QLoRA training) needs a GPU. Free T4 GPUs come from Colab (run inside VS
Code) and Kaggle (run in the background, watched from VS Code), with a private Hugging Face
dataset repo as the shared drive. Every question is independent, so each GPU takes a *slice* of
the questions; a slice saves progress every ~100 questions and resumes after a timeout; the laptop
merges the slices and runs every other stage on CPU in minutes. All generation runs on the same
GPU type (T4) in float16 with pinned model and library versions, because different hardware gives
slightly different numbers. Step-by-step setup: [gpu-setup.md](gpu-setup.md).

## 14. What is built and what is not

| Built and tested in CI | Written but never run | Not built yet (see [plan.md](plan.md)) |
|---|---|---|
| The pipeline, 4 detectors, all metrics and statistics, the Kenya-vs-UK twin comparison, the API, Docker, CI, CD, registry and gates, config system, data contract, leakage audit | The real-model backend, QLoRA fine-tuning, MLflow tracking, DVC stages, CT | Split GPU runs and resume, the runner notebook, the sheet importer, the document-collection builder, label-noise rate, monitoring and drift, serving the champion, the real-model Docker image, probe transfer, LLM judge, semantic entropy, LSTM |
