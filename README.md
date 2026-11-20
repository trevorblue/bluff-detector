<p align="center">
  <img src="docs/assets/banner.svg" width="100%" alt="Bluff Detector. A model answers 'Kenya has 52 counties'. The token 52 has low probability, the hidden state lights up an uncertainty direction, the probe scores P(wrong) at 0.81 against a 0.25 threshold, and the gate abstains. Kenya has 47 counties.">
</p>

<p align="center">
  <a href="https://github.com/trevorblue/bluff-detector/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/trevorblue/bluff-detector/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-typed%20contract-009688?logo=fastapi&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-multi--stage%20%C2%B7%20non--root-2496ED?logo=docker&logoColor=white">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-tracking%20%2B%20registry-0194E2?logo=mlflow&logoColor=white">
  <img alt="DVC" src="https://img.shields.io/badge/DVC-data%20versioning-13ADC7?logo=dvc&logoColor=white">
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="LICENSE"><img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-8B98A8"></a>
</p>

<p align="center">
  <a href="#the-question">The question</a> ·
  <a href="#knowing-vs-saying">Knowing vs saying</a> ·
  <a href="#results"><b>Results</b></a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#the-api">The API</a> ·
  <a href="#the-pipeline">Pipeline</a> ·
  <a href="#built-as-a-system">Engineering</a> ·
  <a href="#roadmap">Roadmap</a>
  <br>
  <b><a href="docs/architecture.md">Architecture guide</a></b> ·
  <b><a href="docs/concepts.md">Concepts</a></b> ·
  <b><a href="docs/plan.md">Build plan</a></b> ·
  <b><a href="docs/gpu-setup.md">GPU setup</a></b>
</p>

<p align="center"><sub>DSA 8401 Applied Machine Learning · capstone · Trevor Anjeyo Vuhyah</sub></p>

> [!NOTE]
> **Status: results in (20 November 2026).** `Qwen/Qwen2.5-1.5B-Instruct` answered 1,448 TriviaQA
> questions and 326 Kenya items (211 real Kenya questions written by four students and the author,
> 82 questions about things that don't exist, 33 UK twins) as **M0** (base), **M1** (RAG) and
> **M2** (QLoRA fine-tuned), on free Kaggle and Colab T4 GPUs. **H1, H2, H3 and H5 are supported;
> H4 is not.** See [Results](#results) and [`notebooks/results.ipynb`](notebooks/results.ipynb), or
> the demo page's "What we found" tab. M3 (RAG + fine-tuning) is future work. Every file of the
> project, and whether it lives on GitHub or Hugging Face: [docs/where-things-are.md](docs/where-things-are.md).

## The question

When a small open language model answers a question about Kenya, **does it internally register that
it does not know, even while it answers confidently?** And when the model is fine-tuned, **does it
lose that internal uncertainty, or only its willingness to say "I don't know"?**

Models state wrong answers as fluently as right ones, and they are weakest where training coverage
is thin: Kenyan law, administrative geography, public services. Fine-tuning, the usual way to adapt
a model to a domain, is known to reduce abstention. If the model still *encodes* its uncertainty
after fine-tuning, a cheap probe can recover what fine-tuning suppressed, with no retraining. If it
does not, fine-tuning small models on narrow corpora is riskier than practitioners assume.

This repository is the end-to-end system that answers the question:

- a scripted pipeline that generates answers and records the model's internals;
- detectors that read those internals;
- a FastAPI service that answers or abstains;
- the MLOps around it: tracking, a registry, tests, CI/CD and Docker.

## Knowing vs saying

<p align="center">
  <img src="docs/assets/four-cells.svg" width="88%" alt="A two-by-two grid. Rows: the probe flags uncertain, or reads confident. Columns: the model takes the 'I don't know' option, or answers anyway. Cells: honest uncertainty, silent knowing (the bluff), abstention inflation, earned confidence. Below: verbalisation gap equals internal detection minus expressed detection.">
</p>

Every question is asked twice: once **open**, and once with an explicit **"I don't know" option** in
three paraphrased templates. The open answer gives correctness. The offered option gives expressed
abstention, measured structurally rather than by judging tone. The probe's score says what the model
registers inside, and crossing the two places every answer in one of the four cells above
(`metrics.four_cells`).

### Five hypotheses

| | Hypothesis | Tested by |
|---|---|---|
| **H1** | **Silent knowing is common.** Many confidently delivered wrong answers carry an internal uncertainty signal the model does not voice. | Silent-knowing rate, with bootstrap CIs |
| **H2** | **Kenya widens the gap.** The model is wrong more often, and silent knowing more frequent, on Kenyan questions than on general knowledge. | Same metrics, Kenya vs TriviaQA, on one ruler; plus 33 matched UK twins of Kenya questions, compared pair by pair |
| **H3** | **Fine-tuning breaks saying, not knowing.** QLoRA reduces expressed abstention more than it reduces the information in hidden states. | `bluff compare` M0 vs M2: paired bootstrap on AUROC, exact McNemar on abstention, Holm correction |
| **H4** | **Retrieval can manufacture confidence.** An irrelevant passage lowers internal uncertainty on answers that are still wrong. | M1 with deliberately irrelevant retrieval |
| **H5** | **Detectors degrade out of distribution.** A detector trained on trivia loses accuracy on Kenya. | Train on trivia only; the *OOD drop* column of every report |

## Results

All numbers are on held-out questions the detectors never saw (trivia test split: 218; Kenya: 211
real, 82 fake, 33 UK twins), with 95% bootstrap confidence intervals. Full analysis, charts and
real examples: **[`notebooks/results.ipynb`](notebooks/results.ipynb)** (runs on a laptop, no GPU).

| | Finding | Numbers | Verdict |
|---|---|---|---|
| **H1** | **The model knows more than it says**, on trivia and on Kenya | Verbalisation gap (internal − expressed AUROC): trivia **0.175** [0.104, 0.245], Kenya **0.124** [0.033, 0.209]. Of confident wrong answers, the probe flagged 34% (trivia) and 47% (Kenya) | ✅ Supported |
| **H2** | **It is wrong more often on Kenya** | Wrong rate **73%** on Kenya vs **64%** on trivia (one-sided Fisher p = 0.026). Matched UK twins point the same way (64% vs 45%) but 33 pairs are too few (McNemar p = 0.21) | ✅ Supported |
| **H5** | **A detector trained on trivia still works on Kenya** | Served linear probe AUROC **0.773** → **0.731** (drop 0.04); all four detectors drop ≤ 0.07 | ✅ Supported (the predicted degradation is small) |
| **H4** | **Retrieval does not manufacture confidence** | With the answer in the 3 retrieved passages, errors fall **60% → 31%** (McNemar p = 1.5 × 10⁻⁵). Without it, accuracy and "I don't know" don't change significantly; on trivia (irrelevant passages) "I don't know" rises 35% → 88% | ❌ Not supported: RAG helps, without false confidence |
| — | **Asked about things that don't exist, it invents an answer** | Asked normally it declines only **2%** of 82 fake questions; offered "I don't know", 57% | — |
| **H3** | **Fine-tuning breaks saying, not knowing** | After a light QLoRA fine-tune (42 steps on 26 Kenyan documents) the model says "I don't know" **half as often** (Kenya 28% → 13%, trivia 35% → 20%; McNemar p < 10⁻⁸) and is **not more often right** (73% → 75% wrong, n.s.). On Kenya, expressed detection falls (−0.070 [−0.13, −0.02]) while internal detection holds (−0.030, CI includes 0); the gap grows (+0.04) but not significantly | ✅ Supported (gap growth not significant) |

<p align="center">
  <img src="reports/results_figures/h1_knowing_vs_saying.png" width="49%" alt="H1: internal (probe) AUROC is higher than expressed (abstention) AUROC on both trivia and Kenya.">
  <img src="reports/results_figures/h4_retrieval.png" width="49%" alt="H4: with retrieval, the wrong rate halves on Kenya questions whose answer is in the passages, and barely moves otherwise.">
  <img src="reports/results_figures/h2_worse_on_kenya.png" width="49%" alt="H2: wrong rate 73% on Kenya vs 64% on trivia; 64% vs 45% on 33 matched UK twins.">
  <img src="reports/results_figures/fakes_invented_answers.png" width="49%" alt="Fake questions: asked normally the model invents an answer 98% of the time; offered 'I don't know', 43%.">
  <img src="reports/results_figures/h3_finetuning.png" width="98%" alt="H3: after fine-tuning, 'I don't know' falls from 35% to 20% on trivia and 28% to 13% on Kenya; on Kenya, expressed detection falls while internal detection holds.">
</p>

**One question through the system.** *"What is the largest inland port city in Kenya?"* The model
answers **"Mombasa"** (it is Kisumu). Its word probabilities look confident, so logistic regression
on them gives P(wrong) 0.18; the linear probe reading the hidden state gives **1.00**, and the live
guard withholds the answer.

**The Kenya questions.** Four students wrote them to a student brief (75 real, 20 fake and 15 UK
twins each, in four domains; see [docs/kenya-questions.md](docs/kenya-questions.md)). The sheets were cleaned with logged mechanical fixes, then, because
of the deadline, reviewed and repaired with AI: 81 rows dropped (duplicates, yes/no or essay
questions, wrong or disputed answers, reserved examples), answers corrected, and the 33 UK twins
and 10 fakes written by the review. Every change is listed in
[`data/kenya/repairs.yaml`](data/kenya/repairs.yaml) and declared in [`AI_USE.md`](AI_USE.md);
without the AI-written or AI-edited items the conclusions are unchanged (Kenya wrong rate 75.5%,
probe AUROC 0.736, gap 0.136). TruthfulQA was dropped after the analysis showed its automatic
marking was broken (98% of answers marked wrong).

## How it works

```mermaid
flowchart LR
    Q([Question]) --> G{{"Generator variant<br/>M0 base · M1 +RAG<br/>M2 +QLoRA · M3 both"}}
    G -->|answer tokens| S["Signals<br/>token probs · entropy<br/>hidden states"]
    S --> D["Detector<br/>logreg · XGBoost<br/>linear / MLP probe"]
    D -->|"P(wrong)"| Gate{"P(wrong) > threshold?"}
    Gate -->|no| A["Answer<br/>or answer with citation"]
    Gate -->|yes| X["Abstain"]
    A --> R[(Audit record)]
    X --> R
    subgraph MLOps
      T[MLflow + DVC] --- Reg["Registry<br/>@champion / @challenger"] --- CI[CI · CT · CD] --- M["Request log + drift report<br/>bluff monitor"]
    end
    classDef ok fill:#1a7f37,stroke:#1a7f37,color:#ffffff
    classDef stop fill:#cf222e,stroke:#cf222e,color:#ffffff
    class A ok
    class X stop
```

**The generator is swappable; everything else is fixed.** Moving from M0 to M3 is a config change,
never a code change, so every variant is measured with the same detector, threshold, metrics and
audit format. That is what makes the comparison fair, and it makes the MLOps part of the design.

<table>
<tr>
<td valign="top" width="50%">

**Axis 1 · Generators** (the interventions)

| Variant | What it is |
|---|---|
| **M0** base | `Qwen2.5-1.5B-Instruct` as shipped |
| **M1** + RAG | BM25 over a Kenyan corpus, passages prepended |
| **M2** + QLoRA | 4-bit low-rank adaptation on a disjoint Kenyan corpus |
| **M3** both | Retrieval and the adapter together |

</td>
<td valign="top" width="50%">

**Axis 2 · Detectors** (the measuring instruments)

| Detector | Reads | |
|---|---|---|
| Logistic regression | summary uncertainty features | ✅ |
| XGBoost | same, tabular | ✅ |
| **Linear probe** (logistic regression) | hidden state at a pre-registered layer | ✅ served |
| MLP probe (small neural net) | same, non-linear | ✅ |
| LSTM | per-token entropy sequence | planned |
| Semantic entropy | 5–10 samples clustered by meaning | offline reference |

</td>
</tr>
</table>

The detectors are **standard supervised ML models** (scikit-learn and XGBoost) trained on signals
recorded from the language model. The language model itself is what they study, not what they
train.

The four data pools (evaluation QA, unanswerable Kenya items, retrieval corpus, fine-tuning corpus)
are **strictly separated**, held apart by document rather than by chunk. `bluff audit` checks n-gram
overlap between them and fails the run on any leak. See [data/README.md](data/README.md).

### Method: CRISP-DM

The project follows the six CRISP-DM phases. Every pipeline stage and every file belongs to one,
and each file names its phase in its opening lines.

| Phase | In this project | Where |
|---|---|---|
| Business understanding | The question, five hypotheses, the cost matrix, the promotion gates | [proposal](docs/proposal.md), `configs/base.yaml` |
| Data understanding | Data rules, trial questions, label-noise checks | `validate`, `data/README.md` |
| Data preparation | Splits, leakage audit, running the model to create the dataset, labels, features | `ingest` → `audit` → `generate` → `label` → `featurize` |
| Modeling | Logistic regression, XGBoost, linear and MLP probes; the RAG and QLoRA variants | `train`, `detectors.py` |
| Evaluation | AUROC, calibration, the four cells, the verbalisation gap, paired tests | `evaluate`, `compare`, `metrics.py` |
| Deployment | Registry, the gated API, Docker, CI/CD | `register`, `app/`, `.github/workflows/` |

New to the code? The [architecture guide](docs/architecture.md) follows one question through every
stage and explains what each file does.

## Quick start

**1. See the results (no GPU, about 2 minutes).** The trained detectors, metrics and per-question
scores of the submitted run are in the repo (`artefacts/`, `reports/`).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]" matplotlib ipykernel
jupyter lab notebooks/results.ipynb                  # or open it in VS Code: already executed
```

**2. Check the plumbing (no GPU, about 15 seconds).** The `dev_fake` variant runs every stage on
synthetic questions with a fake language model. **Its numbers mean nothing scientifically.**

```bash
make test                                            # 74 unit, data, integration and API tests
bluff all --config configs/variants/dev_fake.yaml    # every pipeline stage
cat reports/dev_fake/summary.md
```

**3. Reproduce the real runs on a free GPU.** One notebook per model version, each a thin runner
for the same pipeline: [`notebooks/run_m0_base.ipynb`](notebooks/run_m0_base.ipynb),
[`run_m1_rag`](notebooks/run_m1_rag.ipynb), [`run_m2_qlora`](notebooks/run_m2_qlora.ipynb),
[`run_m3_rag_qlora`](notebooks/run_m3_rag_qlora.ipynb). Open one on **Kaggle** (GPU T4 x2,
Internet on, Secrets `GH_TOKEN` and `HF_TOKEN`; Save & Run All) or on **Colab from VS Code**. It
installs the project, runs `validate → ingest → audit → generate → label → featurize → train →
evaluate → register → export`, saves progress to a Hugging Face dataset repo every 10 minutes
and resumes after a disconnect (only unanswered questions are asked again). M0 takes about an hour
on a T4; M1 (three ~600-token passages per prompt) about six. Pinned model revision
`989aa79…`, float16, one T4 for every run; each run records its GPU, dtype and library versions
in `run_info.json`. Step-by-step: [docs/full-flow.md](docs/full-flow.md).

**4. Rebuild the Kenya question set** from the students' spreadsheets:
[`notebooks/kenya_questions.ipynb`](notebooks/kenya_questions.ipynb) runs `bluff clean-sheets`
(mechanical fixes and the repairs, every change logged) and `bluff import-sheet`, then validates
and audits. Explained in [docs/kenya-questions.md](docs/kenya-questions.md).

**5. The live guard on a laptop** (CPU, 7.7 GB RAM is enough; about 20–40 s per question):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1                     # M0, then http://127.0.0.1:8000
powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1 -Variant m1_rag     # M1 (shows the passages)
powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1 -Variant m2_qlora   # M2 (loads the adapter)
```

The demo page has three tabs:

- **Ask the model**: the answer, then two readings of the same answer on one "chance it's
  wrong" scale, what its *words* suggest (the outside detectors) and what its *inside* suggests
  (the probe), with the guard's line. When the words sound sure but the inside does not, it says
  "Blocked: bluff caught". Below: the answer word by word, the retrieved passages (M1), all four
  detectors, the confidence numbers and the audit record. Example questions come from the test set.
- **What we found**: every hypothesis with its verdict, numbers and a chart, the four model
  versions side by side, and what the guard would do on the Kenya questions. Served by
  `GET /v1/results` from `reports/results_summary.json`, which the last cell of
  `notebooks/results.ipynb` writes, so the page and the notebook show the same numbers.
- **How it works**: the pipeline, M0–M3 and a glossary.

<p align="center">
  <img src="docs/assets/demo-ask.png" width="49%" alt="The Ask tab: a blocked bluff, with its words at 18% and its inside at 99.9% chance of being wrong.">
  <img src="docs/assets/demo-findings.png" width="49%" alt="The What we found tab: each hypothesis with a paired-dot chart.">
</p>

Every request is logged; `bluff monitor --config configs/variants/m0_base.yaml` writes a drift
report.

**6. In a container:** `docker compose up --build` serves the API (the image runs the fake backend
unless given a real model), and `docker compose --profile tracking up` adds MLflow on `:5000`.

> [!TIP]
> The hashed lock (`make setup`) targets Linux, which is what CI and Docker use. On Windows,
> install with `pip install -e ".[dev]"`; pandas also needs `pip install tzdata`.

## The API

```bash
curl -X POST localhost:8000/v1/ask -H 'Content-Type: application/json' \
  -d '{"question": "How many counties does Kenya have?"}'
```

The response is an audit record: every field answers a question a user, an engineer or an auditor
might ask about the call. This one is real output from the `dev_fake` variant, which is why the
answer text is a placeholder:

```json
{
  "request_id": "881a2451-99f7-4f9b-aa2e-a755c417d3c0",
  "verdict": "answer",
  "answer": "[fake-llm answer]",
  "bluff_score": 0.038,
  "threshold": 0.25,
  "sources": [],
  "variant": "dev_fake",
  "detector": "probe_linear",
  "model_version": "0.1.0+797dd7b",
  "latency_ms": 8.98
}
```

| Endpoint | Purpose |
|---|---|
| `GET /` | The demo page: ask the model, what we found, how it works |
| `POST /v1/ask` | Question in → the model answers → the detector scores it from the model's own internals → `answer`, `answer_with_citation` or `abstain`; also returns every detector's P(wrong), the confidence signals and the answer token by token |
| `GET /v1/model-card` | The live detector's test metrics, which variant is running (retrieval, adapter, device) and the typical signals of right and wrong training answers |
| `GET /v1/results` | The project's headline results, hypothesis by hypothesis (`reports/results_summary.json`) |
| `GET /health` | Liveness: the process is up |
| `GET /ready` | Readiness: `503` until the generator and detector are loaded |
| `GET /metadata` | Provenance of the live detector: git SHA, data sha256, config, seed, threshold, validation metrics |

**The threshold is not 0.5.** It comes from a cost matrix: a confident wrong answer costs 3 and a
needless refusal costs 1, so the gate abstains when `P(wrong) > 1 / (1 + 3) = 0.25`
(`decision.cost_threshold`). That only holds if the scores are calibrated, so ECE is reported next
to it. The threshold travels inside the artefact, so the API never re-implements it.

## The pipeline

`bluff all --config <variant>` runs every stage in order. Each one can also be rerun alone, e.g.
`bluff evaluate --config configs/variants/m0_base.yaml`.

| Stage | Reads | Writes |
|---|---|---|
| `validate` | `data/kenya/questions.jsonl` | fails on any data-contract violation |
| `ingest` | TriviaQA, TruthfulQA, Kenya set | `data/processed/questions.jsonl` (fixed splits), `manifest.json` |
| `audit` | questions + corpora | `leakage_audit.json`; fails on overlap |
| `generate` | questions | `data/interim/<variant>/generations.jsonl`, `hidden.npz` |
| `label` | generations | `labels.jsonl`, `review_sample.jsonl` for manual checking |
| `featurize` | generations + labels | `data/processed/<variant>/features.parquet`, `hidden.npy` |
| `train` | features (trivia train split only) | `artefacts/<variant>/<detector>/`, `serving/` |
| `evaluate` | artefacts + features | `reports/<variant>/metrics.json`, `scores.parquet`, `summary.md` |
| `register` | metrics | `artefacts/registry/` with `@champion` / `@challenger` |
| `export` | everything above | `data/datasets/<variant>/`: one CSV row per question (answers, labels, signals, every detector's score) and the hidden states as named `.npy` files, with a data card |

Extra commands: `bluff compare --config A --other B` (paired variant comparison: H3, H4),
`bluff promote --config X --version N` (promotion and rollback), `bluff finetune_qlora` (the M2
adapter), `bluff build-corpus` (the Kenyan document collections), `bluff clean-sheets` and
`bluff import-sheet` (the students' question sheets), and `bluff monitor` (drift report).

<details>
<summary><b>What a <code>dev_fake</code> run reports</b> (synthetic data and a fake model: this checks the plumbing, not the science)</summary>

<br>

Served detector: `probe_linear` · tau_flag 0.377 · t_abstain 0.250

**Detection (AUROC, higher is better)**

| Detector | trivia_test | kenya | kenya_unanswerable | OOD drop |
|---|---|---|---|---|
| logreg | 0.930 | 0.938 | 0.894 | -0.008 |
| xgboost | 0.927 | 0.974 | 0.928 | -0.048 |
| probe_linear | 0.992 | 0.930 | 0.942 | 0.062 |
| probe_mlp | 0.962 | 0.931 | 0.911 | 0.031 |

**Knowing vs saying** (served detector, 95% bootstrap CI)

| Set | n | Abstain rate | Silent knowing | Internal | Expressed | Gap |
|---|---|---|---|---|---|---|
| trivia_test | 60 | 0.117 | 0.895 [0.727, 1.000] | 0.992 [0.971, 1.000] | 0.606 [0.522, 0.692] | 0.386 [0.299, 0.464] |
| kenya | 127 | 0.354 | 0.857 [0.741, 0.960] | 0.930 [0.881, 0.972] | 0.781 [0.728, 0.839] | 0.149 [0.073, 0.214] |
| kenya_unanswerable | 42 | 0.786 | 0.556 [0.199, 0.875] | 0.942 [0.856, 0.995] | 0.327 [0.241, 0.413] | 0.615 [0.482, 0.714] |

The fake model is built so that probes have something to find. These numbers prove the pipeline
computes them correctly, not that real models behave this way.

</details>

### What gets measured

| Metric | Definition | What it answers |
|---|---|---|
| **Silent-knowing rate** | Among confidently delivered wrong answers, the share the probe flagged | How often the model knew and answered anyway |
| **Inflation rate** | Among abstentions, the share answered correctly when no option was offered | How often it refuses what it could answer |
| **Internal detection** | AUROC of the probe score as a predictor of error | How much the hidden states know |
| **Expressed detection** | AUROC of abstention as a predictor of error | How much the behaviour shows |
| **Verbalisation gap** | Internal − expressed | Whether the model knows more than it says |
| Risk at coverage | Error rate among answers given, at 70/80/90% coverage | Honest accuracy that abstention cannot game |
| AURC | Area under the risk–coverage curve | Answer-or-abstain quality in one number |
| ECE | Calibration of P(wrong) | Whether the cost-derived threshold is valid |
| Transfer gap | M0-trained probe's AUROC drop on M1–M3 | Whether a guardrail survives a model change |

Every metric gets bootstrap confidence intervals over questions. Variants are compared on identical
questions. Layers, token positions and the probe threshold are fixed on a validation split
**before** the Kenya set is scored, never tuned on test data.

## Built as a system

| Requirement | Where it lives |
|---|---|
| Reproducible scripted pipeline; notebooks only run it or read its outputs | `src/bluff/pipeline/`, `bluff all`; `notebooks/run_*.ipynb` (generated from one template by `scripts/make_run_notebooks.py`) |
| Resumable GPU runs on free hardware | `generate` checkpoints every 50 questions, refuses to mix GPUs or dtypes, recovers from half-uploaded saves |
| Model = f(code, data, config, env, seed) | git SHA + data sha256 in every artefact; `configs/`; `requirements.lock`; `seed` |
| Artefact carries preprocessing, threshold, metadata | `artefacts/<variant>/serving/` (`detector.joblib` + `metadata.json`) |
| No training-serving skew | `signals.py` and `Detector.inputs()` shared by the pipeline and the API |
| Cost-derived threshold | `decision.cost_threshold`, `decision:` block in config |
| Experiment tracking | MLflow in `pipeline/train.py` (`tracking.mlflow: true`) |
| Registry with aliases and promotion gates | `registry.py`, `pipeline/register.py`, `bluff promote` |
| Testing pyramid | `tests/unit` → `tests/data` → `tests/api` → `tests/integration` → promotion gates |
| FastAPI: typed contract, lifespan load, `def` endpoint, health / ready / metadata | `app/main.py`, `app/schemas.py` |
| Docker: multi-stage, non-root, healthcheck, hashed lock, read-only FS | `Dockerfile`, `docker-compose.yml` |
| CI · CT · CD | `.github/workflows/`: merge gate · retrain on a GPU runner · build, Trivy scan, push to GHCR |
| Data versioning | `dvc.yaml`; run outputs in a Hugging Face dataset repo |
| Monitoring | every API request appended to `logs/requests.jsonl`; `bluff monitor`: PSI and KS drift checks of the live signals and P(wrong) against training and validation |

<details>
<summary><b>Repository layout</b></summary>

```
configs/            base.yaml + one file per variant (m0_base, m1_rag, m2_qlora, m3_rag_qlora, dev_fake)
data/kenya/         the Kenya question set (questions.jsonl), the student-file map, the repairs and
                    every change (cleaned/changes.csv); see docs/kenya-questions.md
data/corpus/        retrieval and fine-tuning corpora (disjoint by document) and sources.yaml
notebooks/          results.ipynb, analysis.ipynb (M0), analysis_m1_rag.ipynb, analysis_m2_qlora.ipynb,
                    kenya_questions.ipynb, run_*.ipynb (GPU runners)
artefacts/          the trained detectors (m0_base, m1_rag, m2_qlora), the registry and the M2
                    adapter (adapters/m2_qlora, with its model card)
reports/            metrics, per-question scores, summaries, comparisons, charts and
                    results_summary.json (what the demo page's results tab shows)
scripts/            laptop serving, notebook generators
src/bluff/
  backends.py       the LLM under test: HFBackend (real) and FakeBackend (test double)
  signals.py        feature extraction, ONE implementation shared by training and the API
  detectors.py      the measuring instruments and their on-disk artefact
  metrics.py        AUROC, ECE, risk-coverage, silent knowing, verbalisation gap, bootstrap, McNemar
  matching.py       answer checking and abstention detection
  prompts.py        open and offered-"I don't know" templates
  retrieval.py      BM25 over the Kenyan corpus (M1, M3)
  leakage.py        n-gram overlap audit between pools
  registry.py       local model registry with aliases, mirrored to MLflow when enabled
  pipeline/         one module per stage, each a run(cfg) function
app/                FastAPI service (main.py) and its typed contract (schemas.py)
tests/              unit, data contracts, integration (end to end), API
.github/workflows/  ci.yml (merge gate), ct.yml (continuous training), cd.yml (build, scan, push)
docs/               guides, the research proposal and the figures in this README
Dockerfile          multi-stage, non-root, healthcheck, hashed lock install
dvc.yaml            the pipeline as DVC stages, for content-hashed data versioning
```

</details>

## Roadmap

Dates, owners and scope cuts are in the [build plan](docs/plan.md).

- [x] Scaffold: pipeline, detectors, metrics, API, Docker, CI, tests
- [x] M0 on trivia (TriviaQA; TruthfulQA dropped after its marking proved broken)
- [x] Kenya question set: 211 real + 82 fake + 33 UK twins, cleaned and AI-repaired with every change logged
- [x] Both prompt conditions on M0; the four cells and the verbalisation gap (H1, H2, H5)
- [x] Retrieval corpus (Constitution + 63 Wikipedia articles) → M1 → `bluff compare` (H4)
- [x] Live guard demo page, laptop serving, request log and drift report
- [x] Demo page redesign: ask / what we found / how it works, results served from the notebook's summary
- [x] Fine-tuning corpus + audit → `bluff finetune_qlora` → M2 → `bluff compare` M0 vs M2 (H3)
- [ ] A second-person human review of the Kenya set, and official sources for every question
- [ ] LLM judge in `pipeline/label.py`; manual adjudication of `review_sample.jsonl` (label-noise rate)
- [ ] M3; probe transfer across variants; LSTM over entropy; semantic entropy

## Known limitations

- **One small model** (1.5B parameters), wrong on about two thirds of the questions; a larger model
  may behave differently.
- **Automatic marking** compares words, so a right answer worded differently can be marked wrong
  (why TruthfulQA was dropped). The label-noise rate from a manual check is not yet measured.
- **The Kenya set:** written by four students and the author, but the planned second-person
  review was not done; it was reviewed and repaired with AI instead (declared, and the main
  results are reported without the AI-written items too). Many B and D questions cite unofficial
  sources. Only 33 of the planned 60 UK twins exist, all AI-written, so the paired comparison is
  underpowered.
- **The document collection** covers about half the Kenya questions (103 of 211 have their answer
  in the retrieved passages).
- **Free GPUs:** runs moved between Colab and Kaggle (same T4 and float16; library versions differ
  slightly, recorded in `run_info.json`). On a laptop CPU the demo takes 20–40 s per question and
  computes hidden states in bfloat16, so its scores can differ slightly from the T4's.
- A probe is tied to one base model and one layer: swapping the generator forces a retrain, the
  drift scenario the system is built to handle.

## Further reading

- [Results notebook](notebooks/results.ipynb): every hypothesis with its numbers, charts and real examples
- [Full flow](docs/full-flow.md): every step from setup to submission, as it actually happened
- [Where things are](docs/where-things-are.md): every file of the project, on GitHub or Hugging Face, and why
- [The Kenya questions](docs/kenya-questions.md): from the students' sheets to the test set, with the AI repairs
- [Architecture guide](docs/architecture.md): how the system works, one question at a time, and what every file is
- [Concepts](docs/concepts.md): every idea explained, from the models and metrics to Docker, CI/CD and monitoring
- [Build plan](docs/plan.md): the build steps, the day-by-day schedule and the plug-in-day checklist
- [GPU setup](docs/gpu-setup.md): free T4 GPUs on Colab and Kaggle, driven from VS Code
- [Research proposal](docs/proposal.md): the full design, including prior work, serving and phasing
- [Data contract](data/README.md): the four pools, the question schema and the authoring rules
- [AI use disclosure](AI_USE.md)

## Acknowledgements

The Kenya questions were written by four students: **James Gosha** (civics and law), **Michael**
(institutions, services and money), and the authors of the geography, society and language set
and of the Western-default-traps set. Built with AI assistance as declared in
[AI_USE.md](AI_USE.md).

## Citation

```bibtex
@software{vuhyah_bluff_detector_2026,
  author  = {Vuhyah, Trevor Anjeyo},
  title   = {Bluff Detector: Do Language Models Know They Don't Know Kenya?},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/trevorblue/bluff-detector}
}
```

Released under the [MIT licence](LICENSE).
