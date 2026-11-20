# Where everything is stored

There are three places. **GitHub** (this repo) holds the code, the questions, the trained models
and the results, everything needed to read, check and serve the project. The **Hugging Face
dataset repo** [`vuhyeahhh/bluff-detector-runs`](https://huggingface.co/datasets/vuhyeahhh/bluff-detector-runs)
holds the same results plus the large raw run files that are too big or too many for git. **Kaggle
and Colab** only *ran* the GPU work; nothing is kept there that is not also in one of the other two.

Last checked: 20 November 2026, after the M2 run.

## The short version

| What | GitHub | Hugging Face | Why there |
|---|:---:|:---:|---|
| Code, configs, tests, Docker, CI | ✅ | | It is the project |
| Questions (Kenya set, trivia list) | ✅ | ✅ | Small; the run needs them |
| Retrieval and fine-tuning corpus | ✅ | | Small (text), rebuilt by `bluff build-corpus` |
| Raw model answers and hidden states | | ✅ | 20–80 MB per run; rebuilt only with a GPU |
| Features (8 numbers + hidden states per question) | | ✅ | Derived from the raw files |
| Labelled datasets for people (CSV + .npy) | | ✅ | ~20 MB per run; rebuilt by `bluff export` |
| Trained detectors (4 per run) | ✅ | ✅ | Small (~3 MB per run); needed to serve |
| Model registry (versions, champion) | ✅ | ✅ | Small; the API reads the champion |
| M2 fine-tuning adapter | ✅ | ✅ | 74 MB; in git so M2 runs without a Hugging Face account |
| Metrics, per-question scores, summaries | ✅ | ✅ | The results |
| Charts | ✅ | ✅ (per run) | For the README and report |
| The base model (Qwen2.5-1.5B, ~3 GB) | | Qwen's own repo | Downloaded automatically, pinned to one revision |

## In detail, by stage

### 1. Questions
- `data/kenya/questions.jsonl`: the final 326 Kenya questions (211 real, 82 fake, 33 UK
  twins). How they were made and cleaned: [kenya-questions.md](kenya-questions.md).
- `data/kenya/cleaned/cleaned_questions.xlsx`, `cleaned/changes.csv`, `cleaning_report.md`, `import_report.md`,
  `repairs.yaml`, `student_files.yaml`: the cleaning trail.
- The students' original spreadsheets are **outside the repo**, in the folder above it
  (`Bluff_Detector/*.xlsx`). They are the raw source; `student_files.yaml` names each one.
- TriviaQA questions are downloaded from Hugging Face (`mandarjoshi/trivia_qa`) by `bluff ingest`;
  the questions actually used are saved in `data/processed/questions.jsonl` below.
- `data/processed/questions.jsonl`: all questions after ingest, with their train/val/test split
  (Hugging Face).

### 2. Corpus for RAG (M1) and fine-tuning (M2)
- `data/corpus/sources.yaml`: which documents and why. `retrieval_chunks.jsonl` (441 passages for
  M1), `finetune_docs.jsonl` (26 documents for M2), `manifest.json`. All in GitHub.
- `data/processed/leakage_audit.json`: proof that no fine-tuning document is used for retrieval
  or answers a test question (Hugging Face).

### 3. Runs (one folder per variant: `m0_base`, `m1_rag`, `m2_qlora`)
| Folder | Contents | Where |
|---|---|---|
| `data/interim/<variant>/` | `generations.jsonl` (4 answers per question), `labels.jsonl` (right/wrong, abstained), `hidden.npz` (hidden states), `run_info.json` (GPU, versions), `review_sample.jsonl` | Hugging Face |
| `data/processed/<variant>/` | `features.parquet`, `hidden.npy` | Hugging Face |
| `data/datasets/<variant>/` | the same records for people: 5 CSV files, 4 `.npy` files named by layer, and a README data card | Hugging Face |
| `artefacts/<variant>/` | the four detectors (`logreg`, `xgboost`, `probe_linear`, `probe_mlp`) and `serving/` (the one the API uses), each with `metadata.json` | GitHub + Hugging Face |
| `reports/<variant>/` | `metrics.json`, `scores.parquet`, `summary.md`, `figures/` (13 charts) | GitHub + Hugging Face |

`m0_smoke` and `dev_fake` are test runs (a 30-question smoke test and the fake model). They are
on Hugging Face and in the registry for the record but are not results.

### 4. Models
- `artefacts/registry/registry.json` + `artefacts/registry/bluff-detector-<variant>/v<N>/`: every
  registered detector version, its metrics, the gates it passed and which version is
  **@champion**. Now: M0 v2, M1 v1 (v2 is the @challenger), M2 v1.
- `artefacts/adapters/m2_qlora/`: the M2 adapter, its settings and its training log. Its
  [README](../artefacts/adapters/m2_qlora/README.md) is the model card.

### 5. Comparisons and final results
- `reports/compare/m0_base__vs__m1_rag.json`, `m0_base__vs__m2_qlora.json`: the paired tests
  (`bluff compare`).
- `reports/results_figures/`: the five headline charts in the README.
- `notebooks/results.ipynb`: every hypothesis, its numbers and its chart, the notebook to
  demo.
- `notebooks/analysis.ipynb` (M0), `analysis_m1_rag.ipynb`, `analysis_m2_qlora.ipynb`: the
  detailed look at one run each (data, detectors, curves, examples).

### 6. Live system
- `app/`: the API and the demo page (`app/static/index.html`).
- `logs/requests.jsonl`: the live request log the drift monitor reads. Local only (git ignores
  it); `bluff monitor` writes `reports/<variant>/monitoring/`.

## Getting the Hugging Face files back

From inside `bluff-detector`:

```
hf download vuhyeahhh/bluff-detector-runs --repo-type dataset --local-dir . --exclude .gitattributes
```

This puts every file in the same folder it came from, so the notebooks and `bluff` commands find
them.

## What Kaggle and Colab keep

Nothing that matters. Each run notebook pushes its results to Hugging Face (and the code to GitHub)
at the end. A Kaggle "version" also keeps a copy of `/kaggle/working`, which is why a finished
Kaggle session can sit at "Running" for a long time while it saves; it can be stopped safely once
the push to Hugging Face is done.
