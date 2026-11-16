"""Stage 10 - export one variant's records as clearly named datasets, one row per question.

The pipeline's own files are built for the code (JSONL, parquet, compressed arrays indexed by
position). This stage rewrites them for people: CSV files that open in Excel, hidden-state
arrays named by layer and position, an index linking every array row to its question, and a
README (a data card) explaining every column and recording which model, GPU and settings
produced the data.

CRISP-DM: deployment (handing the data over).
In the pipeline: register -> [export]
Reads:  data/processed/questions.jsonl, data/interim/<variant>/generations.jsonl, labels.jsonl,
        hidden.npz, run_info.json, data/processed/<variant>/features.parquet,
        reports/<variant>/scores.parquet
Writes: data/datasets/<variant>/ (CSV files, .npy hidden-state files, README.md)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bluff.config import paths
from bluff.signals import SUMMARY_FEATURES
from bluff.utils import ensure_dir, read_json, read_jsonl

log = logging.getLogger(__name__)

POSITION_NAMES = {"last_answer": "last_word", "mean_answer": "average_of_answer_words"}


def _layer_name(layer: int) -> str:
    """-8 -> '8th_from_top'. Hidden-state index -8 of 29 (word table + 28 layers) is layer 21."""
    n = -layer
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th")
    return f"{n}{suffix}_from_top" if layer < 0 else f"layer{layer}"


def _csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel shows accents right


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    p = paths(cfg)
    vid = cfg["variant"]["id"]
    out = ensure_dir(Path(cfg["paths"]["data_dir"]) / "datasets" / vid)

    questions = {r["id"]: r for r in read_jsonl(p.questions)}
    labels = {r["qid"]: r for r in read_jsonl(p.interim / "labels.jsonl")}
    by_q: dict[str, dict] = defaultdict(dict)
    for g in read_jsonl(p.interim / "generations.jsonl"):
        key = "open" if g["condition"] == "open" else f"offered_{g['template'] + 1}"
        by_q[g["qid"]][key] = g
    qids = [q for q in questions if q in labels and "open" in by_q[q]]

    # 01: questions and the model's answers
    rows = []
    for q in qids:
        it, gens, lab = questions[q], by_q[q], labels[q]
        row = {"id": q, "source": it["pool"], "split": it["split"], "question": it["question"],
               "correct_answer": it.get("gold"), "accepted_answers": " | ".join(it["aliases"]),
               "model_answer": gens["open"]["answer"]}
        votes = lab.get("abstain_votes") or []
        for i in range(1, 4):
            g = gens.get(f"offered_{i}")
            row[f"answer_with_idk_option_{i}"] = g["answer"] if g else None
            row[f"said_idk_{i}"] = int(votes[i - 1]) if i <= len(votes) else None
        rows.append(row)
    _csv(pd.DataFrame(rows), out / "01_questions_and_answers.csv")

    # 02: labels
    _csv(pd.DataFrame([{
        "id": q, "source": questions[q]["pool"], "split": questions[q]["split"],
        "wrong": int(not labels[q]["correct"]), "abstained": int(labels[q]["abstained"]),
        "said_idk_without_option": int(labels[q]["open_abstained"]),
        "marking_method": labels[q].get("method"),
    } for q in qids]), out / "02_labels.csv")

    # 03: the 8 confidence numbers
    feats = pd.read_parquet(p.features / "features.parquet")
    keep = ["qid", "pool", "split", *SUMMARY_FEATURES, "latency_ms"]
    _csv(feats[[c for c in keep if c in feats.columns]]
         .rename(columns={"qid": "id", "pool": "source"}), out / "03_confidence_features.csv")

    # 04: every detector's P(wrong), per question (test questions only have honest scores)
    scores_path = p.reports / "scores.parquet"
    if scores_path.exists():
        sc = pd.read_parquet(scores_path).rename(columns={
            "qid": "id", "pool": "source", "y_wrong": "wrong",
            "open_abstained": "said_idk_without_option", "flag": "probe_flagged_unsure"})
        _csv(sc, out / "04_detector_scores.csv")

    # 05: hidden states, one file per (layer, position), rows in the order of 05_hidden_index.csv
    with np.load(p.interim / "hidden.npz") as f:
        hidden, layers, positions = f["hidden"], list(f["layers"]), list(f["positions"])
    idx = [by_q[q]["open"]["hidden_index"] for q in qids]
    hidden_files = []
    for li, layer in enumerate(layers):
        for pi, pos in enumerate(positions):
            name = f"05_hidden_{_layer_name(int(layer))}_{POSITION_NAMES.get(str(pos), pos)}.npy"
            np.save(out / name, hidden[idx, li, pi, :].astype(np.float16))
            hidden_files.append(name)
    _csv(pd.DataFrame({"row": range(len(qids)), "id": qids,
                       "source": [questions[q]["pool"] for q in qids],
                       "split": [questions[q]["split"] for q in qids]}),
         out / "05_hidden_index.csv")

    info = read_json(p.interim / "run_info.json") if (p.interim / "run_info.json").exists() else {}
    counts = pd.crosstab(pd.Series([questions[q]["pool"] for q in qids], name="source"),
                         pd.Series([questions[q]["split"] for q in qids], name="split"))
    _write_readme(out, vid, info, counts, hidden_files, hidden.shape[-1], len(qids))
    log.info("exported %d questions to %s", len(qids), out)
    return {"variant": vid, "questions": len(qids), "folder": str(out),
            "files": sorted(x.name for x in out.iterdir())}


def _write_readme(out: Path, vid: str, info: dict, counts: pd.DataFrame,
                  hidden_files: list[str], dim: int, n: int) -> None:
    table = "| source | " + " | ".join(counts.columns) + " |\n|---|" + "---|" * len(counts.columns)
    for src, r in counts.iterrows():
        table += f"\n| {src} | " + " | ".join(str(int(v)) for v in r) + " |"
    hidden_list = "\n".join(f"- `{h}`: {n} rows × {dim} numbers" for h in hidden_files)
    run = "\n".join(f"| {k} | {v} |" for k, v in info.items()) or "| (no run record) | |"
    (out / "README.md").write_text(f"""# Bluff Detector dataset: {vid}

One row per question: what the model ({info.get('model', 'unknown')}) answered, whether it was
right, how confident it was, its hidden states, and what each detector predicted. Exported
{date.today().isoformat()} by `bluff export`.

## How it was produced

| Setting | Value |
|---|---|
{run}

## Questions per source and split

{table}

Sources: `trivia` = TriviaQA, `truthfulqa` = TruthfulQA (trick questions), `kenya` = hand-built
Kenya questions, `kenya_unanswerable` = fake Kenya questions with no true answer, `western` =
UK twins. Detectors train on `train`, pick thresholds on `val`, and are scored on `test`;
Kenya and twin questions are always `test`.

## Files

| File | One row per question | Columns |
|---|---|---|
| `01_questions_and_answers.csv` | The question and every answer | `correct_answer`, `accepted_answers` (any of these counts as right), `model_answer` (asked normally), `answer_with_idk_option_1..3` (asked 3 ways with an "I don't know" option), `said_idk_1..3` (1 = it said it didn't know) |
| `02_labels.csv` | The marks | `wrong` (1 = the normal answer was wrong), `abstained` (1 = said "I don't know" in at least 2 of the 3 wordings), `said_idk_without_option`, `marking_method` |
| `03_confidence_features.csv` | The 8 confidence numbers | `mean_logprob`, `min_logprob` (average and lowest log-probability of the answer's words), `mean_entropy`, `max_entropy`, `last_entropy` (how torn it was), `n_tokens` (answer length), `frac_low_prob` (share of words under 0.5 probability), `retrieval_top_score` (RAG only, else 0), `latency_ms` |
| `04_detector_scores.csv` | Every detector's verdict | `p_logreg`, `p_xgboost`, `p_probe_linear`, `p_probe_mlp` = P(wrong) from each detector; `probe_flagged_unsure` (1 = the served probe's P(wrong) is above tau_flag) |
| `05_hidden_*.npy` | The hidden states | See below |
| `05_hidden_index.csv` | Which question each hidden-state row belongs to | `row`, `id`, `source`, `split` |

All files share the `id` column, so they can be joined.

## Hidden states

{hidden_list}

`8th_from_top` is hidden-state index −8 of 29 (the word table plus 28 layers): **layer 21 of 28**
for Qwen2.5-1.5B; `4th_from_top` is layer 25. `last_word` is the model's state at the final word
of its answer; `average_of_answer_words` is the average over all its answer words. The served
probe reads `05_hidden_8th_from_top_last_word.npy`. Load one in Python with
`numpy.load("05_hidden_8th_from_top_last_word.npy")`; row *i* is the question in row *i* of
`05_hidden_index.csv`. Stored as float16.
""", encoding="utf-8")
