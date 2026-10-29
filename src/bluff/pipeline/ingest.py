"""Stage 1 - assemble the evaluation question set and fix the splits.

Trivia-style pools are split into train/val/test. Kenya pools are test-only: detectors
never see a Kenyan question during training, which is what makes it an OOD test.
The output is shared by every variant, so all variants are scored on the same questions.

CRISP-DM: data preparation.
In the pipeline: validate -> [ingest] -> audit
Reads:  TriviaQA + TruthfulQA (Hugging Face) or synthetic questions,
        data/kenya/questions.jsonl
Writes: data/processed/questions.jsonl, data/processed/manifest.json
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bluff import synthetic
from bluff.config import paths
from bluff.pipeline.validate import load_kenya
from bluff.schema import OOD_POOLS, QuestionItem, validate_questions
from bluff.utils import file_sha256, write_json, write_jsonl

log = logging.getLogger(__name__)


def load_triviaqa(n: int, seed: int) -> list[QuestionItem]:
    """TriviaQA (rc.nocontext, validation split), sampled. Needs `pip install .[data]`."""
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    out = []
    for i in idx:
        row = ds[int(i)]
        ans = row["answer"]
        out.append(
            QuestionItem(
                id=f"tqa-{row['question_id']}",
                pool="trivia",
                question=row["question"],
                gold=ans["value"],
                aliases=sorted(set(ans["aliases"]) | set(ans.get("normalized_aliases", []))),
            )
        )
    return out


def load_truthfulqa() -> list[QuestionItem]:
    """TruthfulQA generation split. Its answers are long, so alias matching is weak here:
    plan to adjudicate these with the LLM judge or by hand."""
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    return [
        QuestionItem(
            id=f"tfqa-{i:04d}",
            pool="truthfulqa",
            question=row["question"],
            gold=row["best_answer"],
            aliases=list(row["correct_answers"]),
        )
        for i, row in enumerate(ds)
    ]


def assign_splits(items: list[QuestionItem], split: dict[str, float], seed: int) -> None:
    trainable = [it for it in items if it.pool not in OOD_POOLS]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trainable))
    n_train = int(split["train"] * len(trainable))
    n_val = int(split["val"] * len(trainable))
    for rank, i in enumerate(order):
        trainable[i].split = (
            "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
        )
    for it in items:
        if it.pool in OOD_POOLS:
            it.split = "test"


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    d, seed = cfg["data"], cfg["seed"]
    items: list[QuestionItem] = []
    if d["source"] == "synthetic":
        items += synthetic.make_trivia(d["synthetic_trivia"], seed)
        items += synthetic.make_kenya(d["synthetic_kenya"], d["synthetic_unanswerable"], seed + 1)
        n_twins = min(d.get("synthetic_western", 0), d["synthetic_kenya"])
        items += synthetic.make_western(n_twins, seed + 2)
    elif d["source"] == "hf":
        items += load_triviaqa(d["trivia_sample"], seed)
        if d.get("truthfulqa"):
            items += load_truthfulqa()
    else:
        raise ValueError(f"unknown data.source {d['source']!r}")
    items += load_kenya(d["kenya_questions"])

    assign_splits(items, d["split"], seed)
    errors = validate_questions(items)
    if errors:
        raise ValueError(f"data contract failed: {errors[:5]} (+{max(0, len(errors) - 5)} more)")

    p = paths(cfg)
    write_jsonl(p.questions, (it.model_dump() for it in items))
    counts: dict[str, dict[str, int]] = {}
    for it in items:
        counts.setdefault(it.pool, {}).setdefault(it.split, 0)
        counts[it.pool][it.split] += 1
    manifest = {"n": len(items), "counts": counts, "sha256": file_sha256(p.questions)}
    write_json(p.manifest, manifest)
    log.info("wrote %d questions to %s", len(items), p.questions)
    return manifest
