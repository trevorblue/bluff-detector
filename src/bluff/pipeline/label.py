"""Stage 3 - mark each answer right or wrong, and record whether the model abstained.

correct   : the open-condition answer, checked against gold aliases
abstained : majority vote over the offered-"I don't know" templates
Also writes a random sample for manual adjudication; the disagreement rate between
automated and manual marking is reported as the label-noise floor.

CRISP-DM: data preparation (labelling). review_sample.jsonl is for data understanding: a
person checks it to measure how noisy the automatic labels are.
In the pipeline: generate -> [label] -> featurize
Reads:  data/processed/questions.jsonl, data/interim/<variant>/generations.jsonl
Writes: data/interim/<variant>/labels.jsonl, data/interim/<variant>/review_sample.jsonl
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from bluff.config import paths
from bluff.matching import is_abstention, is_correct
from bluff.schema import Label, QuestionItem
from bluff.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def judge(answer: str, item: QuestionItem, cfg: dict[str, Any]) -> bool | None:
    """Hook for an LLM judge. Return None to defer to alias matching.

    TODO (Core tier): implement with a strong model, using a fixed rubric prompt that
    sees the question, gold aliases and the answer, and returns CORRECT / INCORRECT.
    Log every disagreement with alias matching for manual adjudication.
    """
    if cfg["labelling"]["judge"] == "none":
        return None
    raise NotImplementedError("LLM judge not wired yet - set labelling.judge: none")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    p = paths(cfg)
    items = {r["id"]: QuestionItem(**r) for r in read_jsonl(p.questions)}
    gens = read_jsonl(p.interim / "generations.jsonl")
    by_q: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for g in gens:
        by_q[g["qid"]][g["condition"]].append(g)

    labels, review = [], []
    for qid, conds in by_q.items():
        item = items[qid]
        open_answer = conds["open"][0]["answer"]
        auto = is_correct(open_answer, item)
        judged = judge(open_answer, item, cfg)
        correct = auto if judged is None else judged
        votes = [is_abstention(g["answer"]) for g in sorted(conds["offered"], key=_tmpl)]
        abstained = sum(votes) > len(votes) / 2 if votes else False
        labels.append(
            Label(
                qid=qid,
                pool=item.pool,
                split=item.split,
                correct=correct,
                open_abstained=is_abstention(open_answer),
                abstained=abstained,
                abstain_votes=votes,
                method="alias" if judged is None else "judge",
            ).model_dump()
        )
        review.append(
            {
                "qid": qid,
                "question": item.question,
                "gold": item.gold,
                "answer": open_answer,
                "auto": auto,
                "manual": None,
            }
        )

    write_jsonl(p.interim / "labels.jsonl", labels)
    rng = np.random.default_rng(cfg["seed"])
    n_review = min(cfg["labelling"]["review_sample"], len(review))
    sample = [review[i] for i in rng.choice(len(review), n_review, replace=False)]
    write_jsonl(p.interim / "review_sample.jsonl", sample)

    wrong = sum(not lab["correct"] for lab in labels)
    template_spread = _template_spread(labels)
    log.info("labelled %d answers: %d wrong; abstention by template %s", len(labels), wrong,
             template_spread)
    return {"n": len(labels), "wrong": wrong, "abstain_rate_by_template": template_spread}


def _tmpl(g: dict) -> int:
    return g["template"]


def _template_spread(labels: list[dict]) -> list[float]:
    """Abstention rate per offered template - a large spread means a prompt artifact."""
    votes = [lab["abstain_votes"] for lab in labels if lab["abstain_votes"]]
    if not votes:
        return []
    arr = np.asarray(votes, dtype=float)
    return [round(float(x), 3) for x in arr.mean(axis=0)]
