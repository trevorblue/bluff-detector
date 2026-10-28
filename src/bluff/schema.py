"""Typed records that flow between pipeline stages, and the question-set data contract.

CRISP-DM: data understanding (the rules every question must pass) and data preparation
(the record formats each stage writes).
Used by: validate, ingest, generate, label, train, evaluate, matching and tests/data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Pool = Literal["trivia", "truthfulqa", "kenya", "kenya_unanswerable", "synthetic"]
Split = Literal["train", "val", "test"]
Condition = Literal["open", "offered"]

TRAIN_POOLS = ("trivia", "truthfulqa", "synthetic")  # detectors are trained on these only
OOD_POOLS = ("kenya", "kenya_unanswerable")  # test-only


class QuestionItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    pool: Pool
    question: str = Field(min_length=3)
    gold: str | None = None
    aliases: list[str] = Field(default_factory=list)
    answerable: bool = True
    split: Split | None = None
    domain: str | None = None
    language: str | None = None
    source_url: str | None = None
    source_note: str | None = None
    source_date: str | None = None
    valid_until: str | None = None
    author: str | None = None
    reviewer: str | None = None
    verified: bool = False


class Generation(BaseModel):
    qid: str
    variant: str
    condition: Condition
    template: int
    prompt: str
    answer: str
    token_probs: list[float]
    token_entropies: list[float]
    latency_ms: float
    retrieved_ids: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    hidden_index: int | None = None  # row in hidden.npz (open condition only)


class Label(BaseModel):
    qid: str
    pool: Pool
    split: Split
    correct: bool  # open-condition answer judged correct
    open_abstained: bool  # the open answer itself was a refusal
    abstained: bool  # majority vote over offered-"I don't know" templates
    abstain_votes: list[bool]
    method: str


def validate_questions(items: list[QuestionItem]) -> list[str]:
    """The data contract. Returns a list of violations; empty means the set is valid."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_q: set[str] = set()
    for it in items:
        if it.id in seen_ids:
            errors.append(f"{it.id}: duplicate id")
        seen_ids.add(it.id)
        key = " ".join(it.question.lower().split())
        if key in seen_q:
            errors.append(f"{it.id}: duplicate question text")
        seen_q.add(key)
        if it.answerable:
            if not it.gold:
                errors.append(f"{it.id}: answerable item has no gold answer")
            if it.pool == "kenya_unanswerable":
                errors.append(f"{it.id}: pool kenya_unanswerable but answerable=true")
        else:
            if it.gold is not None:
                errors.append(f"{it.id}: unanswerable item must have gold=null")
            if it.pool != "kenya_unanswerable":
                errors.append(f"{it.id}: unanswerable items belong in pool kenya_unanswerable")
        if it.verified and it.pool in OOD_POOLS:
            if not (it.source_url or it.source_note):
                errors.append(f"{it.id}: verified Kenya item needs a source")
            if not it.reviewer:
                errors.append(f"{it.id}: verified Kenya item needs a second-person reviewer")
            if it.reviewer and it.reviewer == it.author:
                errors.append(f"{it.id}: reviewer must differ from author")
        if it.split is not None and it.pool in OOD_POOLS and it.split != "test":
            errors.append(f"{it.id}: Kenya pools are test-only")
    return errors
