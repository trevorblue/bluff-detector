"""Synthetic question sets for the fake backend: CI, smoke tests and local dry runs.

CRISP-DM: data preparation (test data only).
Used by: pipeline/ingest.py when `data.source: synthetic` (the dev_fake configs).
"""

from __future__ import annotations

import numpy as np

from bluff.schema import QuestionItem

_WORDS = (
    "amber", "baobab", "cobalt", "delta", "ember", "falcon", "granite", "harbour",
    "indigo", "jasper", "kestrel", "lagoon", "marble", "nectar", "onyx", "papyrus",
)  # fmt: skip


def _code(rng: np.random.Generator) -> str:
    return f"{rng.choice(_WORDS)}-{int(rng.integers(10, 99))}"


def make_trivia(n: int, seed: int = 42) -> list[QuestionItem]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        gold = _code(rng)
        out.append(
            QuestionItem(
                id=f"syn-tri-{i:04d}",
                pool="synthetic",
                question=f"Synthetic general question {i}: which code is filed under item {i}?",
                gold=gold,
                aliases=[gold],
            )
        )
    return out


def make_kenya(n: int, n_unanswerable: int, seed: int = 43) -> list[QuestionItem]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        gold = _code(rng)
        out.append(
            QuestionItem(
                id=f"syn-ke-{i:04d}",
                pool="kenya",
                question=f"Synthetic Kenyan question {i}: which county code is filed under {i}?",
                gold=gold,
                aliases=[gold],
                language="en",
            )
        )
    for i in range(n_unanswerable):
        out.append(
            QuestionItem(
                id=f"syn-una-{i:04d}",
                pool="kenya_unanswerable",
                question=f"Synthetic Kenyan question {i}: who governs the non-existent county {i}?",
                gold=None,
                answerable=False,
            )
        )
    return out


def make_western(n: int, seed: int = 44) -> list[QuestionItem]:
    """UK twins of the first n synthetic Kenya questions (call make_kenya with at least n).

    The wording avoids the fake backend's Kenya keywords, so it treats twins as general
    knowledge, as the hypothesis expects a real model to.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        gold = _code(rng)
        out.append(
            QuestionItem(
                id=f"syn-uk-{i:04d}",
                pool="western",
                question=f"Synthetic UK question {i}: which borough code is filed under {i}?",
                gold=gold,
                aliases=[gold],
                language="en",
                twin_of=f"syn-ke-{i:04d}",
            )
        )
    return out
