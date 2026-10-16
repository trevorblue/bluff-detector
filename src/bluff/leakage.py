"""Leakage audit between the data pools.

If the fine-tuning corpus or the retrieval index contains the evaluation questions, every
result is invalid. This check runs as a data contract.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+", re.UNICODE)


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def find_overlaps(
    questions: list[dict[str, str]], docs: list[dict[str, str]], n: int = 8
) -> list[tuple[str, str]]:
    """Return (question_id, doc_id) pairs sharing at least one word n-gram.

    Short questions (fewer than n words) are checked with n = their length, so a short
    question copied verbatim into a document is still caught.
    """
    hits: list[tuple[str, str]] = []
    cache: dict[int, dict[str, set[tuple[str, ...]]]] = {}
    for q in questions:
        words = _WORD.findall(q["question"].lower())
        k = min(n, len(words))
        if k < 4:
            continue  # too short to count an overlap as evidence of leakage
        if k not in cache:
            cache[k] = {d["id"]: ngrams(d["text"], k) for d in docs}
        q_grams = ngrams(q["question"], k)
        for doc_id, grams in cache[k].items():
            if q_grams & grams:
                hits.append((q["id"], doc_id))
    return hits
