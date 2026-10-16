"""BM25 retrieval over the Kenyan corpus (variants M1 and M3).

Start simple: BM25 is a strong lexical baseline. A dense multilingual retriever and a
reranker are later upgrades (see the build guide), not day-one requirements.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from bluff.utils import read_jsonl

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Passage:
    id: str
    text: str
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, str]]) -> None:
        if not chunks:
            raise ValueError("retrieval corpus is empty")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int = 3) -> list[Passage]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Passage(self.chunks[i]["id"], self.chunks[i]["text"], float(scores[i])) for i in order
        ]


def build_retriever(cfg: dict[str, Any]) -> BM25Retriever | None:
    if not cfg["retrieval"]["enabled"]:
        return None
    path = Path(cfg["data"]["retrieval_corpus"])
    if not path.exists():
        raise FileNotFoundError(
            f"retrieval is enabled but {path} does not exist. "
            "Build the corpus first (see data/README.md)."
        )
    log.info("loading retrieval corpus %s", path)
    return BM25Retriever(read_jsonl(path))
