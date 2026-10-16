"""Leakage audit - evaluation questions must not appear in the retrieval or fine-tuning
corpora. Skipped (with a warning) for corpora that do not exist yet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bluff.config import paths
from bluff.leakage import find_overlaps
from bluff.utils import read_jsonl, write_json

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any], n: int = 8) -> dict[str, Any]:
    p = paths(cfg)
    questions = read_jsonl(p.questions)
    report: dict[str, Any] = {}
    for key in ("retrieval_corpus", "finetune_corpus"):
        corpus = Path(cfg["data"][key])
        if not corpus.exists():
            log.warning("leakage audit: %s not found, skipped", corpus)
            report[key] = {"checked": False}
            continue
        hits = find_overlaps(questions, read_jsonl(corpus), n=n)
        report[key] = {"checked": True, "overlaps": len(hits), "examples": hits[:20]}
        if hits:
            log.error("leakage: %d question/document overlaps in %s", len(hits), corpus)
    write_json(p.manifest.parent / "leakage_audit.json", report)
    if any(r.get("overlaps") for r in report.values()):
        raise ValueError("leakage audit failed - remove overlapping documents before training")
    return report
