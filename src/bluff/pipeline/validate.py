"""Stage 0 - validate the hand-built Kenya question set against the data contract."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bluff.schema import QuestionItem, validate_questions
from bluff.utils import read_jsonl

log = logging.getLogger(__name__)


def load_kenya(path: str | Path) -> list[QuestionItem]:
    path = Path(path)
    if not path.exists():
        log.warning("no Kenya question file at %s", path)
        return []
    return [QuestionItem(**row) for row in read_jsonl(path)]


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    items = load_kenya(cfg["data"]["kenya_questions"])
    errors = validate_questions(items)
    if errors:
        for e in errors:
            log.error(e)
        raise ValueError(f"{len(errors)} data-contract violations in the Kenya question set")
    report = {
        "n": len(items),
        "answerable": sum(i.answerable for i in items),
        "unanswerable": sum(not i.answerable for i in items),
        "verified": sum(i.verified for i in items),
    }
    log.info("Kenya question set OK: %s", report)
    if report["n"] and report["verified"] < report["n"]:
        log.warning(
            "%d of %d Kenya items are not yet verified by a second reviewer",
            report["n"] - report["verified"],
            report["n"],
        )
    return report
