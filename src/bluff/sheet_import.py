"""Import the students' Kenya Question Sheet (.xlsx) into data/kenya/questions.jsonl.

The sheet has one tab per student ("Student A" ... "Student D"), one row per question, with the
columns of the template: Question, Answer, Other accepted answers, Type (real / fake / twin),
Twin of, Domain, Language, Source link, Source note, Date checked, Valid until, Author, Reviewer,
Verified, Review note.

How each row becomes a QuestionItem:
- real -> pool `kenya`; fake -> pool `kenya_unanswerable` (no answer); twin -> pool `western`,
  linked through `Twin of` (the Kenya question's text, copied) to that question's id.
- Answer + "Other accepted answers" (split on ";") become the accepted answers.
- Ids come from the question text (e.g. ke-civ-3f2a9c1b), so re-importing a sheet with more rows
  keeps every earlier id: a resumed `generate` asks only the new questions.
- A bad row is skipped, never fatal, and listed with its tab, row number and reason. A half-
  finished sheet still imports. The hand-numbered starter questions (e.g. ke-civ-0001) already in
  the file are kept.

CRISP-DM: data preparation (collecting the hand-built test set).
Run:    bluff import-sheet --config configs/variants/m0_base.yaml --sheet "<path to .xlsx>"
Reads:  the .xlsx, data/kenya/questions.jsonl (for the starter questions)
Writes: data/kenya/questions.jsonl, data/kenya/import_report.md
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from bluff.schema import QuestionItem, validate_questions
from bluff.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

COLUMNS = ["Question", "Answer", "Other accepted answers", "Type", "Twin of", "Domain",
           "Language", "Source link", "Source note", "Date checked", "Valid until", "Author",
           "Reviewer", "Verified", "Review note"]
DOMAIN_CODES = {"civics_law": "civ", "geography_society": "geo", "language": "sw",
                "institutions_services": "ins", "money_livelihoods": "mon",
                "western_default_traps": "trap"}
STARTER_ID = re.compile(r"-\d{4}$")  # hand-numbered starter ids are kept on every import


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def _cell(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    return text or None


def _make_id(prefix: str, domain: str | None, question: str) -> str:
    code = DOMAIN_CODES.get(domain or "", "gen")
    digest = hashlib.sha1(_norm(question).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{code}-{digest}"


def read_sheet(path: str | Path) -> list[dict[str, Any]]:
    """Every non-empty row of every student tab, with its tab name and row number."""
    from openpyxl import load_workbook  # optional dependency: pip install openpyxl

    wb = load_workbook(path, data_only=True, read_only=True)
    rows = []
    for ws in wb.worksheets:
        values = ws.iter_rows(values_only=True)
        header = [(_cell(h) or "").replace("(reviewer)", "").strip() for h in next(values, [])]
        if not ws.title.lower().startswith("student") or "Question" not in header:
            continue  # Instructions, Documents, Starter questions
        for number, raw in enumerate(values, start=2):
            row = {h: _cell(v) for h, v in zip(header, raw, strict=False) if h}
            if any(row.get(c) for c in ("Question", "Answer", "Type")):
                rows.append({"tab": ws.title, "row": number, **row})
    wb.close()
    return rows


def rows_to_items(rows: list[dict[str, Any]], starters: list[QuestionItem]
                  ) -> tuple[list[QuestionItem], list[dict[str, Any]]]:
    """Turn sheet rows into QuestionItems. Returns (items, problems)."""
    problems: list[dict[str, Any]] = []
    seen = {_norm(it.question): it.id for it in starters}
    items: list[QuestionItem] = []
    twins: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def skip(r: dict[str, Any], reason: str) -> None:
        problems.append({"tab": r["tab"], "row": r["row"], "question": r.get("Question"),
                         "problem": reason, "action": "skipped"})

    def warn(r: dict[str, Any], reason: str) -> None:
        problems.append({"tab": r["tab"], "row": r["row"], "question": r.get("Question"),
                         "problem": reason, "action": "imported anyway"})

    for r in rows:
        q, kind = r.get("Question"), (r.get("Type") or "").lower()
        if not q or len(q) < 3:
            skip(r, "no question text")
            continue
        if kind not in {"real", "fake", "twin"}:
            skip(r, f"Type must be real, fake or twin (got {r.get('Type')!r})")
            continue
        if _norm(q) in seen:
            skip(r, f"duplicate of {seen[_norm(q)]}")
            continue
        if not q.rstrip().endswith("?"):
            warn(r, "question doesn't end with '?'")

        answer = r.get("Answer")
        others = [a.strip() for a in (r.get("Other accepted answers") or "").split(";")]
        aliases = sorted({a for a in [answer, *others] if a}, key=str.lower)
        verified = (r.get("Verified") or "").lower() == "yes"
        reviewer, author = r.get("Reviewer"), r.get("Author") or r["tab"]
        if verified and not (r.get("Source link") or r.get("Source note")):
            warn(r, "Verified = yes but no source: imported as unverified")
            verified = False
        if verified and (not reviewer or reviewer == author):
            warn(r, "Verified = yes needs a reviewer other than the author: imported as unverified")
            verified = False

        fields = {
            "question": q, "domain": r.get("Domain"), "language": r.get("Language") or "en",
            "source_url": r.get("Source link"), "source_note": r.get("Source note"),
            "source_date": r.get("Date checked"), "valid_until": r.get("Valid until"),
            "author": author, "reviewer": reviewer, "verified": verified,
        }
        if kind == "real":
            if not answer:
                skip(r, "real question with no Answer")
                continue
            item = QuestionItem(id=_make_id("ke", r.get("Domain"), q), pool="kenya",
                                gold=answer, aliases=aliases, **fields)
        elif kind == "fake":
            if answer:
                warn(r, "fake question has an Answer: the answer was ignored")
            item = QuestionItem(id=_make_id("ke-una", r.get("Domain"), q),
                                pool="kenya_unanswerable", answerable=False, gold=None,
                                aliases=[], **fields)
        else:
            if not answer:
                skip(r, "twin question with no Answer")
                continue
            twins.append((r, {**fields, "gold": answer, "aliases": aliases}))
            seen[_norm(q)] = "a twin in this sheet"
            continue
        seen[_norm(q)] = item.id
        items.append(item)

    kenya_by_text = {_norm(it.question): it.id for it in [*starters, *items] if it.pool == "kenya"}
    for r, fields in twins:
        original = kenya_by_text.get(_norm(r.get("Twin of") or ""))
        if not original:
            skip(r, "Twin of doesn't match any real Kenya question's text exactly")
            continue
        items.append(QuestionItem(id=_make_id("uk", r.get("Domain"), fields["question"]),
                                  pool="western", twin_of=original, **fields))
    return items, problems


def run(cfg: dict[str, Any], sheet: str | Path) -> dict[str, Any]:
    out = Path(cfg["data"]["kenya_questions"])
    previous = [QuestionItem(**r) for r in read_jsonl(out)] if out.exists() else []
    starters = [it for it in previous if STARTER_ID.search(it.id)]
    rows = read_sheet(sheet)
    items, problems = rows_to_items(rows, starters)
    final = [*starters, *items]

    errors = validate_questions(final)
    if errors:  # rows_to_items should prevent these; refuse rather than write a broken file
        raise ValueError(f"imported set breaks the data contract: {errors[:5]}")
    write_jsonl(out, (it.model_dump() for it in final))

    counts: dict[str, dict[str, int]] = {}
    for it in final:
        who = it.author or "?"
        counts.setdefault(who, {"kenya": 0, "kenya_unanswerable": 0, "western": 0, "verified": 0})
        counts[who][it.pool] += 1
        counts[who]["verified"] += int(it.verified)
    _write_report(out.parent / "import_report.md", sheet, rows, final, counts, problems,
                  len(previous))
    skipped = sum(p["action"] == "skipped" for p in problems)
    log.info("imported %d questions (%d starters kept); %d rows skipped, %d warnings",
             len(items), len(starters), skipped, len(problems) - skipped)
    return {"sheet_rows": len(rows), "imported": len(items), "starters_kept": len(starters),
            "total": len(final), "previous_total": len(previous), "skipped": skipped,
            "warnings": len(problems) - skipped, "by_author": counts,
            "report": str(out.parent / "import_report.md")}


def _write_report(path: Path, sheet: str | Path, rows: list, final: list[QuestionItem],
                  counts: dict, problems: list, n_previous: int) -> None:
    lines = [
        "# Kenya question import report", "",
        f"Imported {datetime.now():%Y-%m-%d %H:%M} from `{Path(sheet).name}`: "
        f"{len(rows)} filled rows; the question file now holds {len(final)} questions "
        f"(it held {n_previous}).", "",
        "## Questions per author", "",
        "| Author | Real | Fake | Twin | Verified |", "|---|---|---|---|---|",
        *[f"| {a} | {c['kenya']} | {c['kenya_unanswerable']} | {c['western']} | {c['verified']} |"
          for a, c in sorted(counts.items())],
        "", f"## Rows with problems ({len(problems)})", "",
    ]
    if problems:
        lines += ["| Tab | Row | Question | Problem | Action |", "|---|---|---|---|---|"]
        lines += [f"| {p['tab']} | {p['row']} | {(p['question'] or '')[:60]} | {p['problem']} "
                  f"| {p['action']} |" for p in problems]
    else:
        lines.append("None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
