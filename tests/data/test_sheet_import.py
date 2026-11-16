"""The students' sheet importer: every row type, every problem, and stable ids across imports."""

import pytest

from bluff import sheet_import
from bluff.schema import validate_questions
from bluff.utils import read_jsonl, write_jsonl

openpyxl = pytest.importorskip("openpyxl")

STARTER = {"id": "ke-civ-0001", "pool": "kenya", "question": "How many counties does Kenya have?",
           "gold": "47", "aliases": ["47"], "author": "Trevor"}


def _row(question, answer=None, others=None, kind="real", twin_of=None, domain="civics_law",
         verified=None, source=None, reviewer=None, author="Amina"):
    return [question, answer, others, kind, twin_of, domain, "en", source, None, "2026-11-02",
            None, author, reviewer, verified, None]


def _sheet(path, tabs):
    wb = openpyxl.Workbook()
    wb.active.title = "Instructions"
    for name, rows in tabs.items():
        ws = wb.create_sheet(name)
        ws.append(sheet_import.COLUMNS)
        for r in rows:
            ws.append(r)
        ws.append([None] * 15)  # an empty row, as in the template
    wb.save(path)
    return path


def _cfg(tmp_path):
    out = tmp_path / "questions.jsonl"
    write_jsonl(out, [STARTER])
    return {"data": {"kenya_questions": str(out)}}, out


def test_import_every_row_type_and_problem(tmp_path):
    cfg, out = _cfg(tmp_path)
    sheet = _sheet(tmp_path / "s.xlsx", {"Student A": [
        _row("Who is the head of state of Kenya?", "The President", "president; the president"),
        _row("In which year was the Kenyan county of Lakeside formed?", kind="fake"),
        _row("Who is the head of state of the United Kingdom?", "The monarch", "king; the king",
             kind="twin", twin_of="Who is the head of state of Kenya?"),
        _row("How many counties does Kenya have?", "47"),                  # repeats a starter
        _row("What is Kenya's national bird?"),                            # real, no answer
        _row("What is the UK's national bird?", "Robin", kind="twin",
             twin_of="A question that does not exist?"),                    # broken twin link
        _row("Which article of the Constitution covers devolution?", "Article 174",
             verified="yes", reviewer="Brian"),                            # verified, no source
    ]})
    result = sheet_import.run(cfg, sheet)
    items = {r["question"]: r for r in read_jsonl(out)}

    assert result["imported"] == 4 and result["starters_kept"] == 1 and result["skipped"] == 3
    assert items["Who is the head of state of Kenya?"]["pool"] == "kenya"
    assert set(items["Who is the head of state of Kenya?"]["aliases"]) >= {"The President",
                                                                           "president"}
    fake = items["In which year was the Kenyan county of Lakeside formed?"]
    assert fake["pool"] == "kenya_unanswerable" and fake["gold"] is None
    twin = items["Who is the head of state of the United Kingdom?"]
    assert twin["pool"] == "western"
    assert twin["twin_of"] == items["Who is the head of state of Kenya?"]["id"]
    assert items["Which article of the Constitution covers devolution?"]["verified"] is False
    report = (tmp_path / "import_report.md").read_text(encoding="utf-8")
    assert "duplicate of ke-civ-0001" in report and "real question with no Answer" in report
    assert "Twin of doesn't match" in report
    from bluff.schema import QuestionItem
    assert validate_questions([QuestionItem(**r) for r in read_jsonl(out)]) == []


def test_reimport_keeps_ids_and_adds_new_rows(tmp_path):
    cfg, out = _cfg(tmp_path)
    first = [_row("What is the capital of Kisumu County?", "Kisumu")]
    sheet_import.run(cfg, _sheet(tmp_path / "a.xlsx", {"Student B": first}))
    ids_before = {r["question"]: r["id"] for r in read_jsonl(out)}
    more = [*first, _row("Which lake borders Kisumu?", "Lake Victoria")]
    result = sheet_import.run(cfg, _sheet(tmp_path / "b.xlsx", {"Student B": more}))
    ids_after = {r["question"]: r["id"] for r in read_jsonl(out)}
    assert result["imported"] == 2 and result["starters_kept"] == 1
    assert all(ids_after[q] == i for q, i in ids_before.items())
