"""The sheet clean-up: each fix on its own, then a whole sheet with its change log."""

import csv
from datetime import datetime

import pytest

from bluff import sheet_clean as sc

openpyxl = pytest.importorskip("openpyxl")


class Cell:
    def __init__(self, value, number_format="General"):
        self.value, self.number_format = value, number_format


def test_question_mark():
    assert sc.question_mark("what is the total area of kenya") == "what is the total area of kenya?"
    assert sc.question_mark("Who is the head of state?") == "Who is the head of state?"
    assert sc.question_mark("Name the capital.") == "Name the capital?"


def test_name_kenya_only_when_the_country_is_missing():
    assert (sc.name_kenya("Who is in charge of voter education?", "en")
            == "In Kenya, who is in charge of voter education?")
    assert sc.name_kenya("What is the capital of Kenya?", "en") == "What is the capital of Kenya?"
    assert sc.name_kenya("Which body regulates NTSA fees?", "en").startswith("Which")  # names a body
    assert sc.name_kenya("Siku ya Madaraka ni lini?", "sw") == "Siku ya Madaraka ni lini?"
    assert sc.name_kenya("IEBC was formed when?", "en") == "IEBC was formed when?"
    assert sc.name_kenya("Under which law is the KNBS set up?", "en").startswith("Under")


def test_excel_value_shows_what_was_typed():
    year = datetime.now().year
    assert sc.excel_value(Cell(0.16, "0%")) == "16%"
    assert sc.excel_value(Cell(0.0275, "0.00%")) == "2.75%"
    assert sc.excel_value(Cell(datetime(year, 6, 1))) == "1 June"
    assert sc.excel_value(Cell(datetime(2010, 8, 27))) == "27 August 2010"
    assert sc.excel_value(Cell(300.0)) == "300"
    assert sc.excel_value(Cell("  Lake Turkana ")) == "Lake Turkana"


def test_tidy_domain():
    assert sc.tidy_domain("civis-law", "x") == "civics_law"
    assert sc.tidy_domain("Language", "x") == "language"
    assert sc.tidy_domain("institutions_services, money_livelihoods", "money_livelihoods") \
        == "money_livelihoods"
    assert sc.tidy_domain(None, "western_default_traps") == "western_default_traps"


def test_split_answer():
    assert sc.split_answer("999/112", "999 or 112") == ("999/112", "999 or 112; 999; 112")
    assert sc.split_answer("Nairobi", None) == ("Nairobi", None)


def test_whole_sheet_is_cleaned_and_every_change_logged(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["QUESTION", "ANSWER", "OTHER ACCEPTED ANSWER", "TYPE", "TWIN OF", "DOMAIN",
               "LANGUAGE", "SOURCE LINK", "SOURCE NOTES"])
    ws.append(["Who is in charge of voter education", "IEBC", None, "Real", None, "civis-law",
               "en", "https://www.iebc.or.ke", None])
    ws.append(["What is the standard VAT rate in Kenya?", 0.16, "16 percent", "real", None,
               "money_livelihoods", "en", "https://kra.go.ke", None])
    ws.cell(row=3, column=2).number_format = "0%"
    ws.append(["LAKES AND RIVERS", None, None, None, None, None, None, None, None])  # a heading
    ws.append(["What is the capital city of Kenya?", "Nairobi", None, "real", None, None,
               "en", None, None])  # repeats a starter
    ws.append([None, "Miles", None, "real", None, None, None, None, None])  # answer-only row
    ws.append(["Who was the fifth president of the UK?", None, None, "fake", None, None, "en",
               None, None])  # about another country
    ws.append(["What is the largest county in Switzerland?", None, None, "fake", None, None, "en",
               None, None])  # "county" alone doesn't make it Kenyan
    ws.append(["CENSUS AND POPULATION FIGURES", None, None, "real", None, None, None, None,
               None])  # a heading typed as real
    wb.save(tmp_path / "s.xlsx")
    (tmp_path / "files.yaml").write_text(
        f"- {{file: '{(tmp_path / 's.xlsx').as_posix()}', tab: Sheet1, student: B, "
        "author: Amina, domain: geography_society}\n", encoding="utf-8")
    cfg = {"data": {"kenya_questions": str(tmp_path / "kenya" / "questions.jsonl")}}

    result = sc.run(cfg, tmp_path / "files.yaml")

    out = openpyxl.load_workbook(result["cleaned"])["Student B"]
    rows = [dict(zip(sc.COLUMNS, r, strict=False)) for r in out.iter_rows(min_row=2, values_only=True)]
    assert [r["Question"] for r in rows] == ["In Kenya, who is in charge of voter education?",
                                             "What is the standard VAT rate in Kenya?"]
    assert rows[0]["Domain"] == "civics_law" and rows[0]["Author"] == "Amina"
    assert rows[1]["Answer"] == "16%"
    with open(tmp_path / "kenya" / "cleaned" / "changes.csv", encoding="utf-8-sig") as f:
        why = [c["why"] for c in csv.DictReader(f)]
    for expected in ["added '?'", "added 'In Kenya, '", "Excel value shown as typed",
                     "domain mapped to the brief's six", "skipped: no Type (a heading or unfinished row)",
                     "skipped: repeats an earlier question or a starter", "skipped: no question text",
                     "skipped: about another country, not Kenya",
                     "skipped: no answer (a real or twin question)"]:
        assert expected in why
    assert (tmp_path / "kenya" / "cleaning_report.md").exists()


def test_repairs_drop_edit_twin_add_and_label_ai(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Student A"
    ws.append(sc.COLUMNS)
    ws.append(["What is the highest court in Kenya?", "Supreme Court", None, "real", None,
               "civics_law", "en", "https://judiciary.go.ke", None, None, None, "Amina"])
    ws.append(["Is the sky green?", "no", None, "real", None, "civics_law", "en", None, None,
               None, None, "Amina"])
    ws.append(["Who copies the Constitution word for word?", "Nobody", None, "real", None,
               "civics_law", "en", None, None, None, None, "Amina"])
    wb.save(tmp_path / "s.xlsx")
    (tmp_path / "files.yaml").write_text(
        f"- {{file: '{(tmp_path / 's.xlsx').as_posix()}', tab: Student A, student: A, "
        "author: Amina, domain: civics_law}\n", encoding="utf-8")
    (tmp_path / "repairs.yaml").write_text("""
A:
  drop: [{match: "sky green", why: "a yes/no question"}]
  edit: [{match: "copies the Constitution", why: "reworded", set: {Question: "Who rewrites Kenya's laws in plain words?"}}]
  twin: [{match: "highest court in Kenya", question: "What is the highest court in the United Kingdom",
          answer: "The Supreme Court", source: "https://www.supremecourt.uk/"}]
  add: [{type: fake, question: "Who heads Kenya's sub-county assemblies", why: "no such body"}]
""", encoding="utf-8")
    cfg = {"data": {"kenya_questions": str(tmp_path / "kenya" / "questions.jsonl")}}
    result = sc.run(cfg, tmp_path / "files.yaml")
    rows = [dict(zip(sc.COLUMNS, r, strict=False)) for r in
            openpyxl.load_workbook(result["cleaned"])["Student A"].iter_rows(min_row=2, values_only=True)]
    by_q = {r["Question"]: r for r in rows}
    assert "Is the sky green?" not in by_q
    assert by_q["Who rewrites Kenya's laws in plain words?"]["Reviewer"] == sc.AI_AUTHOR
    twin = by_q["What is the highest court in the United Kingdom?"]
    assert twin["Type"] == "twin" and twin["Author"] == sc.AI_AUTHOR
    assert twin["Twin of"] == "What is the highest court in Kenya?"
    assert by_q["Who heads Kenya's sub-county assemblies?"]["Type"] == "fake"


def test_a_repair_must_match_exactly_one_row():
    rows = [{"row": 2, "_orig": "Which body issues licences?"},
            {"row": 3, "_orig": "Is NTSA the body that issues licences?"}]
    with pytest.raises(ValueError, match="matches 2 rows"):
        sc._find(rows, "issues licences", "Student C")
