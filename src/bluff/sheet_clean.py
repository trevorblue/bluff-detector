"""Clean the students' question sheets before import, and record every change.

The four students each sent their own .xlsx, in slightly different layouts (data/kenya/
student_files.yaml says which file is which student). This step reads them, applies only
mechanical fixes that don't change what a question asks, and writes a cleaned workbook in the
template's layout, one tab per student, which `bluff import-sheet` then imports.

The fixes (each is one function below):
1. question_mark   add a missing "?" at the end of a question
2. name_kenya      add "In Kenya, " to a question that never names Kenya, so the model knows
                   which country it is about (e.g. "Who is in charge of voter education?")
3. excel_value     show an answer the way the student typed it: Excel stores "16%" as 0.16 and
                   "1 June" as a full date
4. tidy_domain     map a Domain value onto the brief's six (e.g. "civis-law" -> civics_law);
                   anything else becomes the student's own domain
5. skip rows       empty rows, section headings (no Type, or "real" with no answer), twin rows
                   without a question, questions about another country that never mention
                   Kenya, and exact repeats (of an earlier row, or of Trevor's starter questions)
Also: split_answer turns "999/112" or "X or Y" into separate accepted answers.

Nothing else is changed: wrong facts, weak sources, missing twins and the like are for the
students to fix (see "Kenya questions - fixes per student.md").

CRISP-DM: data preparation.
Run:    bluff clean-sheets --config configs/variants/m0_base.yaml
Reads:  data/kenya/student_files.yaml and the .xlsx files it lists
Writes: data/kenya/cleaned/cleaned_questions.xlsx (tabs "Student A".."Student D", plus a
        "Changes" tab), data/kenya/cleaned/changes.csv, data/kenya/cleaning_report.md
"""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from bluff.sheet_import import COLUMNS
from bluff.utils import ensure_dir

log = logging.getLogger(__name__)

DOMAINS = ["civics_law", "geography_society", "language", "institutions_services",
           "money_livelihoods", "western_default_traps"]
DOMAIN_MAP = {  # values the students wrote -> the brief's sheet values
    "civis-law": "civics_law", "civics": "civics_law", "constitutional": "civics_law",
    "geography": "geography_society", "administrative divisions": "geography_society",
    "demographic": "geography_society", "cultural heritage": "geography_society",
    "language": "language", "institutions": "institutions_services",
    "money": "money_livelihoods",
}
HEADER_ALIASES = {"other accepted answer": "Other accepted answers", "source notes": "Source note"}
STARTERS = [
    "How many counties does Kenya have?", "What is the capital city of Kenya?",
    "What is the official currency of Kenya?",
    "Which public body replaced the NHIF as Kenya's national health insurer?",
    "What telephone number do you dial for police, fire or ambulance in Kenya?",
    "In which year did Kenya become independent?", "Mji mkuu wa Kenya ni upi?",
]
# A question "names Kenya" if it mentions Kenya, a Kenyan place, body or term, or is in Kiswahili.
KENYAN = re.compile(
    r"kenya|swahili|kiswahili|nairobi|mombasa|kisumu|nakuru|eldoret|county|counties|\bkra\b|ntsa|"
    r"kuccps|iebc|\bsha\b|nhif|helb|knec|kephis|harambee|madaraka|mashujaa|jamhuri|shilling|"
    r"m-pesa|mpesa|safaricom|\bksh\b|\bkes\b|\bkcse\b|\bkcpe\b|huduma|ecitizen|maasai|turkana|"
    r"knbs|kws|kenyatta|uhuru|\bsiku\b|\bya\b|\bwa\b|\bni\b|gani|ngapi|nini|mkoa|mlima", re.I)
OTHER_COUNTRY = re.compile(
    r"\b(uk|u\.k\.?|united kingdom|britain|british|england|usa|u\.s\.a?|america|american|germany|"
    r"iceland|poland|switzerland|spain|greece|france|malta|portugal|norway|hungary|italy|japan|"
    r"china|india|canada|australia)\b", re.I)
NAMES_KENYA = re.compile(r"kenya|kiswahili|swahili|nairobi|mombasa|kisumu", re.I)  # strict
QUESTION_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "how", "why", "is",
                  "are", "was", "were", "does", "do", "did", "can", "name", "on", "in", "at",
                  "by", "from", "for", "under", "after", "approximately", "according", "must"}


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9 ]", "", " ".join(str(text or "").lower().split()))


# ---------------------------------------------------------------- the five fixes, one each

def excel_value(cell: Any) -> str | None:
    """Fix 3: the value as the student typed it. Excel turns "16%" into 0.16 and "1 June" into
    a full date (with the current year when none was typed)."""
    v, fmt = cell.value, (getattr(cell, "number_format", "") or "")
    if v is None:
        return None
    if isinstance(v, datetime | date):
        day = f"{v.day} {v.strftime('%B')}"
        return day if v.year == datetime.now().year else f"{day} {v.year}"
    if isinstance(v, int | float) and "%" in fmt:
        return f"{v * 100:g}%"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    text = str(v).replace(" ", " ").strip()
    return re.sub(r"\s+", " ", text) or None


def question_mark(q: str) -> str:
    """Fix 1: end the question with a single "?"."""
    q = q.rstrip(" .")
    return q if q.endswith("?") else q + "?"


def name_kenya(q: str, language: str | None) -> str:
    """Fix 2: prefix "In Kenya, " when the question never says which country it is about."""
    if (language or "").lower().startswith("sw") or KENYAN.search(q):
        return q
    first = q.split()[0] if q.split() else ""
    rest = q[0].lower() + q[1:] if first.lower() in QUESTION_WORDS else q
    return "In Kenya, " + rest


def tidy_domain(value: str | None, default: str) -> str:
    """Fix 4: one of the brief's six domain values."""
    v = (value or "").strip().lower()
    if v in DOMAINS:
        return v
    return DOMAIN_MAP.get(v, default)


def split_answer(answer: str | None, others: str | None) -> tuple[str | None, str | None]:
    """Answers like "999/112" or "999 or 112": keep the answer, and add each part as its own
    accepted answer, so a model that says just "999" is marked right."""
    if not answer:
        return answer, others
    parts = [p.strip() for p in re.split(r"\s*/\s*|\s+or\s+", answer) if p.strip()]
    extra = [p for p in parts if len(parts) > 1 and p.lower() != answer.lower()]
    have = [o.strip() for o in (others or "").split(";") if o.strip()]
    new = have + [p for p in extra if p.lower() not in {h.lower() for h in have}]
    return answer, "; ".join(new) or None


# ---------------------------------------------------------------- reading and writing

def read_rows(path: Path, tab: str) -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    ws = load_workbook(path, data_only=True)[tab]
    rows = list(ws.iter_rows())
    header = []
    for i, c in enumerate(rows[0]):
        h = str(c.value or "").replace("(reviewer)", "").strip()
        h = HEADER_ALIASES.get(h.lower(), next((x for x in COLUMNS if x.lower() == h.lower()), h))
        header.append(h or (COLUMNS[i] if i < len(COLUMNS) else ""))  # a blank header cell
    out = [{"row": r[0].row, **{h: excel_value(c) for h, c in zip(header, r, strict=False) if h},
            "_raw": {h: c.value for h, c in zip(header, r, strict=False) if h}}
           for r in rows[1:]]
    for r in out:
        r["_orig"] = r.get("Question")  # repairs.yaml matches the question as the student wrote it
    return out


AI_AUTHOR = "Claude (AI)"  # author / reviewer of everything written or changed by repairs.yaml
EDITABLE = ("Question", "Answer", "Other accepted answers", "Source link", "Source note",
            "Valid until", "Type", "Language")


def _find(rows: list[dict[str, Any]], match: str, who: str) -> dict[str, Any]:
    """The one row whose question contains `match` (case and spacing ignored)."""
    hits = [r for r in rows if r.get("_orig") and norm(match) in norm(r["_orig"])]
    if len(hits) != 1:
        raise ValueError(f"repairs.yaml, {who}: {match!r} matches {len(hits)} rows, "
                         "needs exactly 1")
    return hits[0]


def clean_student(spec: dict[str, Any], rows: list[dict[str, Any]], seen: set[str],
                  repairs: dict[str, Any] | None = None
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    """Apply the fixes, then this student's repairs (repairs.yaml), to one student's rows.
    Returns (kept rows, change log, counts)."""
    kept, changes, counts = [], [], Counter()
    who = f"Student {spec['student']}"
    repairs = repairs or {}
    drops = {id(_find(rows, d["match"], who)): d["why"] for d in repairs.get("drop", [])}
    edits = {id(_find(rows, e["match"], who)): e for e in repairs.get("edit", [])}

    def log_change(r, column, before, after, why):
        changes.append({"student": who, "row": r["row"], "column": column,
                        "before": before, "after": after, "why": why})
        counts[why if not why.startswith("AI ") else why.split(":")[0]] += 1

    final_question: dict[int, str] = {}  # sheet row -> the question as kept, for twins
    for r in rows:
        q, kind = r.get("Question"), (r.get("Type") or "").strip().lower()
        if not q and not r.get("Answer") and not kind:
            continue  # an empty row: nothing to record
        if not q:
            log_change(r, "row", r.get("Answer") or kind, "", "skipped: no question text")
            continue
        if r["row"] in (spec.get("exclude") or {}):  # listed in student_files.yaml, with a reason
            reason = spec["exclude"][r["row"]]
            log_change(r, "row", q, "", f"excluded in student_files.yaml: {reason}")
            continue
        if id(r) in drops:
            log_change(r, "row", q, "", f"AI review, dropped: {drops[id(r)]}")
            continue
        if id(r) in edits:
            e = edits[id(r)]
            for column, value in e["set"].items():
                if column not in EDITABLE:
                    raise ValueError(f"repairs.yaml, {who}: can't edit column {column!r}")
                if (r.get(column) or "") != (value or ""):
                    log_change(r, column, r.get(column), value, f"AI review, edited: {e['why']}")
                    r[column] = value
            r["Reviewer"] = AI_AUTHOR
            q, kind = r.get("Question"), (r.get("Type") or "").strip().lower()
        if kind not in {"real", "fake", "twin"}:
            log_change(r, "row", q, "", "skipped: no Type (a heading or unfinished row)")
            continue
        if kind in {"real", "fake"} and OTHER_COUNTRY.search(q) and not NAMES_KENYA.search(q):
            log_change(r, "row", q, "", "skipped: about another country, not Kenya")
            continue
        if kind in {"real", "twin"} and not r.get("Answer"):
            log_change(r, "row", q, "", "skipped: no answer (a real or twin question)")
            continue
        if norm(q) in seen:
            log_change(r, "row", q, "", "skipped: repeats an earlier question or a starter")
            continue
        seen.add(norm(q))
        out = {c: r.get(c) for c in COLUMNS}
        out["Type"] = kind
        for column in ("Answer", "Other accepted answers"):
            raw = r["_raw"].get(column)
            if raw is not None and not isinstance(raw, str) and str(raw) != (out[column] or ""):
                log_change(r, column, str(raw), out[column], "Excel value shown as typed")
        fixed = question_mark(q)
        if fixed != q:
            log_change(r, "Question", q, fixed, "added '?'")
        named = name_kenya(fixed, out.get("Language")) if kind != "twin" else fixed
        if named != fixed:
            log_change(r, "Question", fixed, named, "added 'In Kenya, '")
        out["Question"] = named
        domain = tidy_domain(out.get("Domain"), spec["domain"])
        if domain != (out.get("Domain") or ""):
            log_change(r, "Domain", out.get("Domain"), domain, "domain mapped to the brief's six")
        out["Domain"] = domain
        if kind == "fake" and out.get("Answer"):
            log_change(r, "Answer", out["Answer"], "", "fake question: answer left blank (brief)")
            out["Answer"] = None
        if kind != "fake":
            answer, others = split_answer(out.get("Answer"), out.get("Other accepted answers"))
            if others != out.get("Other accepted answers"):
                log_change(r, "Other accepted answers", out.get("Other accepted answers"), others,
                           "split '/' or 'or' answer into accepted answers")
            out["Other accepted answers"] = others
        if kind in {"real", "fake"} and out.get("Twin of"):
            log_change(r, "Twin of", out["Twin of"], "",
                       "removed: a twin written in the Kenya row (proper twins: repairs.yaml)")
            out["Twin of"] = None
        if not out.get("Author"):
            out["Author"] = spec["author"]
            counts["Author filled from student_files.yaml"] += 1
        kept.append(out)
        final_question[r["row"]] = out["Question"]
        counts[f"kept: {kind}"] += 1

    # New rows written by the AI review: UK twins of kept Kenya questions, and replacement fakes.
    today = datetime.now().strftime("%Y-%m-%d")
    for t in repairs.get("twin", []):
        kenya = _find(rows, t["match"], who)
        if kenya["row"] not in final_question:
            raise ValueError(f"repairs.yaml, {who}: twin of row {kenya['row']}, which isn't kept")
        new = {"Question": question_mark(t["question"]), "Answer": t["answer"],
               "Other accepted answers": t.get("others"), "Type": "twin",
               "Twin of": final_question[kenya["row"]], "Domain": spec["domain"],
               "Language": "en", "Source link": t.get("source"), "Source note": t.get("note"),
               "Date checked": today, "Author": AI_AUTHOR}
        kept.append(new)
        seen.add(norm(new["Question"]))
        changes.append({"student": who, "row": f"new (twin of row {kenya['row']})",
                        "column": "row", "before": "", "after": new["Question"],
                        "why": "AI-written UK twin"})
        counts["AI-written UK twin"] += 1
    for a in repairs.get("add", []):
        if norm(a["question"]) in seen:
            raise ValueError(f"repairs.yaml, {who}: added question repeats one already kept")
        seen.add(norm(a["question"]))
        new = {"Question": question_mark(a["question"]), "Answer": None, "Type": a["type"],
               "Domain": spec["domain"], "Language": "en", "Source note": a.get("note"),
               "Date checked": today, "Author": AI_AUTHOR}
        kept.append(new)
        changes.append({"student": who, "row": "new", "column": "row", "before": "",
                        "after": new["Question"], "why": f"AI-written {a['type']}: {a['why']}"})
        counts[f"AI-written {a['type']}"] += 1
    return kept, changes, counts


def run(cfg: dict[str, Any], files: str | Path = "data/kenya/student_files.yaml") -> dict[str, Any]:
    from openpyxl import Workbook

    specs = yaml.safe_load(Path(files).read_text(encoding="utf-8"))
    repairs_path = Path(files).parent / "repairs.yaml"
    repairs = (yaml.safe_load(repairs_path.read_text(encoding="utf-8")) or {}
               if repairs_path.exists() else {})
    out_dir = ensure_dir(Path(cfg["data"]["kenya_questions"]).parent / "cleaned")
    seen = {norm(s) for s in STARTERS}
    wb = Workbook()
    wb.remove(wb.active)
    all_changes, summary = [], {}
    for spec in specs:
        rows = read_rows(Path(spec["file"]), spec["tab"])
        kept, changes, counts = clean_student(spec, rows, seen, repairs.get(spec["student"]))
        ws = wb.create_sheet(f"Student {spec['student']}")
        ws.append(COLUMNS)
        for k in kept:
            ws.append([k.get(c) for c in COLUMNS])
        all_changes += changes
        summary[f"Student {spec['student']} ({spec['author']})"] = dict(counts)
    cws = wb.create_sheet("Changes")
    keys = ["student", "row", "column", "before", "after", "why"]
    cws.append(keys)
    for ch in all_changes:
        cws.append([ch[k] for k in keys])
    wb.save(out_dir / "cleaned_questions.xlsx")
    with open(out_dir / "changes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_changes)
    report = Path(cfg["data"]["kenya_questions"]).parent / "cleaning_report.md"
    _write_report(report, specs, summary, all_changes)
    log.info("cleaned %d students: %d changes recorded", len(specs), len(all_changes))
    return {"students": summary, "changes": len(all_changes),
            "cleaned": str(out_dir / "cleaned_questions.xlsx"), "report": str(report)}


def _write_report(path: Path, specs: list, summary: dict, changes: list) -> None:
    kinds = sorted({k for c in summary.values() for k in c})
    lines = [
        "# Kenya questions: cleaning report", "",
        f"Made by `bluff clean-sheets` on {datetime.now():%Y-%m-%d %H:%M}. Every change is listed "
        "row by row in `cleaned/changes.csv` and in the **Changes** tab of "
        "`cleaned/cleaned_questions.xlsx`; the cleaned questions are in that workbook's student "
        "tabs. Nothing is imported until you run `bluff import-sheet`.", "",
        "## Sources", "",
        *[f"- **Student {s['student']}** ({s['author']}): `{s['file']}`, tab \"{s['tab']}\""
          for s in specs], "",
        "## What was done, per student", "",
        "| | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary),
        *[f"| {k} | " + " | ".join(str(c.get(k, 0)) for c in summary.values()) + " |"
          for k in kinds], "",
        "## Examples (first 3 of each kind)", "",
    ]
    by_kind: dict[str, list] = {}
    for ch in changes:
        by_kind.setdefault(ch["why"], []).append(ch)
    for why, items in by_kind.items():
        lines += [f"**{why}** ({len(items)})", ""]
        lines += [f"- {c['student']}, row {c['row']}: \"{str(c['before'])[:70]}\" → "
                  f"\"{str(c['after'])[:70]}\"" for c in items[:3]] + [""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
