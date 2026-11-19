# The Kenya questions: from the students' sheets to the model

How the students' spreadsheets become the Kenya test set, step by step, with the commands to run
it yourself. Nothing here runs in the background: each step writes files you can open and check
before the next one.

```
students' .xlsx files ──► 1. clean-sheets ──► cleaned_questions.xlsx + every change logged
                                                        │  (you check it)
                                                        ▼
                         2. import-sheet ──► data/kenya/questions.jsonl + import_report.md
                                                        │  (you check it)
                                                        ▼
                         3. validate + audit ──► 4. GitHub ──► 5. Kaggle/Colab run ──► results
```

All commands run in VS Code's terminal, inside the `bluff-detector` folder.

## 0. Put the sheets in place and say who is who

1. Download each student's sheet as **.xlsx** into the `Bluff_Detector` folder (next to
   `bluff-detector`).
2. Open **`data/kenya/student_files.yaml`**. One entry per student: which file, which tab, which
   letter in the brief (A–D), their name, and their domain. Update it when a student sends a new
   file, or when you learn a name (B and D are still "Student B" / "Student D").

As of 19 November:

| Student (brief) | Domain | File, tab | Name |
|---|---|---|---|
| A | Civics and law | `Bluff Detector - Kenya Question Sheet student b.xlsx`, tab "Student B" (he typed in the wrong tab) | James Gosha |
| B | Geography, society, language | `Geography_Questions.xlsx`, tab "Sheet1" | not given |
| C | Institutions, services, money | `Bluff Detector - Kenya Question Sheet 1.xlsx`, tab "Student C" | Michael |
| D | Western default traps | `project 2.xlsx`, tab "Student D" (its "Student A" tab is an earlier draft: ignored) | not given |

## 1. Clean: `bluff clean-sheets`

```
.venv\Scripts\bluff clean-sheets --config configs/variants/m0_base.yaml
```

Code: `src/bluff/sheet_clean.py` (each fix is one small function); tests:
`tests/data/test_sheet_clean.py`. It only makes **mechanical** fixes that don't change what a
question asks:

| Fix | Example (from the real sheets) |
|---|---|
| Add a missing "?" | "what is the total area of kenya" → "what is the total area of kenya?" |
| Add "In Kenya, " when the question never says which country | "Who is in charge of voter education?" → "In Kenya, who is in charge of voter education?" Without it, the model can't know the question is about Kenya and could fairly answer for another country. Not added to Kiswahili questions or ones that name a Kenyan place, body or term |
| Show Excel-damaged answers as typed | VAT "16%" stored as 0.16 → "16%"; "1 June" stored as 2026-06-01 → "1 June"; "27 August 2010" kept with its year |
| Map the Domain onto the brief's six | "civis-law" → civics_law; "Constitutional" → civics_law; blank or unknown → the student's domain |
| Split "X/Y" or "X or Y" answers | "999/112" → also accepts "999" and "112" |
| Fill Author when blank | from `student_files.yaml` |
| **Skip** rows that can't be used | empty rows; section headings ("CITIES AND TOWNS"); rows with no question text; real questions with no answer; questions about another country that never mention Kenya ("who was the fifth president of the UK"); exact repeats of an earlier question or of the 9 starter questions |

**What it writes (open these):**

- `data/kenya/cleaned/cleaned_questions.xlsx`: the cleaned questions, one tab per student in the
  template's layout, and a **Changes** tab listing **every change**: student, row, column,
  before, after, why.
- `data/kenya/cleaned/changes.csv`: the same list, for Excel.
- `data/kenya/cleaning_report.md`: counts per student, and 3 examples of each kind of change.

**First run (19 November):** kept **282 real + 70 fake = 352** questions (A 71 + 20, B 73 + 20,
C 66 + 20, D 72 + 10). Changes: 48 "In Kenya, " added, 87 "?" added, 13 Excel answers restored,
3 answers split; skipped: 16 rows with no question, 9 headings, 2 real questions with no answer,
10 of D's fakes about other countries, 4 repeats, and 3 of A's questions excluded in
`student_files.yaml` because they copy the Constitution's wording (found by the leakage audit,
step 3: "a code of conduct for candidates and parties", "land on behalf of the national and
county governments", "Commander-in-Chief of the Kenya Defence Forces").

**Imported and checked (19 November):** the question file holds **361 Kenya questions** (289 real,
72 fake: the students' 352 plus the 9 starters). `validate` passes (none verified by a second
reviewer yet); the leakage audit passes with 0 overlaps.

**Not fixed here (the students must):** UK twins (none usable: written in the wrong column with
no answers, or empty), unofficial sources, over-long answers, changing facts without "Valid
until", reused brief examples, missing names, the review step. Row by row:
`..\Kenya questions - fixes per student.md`.

## 1b. The AI review and repairs (19 November): `data/kenya/repairs.yaml`

Because of the deadline, Trevor had the gaps the students left repaired with AI (Claude) instead
of waiting for the students. `bluff clean-sheets` applies `data/kenya/repairs.yaml` after the
mechanical fixes and logs every repair. Per student: **drop** (a row left out, with the reason),
**edit** (changed cells, with the reason), **twin** (a UK twin of a kept Kenya question, with its
answer and an official UK source) and **add** (new questions). Each entry finds its row by words
from the student's own question and must match exactly one row, or the step stops.

| | A | B | C | D | Total |
|---|---|---|---|---|---|
| Dropped (duplicates across students, yes/no, essay or list answers, reserved brief examples, wrong or disputed answers, changing facts, unclear questions) | 3 | 13 | 34 | 31 | 81 |
| Edited (reworded the 3 questions that copied the Constitution, shortened answers, corrected or added accepted answers, spelling) | yes | yes | yes | yes | |
| UK twins written | 16 | 8 | 5 | 4 | 33 |
| Fakes written (replacing D's fakes about other countries) | | | | 10 | 10 |

**Result:** 326 items: 211 real Kenya questions, 82 fakes, 33 UK twins (including the 9
starters). `validate` passes; the leakage audit finds 0 overlaps. Every new row has Author
"Claude (AI)" and every edited row Reviewer "Claude (AI)", so results can be reported with and
without AI-written items (declared in `AI_USE.md`). To undo a repair, delete its entry and
re-run `notebooks/kenya_questions.ipynb`.

## 2. Import: `bluff import-sheet`

```
.venv\Scripts\bluff import-sheet --config configs/variants/m0_base.yaml --sheet data/kenya/cleaned/cleaned_questions.xlsx
```

Code: `src/bluff/sheet_import.py`. Turns each cleaned row into a question in
`data/kenya/questions.jsonl` (real → `kenya`, fake → `kenya_unanswerable`, twin → `western`),
gives it an id made from its text (so re-importing keeps every id), keeps the 9 starter
questions, and writes **`data/kenya/import_report.md`**: questions per author and any row it
skipped or warned about.

## 3. Check: `validate` and `audit`

```
.venv\Scripts\bluff validate --config configs/variants/m0_base.yaml
.venv\Scripts\bluff audit --config configs/variants/m1_rag.yaml
```

`validate` checks the data contract (ids, answers, pools). `audit` checks that no question's
wording appears in the RAG or fine-tuning documents (it needs the TriviaQA questions, so it runs
`ingest` first if needed).

## 4. Upload to GitHub

The Kaggle/Colab run downloads the code and the question file from GitHub:

```
git add data/kenya
git commit -m "Kenya questions: students' sets imported"
git push
```

## 5. Run the model on the new questions

On Kaggle (or Colab when it has GPU time): **`notebooks/run_m0_base.ipynb`** → Run All. It
resumes M0's saved run and asks **only the new questions**, then re-marks, retrains the
detectors and re-measures: the real **H2** (worse on Kenya?) and **H5** (does the detector
travel?). Then the same with **`notebooks/run_m1_rag.ipynb`** for M1.

## 6. When students send updates

Put the new file in `Bluff_Detector`, point `student_files.yaml` at it, and repeat steps 1–5.
Unchanged questions keep their ids and are not asked again; changed or new ones are.
