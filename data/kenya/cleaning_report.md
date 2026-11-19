# Kenya questions: cleaning report

Made by `bluff clean-sheets` on 2026-11-19 12:39. Every change is listed row by row in `cleaned/changes.csv` and in the **Changes** tab of `cleaned/cleaned_questions.xlsx`; the cleaned questions are in that workbook's student tabs. Nothing is imported until you run `bluff import-sheet`.

## Sources

- **Student A** (James Gosha): `../Bluff Detector - Kenya Question Sheet student b.xlsx`, tab "Student B"
- **Student B** (Student B): `../Geography_Questions.xlsx`, tab "Sheet1"
- **Student C** (Michael): `../Bluff Detector - Kenya Question Sheet 1.xlsx`, tab "Student C"
- **Student D** (Student D): `../project 2.xlsx`, tab "Student D"

## What was done, per student

| | Student A (James Gosha) | Student B (Student B) | Student C (Michael) | Student D (Student D) |
|---|---|---|---|---|
| Author filled from student_files.yaml | 0 | 93 | 0 | 82 |
| Excel value shown as typed | 9 | 2 | 1 | 1 |
| added '?' | 0 | 0 | 5 | 82 |
| added 'In Kenya, ' | 16 | 11 | 17 | 3 |
| domain mapped to the brief's six | 45 | 83 | 86 | 82 |
| excluded in student_files.yaml: copies the Constitution's wording (Chapter 5): leakage audit; rewrite in own words | 1 | 0 | 0 | 0 |
| excluded in student_files.yaml: copies the Constitution's wording (Chapter 7): leakage audit; rewrite in own words | 1 | 0 | 0 | 0 |
| excluded in student_files.yaml: copies the Constitution's wording (Chapter 9): leakage audit; rewrite in own words | 1 | 0 | 0 | 0 |
| kept: fake | 20 | 20 | 20 | 10 |
| kept: real | 71 | 73 | 66 | 72 |
| skipped: about another country, not Kenya | 0 | 0 | 0 | 10 |
| skipped: no Type (a heading or unfinished row) | 0 | 8 | 0 | 1 |
| skipped: no answer (a real or twin question) | 0 | 1 | 1 | 0 |
| skipped: no question text | 15 | 0 | 0 | 1 |
| skipped: repeats an earlier question or a starter | 0 | 1 | 2 | 1 |
| split '/' or 'or' answer into accepted answers | 0 | 0 | 0 | 3 |

## Examples (first 3 of each kind)

**added 'In Kenya, '** (47)

- Student A, row 2: "Who is incharge of the registration of candidates for election?" → "In Kenya, who is incharge of the registration of candidates for electi"
- Student A, row 3: "Who is in charge of the settlement of electoral disputes?" → "In Kenya, who is in charge of the settlement of electoral disputes?"
- Student A, row 4: "Who is in charge of voter education?" → "In Kenya, who is in charge of voter education?"

**domain mapped to the brief's six** (296)

- Student A, row 2: "civis-law" → "civics_law"
- Student A, row 3: "civis-law" → "civics_law"
- Student A, row 4: "civis-law" → "civics_law"

**excluded in student_files.yaml: copies the Constitution's wording (Chapter 7): leakage audit; rewrite in own words** (1)

- Student A, row 7: "Who is in charge of developing a code of conduct for candidates and pa" → ""

**excluded in student_files.yaml: copies the Constitution's wording (Chapter 5): leakage audit; rewrite in own words** (1)

- Student A, row 16: "who manages land on behalf of the national and county governments?" → ""

**Excel value shown as typed** (13)

- Student A, row 24: "2010-08-27 00:00:00" → "27 August 2010"
- Student A, row 35: "2026-06-01 00:00:00" → "1 June"
- Student A, row 36: "2026-10-20 00:00:00" → "20 October"

**excluded in student_files.yaml: copies the Constitution's wording (Chapter 9): leakage audit; rewrite in own words** (1)

- Student A, row 75: "Who is the Commander-in-Chief of the Kenya Defence Forces under the Co" → ""

**skipped: no question text** (16)

- Student A, row 96: "twin" → ""
- Student A, row 97: "twin" → ""
- Student A, row 98: "twin" → ""

**skipped: no Type (a heading or unfinished row)** (9)

- Student B, row 13: "CITIES AND TOWNS" → ""
- Student B, row 41: "COMMUNITIES AND CULTURE" → ""
- Student B, row 54: "NATIONAL SYMBOLS OF KENYA" → ""

**skipped: repeats an earlier question or a starter** (4)

- Student B, row 14: "What is the capital city of Kenya?" → ""
- Student C, row 42: "Which department issues Kenyan passports?" → ""
- Student C, row 56: "Which agency certifies horticultural exports for phytosanitary complia" → ""

**skipped: no answer (a real or twin question)** (2)

- Student B, row 25: "CENSUS AND POPULATION FIGURES" → ""
- Student C, row 21: "when was the establishment of NTSA?" → ""

**added '?'** (87)

- Student C, row 57: "What is Kenya's leading foreign-exchange-earning agricultural export" → "What is Kenya's leading foreign-exchange-earning agricultural export?"
- Student C, row 58: "Which regulator licenses the NSE" → "Which regulator licenses the NSE?"
- Student C, row 66: "Which Act of Parliament established the capital markets authority" → "Which Act of Parliament established the capital markets authority?"

**split '/' or 'or' answer into accepted answers** (3)

- Student D, row 49: "None" → "National government; County government"
- Student D, row 51: "Mkuki" → "Mkuki; Shield; Spears"
- Student D, row 53: "swahili/kiingeereza" → "swahili/kiingeereza; English; Kiswahili"

**skipped: about another country, not Kenya** (10)

- Student D, row 82: "what is the tallest mountain in the UK" → ""
- Student D, row 83: "who was the fifth president of the UK" → ""
- Student D, row 84: "how how often do they celebrate labour day in germany" → ""

