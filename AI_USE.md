# AI use disclosure

Required by the DSA 8401 capstone brief. Edit this as the project evolves.

| Date | Tool | What it helped with | What I did |
|---|---|---|---|
| Oct 2026 | Claude (Anthropic) | Brainstorming project ideas; literature search on hallucination probes and abstention; drafting the proposal | Chose the project and its framing; rewrote the proposal in my own words |
| Oct 2026 | Claude (Anthropic) | Generating the initial repository scaffold: pipeline stages, detectors, metrics, API, Docker, CI and tests | Reviewed, ran and modified the code; I am responsible for every decision it encodes |
| Oct 2026 | Claude (Anthropic) | Publishing the repository: README rewrite, banner and diagram SVGs, LICENSE, CITATION.cff; pinning the Trivy action in CD by commit SHA | Asked for the repository to be published; I am responsible for its content |
| Oct 2026 | Claude (Anthropic) | Documentation: the architecture guide, the three-week plan, and the description at the top of each file, organised by CRISP-DM; the UK-twin comparison (code and tests) | Set the deadline, the CRISP-DM structure and the scope; I am responsible for the plan |
| 15–16 Nov 2026 | Claude Code | GPU runs on free hardware: automatic float16/bfloat16, pinned model revision, resumable generation (and its fixes), the Colab/Kaggle runner notebooks (`notebooks/run_*.ipynb`, `scripts/make_run_notebooks.py`) | Ran every GPU job (Colab, Kaggle) and reported each error; chose the 5-day scope |
| 16 Nov 2026 | Claude Code | Found that TruthfulQA's automatic marking was broken (98% marked wrong) and retrained on TriviaQA only; the analysis notebook (`notebooks/analysis.ipynb`) and its charts; `bluff export` | Decided to drop TruthfulQA; I interpret the results in the report |
| 16 Nov 2026 | Claude Code | The live-guard demo page (`app/static/index.html`), laptop serving (`scripts/serve_laptop.ps1`), monitoring (`bluff monitor`), the Kenyan document collection for RAG and fine-tuning (`bluff build-corpus`, `data/corpus/sources.yaml`) | Chose to build M1 (RAG) |
| 18–19 Nov 2026 | Claude Code | Importing and cleaning the students' sheets (`bluff import-sheet`, `bluff clean-sheets`, `notebooks/kenya_questions.ipynb`) and per-student fix lists | Collected the sheets; matched files to students |
| 19 Nov 2026 | Claude Code | **An AI review and repair of the Kenya question set** (`data/kenya/repairs.yaml`; see below) | Decided, because of the deadline, to repair the gaps with AI instead of waiting for the students |
| 19–20 Nov 2026 | Claude Code | Fixes that let the M2 fine-tuning run on Kaggle (`finetune_qlora.py` settings, a peft/torchao workaround); the results notebook (`notebooks/results.ipynb`: H1–H5, the H4 split by whether the answer was in the passages, the H3 comparison); plain-language explanations in `docs/concepts.md` | Ran M1 and M2 on Kaggle; asked for the H4 split; I interpret the results in the report |
| 20 Nov 2026 | Claude Code | The redesigned demo page (three tabs, the "words vs inside" readout, the results tab and `GET /v1/results`); storing every artefact (the M2 adapter in git, M1/M2 dataset exports, M1/M2 analysis notebooks and charts, `docs/where-things-are.md`) | Asked for the redesign and for everything to be stored; reviewed the page |
| 20 Nov 2026 | Claude Code | Serving M0, M1 and M2 from one model with any detector deciding (`BLUFF_VARIANTS`, `GET /v1/variants`, the adapter switch); plain-language explanations of every reading on each answer; every metric by version on the results tab | Asked for the version and detector choice, the explanations and all the metrics on the page |
| 20 Nov 2026 | Claude Code | The M3 analysis: the comparisons with M1 and M2, the M3 section of `notebooks/results.ipynb`, `analysis_m3_rag_qlora.ipynb`, M3 on the demo page | Ran M3 on Kaggle; I interpret the results in the report |
| 20 Nov 2026 | Claude Code | A study guide explaining every result and measurement (`docs/results-explained.md`) | Asked for it to learn from; the report itself, and its interpretation, are in my own words |

## The Kenya question set: what is human-written and what is not

The questions were written by four students and me, following the Student Brief, which asked for
a human-written set (rule 9). The students' review step was not done, and the UK twins were
unusable, so on 19 November I had the set reviewed and repaired by Claude. Every change is in
`data/kenya/repairs.yaml` (with its reason) and logged row by row in
`data/kenya/cleaned/changes.csv`; provenance is kept in the data itself:

| Kind | Count | How to tell in `data/kenya/questions.jsonl` |
|---|---|---|
| Human-written, unchanged except mechanical fixes ("?", "In Kenya, ", Excel values, domain names) | most real and fake questions | `author` = a student or Trevor |
| Human-written, edited by the AI review (reworded, shortened answers, added or corrected accepted answers) | see `repairs.yaml`, `edit:` | `reviewer` = "Claude (AI)" |
| Written by the AI review | 33 UK twins, 10 fake questions (replacing Student D's fakes about other countries) | `author` = "Claude (AI)" |
| Dropped by the AI review (duplicates, yes/no, essay or list answers, reserved examples, wrong or disputed answers, changing facts) | 81 | not in the file; listed in `repairs.yaml`, `drop:` |

Because AI-written test items could bias a test of an AI model (the reason for rule 9), the
results should be reported **with and without** the AI-written items (`author` = "Claude (AI)").

Not AI-generated:

- Experimental results, their interpretation, and the technique justifications in the final report.
