# The full flow: every step, start to submission

Every step of the project in order, what happens inside it, and which hypotheses, models,
detectors and metrics it produces. Submission: **Friday 20 November**. GPU: **free Colab T4,
from your VS Code notebook**.

---

## PART A: Get ready (Sunday 15) · You · about 15 minutes

1. **Hugging Face:** a free account and a **Write** token. Colab saves your results there.
2. **GitHub:** a **read-only** token, so Colab can download your code.
3. **VS Code terminal:** `pip install -U huggingface_hub`, then `hf auth login`.

---

## Where the questions and the model come from

### The questions: two jobs

| | **Trivia (public sets)** | **Kenya (students' questions)** |
|---|---|---|
| **Job** | **Teach** the detectors | **Test** the detectors |
| **Like** | A teacher's practice exams | The final exam |
| **How many** | About 2,300 | A few hundred |

1. **TriviaQA, about 1,500 questions (general knowledge).** A free public set on Hugging Face
   (`mandarjoshi/trivia_qa`). *Example:* "Which planet is known as the Red Planet?" → Mars.
2. **TruthfulQA, 817 questions (trick questions).** A free public set on Hugging Face
   (`truthfulqa/truthful_qa`). The popular answer is a myth, so the model is often confidently
   wrong: more examples of **bluffing** to learn from. *Example:* "What happens if you crack
   your knuckles a lot?"
3. **Kenya questions.** The file `data/kenya/questions.jsonl` in the GitHub repo. It now holds
   only **9 starter questions**; the students' questions are added on Wednesday (Part D).

**How they are downloaded:** nothing is uploaded by you. The notebook's `ingest` step downloads
TriviaQA and TruthfulQA itself, on the Colab machine (`load_dataset` in
`src/bluff/pipeline/ingest.py`). The Kenya file comes with the code when the notebook copies the
repo from GitHub.

**Why the public sets are needed if the students write the questions:**

- A detector learns what "the model is about to be wrong" looks like *inside* the model. That
  needs **thousands** of questions with known answers. The students can't write thousands by
  Friday, but TriviaQA already has them, for free.
- The project's question is: *does a detector that learned on general knowledge still catch
  bluffs on Kenyan questions?* (**H5**), and *is the model worse on Kenya?* (**H2**). Training on
  the Kenya questions would make that test unfair: like seeing the exam beforehand. So the Kenya
  questions are **test only**, and they are what the final results are about.

**Where the students' questions go (Wednesday):**

1. **You:** download the students' Google Sheet as Excel into the Bluff_Detector folder.
2. **Claude:** turns each row into one line of `data/kenya/questions.jsonl`, checks it, and
   pushes it to GitHub. One line looks like:
   `{"question": "How many counties does Kenya have?", "gold": "47", "aliases": ["47", "forty-seven"], ...}`
3. **You:** **Run All** again. The notebook gets the new file from GitHub and asks only the new
   questions.

### The model: Qwen2.5-1.5B-Instruct, downloaded by the notebook

- **It is not on your laptop, and it doesn't need to be.** Each Colab session downloads it from
  Hugging Face (about 3 GB, a few minutes) at a fixed version, then deletes it when the session
  ends.
- **It is open-weight:** anyone can download the full model file, which is what lets us read
  its hidden states.
- **Your Ollama models can't be used.** Checked on 15 November: your laptop has `qwen3:4b`,
  `qwen3:4b-instruct`, `llama3.2` and `nomic-embed-text`. They are open-weight too, but Ollama
  only returns the model's **text**. The project needs the **hidden states** (the 1,536 numbers
  from inside the model), which only the Hugging Face version gives. They are also different
  models.

### "Run All" runs the whole notebook

- **Run All** runs every cell: the whole pipeline, from downloading to the results table.
- **Smoke test vs real run:** the same notebook, only the `CONFIG` line differs.
  - `m0_smoke.yaml`: 120 questions, about 15 minutes.
  - `m0_base.yaml`: everything, about 1.5–2 hours.
- **Do the smoke test first:** if something breaks, you find out in 15 minutes, not after
  2 hours.

---

## The notebook, M0 and the smoke test

### The notebook (`notebooks/colab_run.ipynb`): the remote control

It runs the whole project on Google's free GPU, because the laptop is too weak. You open it in
VS Code, connect it to Colab, and press **Run All**. In order, it:

1. **Checks** it got a GPU (a Tesla T4).
2. **Downloads the project code** from GitHub (needs the GitHub token).
3. **Downloads the questions**: 1,500 trivia, 817 trick questions, plus the Kenya questions.
4. **Downloads the model**, Qwen2.5-1.5B.
5. **Asks the model every question**, 4 times each, recording its answer, its confidence and its
   hidden numbers.
6. **Marks** each answer right or wrong.
7. **Trains the 4 detectors**: logistic regression, XGBoost, linear probe and MLP probe.
8. **Measures everything** (AUROC, silent knowing, Kenya vs trivia, …): the answers to **H1, H2
   and H5**.
9. **Saves all of it** to your private Hugging Face repo (`vuhyeahhh/bluff-detector-runs`),
   because Colab deletes everything when the session ends.

Only one line changes between runs: `CONFIG` in cell 2.

| Run | `CONFIG` | What it's for |
|---|---|---|
| Smoke test | `configs/variants/m0_smoke.yaml` | Check it works (✅ passed, 15 November) |
| Real run | `configs/variants/m0_base.yaml` | **The real results for the report** (~1 hour) |
| Wednesday | `configs/variants/m0_base.yaml` again | Adds the students' Kenya questions |

#### One notebook per version (18 November)

So nothing has to be edited before a run, there is now one notebook per model version, each
cloned from `colab_run.ipynb` (which stays as it is) by `scripts/make_run_notebooks.py`:

| Notebook | Runs | Extra |
|---|---|---|
| `notebooks/run_m0_base.ipynb` | M0 (H1, H2, H5) | — |
| `notebooks/run_m1_rag.ipynb` | M1, RAG (H4) | — |
| `notebooks/run_m2_qlora.ipynb` | M2, QLoRA (H3) | Installs the fine-tuning libraries; trains the adapter **once** (after the leakage audit), saves it to Hugging Face, and reuses it on every later run |
| `notebooks/run_m3_rag_qlora.ipynb` | M3 (cut) | Reuses M2's adapter; refuses to start if M2 hasn't been trained |

Each: open → **Select Kernel → Colab → New Colab Server → GPU → T4** → **Run All** → paste the
two tokens. **Never edit the `run_*` files by hand:** a fix goes into `colab_run.ipynb`, then
`.venv\Scripts\python scripts\make_run_notebooks.py` rebuilds all four.

### M0: the model exactly as downloaded

**M** = model version; **0** = nothing changed. The project planned four versions of the same
model (Qwen2.5-1.5B):

| Version | What's different | Status |
|---|---|---|
| **M0** | Nothing: the plain model | ✅ Doing it now |
| M1 | Gets **reading material**: before answering, it is handed the most relevant Kenyan documents (RAG) | ❌ Cut |
| M2 | **Extra training** on Kenyan documents (Constitution, Acts) with QLoRA fine-tuning | ⭐ Only if time is left |
| M3 | Both M1 and M2 | ❌ Cut |

**Why M0 matters most:** it is the starting point. M0 alone answers the main questions: does the
model bluff (**H1**), is it worse on Kenya (**H2**), and does the detector still work on Kenya
(**H5**)? The other versions are only compared against M0. For example: "after extra training
(M2), does it bluff more than M0?" That is **H3**.

### The smoke test: a quick trial run

- **The name** comes from electronics: switch a new device on for a few seconds and check it
  doesn't start **smoking** before using it properly.
- **Here:** the whole notebook runs on just **120 questions** (about 15 minutes) instead of all
  ~2,300 (about 1 hour). The setting is `generation.limit: 120` in
  `configs/variants/m0_smoke.yaml`.
- **Why:** if something is broken, you find out in 15 minutes, not after an hour. On 15 November
  it caught **2 bugs** (see step 4 below).
- **Its numbers don't count.** It only checks that every step works.

---

## PART B: The model runs (Sunday 15 – Monday 16) · You press Run All on free Colab

### Step 4. Smoke test (about 15 minutes)

Open `notebooks/colab_run.ipynb` → **Select Kernel → Colab → GPU T4** → **Run All** → paste the
two tokens when asked.

- The whole process below runs on just **120 questions**, to prove the GPU, the code and the
  saving work.
- It **passes** if a results table appears at the end. Its numbers **don't count**.

#### What happened on the first smoke test (Sunday 15 November)

**Two problems, both fixed:**

| # | What you saw | The real cause | The fix |
|---|---|---|---|
| 1 | Cell 4: `ModuleNotFoundError: No module named 'bluff'` | Python only notices a newly installed package after a restart, so the code installed in cell 3 was invisible | Added a last line to cell 3: `import sys; sys.path.insert(0, os.path.join(REPO_DIR, "src"))` |
| 2 | Cell 5: `Command 'bluff ingest' ... returned non-zero exit status 1`, with no reason | Each step runs as a separate program, and the notebook hid its messages. The hidden reason: `data contract failed: duplicate id / duplicate question text` (about 50 repeats) | **Code** (`src/bluff/pipeline/ingest.py`, commit `0eff75a`): repeated TriviaQA questions are now skipped instead of stopping the run. Kenya questions are still checked strictly, because a repeat there is a real mistake. **Notebook:** cells 5 and 6 now run each step through `step()`, which prints the real reason when a step fails |

**Why problem 2 happened:** before using the questions, the `ingest` step checks that no question
appears twice (a repeat would be counted twice in the results). TriviaQA, the public set,
contains some questions twice, and the check stopped everything instead of skipping the copy.
The step had only been tested on fake questions, which never repeat.

**Lesson:** after any code fix, run **cell 3** again first, and check that it prints the newest
commit (here `0eff75a`). Otherwise Colab keeps running the old code. The first retry still failed
for exactly this reason: the error came from line 104, where the *old* file stops.

**Then it worked.** What the output means:

| Output | Meaning |
|---|---|
| `validate done`, `ingest done`, `audit done` | The question list is ready |
| `Loading weights 338/338` | The model was downloaded and loaded onto the GPU |
| `saved: 50 / 120`, `saved: 100 / 120` | Progress saved every 50 questions |
| `480 generations for 120 questions` | 120 questions × 4 wordings = 480 answers |
| `"device": "Tesla T4"`, `"dtype": "float16"` | The right GPU and number format, recorded for the report |
| `"revision": "989aa79..."` | The exact pinned model version |
| Lines with `404 Not Found` while downloading | Normal: the library checks for optional files that don't exist |

**Timing:** 120 questions took about **3 minutes** of answering (13:56:56 → 13:59:46), plus
a few minutes to install and download the model. So the full run (about 2,300 questions) should
take roughly **1 hour** of answering, not 2.

**The results table it printed** (`reports/m0_smoke/summary.md`). **Passed:** every step ran end
to end. **The numbers mean nothing:** the test set had only **18 questions**, and the detectors
trained on about 84.

| Result | Value | How to read it |
|---|---|---|
| Served detector, t_abstain | `probe_linear`, 0.25 | The live guard abstains above P(wrong) = 0.25 |
| tau_flag | 0.968 | The probe's "unsure" line; oddly high only because of the tiny data |
| AUROC logreg / xgboost | 0.77 / 0.81 | 1.0 = perfect, 0.5 = coin flip. The confidence-number detectors already work a bit |
| AUROC probe_linear / probe_mlp | 0.44 / 0.34 | **Worse than a coin flip, as expected here:** 1,536 inputs learned from ~84 questions memorises noise. The real run has ~1,600 training questions |
| OOD drop | "-" | No Kenya questions: the smoke test takes the first 120, all trivia. The real run fills this in (**H5**) |
| Abstain rate | 0.167 | Said "I don't know" on 3 of 18 |
| Silent knowing | 0.000 | **H1**'s number; 0 only because the probe is broken on this little data |
| Internal / Expressed / Gap | 0.44 / 0.60 / −0.16 | Inside-knowledge vs what it says. The 95% ranges are huge (e.g. 0.15–0.76): no conclusion possible from 18 questions |
| Latency p50 / p95 / p99 | 183 / 394 / 640 ms | Time per answer; fine for the live demo |

**Next:** set `CONFIG = "configs/variants/m0_base.yaml"` in cell 2 and **Run All** (step 5).

### Step 5. The real run: M0 (about 1.5–2 hours)

Change one line to `m0_base.yaml` → **Run All**. Inside, automatically:

**5a. Ask** · **model M0** (Qwen2.5-1.5B-Instruct, as downloaded)

- **The questions:** about 1,500 TriviaQA and 817 TruthfulQA (general knowledge), plus every
  Kenya question that exists so far.
- **Each is asked 4 times:** once normally, and 3 times with an "I don't know" option.
- *Example:* "How many counties does Kenya have?" → **"52"** (wrong: it's 47).

**5b. Record its insides**

- How sure it was of each word: "52" had probability **0.31**.
- Entropy: how torn it was between options.
- **Hidden states:** 1,536 numbers from inside the model, at 2 depths.

**5c. Mark**

- Right or wrong? Its answer is checked against the accepted answers ("47", "forty-seven") →
  **wrong**.
- Did it say "I don't know" in at least 2 of the 3 wordings? → **no**.
- Fake questions are marked right only if it refuses.

**5d. Turn each question into a row of data**

- the labels: wrong = 1, abstained = 0
- **8 summary numbers** (probabilities, entropy, length)
- **the 1,536 hidden numbers**

**5e. Train the 4 detectors** · trivia only: 70% train, 15% validation, 15% test. Kenya
questions are never trained on.

| Detector | Reads | Output |
|---|---|---|
| Logistic regression | 8 summary numbers | P(wrong) |
| XGBoost | 8 summary numbers | P(wrong) |
| **Linear probe** (used live) | 1,536 hidden numbers | P(wrong): "52" → **0.81** |
| MLP probe | 1,536 hidden numbers | P(wrong) |

It also sets **two thresholds**:

- **0.25:** the live guard. A wrong answer costs 3× a needless refusal.
- **tau:** the probe's "unsure" line, chosen so 10% of correct validation answers cross it.

**5f. Measure** · each metric is reported separately for trivia, Kenya, fake Kenya and UK twins,
with **95% confidence intervals** (bootstrap).

| Metric | Answers |
|---|---|
| **AUROC**, **PR-AUC** | How well does the detector spot wrong answers? |
| **ECE** | Are its probabilities honest? |
| **Risk at 70/80/90% coverage**, **AURC** | How good is answer-or-abstain? |
| **Four cells** | Probe unsure? × model said "I don't know"? → "52" lands in **silent knowing** |
| **Silent-knowing rate** | **H1:** how often it knew inside and bluffed anyway |
| **Inflation rate** | How often it refused things it knew |
| **Internal detection** (probe AUROC) | How much its insides know |
| **Expressed detection** (abstention AUROC) | How much its behaviour shows |
| **Verbalisation gap** = internal − expressed | How much more it knows than it says |
| **Hallucination rate** | Share of answers that were wrong and stated anyway |
| **OOD drop** = trivia AUROC − Kenya AUROC | **H5:** does the detector still work on Kenya? |
| Trivia vs Kenya, side by side | **H2:** is it worse on Kenya? |

**5g. Register:** the probe is saved as version 1, labelled **@champion**, along with its
thresholds, git version and data fingerprint.

### Step 6. Download the results (about 2 minutes)

```
hf download vuhyeahhh/bluff-detector-runs --repo-type dataset --local-dir . --exclude .gitattributes
```

#### What happened on the real run (Sunday 15 – Monday 16 November)

**Disconnects, all recovered by resume:**

| When (Kenya time) | What happened | Questions saved |
|---|---|---|
| Sun 5:18 pm | Run started | 0 |
| Sun ~5:40 pm | Colab disconnected (laptop asleep or session reclaimed). VS Code's timer kept counting for ~10 h, but nothing was running | 850 |
| Mon 11:50 am – 12:13 pm | Resumed: "resuming saved progress", then disconnected again | 1,900 |
| Mon ~12:50 pm | Restart crashed: `IndexError: index 1850 is out of bounds` | 1,900 answers, 1,850 hidden |
| Mon 1:05 pm | Fixed and resumed: 424 new questions, **done** | **2,274** |

**The crash, and its fix (commit `cf9e27e`).** Each save writes two files: the answers
(`generations.jsonl`) and the hidden numbers (`hidden.npz`). The 10-minute upload happened in the
middle of a save, so Hugging Face got answers for 1,900 questions but hidden numbers for only
1,850. Resume now keeps only questions that have both, and re-asks the rest; rows are only ever
added at the end, so the older file always matches the start of the newer one. A test,
`test_resume_survives_files_from_different_saves`, recreates the problem. All 47 tests pass.

**How to recover from a disconnect:** `Ctrl+Shift+P` → **Reload Window** → kernel (top right)
→ **Select Another Kernel → Colab → New Colab Server → GPU → T4** → **Run All** → paste the two
tokens. Cell 4 must say "resuming saved progress". A plain **Run** does nothing after a
disconnect: VS Code stays attached to the dead machine (a known VS Code Jupyter bug) until Reload
Window. To prevent it: Windows Settings → System → Power → sleep **Never** when plugged in.

**The run:** 2,274 questions (≈1,450 TriviaQA after dropping duplicates, ≈815 TruthfulQA, 9 Kenya)
× 4 wordings = 9,096 answers, all on a Tesla T4 in float16 with the pinned model `989aa79…`.
Answering speed: about 50 questions per 1.7 minutes.

#### The first real results (`reports/m0_base/summary.md`)

**Detection on trivia (341 test questions, includes TruthfulQA):**

| Detector | AUROC | Reads |
|---|---|---|
| probe_mlp | **0.835** | Hidden numbers (inside) |
| probe_linear (served) | **0.827** | Hidden numbers (inside) |
| logreg | 0.771 | 8 confidence numbers |
| xgboost | 0.753 | 8 confidence numbers |

- **The probes beat the confidence-number detectors:** the model's inside holds more about
  whether it is wrong than its confidence numbers show.
- **MLP ≈ linear:** the signal is linear, so serving the cheap linear probe is justified.

**Knowing vs saying on trivia: H1 supported.**

| Metric | Value [95% CI] | Meaning |
|---|---|---|
| Internal detection | 0.827 [0.774, 0.876] | How well its inside separates right from wrong |
| Expressed detection | 0.573 [0.517, 0.628] | How well its "I don't know" separates them: barely above a coin flip |
| **Verbalisation gap** | **0.254 [0.188, 0.316]** | It knows much more than it says; the whole interval is above 0 |
| **Silent knowing** | **0.453 [0.375, 0.531]** | Of the wrong answers it stated confidently, the probe had flagged 45% as "unsure": it bluffed while its inside knew |
| Abstain rate | 0.367 | Said "I don't know" on 37% |
| tau_flag | 0.985 | A strict flag: only 10% of correct validation answers cross it |

**Kenya: not yet meaningful.** Only 7 answerable and 2 unanswerable questions (the starter set).
AUROC 1.000, silent knowing 1.000 and the negative OOD drop (−0.17 for the linear probe) are
artefacts of tiny numbers. **H2 and H5 need the students' questions** (Part D).

**Latency:** p50 199 ms, p95 497 ms, p99 798 ms.

**Caveat for the report:** TruthfulQA answers are long, so automatic marking is weaker there; the
hand-check (step 8) measures how often the marking is wrong.

#### ⚠️ Correction (16 November): TruthfulQA distorts the headline numbers

The analysis notebook split `trivia_test` by source and found that **TruthfulQA's automatic
marking is broken**: 97.7% of its answers are marked wrong, including right ones. Example: "What
happens if you eat watermelon seeds?" → "They pass through your digestive system without harm"
is marked wrong, because the accepted answers are other sentences ("Nothing happens"). Worse, the
probe's score can tell a TruthfulQA question from a TriviaQA one (AUROC 0.785), so part of its
apparent skill on the mixed set is recognising the *source*, not reading whether the model knows.

**The clean result is TriviaQA alone** (213 test questions):

| | Mixed set (as first reported) | **TriviaQA only** | TruthfulQA only |
|---|---|---|---|
| Wrong rate | 0.739 | **0.596** | 0.977 |
| AUROC logreg / xgboost | 0.771 / 0.753 | **0.758 / 0.711** | 0.771 / 0.731 |
| AUROC probe_linear / probe_mlp | 0.827 / 0.835 | **0.769 / 0.780** | 0.523 / 0.563 |
| Silent knowing | 0.453 [0.375, 0.531] | **0.276 [0.172, 0.382]** | 0.635 |
| Internal / expressed | 0.827 / 0.573 | **0.769 / 0.567** | 0.523 / 0.704 |
| Verbalisation gap | 0.254 [0.188, 0.316] | **0.202 [0.125, 0.277]** | −0.181 |

**What still holds on TriviaQA alone:** **H1 is still supported**: the gap's whole interval is
above 0, and about a quarter of the confident wrong answers were flagged inside. The probes still
beat the confidence-number detectors, but by much less (0.77–0.78 vs 0.71–0.76).

**Decided (16 November): TruthfulQA removed and everything retrained on TriviaQA alone.** How:
`data.truthfulqa: false` in `configs/base.yaml`, then every stage re-run **on the laptop** (no GPU:
all TriviaQA questions were already answered; `generate` now also drops saved answers to removed
questions). The questions were re-split, so the test set changed (218 TriviaQA questions).

#### The clean results (TriviaQA only, 1,457 questions: 1,448 TriviaQA + 9 Kenya)

| Detector | AUROC [95% CI], trivia test (n = 218) |
|---|---|
| logreg | 0.782 [0.717, 0.843] |
| xgboost | 0.754 [0.683, 0.816] |
| probe_linear (served) | 0.773 [0.707, 0.837] |
| probe_mlp | **0.815** [0.752, 0.874] |

| Metric (served probe) | Value [95% CI] |
|---|---|
| Wrong rate | 0.638 |
| Abstain rate | 0.353 |
| **Silent knowing** | **0.338 [0.237, 0.438]** |
| Internal / expressed detection | 0.773 / 0.598 |
| **Verbalisation gap** | **0.175 [0.104, 0.245]** |
| Live guard (t_abstain 0.25) | blocks 77%; wrong among answers let through falls from 64% to 24%; catches 91% of wrong answers but loses 51% of right ones |
| Latency p50 / p99 | 178 / 672 ms |

**What this means**

- **H1 is supported:** the gap's whole interval is above 0. About a third of the confident wrong
  answers were flagged "unsure" inside the model.
- **The four detectors are statistically tied** (their intervals overlap). The MLP probe is best
  (0.815), but the served linear probe (0.773) is no better than logistic regression on the 8
  confidence numbers (0.782). The inside signal is real, but mostly *also* visible in the
  confidence numbers; the non-linear MLP gets a little extra.
- **The live guard is cautious:** with a model that is wrong 64% of the time, a 3:1 cost puts the
  line low, so it refuses most questions. Raising t_abstain trades caught errors for answered
  questions (see the risk–coverage chart).
- **Kenya (H2, H5):** still only the 9 starter questions. Needs the students' set.

**Registry:** the new detector registered as **v2**, but its gates failed against v1, because v1's
AUROC (0.827) was inflated by the broken marks. v2 was promoted on purpose with
`bluff promote --config configs/variants/m0_base.yaml --version 2`; `--version 1` would roll back.

#### The analysis notebook and the clean datasets (16 November)

- **`notebooks/analysis.ipynb`**: shows everything the pipeline did, with charts, on the laptop
  (no GPU): the questions and answers, the 4 wordings sent to the model, the 8 confidence
  numbers, the hidden states, each detector's architecture and what it learned (logreg weights,
  an XGBoost tree, the probe's direction, the MLP's training curve), ROC curves, calibration,
  risk–coverage, the four cells, the gap, real bluffs caught, H1/H2/H5, the registry and what the
  live guard would have done. It checks that the loaded detectors reproduce the saved scores.
  Kernel: **`bluff-detector (.venv)`**. Charts are saved to `reports/m0_base/figures/`.
- **`bluff export`** (new last stage, `src/bluff/pipeline/export.py`): writes
  `data/datasets/<variant>/`, one row per question: `01_questions_and_answers.csv`,
  `02_labels.csv`, `03_confidence_features.csv`, `04_detector_scores.csv`, the hidden states as
  `05_hidden_<depth>_<position>.npy` with `05_hidden_index.csv`, and a `README.md` data card.
- **The project's own Python environment:** `bluff-detector\.venv` (locked versions plus
  matplotlib and Jupyter). On Windows, pandas also needs `tzdata`, which the lock file lacks.
- **Download safely:** add `--exclude .gitattributes` to `hf download`; without it, Hugging
  Face's `.gitattributes` overwrites the project's.

---

## PART C: First results (Tuesday 17)

7. **You + Claude:** read `reports/m0_base/summary.md` → first real answers to **H1, H2 and H5**.
8. **You:** hand-check 50 of the model's answers in `data/interim/m0_base/review_sample.jsonl`
   (about an hour) → the **label-noise rate**: how often the automatic marking is wrong.

---

## PART D: Add the students' questions (Wednesday 18)

9. **You:** at noon, download the students' Google Sheet as Excel (**File → Download → Microsoft
   Excel**) into the Bluff_Detector folder.
10. **Claude:** import it, check every question against the rules, upload it.
11. **You:** **Run All** again with `m0_base.yaml`. It answers **only the new questions** (10–30
    minutes), then re-runs 5c–5g automatically.
    - Final **H1, H2, H5**, on the full Kenya set.
    - **Kenya vs UK twins** (paired wrong rate, McNemar's test): **H2** on matched questions, if
      the students wrote twins.
12. **You:** download again → **your final numbers**.

#### The sheet importer (built 16 November): `bluff import-sheet`

Code: `src/bluff/sheet_import.py`; tests: `tests/data/test_sheet_import.py`. In the VS Code
terminal, inside `bluff-detector`:

```
.venv\Scripts\bluff import-sheet --config configs/variants/m0_base.yaml --sheet "..\Bluff Detector - Kenya Question Sheet.xlsx"
```

- **Reads** the tabs `Student A`–`Student D` and skips empty rows. **Type** decides the pool:
  real → `kenya`, fake → `kenya_unanswerable` (no answer), twin → `western`, linked through
  **Twin of** (the Kenya question's text, copied exactly) to that question's id.
- **Answer + Other accepted answers** (split on `;`) become the accepted answers.
- **Ids come from the question text** (e.g. `ke-civ-3f2a9c1b`), so importing again with more
  rows keeps every earlier id, and the Colab resume asks only the new questions.
- **A bad row is skipped, never fatal**, and listed in `data/kenya/import_report.md` with its
  tab, row number and reason: no answer on a real question; a twin whose original isn't found; a
  duplicate; Verified = yes without a source or a second-person reviewer (imported as
  unverified). The report also counts real / fake / twin / verified questions per author.
- **The 9 starter questions are kept** (their hand-numbered ids, e.g. `ke-civ-0001`).
- Tried on the real template on 16 November (student tabs still empty): 0 rows, 9 starters kept.

**Wednesday, step by step:** download the sheet as .xlsx into `Bluff_Detector` → Claude runs
the command above, reads the report with you, and uploads `data/kenya/questions.jsonl` to GitHub
→ you **Run All** in Colab with `m0_base.yaml` (it asks only the new questions) → download → run
`notebooks/analysis.ipynb` for the final H1, H2, H5 and twins.

#### M1 (RAG) vs M0 on trivia: first H4 result (19 November)

M1 ran on Kaggle (T4, 1,457 questions, 3 Kenyan passages in every prompt) and finished at 13:34.
`bluff compare --config configs/variants/m0_base.yaml --other configs/variants/m1_rag.yaml`
(paired, the same 218 TriviaQA test questions; report: `reports/compare/m0_base__vs__m1_rag.json`):

| | M0 | M1 (RAG) | Paired test |
|---|---|---|---|
| Says "I don't know" when offered the option | 35% | **88%** | McNemar: 117 questions newly abstained vs 3 the other way, p ≈ 10⁻³⁰ (Holm-adjusted ≈ 10⁻³⁰) |
| Wrong answer (asked normally) | 64% | 69% | 25 newly wrong vs 14 newly right, p = 0.11 (not significant) |
| Hedges when *not* offered "I don't know" | 0% | 0% | — |
| Internal detection (probe AUROC) | 0.773 | 0.806 | Δ +0.03 [−0.03, 0.10] |
| Expressed detection (abstention) | 0.598 | 0.560 | Δ −0.04 [−0.12, 0.04] |
| Verbalisation gap | 0.175 | 0.246 | Δ +0.07 [−0.01, 0.16], p ≈ 0.06 |
| Latency p50 | 0.18 s | 1.5 s | longer prompts |

**What it means (trivia only).** The Kenyan passages are irrelevant to trivia questions, so this
is the "irrelevant reading material" test. **H4 as stated (retrieval gives false confidence) is
not supported here:** offered the option, the model with irrelevant passages says "I don't know"
far *more* often. Asked normally it never hedges and is slightly more often wrong (not
significant). Its inside knows a little more than it says (gap +0.07, not significant). The
relevant-material test is on the Kenya questions (the M1 + Kenya run).

#### What actually happened (19 November)

- The four students' sheets arrived in four different layouts (one file per student, plus one
  draft tab). `bluff clean-sheets` + `bluff import-sheet` (run step by step in
  `notebooks/kenya_questions.ipynb`) cleaned and imported them, logging every change.
- The students' review step wasn't done and their UK twins were unusable, so **an AI review
  repaired the set** (`data/kenya/repairs.yaml`, declared in `AI_USE.md`): 81 rows dropped
  (duplicates, yes/no, essay or list answers, reserved examples, wrong or disputed answers,
  changing facts), answers and wording corrected, 33 UK twins and 10 fakes written.
- **Result:** 326 items: 211 real Kenya questions, 82 fakes, 33 UK twins. `validate` passes, the
  leakage audit finds 0 overlaps. Full story and commands: `docs/kenya-questions.md`.
- GitHub's history had been rewritten (the same files, "Co-Authored-By" lines removed), so the
  laptop now works on branch `kenya-repairs`, which is GitHub's `main`; the old local `main` is
  kept as a backup.

---

## PART E: Extras, only if A–D went smoothly (Wednesday 18 – Thursday 19)

13. **Model M2: fine-tuning (H3, the original headline)**
    - **Claude:** build a small set of Kenyan documents (the Constitution and a few Acts).
    - **You:** one Colab session to fine-tune (QLoRA, 4-bit, about an hour), then one M2 run
      (about 2 hours). That repeats 5a–5g on the fine-tuned model.
    - **Claude:** `bluff compare` M0 vs M2 → **H3**: did the **verbalisation gap grow**? That is,
      did it stop *saying* "I don't know" while still *knowing* inside? Tested with a paired
      bootstrap, McNemar's test and Holm correction.
14. **Model M1: RAG (H4).** Only after M2, if time is still left. Otherwise **cut**.

    **As it happened:** both ran, M1 first. M1 (RAG) ran on Kaggle on 18–19 November (about six
    hours) and M2 (QLoRA) on 19–20 November (8 minutes of fine-tuning, 42 steps, then the full
    run). Results: H4 not supported (retrieval halves errors when the answer is in the passages and
    does no harm otherwise), H3 supported ("I don't know" halved, accuracy unchanged). M3 was not
    run. All in `notebooks/results.ipynb`.

---

## PART F: Finish (Thursday 19 – Friday 20)

15. **Claude: the live guard demo on your laptop.**
    - You ask a question → the model answers → the probe gives P(wrong).
    - **Above 0.25 → abstain; otherwise → the answer.**
    - Each reply includes an audit record, and **latency p50/p99** is measured.
    **Built 16 November.** Start it yourself (VS Code terminal, inside `bluff-detector`):

    ```
    powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1
    ```

    Wait for `Uvicorn running on http://127.0.0.1:8000`, then open **http://127.0.0.1:8000**.
    Stop it with **Ctrl+C**.

    - **The page** (`app/static/index.html`): a question box, example questions, the verdict
      (ANSWERED / BLOCKED), a P(wrong) meter with the 0.25 line, *"the model was about to
      say…"* when it blocks (the `reveal` option, demo only), and this session's questions.
    - **The model** is a local copy of the pinned Qwen2.5-1.5B-Instruct in
      `C:\Users\Admin\models\qwen2.5-1.5b-instruct` (3 GB; Hugging Face's own downloader
      stalled on this connection, so it was fetched with a resumable `curl`). The detector is
      the champion, `artefacts/m0_base/serving/` (v2, TriviaQA only).
    - **Speed: 20–40 s per question** on this laptop (7.7 GB RAM, CPU only, bfloat16), against
      0.2 s on the T4. RAM is the bottleneck; close Chrome and Discord before a demo.
    - **First real answers:** "Capital of Australia?" → Canberra, P(wrong) 0.000, **answered**;
      "King Lear's youngest daughter?" → Goneril (wrong), 0.948, **blocked**; "How many counties
      does Kenya have?" → 47 (right), 0.688, **blocked**: the 0.25 line is cautious, as the
      results predicted (it loses about half the right answers).
    - **Note for the report:** the probe was trained on hidden states from a T4 in float16; the
      laptop computes them in bfloat16 on a CPU, so its scores can shift slightly. Monitoring's
      prediction-drift check is there to catch exactly this.

    **Monitoring (built 16 November):** every request is appended to `logs/requests.jsonl`
    (question, verdict, P(wrong), the 8 confidence numbers, time). Then:

    ```
    .venv\Scripts\bluff monitor --config configs/variants/m0_base.yaml
    ```

    writes `reports/m0_base/monitoring/drift_report.md`: the live block rate and latency against
    the validation baseline, and for each confidence number and for P(wrong) a PSI and a KS test
    against the training data (OK / WATCH / DRIFT). Below 30 requests it says **TOO EARLY**.
    **Redesigned 20 November** (the page you demo). Three tabs:
    - **Ask the model:** the answer, then *two readings on one scale*: the chance it is wrong
      judged from its **words** (orange ring: the average of the two outside detectors) and from
      its **inside** (indigo dot: the linear probe, which decides), with the guard's dashed line
      at 25%. If the words looked fine but the inside crossed the line: **"Blocked: bluff
      caught"**. Then the answer word by word, the passages it was shown (M1 only), all four
      detectors, the confidence numbers and the audit record, each folded away. Examples are real
      test questions (a caught Kenya bluff, a known Kenya fact, a made-up question, a UK twin,
      trivia). Your questions stay in "This session"; click one to see it again.
    - **What we found:** H1, H2, H5, the made-up questions, H4 and H3, each with its verdict, the
      numbers in words and a paired-dot chart (hover a dot for its value; "Show the numbers" gives
      the table); then M0–M3 side by side and what the guard would do on Kenya. The numbers come
      from `GET /v1/results`, which serves `reports/results_summary.json`, written by the last
      cell of `notebooks/results.ipynb`: re-run the notebook and the page follows.
    - **How it works:** the seven steps, M0–M3, and a glossary.
    - Light and dark mode, works on a phone. Run M1 or M2 instead of M0 with
      `scripts\serve_laptop.ps1 -Variant m1_rag` or `-Variant m2_qlora`.
16. **Claude:** put the real results in the README. **Done**, and on 20 November everything was
    checked and stored: the M2 adapter added to git (with a model card), the M1 and M2 datasets
    exported and uploaded, the 13 analysis charts made for M1 and M2
    (`notebooks/analysis_m1_rag.ipynb`, `analysis_m2_qlora.ipynb`). What lives where:
    [where-things-are.md](where-things-are.md).
17. **You:** write the report in your own words. Include what was cut and why: M1/H4, UK twins
    if missing, monitoring, and M2/H3 if not reached.
18. **You:** **submit, Friday 20.**

---

## Summary: what Friday's submission contains

The plan above was written on 15 November; this is what was actually delivered on 20 November.

| | Status |
|---|---|
| **H1, H2, H5** | ✅ Supported (M0) |
| **H3** | ✅ Supported (M0 vs M2) |
| **H4** | ✅ Tested, **not supported** (M0 vs M1): retrieval helps, no false confidence |
| **M0, M1, M2** | ✅ Run, trained, evaluated, registered |
| **M3** | ❌ Not run (future work) |
| **4 detectors** | ✅ For each of M0, M1, M2 |
| **All metrics** | ✅ Per variant, plus the M0 vs M1 and M0 vs M2 comparisons |
| **Live guard, API, Docker, CI/CD, monitoring** | ✅ Built; demo page redesigned in step 15 |
| **Report** | ✍️ Yours, in your own words |

**Next action:** write the report, make the repo public, submit.
