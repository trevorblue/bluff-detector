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

## PART B: The model runs (Sunday 15 – Monday 16) · You press Run All on free Colab

### Step 4. Smoke test (about 15 minutes)

Open `notebooks/colab_run.ipynb` → **Select Kernel → Colab → GPU T4** → **Run All** → paste the
two tokens when asked.

- The whole process below runs on just **120 questions**, to prove the GPU, the code and the
  saving work.
- It **passes** if a results table appears at the end. Its numbers **don't count**.

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
hf download YOUR-USERNAME/bluff-detector-runs --repo-type dataset --local-dir .
```

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

---

## PART F: Finish (Thursday 19 – Friday 20)

15. **Claude: the live guard demo on your laptop.**
    - You ask a question → the model answers → the probe gives P(wrong).
    - **Above 0.25 → abstain; otherwise → the answer.**
    - Each reply includes an audit record, and **latency p50/p99** is measured.
16. **Claude:** put the real results in the README.
17. **You:** write the report in your own words. Include what was cut and why: M1/H4, UK twins
    if missing, monitoring, and M2/H3 if not reached.
18. **You:** **submit, Friday 20.**

---

## Summary: what Friday's submission contains

| | Status |
|---|---|
| **H1, H2, H5** | ✅ From step 5, finalised in step 11 |
| **H3** | ⭐ Only if step 13 happens |
| **H4** | ❌ Cut |
| **M0** | ✅ |
| **M2** / **M1** | ⭐ Extra / ❌ cut |
| **4 detectors** | ✅ Trained in step 5 |
| **All metrics** | ✅ Computed in steps 5 and 11 (the H3 comparison only with M2) |
| **Live guard, API, Docker, CI/CD** | ✅ Already built; demoed in step 15 |

**Next action:** Part A (accounts and tokens), then step 4, the smoke test.
