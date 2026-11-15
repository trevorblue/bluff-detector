# Status and next steps (5-day plan, submit Friday 20 November)

Where the project stands on Sunday 15 November, what was changed today and how, and every step
from here to submission, in order. The GPU work runs from **your VS Code notebook connected to
a free Colab T4**. No paid GPU is needed at any point.

## 1. The plan in one paragraph

Build and run everything now on trivia questions plus the Kenya questions that already exist.
The Kenya questions are only ever *tested*, never trained on, so the students' unfinished
questions plug in at the end with a short extra run. **Cut-off for Kenya questions: Wednesday
18 November, noon.** Must ship: M0 with H1, H2 and H5. Stretch goals, in order: M2 (H3), the
full Kenya set with UK twins, M1 (RAG), monitoring.

## 2. What was done today, and how

| Change | Why | How | Files |
|---|---|---|---|
| **Automatic number format** (`dtype: auto`) | The free Colab T4 has no native bfloat16; the old code would have run slowly or failed | The code checks the GPU: bfloat16 on newer GPUs, **float16 on the T4**, bfloat16 on a CPU (to save memory) | `src/bluff/backends.py`, `configs/base.yaml` |
| **Pinned model version** | The model on Hugging Face can be updated; every run must use the same one | Qwen2.5-1.5B-Instruct fixed at commit `989aa79…` | `configs/base.yaml`, `backends.py` |
| **Resume** | A Colab session can drop; and the students' questions arrive later | `generate` saves its progress every 50 questions. A rerun keeps the questions already answered and asks only the new ones. It refuses to mix runs from different GPUs or settings. | `src/bluff/pipeline/generate.py` |
| **Run record** (`run_info.json`) | To prove every answer came from the same GPU type, settings and versions | Saved next to the answers on every run | `generate.py` |
| **Smoke-test version** | Check the real model works (~15 min) before the long run | `m0_smoke`: M0 limited to the first 120 questions | `configs/variants/m0_smoke.yaml` |
| **Smaller trivia sample** | Fit one free Colab session | 1,500 TriviaQA questions (was 5,000) plus all 817 TruthfulQA | `configs/base.yaml` |
| **Fine-tuning fixes** (for the M2 stretch goal) | It would have crashed on a T4, and it threw away most of each document | float16 on T4; pinned model; each document cut into 1,024-token windows instead of keeping only its first 1,024 tokens | `src/bluff/pipeline/finetune_qlora.py` |
| **The Colab notebook** | Your GPU runs, from VS Code | Checks the GPU, gets the code, asks for two tokens, runs the pipeline, saves to Hugging Face every 10 minutes | `notebooks/colab_run.ipynb` |
| **Tests** | Prove resume works | A run cut off after 50 questions and finished later gives exactly the same answers as one full run; a run on a different GPU is refused | `tests/integration/test_resume.py` |

All **46 tests pass** (44 before, 2 new), and the full pipeline runs end to end on the fake model.

## 3. Why Colab in VS Code, and why Hugging Face

- **Colab in VS Code** is the GPU: you already set it up. The notebook runs on Google's computer
  with a T4; VS Code is only your screen. The main run takes about 1.5–2 hours, well inside a
  free Colab session.
- **Kaggle** is no longer needed. It was suggested only because it keeps running with the laptop
  off; resume makes that unnecessary.
- **Hugging Face** is the storage. A Colab machine is **wiped when its session ends**, and its
  files never appear on your laptop. So the notebook saves the results to your private
  Hugging Face dataset repo every 10 minutes and at the end. From there, one command downloads
  them to your laptop, and a later run picks up the saved progress.

## 4. Step by step from here

### Step 1. One-time setup (about 15 minutes, today)

1. **Hugging Face:** create a free account at huggingface.co. Then **Settings → Access Tokens
   → Create new token**, type **Write**, and copy it. The notebook creates the private
   `bluff-detector-runs` repo itself on the first run.
2. **GitHub:** github.com → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token.** Repository access: only `trevorblue/bluff-detector`.
   Permissions: **Contents: Read-only**. Copy it.
3. **On the laptop**, in VS Code's terminal: `pip install -U huggingface_hub`, then
   `hf auth login` and paste the Hugging Face token.

### Step 2. Smoke test on Colab (about 15 minutes)

1. In VS Code, open `notebooks/colab_run.ipynb`.
2. Top right: **Select Kernel → Colab → New Colab Server → GPU → T4**.
3. In the second cell, set `HF_REPO` to `your-hf-username/bluff-detector-runs`. Leave
   `CONFIG` as `m0_smoke.yaml`.
4. **Run All.** Paste the GitHub token, then the Hugging Face token, when asked.
5. **It worked if** the last cell prints a results table (`summary.md`). Send me any error
   message exactly as shown.

### Step 3. The full M0 run (about 1.5–2 hours)

1. In the same notebook, change `CONFIG` to `"configs/variants/m0_base.yaml"` and **Run All**.
2. Keep VS Code open and the laptop awake and plugged in.
3. If it disconnects: reconnect (**Select Kernel → Colab**) and **Run All** again. It resumes
   from the last save.

### Step 4. Results on your laptop

In VS Code's terminal, inside the `bluff-detector` folder:

```
hf download YOUR-HF-USERNAME/bluff-detector-runs --repo-type dataset --local-dir .
```

Open `reports/m0_base/summary.md`: the **first real results** (H1, H5, and H2 on the Kenya
questions so far). Then hand-check the 50 rows in
`data/interim/m0_base/review_sample.jsonl` (about an hour), writing right or wrong in `manual`.

### Step 5. Plug in the students' questions (Wednesday 18, after noon)

1. Download the students' Google Sheet: **File → Download → Microsoft Excel (.xlsx)**, into the
   `Bluff_Detector` folder.
2. Claude imports it into `data/kenya/questions.jsonl`, checks every row against the rules, and
   pushes it to GitHub. *(The importer is built when the sheet arrives.)*
3. Run the notebook again with `m0_base.yaml`. Resume asks **only the new questions** (about
   10–30 minutes), then retrains and re-evaluates automatically.
4. Download again (step 4): the final M0 results.

### Step 6. Stretch goals, only if steps 1–5 are done

| Order | Goal | What it takes |
|---|---|---|
| 1 | **M2: fine-tuning (H3, the headline)** | A small collection of Kenyan documents (the Constitution and a few Acts); one Colab session to fine-tune (~1 h); one M2 run (~1.5–2 h); `bluff compare` |
| 2 | **UK twins** | Only if the students wrote some; the importer handles them |
| 3 | **M1: RAG** | The document collection, plus one more Colab run |
| 4 | **Monitoring** | Request logging and a drift report on the laptop |

### Step 7. Ship (Thursday 19 – Friday 20)

- Serve the real detector on the laptop (`make serve`), measure latency.
- Real results into the README; update AI_USE.md; make the repo public if you want.
- Write and submit the report.

## 5. Still to build

| Piece | When | Needs |
|---|---|---|
| Sheet importer | When the students' sheet is downloaded | The sheet |
| Document collection builder | Only for M2 / M1 | A list of document links |
| Laptop serving with the real model | Thursday | The trained M0 detector |
