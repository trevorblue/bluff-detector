# GPU setup: free T4s, driven from VS Code

> **Current plan (15 November): Colab in VS Code is the GPU; Kaggle is only a backup.** Use
> `notebooks/colab_run.ipynb` and follow [next-steps.md](next-steps.md). It saves to a private
> Hugging Face repo and resumes after a disconnect, so the split runs across several machines
> described below are not needed for the 5-day plan. Switch to Kaggle only if Colab's free
> quota runs out: upload the same notebook there with Accelerator = **GPU T4 x2** and run it.

Two jobs need a GPU: answering the questions (`generate`, once per variant) and QLoRA training
(once, for M2). Everything else runs on the laptop. Free T4 GPUs come from **Colab, which runs
inside VS Code**, and **Kaggle, which runs in the background and is watched from VS Code's
terminal**. A private Hugging Face dataset repo is the shared drive that connects them.

> **Status.** The runner notebook and the `--shard`, `--resume`, `merge` and upload commands are
> built in Week 1 (plan steps B1–B3). Until then, steps marked *(built in B1–B3)* don't work yet.
> The account setup in section 2 can be done now.

## 1. How it works

```
 YOUR LAPTOP (VS Code)               SHARED DRIVE                      GPU MACHINES (T4, float16)
 bluff ingest ─► questions.jsonl ──► private Hugging Face  ◄── slice 1 · Kaggle GPU 0 (background)
                                     dataset repo           ◄── slice 2 · Kaggle GPU 1 (background)
 bluff merge ◄── slice checkpoints ◄ (questions, slices,    ◄── slice 3 · Colab T4 (inside VS Code)
 label → featurize → train →          adapter)                  each: download questions → run its
 evaluate → compare → serve                                     slice → upload progress every ~100
```

- **VS Code is only your screen and keyboard.** When a notebook runs on Colab or Kaggle, the code
  runs on Google's or Kaggle's computer, which has an NVIDIA T4 GPU with 16 GB of memory. Your
  laptop does no work and can be slow.
- **A session is a borrowed computer.** It starts empty: the notebook installs the project,
  downloads the model (~3 GB), and runs. When the session ends, that computer is wiped. That is
  why every slice uploads its progress to the shared drive.
- **Questions are independent**, so a variant's questions are split into *slices* (e.g. 3), one
  per GPU, all running at once. A slice that times out is simply started again, on any machine:
  it downloads its own progress and continues where it stopped.
- **Afterwards**, `bluff merge` on the laptop downloads the slices and joins them into one file.
  Every later stage runs on the laptop in minutes.

| Where | Inside VS Code? | Runs while your laptop is off? | Best for |
|---|---|---|---|
| **Colab** (free) | Yes: the official Google Colab extension makes a VS Code notebook run on a Colab T4 | No: keep VS Code open and the laptop awake | One slice; quick tests |
| **Kaggle** (free) | Watched and downloaded from VS Code's terminal (Kaggle CLI); started on kaggle.com | **Yes**: "Save & Run All" runs in the background, up to the session limit | Your main workhorse: "GPU T4 x2" = two slices at once |
| **Rented GPU** (paid) | Yes: VS Code Remote-SSH, a full terminal on the GPU machine | Yes | Only if the free tiers run out |

## 2. One-time setup

Do these once. About 30–45 minutes in total.

### 2.1 Hugging Face: the shared drive

1. Create a free account at huggingface.co.
2. **Settings → Access Tokens → Create new token.** Choose *Write* access. Copy it somewhere safe:
   this is `HF_TOKEN`.
3. **New → Dataset.** Name it `bluff-detector-runs`, set it to **Private**, create it.
4. In VS Code's terminal: `pip install -U huggingface_hub`, then `hf auth login` and paste the
   token.

### 2.2 GitHub: let the GPU machines download the private repo

1. github.com → **Settings → Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token.**
2. Repository access: **only `trevorblue/bluff-detector`**. Permissions: **Contents: Read-only**.
   Expiry: 30 days. This is `GH_TOKEN`. It can read this one repo and nothing else.

(If you make the repo public, this step is not needed.)

### 2.3 Kaggle

1. Create a free account at kaggle.com. If Kaggle asks you to verify your phone number, do it:
   GPUs and internet access in notebooks depend on it.
2. Check your weekly GPU allowance under your account settings (about 30 hours is typical, but
   Kaggle changes it, so trust the number shown there).
3. **Settings → API → Create New Token.** This downloads `kaggle.json`. Move it to
   `C:\Users\Admin\.kaggle\kaggle.json`. If Kaggle shows a different token flow, follow its
   instructions.
4. In VS Code's terminal: `pip install kaggle`, then `kaggle kernels list --mine` to check it works.

### 2.4 VS Code

1. Extensions → install **Python** and **Jupyter** (Microsoft) and **Google Colab** (publisher:
   Google, verified). Avoid look-alike extensions from other publishers.
2. Optional, for a rented GPU later: **Remote - SSH** (Microsoft).

## 3. Running a slice on Colab, inside VS Code *(built in B1–B3)*

1. In VS Code, open `notebooks/gpu_runner.ipynb`.
2. Top right: **Select Kernel → Colab → sign in with Google → New Colab Server → GPU → T4.**
3. In the first cell, set the run:
   ```python
   VARIANT = "m0_base"     # m0_base, m1_rag or m2_qlora
   SHARD, N_SHARDS = 3, 3  # this machine runs slice 3 of 3
   HF_REPO = "<your-username>/bluff-detector-runs"
   ```
4. **Run All.** When asked, paste `HF_TOKEN` and `GH_TOKEN` (they are not saved in the notebook).
5. The notebook then: checks the GPU is a T4 (and stops if not) → downloads the repo and installs
   it → downloads `questions.jsonl` from the shared drive → runs
   `bluff generate --shard 3/3 --resume` → uploads progress every ~100 questions → uploads the
   final slice.
6. Keep VS Code open and the laptop awake and plugged in. If Colab disconnects, reconnect and
   **Run All** again with the same settings: it resumes.

## 4. Running slices on Kaggle, in the background *(built in B1–B3)*

**Once per variant**, on kaggle.com:

1. **Create → New Notebook → File → Import Notebook**, and upload `notebooks/gpu_runner.ipynb`.
2. Right panel, **Settings**: Accelerator = **GPU T4 x2** (not P100); Internet = **on**.
3. **Add-ons → Secrets**: add `HF_TOKEN` and `GH_TOKEN`, and attach both to this notebook.
4. In the first cell set `VARIANT`, `N_SHARDS = 3` and `KAGGLE_TWO_GPUS = True`: the notebook then
   runs slice 1 on GPU 0 and slice 2 on GPU 1 at the same time.
5. **Save Version → Save & Run All (Commit).** It now runs in the background. You can close the
   browser and turn off your laptop.

**From VS Code's terminal**, while it runs:

```bash
kaggle kernels status <your-kaggle-username>/<notebook-slug>      # running, complete or error
kaggle kernels output <your-kaggle-username>/<notebook-slug> -p kaggle_out   # logs and files
```

If a run stops early (session limit, error), open the notebook and **Save & Run All** again: the
slices resume from their last upload.

## 5. Your students' Kaggle accounts (optional extra slices)

Each student can run a slice on their own free account, which adds GPU time. Rules:

- One account per person. Kaggle bans people who run several accounts.
- Give them a **separate** Hugging Face fine-grained token with write access to the
  `bluff-detector-runs` repo only, and delete it when the runs finish. Never share your main token.
- They import the same notebook and set a different `SHARD` (e.g. 4/5, 5/5). `N_SHARDS` must be the
  same on every machine for that variant.

## 6. Which machine runs what

Default: **3 slices per variant**. Times are estimates; the pilot run measures the real ones.

| Run | Slice 1 | Slice 2 | Slice 3 | Wall-clock time |
|---|---|---|---|---|
| M0 base (~6,300 questions × 4 prompts) | Kaggle GPU 0 | Kaggle GPU 1 | Colab | ~1–1.5 h |
| M2 QLoRA (same questions, adapter on top) | Kaggle GPU 0 | Kaggle GPU 1 | Colab | ~1–1.5 h |
| M1 RAG (longer prompts) | Kaggle GPU 0 | Kaggle GPU 1 | Colab | ~2–2.5 h |
| QLoRA training (once) | One session on Kaggle or Colab: `bluff finetune_qlora`, adapter uploaded to the shared drive | | | ~1 h |
| Kenya questions added later (resume runs only new ones) | Any one GPU, per variant | | | ~30–45 min each |

Total: about **16 GPU-hours** including pilots, inside Kaggle's free weekly allowance.

## 7. After the runs: merge on the laptop *(built in B1–B3)*

```bash
bluff merge --config configs/variants/m0_base.yaml    # downloads every slice, joins them
bluff label --config configs/variants/m0_base.yaml
bluff featurize --config configs/variants/m0_base.yaml
bluff train --config configs/variants/m0_base.yaml
bluff evaluate --config configs/variants/m0_base.yaml
```

`merge` refuses to continue if a question is missing or duplicated, or if the slices ran on
different GPU types or library versions.

## 8. The rules, and why

| Rule | Why |
|---|---|
| Every generation run on a **T4, in float16** | Different GPUs and number formats give slightly different answers and hidden states; mixing them adds noise to exactly the comparisons the project measures. The runner stops if it lands on another GPU. |
| **Pinned** model version and library versions | The same reason: the model on Hugging Face can be updated, and a newer library can change results |
| Same `N_SHARDS` for every slice of a variant | Slices are defined by `N_SHARDS`; changing it mid-run would overlap or skip questions |
| Never paste tokens into a notebook cell that gets saved or committed | Anyone who sees the notebook would get access. Use Kaggle Secrets, or the runner's prompt. |
| QLoRA training may use any GPU | The adapter is the output; the training hardware does not affect how answers are generated later |

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| "No GPU" or `torch.cuda.is_available()` is False | The accelerator isn't set: Colab → New Colab Server with GPU; Kaggle → Settings → Accelerator |
| The runner stops: "not a T4" | Kaggle gave a P100: set Accelerator to **GPU T4 x2** and run again |
| Session ended mid-run | Run the same slice again with the same settings; it resumes |
| 401 / 403 from Hugging Face | The token is wrong, expired or read-only: make a Write token |
| Kaggle says the GPU quota is used up | Wait for the weekly reset, use Colab, or a student's account |
| Colab disconnects when the laptop sleeps | Keep it plugged in and awake, or move that slice to Kaggle |
| `kaggle` command not found in VS Code | `pip install kaggle`, then reopen the terminal |

## 10. If the free tiers run out: a rented GPU

Services such as RunPod or Vast.ai rent an L4 or A10 by the hour; for this project, a few dollars
to about $20 in total. Rent a machine with a T4 if available, to keep the hardware rule; otherwise
run **all** of a variant's slices on the same rented GPU type. Connect with VS Code **Remote - SSH**
and run the same commands as the notebook in the terminal. Shut the machine down after each run:
it is billed while it is on.
