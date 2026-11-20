# Concepts: how and why everything works

Every idea in the project explained in plain words: the models, the metrics, serving, Docker,
CI/CD/CT, MLOps, monitoring and the GPU setup. Read it before a presentation or viva. For which
file does what, see [architecture.md](architecture.md); for the schedule, see [plan.md](plan.md).

> **Built vs not built.** Every section says plainly what exists today and what does not.
> Section 13 lists it all in one table. Examiners probe exactly these gaps, so know them.

## 1. The big picture

```
 OFFLINE (the pipeline, `bluff all`)                       ONLINE (the API)
 ┌──────────────────────────────────────────────┐          ┌──────────────────────────────┐
 │ questions → run the LLM → label → features → │ trained  │ question → LLM answers →     │
 │ train detectors → evaluate → register        │ detector │ detector scores → answer or  │
 └──────────────────────────────────────────────┘ ───────► │ abstain → audit record       │
                    ▲                                       └──────────────────────────────┘
 OPS around both: git + GitHub Actions (CI, CD, CT) · Docker · MLflow · registry · DVC · tests
```

Where the code came from: the scaffold (pipeline, API, Docker, CI and tests) was generated with
Claude on 16 October 2026 and is disclosed in [AI_USE.md](../AI_USE.md). Later work published
the repository, fixed the CD workflow, wrote these docs and added the UK-twin comparison.

## 2. The models

There are two kinds of model, and keeping them apart is the key to the whole project.

### 2a. The language model under study: Qwen2.5-1.5B-Instruct

- A 1.5-billion-parameter **open-weights** model. Open weights matter because the project reads
  the model's insides; hosted APIs such as ChatGPT do not expose them.
- It builds an answer **one token at a time** (a token is a word or part of a word). At each
  step it gives a probability to every possible next token and picks the most likely one
  (*greedy* decoding, at most 32 tokens).
- Two signals come from each step:
  - **Token probability:** how much probability it gave the token it chose. Saying "52" with
    0.31 means it was far from sure.
  - **Entropy:** how spread out its probability was over all options. High entropy means it was
    torn between many possibilities.
- **Hidden states:** the model has 28 layers. Each layer turns every token into a vector of
  1,536 numbers, its internal "thought" at that point. The code saves this vector at two layers
  (8th and 4th from the top: layers 21 and 25 of 28) and two positions (the last answer token,
  and the average over all answer tokens).

#### Weights vs hidden states: 1.5 billion vs 1,536

They count two different things.

| | **Weights (parameters)** | **Hidden state** |
|---|---|---|
| How many | About 1.5 billion | 1,536 per layer, per token |
| What they are | The model's fixed "wiring", learned in training | Its "thought" while answering *this* question |
| Change per question? | No, never | Yes, every question |
| Analogy | The wiring of a brain | A brain scan taken while it answers |
| Do we read them? | **No** | **Yes**: this is what the probes read |

**Where 1,536 comes from (not a guess):** the model's published settings file. On
huggingface.co/Qwen/Qwen2.5-1.5B-Instruct → *Files* → `config.json`: `"hidden_size": 1536`,
`"num_hidden_layers": 28`. It is the model's *width*: every layer takes in 1,536 numbers per token
and gives out 1,536.

**How 1.5 billion is built from 1,536-sized pieces** (rounded):

| Piece | Size | Weights |
|---|---|---|
| Word table (151,936 tokens × 1,536) | 151,936 × 1,536 | ≈ 0.23 billion |
| Each layer's attention | ≈ 1,536 × 1,536 × 2 + smaller parts | ≈ 5.5 million |
| Each layer's feed-forward block | 3 × 1,536 × 8,960 | ≈ 41 million |
| 28 layers × ≈ 47 million | | ≈ 1.31 billion |
| **Total** | | **≈ 1.54 billion** |

**How the hidden states are read, technically** (`src/bluff/backends.py`, `HFBackend.generate`):

1. Load the model from Hugging Face with the `transformers` library, on the GPU.
2. It answers normally: "How many counties does Kenya have?" → "52".
3. Feed the question plus answer back in once, with `output_hidden_states=True`, a built-in
   `transformers` option. The model returns its "thought" after **every** layer, for **every**
   token: 29 tables (the word table plus 28 layers), each *tokens × 1,536*.
4. Keep a small slice: layers 21 and 25 × (last answer token, average of answer tokens) =
   4 vectors of 1,536 numbers, saved per question.
5. The served probe reads one of them, **layer 21, last answer token**: 1,536 numbers in,
   P(wrong) out. "52" → 0.81.

**Why Ollama can't be used:** it returns only the text "52", so step 3 is impossible. The
models on the laptop (`qwen3:4b`, `llama3.2`) are also different models.

### 2b. The variants: the interventions under test

| Variant | What changes | How |
|---|---|---|
| **M0** | Nothing | The model as downloaded |
| **M1 + RAG** | It gets reading material | Retrieval-augmented generation (section 3) |
| **M2 + QLoRA** | It is fine-tuned on Kenyan documents | Section 4 |
| **M3** | Both | Retrieval and the adapter together (ran 20 November) |

Every variant runs the same pipeline with the same detectors and metrics; only its config file
differs. That is what makes the comparison fair.

### 2c. The detectors: the models you actually train

Standard supervised ML. **Input:** signals recorded from one answer. **Output:** P(wrong).
**Label:** was the answer wrong (1) or right (0).

| Detector | Model | Input |
|---|---|---|
| `logreg` | Logistic regression (scikit-learn), inputs standardised | 8 summary numbers: average and minimum log-probability; average, maximum and last entropy; number of tokens; share of tokens under 0.5 probability; top retrieval score |
| `xgboost` | Gradient-boosted trees: 300 trees, depth 3, learning rate 0.05 | The same 8 numbers |
| `probe_linear` (served) | Logistic regression on the hidden state, strong L2 regularisation (C = 0.05) | The 1,536-number hidden state, 8th layer from the top, last answer token |
| `probe_mlp` | Neural network, one hidden layer of 128 units, early stopping | The same hidden state |

They are trained on **trivia questions only** (70% train, 15% validation, 15% test). Kenya
questions and their UK twins are never used in training, which makes them true
out-of-distribution tests. *Probe* just means a simple classifier that reads a model's hidden
states.

#### How each detector works, with an example

**The training data.** Every trivia question becomes one row:

| Question | Label (wrong?) | 8 summary numbers | 1,536 hidden numbers |
|---|---|---|---|
| "Capital of Australia?" → "Sydney" | 1 (wrong) | min prob 0.22, max entropy 2.1, … | −0.4, 1.7, 0.02, … |
| "Red Planet?" → "Mars" | 0 (right) | min prob 0.97, max entropy 0.1, … | 0.9, −0.3, 0.5, … |

Each detector learns from about 1,600 such rows (70% of the trivia) to output **P(wrong)** for a
new answer.

**1. Logistic regression (`logreg`), the simple baseline.**

- It gives each of the 8 numbers a weight, adds them up, and squashes the total into 0–1.
- It learns rules like "low minimum probability → more likely wrong" (a big weight) and
  "number of tokens → hardly matters" (a weight near 0).
- *Example:* "52", with min probability 0.31 and high entropy → P(wrong) = 0.74.
- **Why:** fast, and you can read its weights to see which signal matters most.

**2. XGBoost (`xgboost`), trees on the same 8 numbers.**

- It builds 300 small decision trees. Each asks up to 3 yes/no questions, e.g. "min prob below
  0.4? → max entropy above 1.5? → probably wrong".
- Each new tree focuses on the rows the earlier trees got wrong ("boosting"). The trees' votes
  are added up into P(wrong).
- **Why:** it catches combinations and cut-offs that logistic regression can't, e.g. "only
  suspicious when *both* are bad".

**3. Linear probe (`probe_linear`), the one used live.**

- The same maths as logistic regression, but on the **1,536 hidden numbers**: 1,536 weights.
- Those weights point out one **direction** in the model's "thought space" that means "I don't
  really know this". The probe measures how far an answer's thought lies along it.
- Strong regularisation (C = 0.05): 1,536 inputs from about 1,600 rows could be memorised, so the
  weights are kept small.
- *Example:* "52" → P(wrong) = 0.81.
- **Why it is served:** one multiply-and-add, so it costs almost nothing per request. If it works,
  that also shows the knowledge is written **plainly (linearly)** inside the model.

**4. MLP probe (`probe_mlp`), a small neural network on the same 1,536 numbers.**

- 1,536 inputs → 128 hidden units → P(wrong). Each of the 128 units learns to spot one pattern;
  the last step combines them.
- It can find curved (non-linear) patterns a straight line misses. Early stopping halts training
  when the validation score stops improving, to avoid memorising.
- **Why:** a check on the linear probe. If the MLP is not clearly better, the signal really is
  linear, and the simple probe is the right one to serve.

**What comparing them tells us.**

| Comparison | Question it answers |
|---|---|
| Probes (inside) vs logreg and XGBoost (outside) | Does the "brain scan" know more than the confidence numbers show? |
| MLP vs linear probe | Is the signal simple (linear) or complex? |
| Probe vs the model saying "I don't know" | **The verbalisation gap**: how much more it knows than it says (H1, H3) |
| Trivia vs Kenya, for every detector | **The OOD drop (H5)**: does what they learned carry over to Kenya? |

Every detector outputs P(wrong) for every test question; the metrics in section 6 (AUROC, ECE
and the rest) are computed from those probabilities. The served probe also gets the live
threshold: P(wrong) above 0.25 → abstain.

### 2d. How the model "decides" to say "I don't know"

**It doesn't really decide.** There is no separate "do I know this?" step. The model only ever
does one thing: predict the next word. At every step it gives each possible next word a
probability and (here, with greedy decoding) picks the most likely one.

**Offered the option**, e.g. "Who is the Commander-in-Chief of Kenya's armed forces? … or reply
I don't know if you are not sure", the *first word* of its reply might get (illustrative numbers):

| First word | M0 (original) | M2 (fine-tuned) |
|---|---|---|
| "I" (as in "I don't know") | **0.40** | 0.15 |
| "The" (as in "The President") | 0.30 | **0.55** |
| others ("General", "Uhuru", …) | 0.30 | 0.30 |
| **It says** | "I don't know" | "The …" |

**Where those probabilities come from.** When a question is familiar, the model's inside points
strongly to one answer, and that answer gets a high probability. When it is unfamiliar, the inside
state is weak or mixed: many answers each get a little. During instruction training the model
learned that in that weak, spread-out situation "I don't know" is the right reply, so "I" becomes
the most likely first word.

**This is the whole project in one picture:**

- **Knowing** is that weak or mixed state *inside* the model. The probe reads it directly from the
  1,536 hidden numbers.
- **Saying** is whether that inside state is strong enough to push "I don't know" to the top of the
  word choice.
- **H1 (the gap):** the inside often signals doubt that isn't strong enough to win the word choice,
  so the model bluffs, but the probe still sees the doubt.
- **H3 (fine-tuning):** the fine-tuned model's word choice leans towards confident words ("The …"),
  so the same inside doubt loses the word choice more often. The doubt is still there for the
  probe; it just doesn't reach the words.

## 3. RAG, step by step (variant M1)

**Retrieval-Augmented Generation:** look things up, then answer. M0 sits a closed-book exam; M1
an open-book one. The question is whether the open book helps a small model on Kenyan facts, or
only makes it sound more sure of itself (H4).

1. **The knowledge base** (`data/corpus/retrieval_chunks.jsonl`): Kenyan documents (the
   Constitution, Acts, county pages, Wikipedia) cut into **chunks** of 300–800 tokens, roughly
   one section each. Chunks, because a whole Act is far too long to paste into a prompt. Each
   line: `{"id": "constitution-art-6", "text": "...", "source": "Constitution of Kenya 2010",
   "doc_id": "constitution"}`. It should hold the *answers* to Kenya questions but never their
   *wording*; the leakage audit checks that.
2. **The search engine** (`src/bluff/retrieval.py`, **BM25**), a classic keyword ranking:
   - Split text into lowercase words: "How many counties does Kenya have?" →
     `how, many, counties, does, kenya, have`.
   - At startup, count how many chunks contain each word, and each chunk's length.
   - Score each chunk. For each question word it contains, add points: **rare words count more**
     ("counties" a lot, "does" almost nothing; this is *IDF*, inverse document frequency);
     **repeats help with diminishing returns** (`k1 = 1.5`); **short, focused chunks win**
     (`b = 0.75`). Per word *q* the score is
     `IDF(q) × f·(k1+1) / (f + k1·(1 − b + b·length/average_length))`, summed over the words,
     where *f* is how often *q* appears in the chunk.
   - Keep the **top 3** chunks (`top_k: 3`).
3. **The generator** (`src/bluff/prompts.py`): the 3 chunks go in front of the question, under
   "Use the context if it is relevant. Context: [1] … [2] … [3] …".

How it plugs in: `m1_rag.yaml` sets `retrieval.enabled: true`; `generate` searches before every
question and records which chunks it used and their scores; the **top retrieval score is the 8th
detector feature**; in the API, a gated answer that used passages comes back as
`answer_with_citation` with the chunk ids as `sources`.

Limits: BM25 matches exact words, so "devolved units" will not find "counties", and Swahili will
not match English. The upgrade is *dense retrieval* (meaning-based embeddings) plus a reranker.
BM25 came first because it is fast, needs no GPU, is a strong baseline and is easy to explain.

### How a run actually works: the notebook is the remote control

The notebook (`notebooks/run_m1_rag.ipynb`) contains no RAG code. Its cells only run commands
such as `bluff generate --config configs/variants/m1_rag.yaml`. `bluff` is the project's own
program (`src/bluff/cli.py`): each command runs one file in `src/bluff/pipeline/`, which uses the
helper files in `src/bluff/`. **The only difference between M0 and M1 is one switch** in
`configs/variants/m1_rag.yaml`: `retrieval: enabled: true, top_k: 3`. Every file below then
behaves accordingly.

| Step | File | What it does |
|---|---|---|
| 1. The reading material (built once) | `src/bluff/corpus.py` + `data/corpus/sources.yaml` | `bluff build-corpus` downloads the documents and cuts them into passages → `data/corpus/retrieval_chunks.jsonl` |
| 2. Load the search engine | `src/bluff/retrieval.py` | `build_retriever` reads the 441 passages and builds the BM25 index, once per run |
| 3. Search | `src/bluff/retrieval.py` | `search(question, 3)`: scores every passage, returns the top 3 with their ids and scores |
| 4. Build the prompt | `src/bluff/prompts.py` | `build_prompt(question, condition, wording, passages)`: "Use the context if it is relevant. Context: [1] … [2] … [3] …" then the usual question and instruction |
| 5. Ask the model | `src/bluff/pipeline/generate.py` | For every question and all 4 wordings: search → prompt → the model answers → token probabilities, entropies and hidden states recorded, plus `retrieved_ids` and `retrieval_scores` |
| 6. The extra signal | `src/bluff/signals.py` | `retrieval_top_score` (the best passage's BM25 score) becomes the 8th confidence number; in M0 it is always 0 |
| 7. Everything after | `label`, `featurize`, `train`, `evaluate`, `register`, `export` | Exactly the same code as M0: M1 gets its own 4 detectors, metrics and registry entry (`bluff-detector-m1_rag`) |
| 8. Compare (H4) | `src/bluff/pipeline/compare.py` | `bluff compare --config m0_base.yaml --other m1_rag.yaml`: paired comparison of the two versions on the same questions |

### The document collection, as built (16 November)

- **Sources** (listed in `data/corpus/sources.yaml`, editable): the **Constitution of Kenya 2010**
  from Wikisource (25 parts: preamble, 18 chapters, 6 schedules) and **63 Kenya-related English
  Wikipedia articles**, spread over the six question domains (civics and law, geography and
  society, language, institutions and services, money and livelihoods, "Western default" traps).
- **Cleaning:** reference-style sections (See also, References, External links, …) removed.
- **Passages:** paragraphs grouped into chunks of about 300–780 tokens (mean ~407 words); a long
  paragraph is split at sentence ends. Each chunk starts with its document's title. **441
  passages from 63 documents.**
- **Kept apart by document:** each document goes to exactly one collection, retrieval (M1) or
  fine-tuning (M2, 26 documents, ~61,000 words). The builder refuses a document listed in both.
  This caught "Ethnic groups in Kenya", a Wikipedia redirect to "Demographics of Kenya", which
  would have put one article in both (swapped for "Kenya Colony").
- **Leakage audit:** no test question's wording (8-word phrases) may appear in either collection.
  It caught "Religion in Kenya", which shares a phrase with a TriviaQA question (swapped for
  "Kikuyu people"). After that: 0 overlaps.
- **Every passage records** its source URL and licence (Wikipedia: CC BY-SA 4.0).

### A real example

"What is the capital city of Kenya?" → BM25 returns *Kenya* (score 20.1), *Nairobi* (18.9) and
*Health in Kenya* (18.2): the answer is in the second. "What is the name of King Lear's youngest
daughter?" → only unrelated Kenyan passages (*Jomo Kenyatta*, *Kenya*, *Nairobi*), because the
collection is about Kenya. **That is the H4 test:** does irrelevant reading material make the
model answer more confidently anyway? An M1 prompt averages ~1,800 tokens (M0: ~50), so an M1
run takes about 1–1.5 hours on the T4.

A known gap: "Which public body replaced the NHIF?" finds nothing useful, because no article on
the new Social Health Authority is in the list. BM25 also can't match synonyms ("devolved
units" vs "counties") or Swahili against English.

**Status (18 November):** code built and tested; document collection built and audited; the M1
run is next (`notebooks/run_m1_rag.ipynb`, waiting for Colab's GPU quota).

*Written with AI help (Claude): `retrieval.py`, `prompts.py`, `corpus.py`, `sources.yaml` and the
notebooks; see `AI_USE.md`.*

## 4. Quantisation, LoRA and QLoRA (variant M2)

### Quantisation

A model is numbers: 1.5 billion **weights**. Memory depends on the bits per weight:

| Format | Bytes per weight | Qwen 1.5B in memory |
|---|---|---|
| float32 | 4 | ~6 GB |
| float16 / bfloat16 | 2 | ~3 GB |
| 8-bit | 1 | ~1.5 GB |
| **4-bit** | 0.5 | **~0.8 GB** |

**Quantisation stores each weight with fewer bits by rounding it to a smaller set of allowed
values**, like recording heights to the nearest 5 cm instead of the nearest millimetre. With 4
bits each weight can take only 16 values. Three tricks make that enough, all switched on in the
code:

1. **Blocks with their own scale:** weights are grouped in blocks of 64; each block stores one
   scale, and each weight is stored as "level 0–15 × my block's scale".
2. **NF4 (NormalFloat-4):** weights follow a bell curve bunched near zero, so NF4 puts its 16
   levels where the weights are: dense near zero, sparse in the tails.
3. **Double quantisation:** the block scales are themselves stored in 8-bit.

Maths cannot be done in 4-bit: each layer's weights are expanded to 16-bit for the moment of the
calculation. **Stored in 4-bit, computed in 16-bit.**

**float16 vs bfloat16:** both 16 bits. bfloat16 keeps float32's *range* (no overflow) with less
precision; float16 is more precise but can overflow. Free T4 GPUs have no native bfloat16, so the
code must use float16 there.

### LoRA

- One Qwen attention matrix (`q_proj`) is 1,536 × 1,536, about **2.36 million** numbers. Full
  fine-tuning changes all of them, in every layer, and needs a lot of GPU memory.
- **LoRA** freezes that matrix and learns a small correction: two thin matrices, B (1,536 × 16)
  times A (16 × 1,536), **49,152** numbers, about 2% of the original. The layer's output becomes
  *original output + 2 × (B·A applied to the input)*: 16 is the **rank** (`r`), 2 is
  `alpha / r` = 32 / 16. B starts at zero, so training starts from exactly the original model.
- It works because the change fine-tuning needs tends to be *low-rank*: a few directions carry it.
- Applied to all 7 projections (q, k, v, o, gate, up, down) in all 28 layers: about **18.5
  million trainable numbers, ~1.2% of the model**. The result is an **adapter**, a folder of
  tens of MB that sits on top of the unchanged base model.

### QLoRA = quantised base + LoRA

The frozen base is stored in **4-bit NF4** (~0.8 GB instead of 3 GB); only the small LoRA
matrices are trained, in 16-bit. **Gradient checkpointing** saves more memory by recomputing
intermediate results instead of storing them. LoRA vs QLoRA in one line: same adapter idea;
QLoRA only changes how the frozen base is *stored*.

### The training settings (`src/bluff/pipeline/finetune_qlora.py`)

| Setting | Meaning |
|---|---|
| Learning rate 2e-4 | How big each correction step is |
| 3% warm-up steps, then cosine | Ramp the step size up gently, then lower it smoothly towards zero |
| Batch 2 × gradient accumulation 4 | Process 2 texts at a time, add up 4 rounds of corrections, update once: 8 texts per update without the memory cost |
| 3 epochs | Three passes over the Kenyan documents |
| Documents cut into 1,024-token windows | Long Acts are split into windows, not truncated, so all of each document is used |
| LoRA r = 16, alpha 32, on all 7 projection layers | 18.5 million trainable weights, 1.2% of the model |
| Objective: next-token prediction | Read Kenyan text and learn to predict each next word. It absorbs facts and phrasing; it is **not** taught to answer questions or to say "I don't know". |

That last row is why H3 is interesting: the model may gain knowledge *and* confidence, and lose
its willingness to abstain. M2 loads the base model in 16-bit and attaches the adapter on top.

**As run (19–20 November, Kaggle T4):** 26 documents → 106 windows → **42 update steps**, about
8 minutes. Training loss 2.47 → 2.26 (`reports/m2_qlora/training_log.json`). The adapter is
saved to the Hugging Face dataset repo and reused by every later M2 or M3 run.

### What fine-tuning did, and why (the H3 result)

**The result.** After the fine-tune, the model says "I don't know" **about half as often**
(Kenya 28% → 13%, trivia 35% → 20%; McNemar p < 10⁻⁸), yet it is **not more often right**
(Kenya 73% → 75% wrong, not significant). On Kenya, what its words reveal about its errors falls
(expressed detection −0.07), while what its inside reveals barely moves (internal −0.03, not
significant). Saying broke; knowing held.

**Quantisation is not the cause.** The "Q" in QLoRA only stores the frozen 1.5 billion weights in
4 bits *during training*, so training fits in a free T4's memory, like saving a photo at lower
quality. When M2 answers questions, the base model runs at normal 16-bit precision. What changes
the behaviour is the **LoRA adapter**: 18.5 million new weights trained on top of the frozen model.

**What the adapter learned.** Its only task was to predict the next word in Kenyan documents.
Those documents never say "I don't know": every sentence is a confident statement of fact ("The
President is the Commander-in-Chief of the Kenya Defence Forces"). So training pulls two ways:

1. **Towards facts.** But 42 short steps over about 80,000 words is far too little to memorise
   facts, so accuracy did not improve.
2. **Towards a style:** confident, declarative sentences. A style is much easier to learn than
   facts, so the model picked it up even in 42 steps.

**What it partly forgot.** The original model learned to say "I don't know" during its
**instruction training** (the stage that taught it to be a helpful, honest assistant). Training on
plain documents wears some of those assistant habits away: a known side effect of fine-tuning,
often called *forgetting*. Research on language models has reported the same pattern: fine-tuning
on new material can make a model more confident without making it more correct.

**RAG vs fine-tuning, side by side:**

| | RAG (M1) | Fine-tuning (M2) |
|---|---|---|
| Where the facts are | In front of the model, in the prompt, at question time | Must be stored in its weights during training |
| How it answers | Reads and copies from the passage | Recalls from memory, which barely changed |
| Effect on "I don't know" | Unchanged on Kenya; more often when the passages are useless | Halved: it learned the confident style |
| Effect on accuracy | Errors halved when the answer is in the passages | No improvement |

In one line: **RAG hands the model the answer; fine-tuning gave it the confidence of someone who
had studied, without the knowledge.** That makes the fine-tuned model riskier, and the probe,
which still reads the inside doubt, more useful.

## 5. Labels: how "right" and "wrong" are decided

- **Correct:** the open answer, lower-cased with punctuation and "a/an/the" removed, contains one
  of the accepted answers as whole words ("47", "forty-seven").
- **Abstained:** each question is also asked with an "I don't know" option in 3 wordings. If the
  model refuses in at least 2 of the 3 ("I don't know", "not sure", "no such", "sijui"…), it
  counts as abstained. Three wordings stop one prompt's phrasing from skewing the result.
- **Label noise:** automatic marking makes mistakes; "Kenya shillings" is marked wrong because the
  plural is not an accepted spelling. A random 50 answers per variant are checked by hand and the
  disagreement rate is reported as the *noise floor*. The code that computes that rate is not
  written yet (plan step B9).

## 6. What is measured, and what each number means

**Is the detector good at spotting wrong answers?**

| Metric | Meaning | Good |
|---|---|---|
| **AUROC** | Pick one wrong and one right answer at random: the chance the detector gives the wrong one the higher P(wrong). Needs no threshold. | 0.5 = coin flip, 1.0 = perfect |
| **PR-AUC** | How well it finds wrong answers without false alarms; more sensitive when wrong answers are rare | Higher; baseline = share of wrong answers |

**Can its probabilities be trusted?**

| Metric | Meaning | Good |
|---|---|---|
| **ECE** (Expected Calibration Error) | Sort answers into 10 bins by P(wrong); in each bin compare the average prediction with the share actually wrong; take the weighted average gap. If it says 30%, are 30% really wrong? | 0 = honest probabilities |
| **Reliability curve** | The per-bin table behind ECE, for plotting | Close to the diagonal |

Calibration matters because the abstain threshold assumes P(wrong) is honest.

**How good is answer-or-abstain?**

| Metric | Meaning | Good |
|---|---|---|
| **Risk–coverage curve** | Answer only the top k% most confident questions. Coverage = share answered; risk = error rate among them. | Risk stays low as coverage grows |
| **Risk at 70/80/90% coverage** | E.g. "answering only the 80% most confident, 12% of answers are wrong" | Lower |
| **AURC** | Average risk across all coverages: one number for answer-or-abstain quality | Lower |

**Knowing vs saying: the core of the research**

| Metric | Meaning |
|---|---|
| **Four cells** | Every answer placed by *probe flags uncertain?* × *took the "I don't know" option?*: honest uncertainty, **silent knowing**, abstention inflation, earned confidence |
| **Silent-knowing rate** (H1) | Among wrong answers the model delivered without abstaining, the share the probe flagged. High = it "knew" and answered anyway. |
| **Inflation rate** | Among abstentions, the share it answered correctly when given no option. High = it refuses things it knows. |
| **Internal detection** | AUROC of the probe score: how much the hidden states know about errors |
| **Expressed detection** | AUROC of abstention (yes/no) as an error predictor: how much the behaviour shows. 0.5 = its refusals are unrelated to its errors. |
| **Verbalisation gap** | Internal − expressed. Positive = **the model knows more than it says**. H3 predicts fine-tuning makes it grow. |
| **Hallucination rate** | Share of all answers that were wrong and stated rather than refused |

**Kenya against general knowledge (H2)**

| Comparison | Meaning |
|---|---|
| **Trivia vs Kenya** | Every metric reported separately for TriviaQA/TruthfulQA test questions and for Kenya, side by side |
| **Kenya vs UK twins** | 60 Kenya questions paired with the same question about the UK. Paired wrong and abstain rates, a bootstrap interval on the difference (Kenya minus UK), and McNemar's test. Pairing removes question style as an explanation. |

**Does it hold up out of distribution?**

| Metric | Meaning |
|---|---|
| **OOD drop** (H5) | AUROC on trivia test − AUROC on Kenya. Big = the detector does not travel well. Also a promotion gate: must be ≤ 0.25. |
| **Transfer gap** | The M0 probe applied unchanged to M1 and M2: the AUROC drop. Not built yet (plan step B12, if time allows). |

**Statistics: how sure are we?**

| Method | What it does |
|---|---|
| **Bootstrap CI** | Resample the questions with replacement 2,000 times (10,000 for the final report), recompute the metric each time; the middle 95% of results is the 95% confidence interval |
| **Paired bootstrap** | For two variants (or Kenya vs twin) on the same questions: resample the same questions for both and take the interval of the difference |
| **McNemar's test** | Did something change a yes/no outcome? It counts only the questions where the outcome changed; if nothing really changed, the two directions should be about equal |
| **Holm correction** | Several tests raise the chance of a false alarm; Holm makes the p-values stricter to compensate |

**Cost:** latency **p50/p95/p99** = median, 95th and 99th percentile response time. "p99 = 4 s"
means 1 request in 100 is slower than 4 seconds. Percentiles, not averages, because slow outliers
are what users feel.

## 7. The two thresholds

- **t_abstain = 0.25** (the API's gate). From costs: a confident wrong answer costs 3, a needless
  refusal costs 1. Answering costs p × 3 on average and refusing (1 − p) × 1, so refuse when
  p > 1/(1+3) = 0.25. A business decision written as maths (CRISP-DM business understanding).
- **tau_flag** (when the probe "says uncertain", for the four cells). Set on validation so that
  10% of *correct* answers are flagged, then fixed before any Kenya question is scored.

## 8. Serving, start to end

1. **Train:** each detector is saved as a folder: `detector.joblib` (the fitted model) plus
   `metadata.json` (thresholds, layer, git SHA, data hash, seed). The served one is copied to
   `artefacts/<variant>/serving/`.
2. **Register:** stored as version N. The first becomes `@champion`; later ones become
   `@challenger` only if they pass the gates.
3. **Package:** the Dockerfile copies that folder into the image (or `make serve` runs locally).
4. **Start:** uvicorn starts FastAPI; a *lifespan* handler loads the config, the detector and the
   language model **once**. `/ready` returns 503 until loading finishes.
5. **Request:** `POST /v1/ask {"question": "..."}`. Pydantic checks it (3–500 characters, no
   unknown fields, else error 422). For M1 the top 3 passages are retrieved. The model answers
   while signals are recorded.
6. **Score:** `signals.py` builds the same features as in training and `Detector.inputs()` picks
   the same layer and position, so there is no **training–serving skew**.
7. **Decide:** above 0.25 the answer is hidden (`abstain`); otherwise `answer`, or
   `answer_with_citation` when passages were used.
8. **Reply:** an audit record: request id, verdict, answer, bluff score, threshold, sources,
   variant, detector, version, latency.

The endpoint is a plain `def`, not `async`, so FastAPI runs the heavy work in a thread pool and
`/health` stays responsive.

**Gaps:** the API loads the `serving/` folder, not "whatever is `@champion`" (plan step B11).
On a laptop with 7.7 GB RAM and no NVIDIA GPU, the real model runs in bfloat16 (~3 GB) at several
seconds per answer: fine for a one-question-at-a-time demo, and by design (the proposal targets
CPU serving). Never shrink or quantise the model just to serve it faster: the probe was trained on
this exact model's hidden states.

### Why not serve with llamafile or Ollama?

[llamafile](https://github.com/mozilla-ai/llamafile) (Mozilla.ai) packs a quantised model and the
llama.cpp engine into one executable that runs fast on an ordinary CPU, with no installs. Ollama
is similar. Neither fits the served system:

- **No hidden states.** Their interface returns the answer text (and token probabilities), not the
  internal vectors at chosen layers that the probe reads. No hidden states means no probe, no bluff
  score and no gate.
- **A different model.** They run quantised weights (e.g. 4-bit), whose numbers differ from the
  model the detectors were trained on: training–serving skew.

The principle: **a guardrail that reads a model's internals ties you to a serving stack that
exposes them.** The project trades some speed for being able to see inside the model. llamafile
is still useful for a **live demo of the raw model bluffing**: it answers instantly on a laptop,
with no GPU and no bluff detector, which is exactly the contrast the system is built to show.

## 9. Docker

- **Image vs container:** an image is a frozen package of code, Python, libraries and the trained
  detector; a container is that image running. It behaves the same on any machine.
- **Multi-stage build:** stage 1 installs everything (with compilers); stage 2 copies only the
  finished environment. Smaller, fewer security holes.
- **Security:** runs as a non-root user (uid 10001); read-only filesystem in compose;
  dependencies installed from the hash-checked lock, so a tampered package fails to install.
- **HEALTHCHECK:** Docker calls `/health` every 30 seconds and marks the container unhealthy if it
  stops answering.
- **The detector is baked into the image**, so the image's unique ID *is* the release.
- **`docker-compose.yml`** runs the API (limited to 2 CPUs and 2 GB) plus an optional MLflow
  server. **`scripts/smoke.sh`** starts the container, waits for `/ready`, asks a question, checks
  the reply and checks that an empty question is rejected with 422.
- **Gap:** the image does not yet hold or cache the real model's weights, and 2 GB is too small for
  them (plan step B11).

## 10. Git, GitHub Actions, and CI / CD / CT

- **Git** records snapshots (commits) of the code; GitHub hosts them; `main` is the official
  branch.
- **GitHub Actions** is GitHub's automation: each YAML file in `.github/workflows/` says "when
  event X happens, start a fresh Linux machine and run these steps".

| | Trigger | Steps | Purpose |
|---|---|---|---|
| **CI** (Continuous Integration), `ci.yml` | Every push or pull request to `main` | Install locked deps → lint (ruff) → check the Kenya data → 44 tests → full pipeline on the fake model → Docker build → smoke test | The **merge gate**: red means don't merge |
| **CD** (Continuous Delivery), `cd.yml` | Every push to `main` | Install → train a detector (fake for now) → build the image tagged version + commit → **Trivy** scans it and fails on critical vulnerabilities → push to **GHCR** (GitHub's image registry) | Every change produces a scanned, versioned image. It publishes the image but runs it nowhere: there is no live server. |
| **CT** (Continuous Training), `ct.yml` | Manual button, pick a variant | On a GPU machine: install → pull data (DVC) → check data → full pipeline → upload reports | Retrain on demand. Never run: it needs a GPU machine registered with GitHub (a *self-hosted runner*). |

Trivy is pinned to a **commit SHA**, not a version tag, because its tags were hijacked in a 2026
supply-chain attack and a SHA cannot be swapped.

## 11. The rest of the MLOps

- **MLflow (experiment tracking):** when switched on (`tracking.mlflow: true`), every training run
  logs its settings, metrics, artefact, git SHA and data hash; registrations are mirrored to
  MLflow's registry. Off by default; needs an MLflow server running.
- **Registry and promotion gates:** numbered versions with movable labels. A challenger needs AUROC
  ≥ the champion's, ECE no worse, and an OOD drop ≤ 0.25. `bluff promote` moves the champion
  label (the human sign-off); **rollback** is promoting the old version again, nothing rebuilt.
- **DVC (Data Version Control):** like git for big data files; stores them by content hash, and
  `dvc repro` reruns only the stages whose inputs changed. Only the M0 stages are defined; no
  remote storage is set up.
- **Reproducibility:** each result is a function of code + data + config + environment + seed.
  Every saved detector records the git SHA, the data's sha256, the config path and the seed (42);
  `requirements.lock` pins the environment.
- **Testing pyramid:** unit (small functions) → data (the Kenya file passes its rules) → API (the
  service's behaviour) → integration (the whole pipeline on the fake model) → promotion gates.
- **Data contract and leakage audit:** `validate` enforces the question rules (unique ids, a
  source, a different reviewer, twins linked to real Kenya questions); `audit` fails the run if a
  question's wording appears in a document collection, so RAG or fine-tuning cannot "cheat".
- **Pre-registration:** the probe's layer and threshold are fixed on trivia validation data
  *before* Kenya is scored, so they cannot be tuned to flatter the results.

## 12. Monitoring and drift

**Exists today:** `/health` (liveness), `/ready` (readiness), `/metadata` (which detector is live
and where it came from), latency in every response, Docker's healthcheck.

**Does not exist yet:** request logging, anything that watches live traffic, drift detection.
Plan step B10 builds a minimal version: the API appends each audit record to a log, and
`bluff monitor` compares live traffic with the training data.

| Kind of drift | What it is here | How to monitor it |
|---|---|---|
| **Data drift** | The questions change, e.g. users start asking about Kenyan taxes when the detector was trained on trivia | Compare the 8 summary features of live answers with the training data: PSI (Population Stability Index) or a KS test |
| **Prediction drift** | The share of answers flagged or refused changes | Track the bluff-score distribution and the abstain rate per day; alert when they move far from the validation baseline |
| **Concept drift / model change** | The language model is swapped or fine-tuned, so the probe reads a different "mind" | The transfer experiment: how far the probe's accuracy falls on a new variant. The system's answer is to retrain and pass the gates. |

The proposal also describes **shadow mode**: a new detector scores live traffic beside the
champion without making decisions, until it is trusted.

## 13. The GPU setup

Only `generate` (and QLoRA training) needs a GPU. Free T4 GPUs come from Colab (run inside VS
Code) and Kaggle (run in the background, watched from VS Code), with a private Hugging Face
dataset repo as the shared drive. Every question is independent, so each GPU takes a *slice* of
the questions; a slice saves progress every ~100 questions and resumes after a timeout; the laptop
merges the slices and runs every other stage on CPU in minutes. All generation runs on the same
GPU type (T4) in float16 with pinned model and library versions, because different hardware gives
slightly different numbers. Step-by-step setup: [gpu-setup.md](gpu-setup.md).

## 14. What is built and what is not

| Built and tested in CI | Written but never run | Not built yet (see [plan.md](plan.md)) |
|---|---|---|
| The pipeline, 4 detectors, all metrics and statistics, the Kenya-vs-UK twin comparison, the API, Docker, CI, CD, registry and gates, config system, data contract, leakage audit | The real-model backend, QLoRA fine-tuning, MLflow tracking, DVC stages, CT | Split GPU runs and resume, the runner notebook, the sheet importer, the document-collection builder, label-noise rate, monitoring and drift, serving the champion, the real-model Docker image, probe transfer, LLM judge, semantic entropy, LSTM |
