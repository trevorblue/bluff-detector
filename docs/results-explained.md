# Bluff Detector: every result explained

**What this is.** A study guide for writing the final report. It goes through the project in
CRISP-DM order and, for every result, says **what was measured, what each number means, and what
it tells us about the model**. It is an explainer, not the report: the report, and its
interpretation of the results, must be written in your own words (see [AI_USE.md](../AI_USE.md)).
Every number here comes from `reports/results_summary.json`, written by
`notebooks/results.ipynb`; check against the notebook when you quote one.

**Contents**

1. The project in one page
2. Business understanding: the problem, the hypotheses, the cost rule
3. Data understanding: the questions and how answers are marked
4. Data preparation: cleaning, splits, the signals recorded
5. Modelling: the language model, the four versions, the four detectors
6. Evaluation, part 1: how to read every measurement
7. Evaluation, part 2: the results, one by one
8. Evaluation, part 3: what all the measurements mean together
9. Deployment: the live guard and the system around it
10. Limitations
11. A suggested outline for the report

---

## 1. The project in one page

**The question.** When a small open language model answers a question about Kenya, does it
*internally* register that it doesn't know, even while it answers confidently? And when the model
is changed (given documents, or fine-tuned), does that inner signal survive?

**The idea.** A language model has two ways of showing doubt:

- **What it says:** offered the option, does it reply "I don't know"?
- **What it knows:** its *hidden state*, the 1,536 numbers inside the model at the moment it
  answers. A simple classifier (a *probe*) can read them and estimate the chance the answer is
  wrong.

If the probe catches wrong answers that the model delivered confidently, the model **knows more
than it says**: it bluffs. That gap is what the project measures, and the probe becomes a
**guard** that withholds answers likely to be wrong.

**The model:** Qwen2.5-1.5B-Instruct (1.5 billion parameters, open weights so its insides can be
read), tested in four versions: as released (M0), with Kenyan documents pasted before each question
(M1, RAG), lightly fine-tuned on other Kenyan documents (M2, QLoRA), and both (M3).

**The answer, in short.**

- Yes, it knows more than it says (H1). Of the wrong answers it gave confidently on Kenya
  questions, its hidden state had already flagged 47%.
- It is worse on Kenya than on general trivia: 73% wrong against 64% (H2).
- A detector trained only on trivia still works on Kenya, losing little (H5).
- Asked about things that don't exist, it invents an answer 98% of the time.
- Giving it documents helps when they contain the answer and does no harm when they don't (H4 not
  supported).
- Fine-tuning halves how often it says "I don't know" without making it more accurate, while the
  inside signal holds (H3).
- Documents plus fine-tuning (M3): the documents repair what fine-tuning broke, but the probe
  travels worst there.

---

## 2. Business understanding (CRISP-DM phase 1)

**The problem.** Small models are cheap enough to run locally, which matters where large hosted
models are expensive or unavailable, as in much of Kenya. But they **hallucinate**: they state wrong
facts in the same confident tone as right ones. On local topics (Kenyan law, institutions,
geography), which are thin in their training data, the problem is worse. A user can't tell a bluff
from a fact.

**The business goal.** A guard in front of the model that withholds answers likely to be wrong,
cheaply enough to run on every request.

**The five hypotheses**

| | Hypothesis | Tested by |
|---|---|---|
| **H1** | The model knows more than it says: its hidden state predicts its errors better than its own "I don't know" | M0: probe AUROC vs "I don't know" AUROC |
| **H2** | It is wrong more often on Kenyan questions | M0: Kenya vs trivia; Kenya vs matched UK questions |
| **H3** | Fine-tuning breaks *saying* but not *knowing*: it says "I don't know" less, while the inside still registers doubt | M0 vs M2 |
| **H4** | Retrieval creates false confidence: given documents that don't contain the answer, it bluffs more | M0 vs M1 |
| **H5** | A detector trained on general trivia still works on Kenyan questions (a small drop) | Every detector, trivia vs Kenya |

**The cost rule (where the 25% line comes from).** A confident wrong answer is judged **3 times
worse** than a needless refusal: a wrong fact can mislead someone, while a refusal only costs them
a search. If the chance an answer is wrong is *p*:

- answering costs, on average, *p* × 3
- refusing costs (1 − *p*) × 1
- refuse when *p* × 3 > (1 − *p*), that is when **p > 0.25**

So the guard withholds any answer whose estimated chance of being wrong is above 25%. This is a
business decision written as arithmetic: a different cost ratio gives a different line.

**Success criteria.** The detector must beat a coin flip clearly on questions it never saw (AUROC
well above 0.5), its probabilities must be roughly honest (low ECE), and it must lose little when
moved from trivia to Kenya (a drop under 0.10 counts as small; above 0.25 blocks promotion).

---

## 3. Data understanding (CRISP-DM phase 2)

**The question sets**

| Set | How many | Where from | What it is for |
|---|---|---|---|
| **Trivia** | 1,448 (218 held out for testing) | TriviaQA (public dataset) | Training and testing the detectors on general knowledge |
| **Kenya, real** | 211 | Written by four students and Trevor | The main test: questions the detectors never saw |
| **Kenya, made up** ("fakes") | 82 | Questions about things that don't exist (a third house of Parliament, a 2,000-shilling note) | Does it admit it doesn't know, or invent? Every answer counts as wrong unless it declines |
| **UK twins** | 33 | The same question as a Kenya question, asked about the UK | H2 with question style held constant |

**Why trivia for training and Kenya for testing.** The detectors learn on trivia only. Kenya
questions are never used in training, so they are a true **out-of-distribution** test: the
situation a real deployment faces when users ask about things the detector wasn't built on.

**How each question is asked: two conditions.**

1. **Open:** the plain question. This answer is the one marked right or wrong and the one whose
   signals the detectors read.
2. **Offered "I don't know":** the question plus an invitation to say "I don't know" if unsure,
   in **3 different wordings**. If the model declines in at least 2 of the 3, the question counts
   as **abstained**. Three wordings stop one phrasing from skewing the result.

So every question produces 4 answers (1 open + 3 offered).

**How answers are marked (labels).** Automatically. The open answer is lower-cased, punctuation
and "a/an/the" are removed, and it counts as **right** if it contains one of the accepted answers
as whole words ("47", "forty-seven"). Otherwise it is **wrong**. Automatic marking makes mistakes
(a right answer worded differently is marked wrong), which is why TruthfulQA was dropped: its
marking was broken, with 98% of answers marked wrong.

**The Kenya set's history (declare this in the report).** The students' sheets were cleaned with
logged mechanical fixes. Because the planned student review didn't happen before the deadline, the
set was then **reviewed and repaired with AI**: 81 rows dropped (duplicates, yes/no or essay
questions, wrong or disputed answers), some answers corrected, and the 33 UK twins and 10 fakes
written by the AI review. Every change is in `data/kenya/repairs.yaml` and declared in AI_USE.md.
**Check:** using only questions written *and not edited* by humans, the conclusions don't change
(Kenya wrong rate 75.5%, probe AUROC 0.736, gap 0.136, against 73.0%, 0.731 and 0.124 on all
questions).

---

## 4. Data preparation (CRISP-DM phase 3)

**Splits.** Trivia is split 70% train, 15% validation, 15% test (218 test questions). Validation is
used to choose settings and the flag threshold; test is touched once, for the final numbers.

**What is recorded from each answer**

*The 8 summary signals (the model's "words").* As the model writes each word piece (token), it
gives a probability to every possible next token. From those:

| Signal | Meaning |
|---|---|
| Average word confidence (mean log-probability) | How sure it was of each word it chose, on average |
| Least-sure word (min log-probability) | Its confidence in its weakest word |
| Unsure words (share under 0.5) | Share of words picked with less than 50% confidence |
| Average hesitation (mean entropy) | How torn it was between possible words; 0 = no doubt |
| Biggest hesitation (max entropy) | Its most torn moment |
| Hesitation at the end (last entropy) | How torn it was finishing the answer |
| Answer length | Number of tokens |
| Top retrieval score | How well the best document matched the question (only meaningful for M1 and M3) |

*The hidden state (the model's "inside").* The model has 28 layers; each turns every token into
1,536 numbers. The probe reads **layer 21 (8th from the top), at the last answer token**: 1,536
numbers per answer. Think of the 8 signals as listening to the tone of voice, and the hidden state
as a brain scan taken while it answers.

**The document collections (for M1, M2, M3).**

- **Retrieval collection (M1, M3):** the Constitution of Kenya and 63 Wikipedia articles, cut into
  441 chunks. For each question, the 3 best-matching chunks (BM25 keyword search) are pasted
  before it.
- **Fine-tuning collection (M2, M3):** 26 *other* Kenyan documents (Acts, articles), none of them
  in the retrieval collection.
- **Leakage audit:** checks that no Kenya test question's wording appears in either collection, so
  the model can't simply copy the question's source.

---

## 5. Modelling (CRISP-DM phase 4)

There are two kinds of model. Keep them apart:

- the **language model** is *studied*, never trained on the test questions;
- the **detectors** are *trained*, to predict when the language model is wrong.

### 5a. The four versions of the language model

| Version | What changes | Technique |
|---|---|---|
| **M0** base | Nothing: the model as released | — |
| **M1** + RAG | 3 Kenyan document chunks pasted before every question | Retrieval-augmented generation, BM25 top-3 |
| **M2** + QLoRA | A small add-on trained on 26 Kenyan documents | QLoRA: model loaded in 4-bit; a low-rank adapter (rank 16, alpha 32, 18.5M trainable parameters) trained for 42 steps; training loss 2.47 → 2.26 |
| **M3** both | The M2 add-on and the M1 documents together | Both |

Each version answers every question, and each gets its **own four detectors trained on its own
answers**. The pipeline, metrics and thresholds are identical; only a config file changes. That is
what makes the comparison fair.

### 5b. The four detectors (the standard ML models)

All four do the same job: input = signals from one answer; output = **P(wrong)**, a number from 0
to 1; trained on trivia with labels wrong = 1, right = 0.

| Detector | Model | Reads | Why it's there |
|---|---|---|---|
| **Logistic regression** | Weighted sum of the inputs, squashed to 0–1 | The 8 word signals | Simple, readable baseline |
| **XGBoost** | 300 small decision trees (depth 3) | The 8 word signals | Catches combinations and cut-offs ("suspicious only when both are low") |
| **Linear probe** (served) | Logistic regression on the hidden state, strongly regularised (C = 0.05) | 1,536 hidden numbers | Reads the inside; cheap enough for every request |
| **MLP probe** | Small neural network, one hidden layer of 128 units, early stopping | 1,536 hidden numbers | Checks whether the inside signal is simple (linear) or needs a curve |

**What comparing them answers**

- Probes vs the word detectors: does the inside know more than the words show?
- MLP vs linear probe: if the MLP isn't clearly better, the signal is written plainly (linearly)
  in the model.
- Probe vs the model's own "I don't know": the gap between knowing and saying (H1, H3).
- Trivia vs Kenya, for each: does what they learned travel (H5)?

**Which one is served.** For all four versions the served detector (the "champion" in the
registry) is the linear probe. The demo lets any of the four decide.

### 5c. The two thresholds

- **t_abstain = 0.25**: the guard's line, from the 3:1 cost rule (section 2).
- **tau_flag**: when the probe counts as "flagging doubt" for the knowing-vs-saying analysis. Set
  on validation so that 10% of *right* answers get flagged, then fixed before any Kenya question
  is scored. (For M3 it came out at 0.979.)

---

## 6. Evaluation, part 1: how to read every measurement

Each entry: what it is, how to read a value, and an example from the results.

### Is the detector good at spotting wrong answers?

**AUROC** (area under the ROC curve).
- *What:* pick one wrong and one right answer at random. AUROC is the chance the detector gives
  the wrong one the higher P(wrong).
- *Read it:* 0.5 = coin flip (useless); 1.0 = perfect; 0.7 = acceptable; 0.8 = good. It needs no
  threshold, so it measures ranking ability alone.
- *Example:* M0's linear probe has AUROC 0.731 on Kenya: given a wrong and a right Kenya answer,
  it ranks the wrong one as more suspicious 73% of the time.

**95% confidence interval [lo, hi]** (bootstrap).
- *What:* the questions are resampled with replacement 2,000 times and the metric recomputed each
  time; the middle 95% of results is the interval.
- *Read it:* the range the true value plausibly lies in, given only ~200 questions. If an
  interval for a *difference* excludes 0, the difference is real at the 95% level.
- *Example:* Kenya probe AUROC 0.731 [0.651, 0.803]: confidently above 0.5, but the exact value is
  uncertain by about ±0.08.

**PR-AUC** (area under the precision–recall curve).
- *What:* how well it finds the wrong answers without false alarms.
- *Read it:* compare with the **share of wrong answers**, which is what a useless detector scores.
  On Kenya (73% wrong), a PR-AUC of 0.86 is good but the floor is already 0.73.
- *Example:* M0 linear probe, Kenya PR-AUC 0.861 against a floor of 0.73.

### Can its probabilities be trusted?

**ECE** (expected calibration error).
- *What:* answers are sorted into 10 bins by P(wrong); in each bin the average prediction is
  compared with the share actually wrong; ECE is the weighted average gap.
- *Read it:* 0 = honest probabilities ("30%" really means 30% are wrong). 0.14 = off by about 14
  percentage points on average. It matters because the 25% line assumes P(wrong) is honest.
- *Example:* M0 linear probe, Kenya ECE 0.141; M1's rises to 0.268 (much less honest on Kenya).

### How good is "answer or abstain"?

**Risk at 80% coverage.**
- *What:* let the detector refuse its 20% most suspicious answers and answer the rest; risk = the
  share wrong among those answered.
- *Read it:* lower is better; compare with the overall wrong rate.
- *Example:* M0 linear probe on Kenya: 69% wrong at 80% coverage, against 73% answering
  everything. Refusing a fifth only helps a little, because so much is wrong.

**AURC** (area under the risk–coverage curve).
- *What:* the average risk over every possible coverage, one number for answer-or-abstain quality.
- *Read it:* lower is better, but it depends heavily on the overall wrong rate, so compare
  detectors *within* a version, not across versions.

### Does it travel?

**OOD drop** (H5).
- *What:* AUROC on trivia test minus AUROC on Kenya.
- *Read it:* small (< 0.10) = the detector carries over to the new kind of question; large = it
  doesn't. Above 0.25 blocks promotion to serving.
- *Example:* M0 linear probe 0.773 → 0.731, a drop of 0.04.

### The model's behaviour

**Wrong rate.** Share of open answers marked wrong. M0 Kenya: 73%.

**"I don't know" rate (abstain rate).** Share of questions where, offered the option, it declined
(2 of 3 wordings). M0 Kenya: 28%.

**"Stated a wrong answer" (hallucination rate).** Share of all questions where, asked normally,
it gave a wrong answer instead of declining. M0 Kenya: 72.5%. Almost equal to the wrong rate,
because asked normally the model almost never declines.

**Inflation rate.** Among the times it said "I don't know", the share it actually got right when
asked normally. High = it refuses things it knows. M0 Kenya: 12% (low: its refusals are mostly
justified). M1 on trivia: 28% (the irrelevant documents make it refuse things it knows).

### Knowing vs saying (the heart of the project)

**Internal detection** = the probe's AUROC: how much the *inside* reveals about errors.

**Expressed detection** = the AUROC of "said I don't know" (yes/no) as a predictor of being wrong:
how much its *behaviour* reveals. 0.5 = its refusals are unrelated to its errors.

**Verbalisation gap** = internal − expressed. **Positive = it knows more than it says.**
- *Example:* M0 Kenya: internal 0.731, expressed 0.607, gap 0.124 [0.033, 0.209]. The interval
  excludes 0, so the gap is real.

**The four cells.** Every answer is placed in a 2 × 2 grid: *did the probe flag doubt?* × *did the
model say "I don't know"?*

| | Said "I don't know" | Answered anyway |
|---|---|---|
| **Probe flagged doubt** | **Honest uncertainty**: unsure inside and out | **Silent knowing**: unsure inside, confident outside (the bluff) |
| **Probe saw no doubt** | **Abstention inflation**: sure inside, but refused | **Earned confidence**: sure inside and out |

M0 on Kenya (211 questions): honest uncertainty 25, **silent knowing 54**, abstention inflation 34,
earned confidence 98.

**Silent-knowing rate.** Among the wrong answers it delivered without declining, the share the
probe had flagged. *Read it:* the share of bluffs the guard could catch. M0 Kenya: **47%**
[37%, 56%].

### The statistical tests

| Test | Used for | How to read it |
|---|---|---|
| **Fisher's exact test** | Wrong rate on Kenya vs trivia (H2), two separate groups | p < 0.05 = the difference is unlikely to be chance |
| **McNemar's test** | Two versions, or Kenya vs twin, on the **same** questions; yes/no outcomes | Counts only the questions whose outcome changed. p < 0.05 = a real change |
| **Paired bootstrap** | Differences in AUROC between versions on the same questions | If the interval excludes 0, the change is real |
| **Holm correction** | Several tests at once | Makes p-values stricter so that running many tests doesn't produce false alarms |

A p-value is the chance of seeing a difference at least this big **if nothing had really
changed**. p = 0.03 means 3%: unlikely, so we call it significant. "n.s." = not significant.

---

## 7. Evaluation, part 2: the results, one by one

### H1. The model knows more than it says — supported

**Measured on M0.**

| | Trivia (218) | Kenya (211) |
|---|---|---|
| Internal (probe AUROC) | 0.773 [0.707, 0.837] | 0.731 [0.651, 0.803] |
| Expressed ("I don't know" AUROC) | 0.598 [0.535, 0.661] | 0.607 [0.551, 0.663] |
| **Gap** | **0.175 [0.104, 0.245]** | **0.124 [0.033, 0.209]** |
| Silent knowing | 34% [24%, 44%] | **47% [37%, 56%]** |

**What each number means**

- *Internal 0.73 on Kenya:* the probe ranks a wrong answer above a right one 73% of the time.
- *Expressed 0.61:* the model's own "I don't know" barely beats a coin flip at telling its wrong
  answers from its right ones; its refusals are only loosely tied to its errors.
- *Gap 0.12, interval above 0:* the inside reveals clearly more than the words, on both sets.
- *Silent knowing 47%:* nearly half of its confident wrong answers on Kenya were already flagged
  inside. Those are the bluffs.

**A real example.** "What is the largest inland port city in Kenya?" M0 answers "Mombasa" (it is
Kisumu; Mombasa is on the coast). Its words look confident: the word detectors give a 14–18%
chance of being wrong. The probe reading its inside gives 99.8%. The guard blocks it.

**What it tells us.** The doubt exists inside the model but often isn't strong enough to win the
word choice, so the model answers anyway. A probe can recover that doubt.

### H2. It is worse on Kenyan questions — supported

| | Wrong |
|---|---|
| Trivia | 64% |
| Kenya | **73%** (one-sided Fisher p = 0.026) |
| UK twins (33 pairs) | 45% UK vs 64% Kenya (McNemar p = 0.21) |

**What it means.** Kenya questions are answered wrongly more often, and the difference is unlikely
to be chance. The twins compare *the same question* about Kenya and about the UK, which removes
question style as an explanation; they point the same way (64% vs 45%), but 33 pairs are too few
for that difference to be significant. Overall, the model is weaker on local knowledge, as
expected from its training data.

**Note.** Both rates are high: a 1.5B model is wrong on about two-thirds of trivia too. That is
why a guard matters.

### H5. A detector trained on trivia still works on Kenya — supported

| Detector (M0) | Trivia AUROC | Kenya AUROC | Drop |
|---|---|---|---|
| Linear probe (served) | 0.773 | 0.731 | 0.04 |
| MLP probe | 0.815 | 0.749 | 0.07 |
| Logistic regression | 0.782 | 0.760 | 0.02 |
| XGBoost | 0.753 | 0.722 | 0.03 |

**What it means.** Every detector loses a little moving to Kenya, none collapses, and all drops are
under the 0.10 "small" line. What they learned about wrong answers on general trivia carries over
to Kenyan questions they never saw.

**A nuance worth stating honestly.** On M0 the probes don't clearly beat the word detectors
(logistic regression on 8 numbers reaches 0.76 on Kenya; the confidence intervals overlap). The
probe's advantage is over the model's *own "I don't know"* (H1), not clearly over a detector that
reads its word confidence.

### Made-up questions: it invents answers

| 82 questions about things that don't exist | Invents an answer |
|---|---|
| Asked normally | **98%** (declines only 2%) |
| Offered "I don't know" | 43% |

**What it means.** Asked normally, it almost never says a thing doesn't exist: it makes up an
answer as if it did. Inviting it to say "I don't
know" helps a lot, but it still invents an answer to 43%. The detectors find these hard: the
model's insides look confident on many fakes (probe AUROC on fakes is below 0.5).

### H4. Retrieval creates false confidence — not supported

**Measured M0 vs M1 on the same questions (McNemar).** The key is to split the Kenya questions by
whether the 3 retrieved chunks actually contain the answer (103 of 211 do).

| Set | n | Wrong M0 → M1 | p | "I don't know" M0 → M1 |
|---|---|---|---|---|
| Kenya, answer **in** the passages | 103 | **60% → 31%** | 0.00002 | 19% → 11% |
| Kenya, answer **not** in the passages | 108 | 85% → 92% | 0.09 (n.s.) | 36% → 44% |
| Kenya, all | 211 | 73% → 62% | 0.004 | 28% → 27% |
| Trivia (documents irrelevant) | 218 | 64% → 69% | 0.11 (n.s.) | **35% → 88%** |

**What each row means**

- *Answer in the passages:* the documents **halve** its errors. Retrieval works when it finds the
  right page.
- *Answer not in the passages:* no significant change in errors, and it says "I don't know" a
  little *more*, not less. No sign of false confidence.
- *Trivia:* given clearly irrelevant Kenyan pages, it says "I don't know" far more (88%). It
  becomes *more* cautious, the opposite of H4.

**What it tells us.** H4 predicted documents would make it bluff; instead they help when relevant
and make it more cautious when not. The limit is coverage: the collection only contains the answer
to about half the Kenya questions.

**One more effect to mention.** With documents, the knowing-vs-saying gap on Kenya disappears
(M1 gap 0.002; change from M0 −0.12 [−0.23, −0.02]). Its "I don't know" becomes more informative
(expressed 0.61 → 0.68) and the probe less (0.73 → 0.68).

### H3. Fine-tuning breaks saying, not knowing — supported

**Measured M0 vs M2 on the same questions** (McNemar with Holm correction; paired bootstrap on the
AUROCs).

| | Trivia | Kenya |
|---|---|---|
| "I don't know" M0 → M2 | **35% → 20%** | **28% → 13%** (p < 10⁻⁸) |
| Wrong M0 → M2 | 64% → 66% (n.s.) | 73% → 75% (n.s.) |
| Change in expressed (saying) | −0.005 (n.s.) | **−0.070 [−0.13, −0.02]** |
| Change in internal (knowing) | +0.047 [0.002, 0.096] | −0.030 [−0.10, 0.05] (n.s.) |
| Change in the gap | +0.052 (n.s.) | +0.041 (n.s.) |

**What each number means**

- *"I don't know" halved, wrong unchanged:* after fine-tuning it admits doubt half as often, but
  it isn't more often right. It has become **more confident, not more knowledgeable**.
- *Expressed −0.07 on Kenya (interval below 0):* its "I don't know" became significantly worse at
  pointing to its errors: **saying broke**.
- *Internal −0.03 (interval includes 0):* the probe's ability didn't change significantly:
  **knowing held**. On trivia it even rose slightly.
- *Gap +0.04:* grows in the predicted direction but not significantly. State this honestly: the
  core of H3 is supported; the gap growth is not proven.

**Why it matters.** Only 42 training steps on 26 documents were enough to halve the model's
willingness to admit doubt. Fine-tuning, routine in practice, can silently make a model a more
confident bluffer, and a probe still sees the doubt its words no longer show.

### M3. Retrieval and fine-tuning together (no hypothesis; an extra experiment)

**Measured M2 vs M3 and M1 vs M3 on the same questions.**

| Set | n | Wrong M2 → M3 | p |
|---|---|---|---|
| Kenya, answer **in** the passages | 103 | **62% → 28%** | < 10⁻⁷ |
| Kenya, answer **not** in the passages | 108 | 88% → 92% | 0.42 (n.s.) |

| Kenya | M0 | M1 | M2 | M3 |
|---|---|---|---|---|
| Wrong | 73% | 62% | 75% | **61%** |
| "I don't know" | 28% | 27% | **13%** | 27% |
| Probe AUROC (inside) | 0.73 | 0.68 | 0.70 | **0.64** |
| Logistic regression AUROC (words) | 0.76 | 0.73 | 0.80 | 0.79 |

**What it means**

1. *Documents still help the fine-tuned model:* with the answer in the passages, errors more than
   halve (62% → 28%), the same effect as M0 → M1.
2. *Documents repair what fine-tuning broke:* "I don't know" on Kenya returns from 13% to 27%
   (p = 0.0001), about M0's 28%, and the words' ability to point to errors recovers (expressed
   +0.16 [0.10, 0.22]).
3. *Fine-tuning adds nothing on top of documents:* M1 → M3 on Kenya, wrong 62% → 61% and "I don't
   know" 27% → 27%. Its only clear effect is on trivia ("I don't know" 88% → 63%).
4. *The cost:* M3 is the most accurate version on Kenya, but its probe is the weakest there (0.64,
   interval 0.56–0.71). The likely reason: each probe was trained on trivia, where the Kenyan
   passages never contain the answer, so it never saw what the inside looks like when they do. The
   word detector, which includes the retrieval score, holds at 0.79.

### The guard in practice (M0, linear probe, line at 25%)

| On the 211 Kenya questions | |
|---|---|
| Wrong if it answers everything | 73% |
| Questions refused | 87% |
| **Wrong answers caught** | **93%** |
| Right answers refused | 70% |
| Wrong among the answers it lets through | **39%** |

**What it means.** With the 3:1 cost rule, the guard is very cautious on Kenya: it catches almost
every wrong answer, but refuses most right ones too, because the model is wrong so often that
almost every answer is above 25%. Of what it does let through, 39% is still wrong, against 73%
without it. A higher line would answer more and catch less; the line is a business choice.

### All four detectors across the four versions (AUROC)

| Detector | M0 trivia / Kenya | M1 | M2 | M3 |
|---|---|---|---|---|
| Linear probe | 0.77 / 0.73 | 0.81 / 0.68 | 0.82 / 0.70 | 0.82 / 0.64 |
| MLP probe | 0.82 / 0.75 | 0.82 / 0.67 | 0.77 / 0.70 | 0.83 / 0.64 |
| Logistic regression | 0.78 / 0.76 | 0.76 / 0.73 | 0.81 / 0.80 | 0.81 / 0.79 |
| XGBoost | 0.75 / 0.72 | 0.75 / 0.72 | 0.79 / 0.79 | 0.79 / 0.76 |

**What it means.**

- On **trivia** (the kind of question they trained on), the best detector is a probe in every
  version.
- On **Kenya**, the probes travel worse than the word detectors whenever the model was changed,
  most of all with documents (M1, M3). The word detectors barely drop.
- The MLP is not consistently better than the linear probe, so the inside signal is mostly simple
  (linear).
- Calibration follows the same pattern: the probes' ECE on Kenya grows from 0.14 (M0) to 0.27 (M1
  and M3).

### Speed (generation time per answer on a T4 GPU, median)

M0 0.19 s, M2 0.40 s, M1 1.5 s, M3 2.1 s. Documents make the prompt about 2,000 tokens long, which
is why M1 and M3 are slower. The probe itself adds almost nothing (one multiply-and-add over 1,536
numbers). On the laptop CPU: M0 and M2 take 15–40 s, M1 and M3 about 8 minutes.

---

## 8. Evaluation, part 3: what all the measurements mean together

Put the readings side by side and a consistent picture of this model appears.

1. **It often doesn't know, and rarely says so.** Wrong on 73% of Kenya questions; asked normally
   it states a wrong answer 72.5% of the time and almost never declines, even about things that
   don't exist (98%).
2. **But its inside does register doubt.** The hidden state predicts its errors clearly better than
   its own "I don't know" (gap 0.12–0.18), and nearly half of its confident Kenya bluffs were
   flagged inside. *Knowing* and *saying* are separate things in this model.
3. **That inside signal is general.** Learned on trivia, it carries over to Kenyan questions with
   a small loss (H5), so a guard can be built without Kenyan training data, on the base model.
4. **Changes to the model affect saying and knowing differently.** Fine-tuning damaged saying (half
   as many "I don't know"s) while knowing held (H3). Documents improved accuracy and saying (H4,
   M3). So a model's willingness to admit doubt can't be trusted to survive a change; it must be
   re-measured after every one.
5. **A probe is tied to the model and the setting it was trained on.** Every time the model was
   changed, the probe trained on trivia lost more on Kenya, most of all with documents (0.64 on
   M3). The word detectors were sturdier. In practice: **retrain the detector whenever the model or
   its setting changes, and train it on questions like the ones it will face** (for a RAG system,
   questions the documents cover). The project's registry and promotion gates exist for exactly
   that.
6. **The guard works, at a price set by the cost rule.** On Kenya it cuts wrong answers from 73% to
   39% of what it says, at the cost of refusing most questions. With a model this weak on local
   facts, caution is the right default; a stronger model would let more through.

**One-sentence conclusion.** A small model's hidden state knows when it is likely to be wrong more
reliably than its words admit, well enough to guard it on Kenyan questions it never trained on;
but that knowledge must be re-learned whenever the model or its setting changes.

---

## 9. Deployment (CRISP-DM phase 6)

| Piece | What it does |
|---|---|
| **API** (FastAPI) | `POST /v1/ask`: question in → answer or refusal, P(wrong), the line, the version, the detector, an audit record. Also `/health`, `/ready`, `/v1/variants`, `/v1/model-card`, `/v1/results` |
| **Demo page** | Three tabs: ask a question (choose the version M0–M3 and the detector; see the words-vs-inside readings explained), what we found (every result and every metric), how it works |
| **One model, four versions** | M0–M3 share one copy of the model in memory; M2 and M3 switch the fine-tuning adapter on, M1 and M3 add the documents |
| **Model registry** | Every trained detector is versioned with its metrics, data hash and git commit; promotion gates (e.g. OOD drop ≤ 0.25) decide the champion |
| **Docker** | The service as a container; CI builds and smoke-tests it |
| **CI** (GitHub Actions) | Every push: lint, data checks, 77 tests, the whole pipeline on a fake model, Docker build |
| **CD** | Every push to main: build, vulnerability scan (Trivy), publish the image |
| **Monitoring** | Every request is logged; `bluff monitor` turns the log into a drift report (are live questions drifting from the training data?) |
| **Storage** | Code, detectors, the M2 adapter and results on GitHub; raw answers, hidden states and datasets on Hugging Face ([where-things-are.md](where-things-are.md)) |

---

## 10. Limitations (state these in the report)

- **One small model** (1.5B), wrong on about two-thirds of questions. A larger model may behave
  differently.
- **Automatic marking** compares words, so some right answers are marked wrong; the label-noise
  rate from a manual check was not measured.
- **The Kenya set** was repaired with AI instead of the planned student review (declared; results
  hold without the AI items). Only 33 of the planned 60 UK twins exist, all AI-written, so H2's
  paired test is underpowered.
- **Sample size.** About 200 test questions per set gives confidence intervals of roughly ±0.08 on
  AUROC; some real differences (the H3 gap growth, the twins) can't be confirmed.
- **The document collection** covers only about half the Kenya questions.
- **Probes trained on trivia** travel worst when documents are added (M1, M3).
- **Free GPUs** (Colab and Kaggle T4s): library versions differ slightly between runs (recorded);
  the laptop demo runs on a CPU, so its scores can differ slightly from the GPU's.

---

## 11. A suggested outline for the report

Use your own words throughout; this is only a skeleton with the figures to include.

1. **Introduction**: the problem (bluffing small models, local knowledge), the question, the
   contribution (a probe-based guard, tested across four model versions).
2. **Background**: hallucination; probes on hidden states; abstention; RAG; QLoRA.
3. **Business understanding**: the goal, the five hypotheses, the 3:1 cost rule.
4. **Data**: the four question sets, the two prompt conditions, automatic marking, the Kenya set's
   history (and the AI repair).
5. **Method**: the four versions; the 8 signals and the hidden state; the four detectors; the
   thresholds; the metrics and statistical tests.
6. **Results**: H1, H2, H5, made-up questions, H4, H3, M3, the guard. Figures, all in
   `reports/results_figures/`:
   - `h1_knowing_vs_saying.png`
   - `h2_worse_on_kenya.png`
   - `fakes_invented_answers.png`
   - `h4_retrieval.png`
   - `h3_finetuning.png`
   - `m3_combined.png`
   - Per-version charts (ROC, calibration, risk–coverage, four cells) in
     `reports/<version>/figures/`.
7. **Discussion**: what the measurements mean together (section 8 here), in your words.
8. **Deployment**: the API, demo, registry, CI/CD, monitoring.
9. **Limitations and future work**: section 10; plus probes trained on Kenyan or retrieval-covered
   questions, a larger model, a human-reviewed Kenya set, the label-noise check.
10. **Conclusion.**
11. **AI use statement**: point to AI_USE.md.
