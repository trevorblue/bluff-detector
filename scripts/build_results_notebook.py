# ruff: noqa: E501  (the notebook's markdown text has long table rows)
"""Builds notebooks/results.ipynb: the project's results, hypothesis by hypothesis, for a demo."""
import sys

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# Bluff Detector: the results

**The question.** When a language model answers confidently but wrongly (a *bluff*), does its
inside already "know" it is unsure? And is it worse on Kenyan questions?

**The setup.** One open model (Qwen2.5-1.5B-Instruct) answered 1,448 general trivia questions
and 326 Kenya items (211 real Kenya questions, 82 fake ones about things that don't exist, and
33 UK "twins" of Kenya questions). Each was asked normally, and three more times with the option
to say "I don't know". A small detector (a *probe*) read the model's hidden state on every answer
and estimated the chance the answer was wrong. Detectors were trained on trivia only; every Kenya
item is a test question they never saw.

**How to run:** top right **Select Kernel → `bluff-detector (.venv)`**, then **Run All**. It reads
the saved results (no GPU). Charts are saved to `reports/results_figures/`.

| # | Hypothesis | In plain words |
|---|---|---|
| H1 | Silent knowing | The model's inside knows more than it says |
| H2 | Worse on Kenya | It is wrong more often on Kenyan questions |
| H5 | The detector travels | A detector trained on trivia still works on Kenya |
| — | Fake questions | Does it admit it doesn't know, or invent an answer? |
| H4 | Retrieval (RAG) | Do Kenyan documents help, or create false confidence? |
| H3 | Fine-tuning | Does extra training make it bluff more? (not run: future work) |
""")

code('''
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import fisher_exact
from sklearn.metrics import roc_curve

from bluff import metrics
from bluff.utils import ensure_dir, read_jsonl

if Path.cwd().name == "notebooks":
    os.chdir("..")
FIG = ensure_dir(Path("reports/results_figures"))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_colwidth", 70)
C = {"trivia": "#5b6577", "kenya": "#1b7a3a", "uk": "#1f4fd1", "bad": "#b3261e"}
SEED = 42


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")


def ci_text(c, pct=False):
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
    return f"{f(c['point'])} (95% CI {f(c['lo'])} to {f(c['hi'])})"


def boot_diff(a, b, n_boot=2000, seed=SEED):
    """Difference of two means (a - b) with a bootstrap 95% CI, independent samples."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"point": a.mean() - b.mean(), "lo": lo, "hi": hi}


def load(variant):
    p = Path(f"reports/{variant}")
    if not (p / "scores.parquet").exists():
        return None
    sc = pd.read_parquet(p / "scores.parquet")
    gens = {g["qid"]: g["answer"] for g in read_jsonl(f"data/interim/{variant}/generations.jsonl")
            if g["condition"] == "open"}
    sc["model_answer"] = sc["qid"].map(gens)
    return sc.merge(Q, left_on="qid", right_on="id", how="left")


Q = pd.DataFrame(read_jsonl("data/processed/questions.jsonl"))[
    ["id", "question", "gold", "author", "reviewer", "twin_of"]]
m0 = load("m0_base")
m1 = load("m1_rag")
M = json.loads(Path("reports/m0_base/metrics.json").read_text())
trivia = m0[(m0.pool == "trivia") & (m0.split == "test")]
kenya, fakes, twins = (m0[m0.pool == p] for p in ("kenya", "kenya_unanswerable", "western"))
print(f"M0 results: {len(trivia)} trivia test questions, {len(kenya)} Kenya, {len(fakes)} fake, "
      f"{len(twins)} UK twins.  M1 results: {'yes' if m1 is not None else 'not downloaded'}")
''')

md("""
## The data

Where every number below comes from. "Written by AI" and "edited by AI" are the Kenya items
repaired by an AI review on 19 November (declared in `AI_USE.md`); section 7 repeats the main
results without them.
""")

code('''
rows = []
for name, d in [("Trivia (test questions)", trivia), ("Kenya, real", kenya),
                ("Kenya, fake", fakes), ("UK twins", twins)]:
    rows.append({"set": name, "questions": len(d),
                 "written by students / Trevor": int((~d.author.eq("Claude (AI)")).sum()),
                 "written by AI": int(d.author.eq("Claude (AI)").sum()),
                 "edited by AI": int(d.reviewer.eq("Claude (AI)").sum())})
display(pd.DataFrame(rows).set_index("set"))
''')

md("""
---
## H1. Does the model know more than it says?

Two ways to tell a right answer from a wrong one:
- **Internal** (inside): the probe reading the model's hidden state. AUROC: 1 = perfect, 0.5 = a coin flip.
- **Expressed** (saying): whether the model itself takes the "I don't know" option on the questions it gets wrong.

**The gap = internal − expressed.** Above 0 means the inside knows more than the model says.
**Silent knowing:** of the wrong answers it stated confidently, the share its inside flagged as unsure.
""")

code('''
A = M["abstention"]
sets = [("trivia_test", "Trivia"), ("kenya", "Kenya")]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
BARS = [("internal_detection", "internal (inside)", C["uk"]),
        ("expressed_detection", "expressed (saying)", C["trivia"])]
for i, (key, _label) in enumerate(sets):
    for j, (m, name, col) in enumerate(BARS):
        c = A[key][m]
        ax.bar(i + (j - 0.5) * 0.35, c["point"], 0.35, color=col,
               yerr=[[c["point"] - c["lo"]], [c["hi"] - c["point"]]], capsize=4,
               label=name if i == 0 else None)
ax.axhline(0.5, ls="--", lw=0.8, color="grey")
ax.text(1.45, 0.51, "coin flip", fontsize=8, color="grey", ha="right")
ax.set_xticks([0, 1], [f"{lab} (n={A[k]['n']})" for k, lab in sets])
ax.set_ylim(0.4, 0.9)
ax.set_ylabel("AUROC (separates right from wrong)")
ax.legend(loc="upper right")
ax.set_title("H1: the inside knows more than the model says")
save(fig, "h1_knowing_vs_saying")
plt.show()
for key, label in sets:
    g, s = A[key]["verbalisation_gap"], A[key]["silent_knowing_rate"]
    verdict = "SUPPORTED" if g["lo"] > 0 else "not shown"
    print(f"{label}: gap {ci_text(g)} -> {verdict};  silent knowing {ci_text(s, pct=True)}")
''')

md("""
### Real bluffs the probe caught on Kenya questions

The model answered wrongly **without hedging**, but its inside was flagged "unsure". Compare the
probe (inside) with logistic regression, which only sees how sure the model was of its words:
where logreg is low and the probe high, the bluff was invisible from the outside.
""")

code('''
caught = kenya[(kenya.flag == 1) & (kenya.y_wrong == 1) & (kenya.abstained == 0)].copy()
caught["inside minus outside"] = caught.p_probe_linear - caught.p_logreg
display(caught.sort_values("inside minus outside", ascending=False)[
    ["question", "gold", "model_answer", "p_probe_linear", "p_logreg"]].head(8).round(2)
    .rename(columns={"gold": "correct answer", "p_probe_linear": "P(wrong): probe (inside)",
                     "p_logreg": "P(wrong): logreg (outside)"}))
''')

md("""
---
## H2. Is the model worse on Kenyan questions?

**Test 1:** wrong rate on Kenya vs on general trivia. **Test 2, the fair one:** the **UK twins**,
the same question asked about Kenya and about the UK ("What is the highest court in Kenya?" /
"...in the United Kingdom?"), compared pair by pair (McNemar's test).
""")

code('''
d = boot_diff(kenya.y_wrong, trivia.y_wrong)
T = M["twins"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
vals = [trivia.y_wrong.mean(), kenya.y_wrong.mean()]
a1.bar(["Trivia", "Kenya"], vals, color=[C["trivia"], C["kenya"]])
for i, v in enumerate(vals):
    a1.text(i, v + 0.01, f"{100 * v:.0f}%", ha="center")
a1.set_ylim(0, 1)
a1.set_title("Wrong rate, all questions")
tw = [T["wrong_rate"]["kenya"], T["wrong_rate"]["uk"]]
a2.bar(["Kenya question", "UK twin"], tw, color=[C["kenya"], C["uk"]])
for i, v in enumerate(tw):
    a2.text(i, v + 0.01, f"{100 * v:.0f}%", ha="center")
a2.set_ylim(0, 1)
a2.set_title(f"Wrong rate, {T['n_pairs']} matched pairs")
fig.suptitle("H2: is the model worse on Kenya?")
save(fig, "h2_worse_on_kenya")
plt.show()
table = [[int(kenya.y_wrong.sum()), int((1 - kenya.y_wrong).sum())],
         [int(trivia.y_wrong.sum()), int((1 - trivia.y_wrong).sum())]]
p_h2 = fisher_exact(table, alternative="greater").pvalue  # H2 predicts Kenya is worse
H2_VERDICT = ("SUPPORTED" if p_h2 < 0.05 else
              "BORDERLINE: as predicted, but not quite significant" if p_h2 < 0.1 else "not shown")
print(f"Kenya minus trivia: {ci_text(d, pct=True)}; one-sided Fisher p = {p_h2:.3f} -> "
      f"{H2_VERDICT}")
wd, p_tw = T["wrong_diff"], T["mcnemar_wrong"]["p_value"]
tw_verdict = "SUPPORTED" if p_tw < 0.05 else "same direction (only 33 pairs), not significant"
print(f"Twins, Kenya minus UK: {100 * wd['point']:.1f} points "
      f"(95% CI {100 * wd['lo']:.1f} to {100 * wd['hi']:.1f}), McNemar p = {p_tw:.3f} -> "
      f"{tw_verdict}")
''')

md("### Some of the pairs: the same question about Kenya and about the UK")

code('''
k = kenya.set_index("id")
pairs = twins.dropna(subset=["twin_of"]).copy()
pairs = pairs[pairs.twin_of.isin(k.index)]
show = pd.DataFrame({
    "Kenya question": k.loc[pairs.twin_of, "question"].values,
    "model said": k.loc[pairs.twin_of, "model_answer"].str[:30].values,
    "right?": np.where(k.loc[pairs.twin_of, "y_wrong"].values == 0, "yes", "no"),
    "UK twin": pairs.question.values, "model said ": pairs.model_answer.str[:30].values,
    "right? ": np.where(pairs.y_wrong.values == 0, "yes", "no")})
display(show.head(12))
''')

md("""
---
## H5. Does a detector trained on trivia still work on Kenya?

The detectors never saw a Kenya question. **OOD drop** = AUROC on trivia − AUROC on Kenya: close
to 0 means the detector travels.
""")

code('''
names = {"logreg": "Logistic regression", "xgboost": "XGBoost",
         "probe_linear": "Linear probe (served)", "probe_mlp": "MLP probe"}
rows = []
for det, label in names.items():
    s = M["detectors"][det]["sets"]
    t, ke = s["trivia_test"]["auroc_ci"], s["kenya"]["auroc_ci"]
    rows.append({"detector": label,
                 "trivia AUROC": f"{t['point']:.3f} [{t['lo']:.2f}-{t['hi']:.2f}]",
                 "Kenya AUROC": f"{ke['point']:.3f} [{ke['lo']:.2f}-{ke['hi']:.2f}]",
                 "OOD drop": round(M["detectors"][det]["ood_auroc_drop"], 3)})
display(pd.DataFrame(rows).set_index("detector"))

fig, ax = plt.subplots(figsize=(5, 4.6))
for d_, col, lab in [(trivia, C["trivia"], "Trivia"), (kenya, C["kenya"], "Kenya")]:
    fpr, tpr, _ = roc_curve(d_.y_wrong, d_.p_probe_linear)
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f"{lab} (AUROC {metrics.auroc(d_.y_wrong, d_.p_probe_linear):.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
ax.set_xlabel("right answers wrongly flagged")
ax.set_ylabel("wrong answers caught")
ax.legend(loc="lower right")
ax.set_title("H5: the served probe on trivia vs Kenya")
save(fig, "h5_detector_travels")
plt.show()
drop = M["detectors"]["probe_linear"]["ood_auroc_drop"]
print(f"Served probe: OOD drop {drop:.3f} -> "
      f"{'SUPPORTED: it travels (small drop)' if drop < 0.1 else 'it drops'}")
''')

md("""
---
## Fake questions: does it admit it doesn't know?

82 questions about things that **don't exist** in Kenya (e.g. a Kenyan "Class Z" driving
licence). The only right answer is to decline.
""")

code('''
normal, offered = fakes.open_abstained.mean(), fakes.abstained.mean()
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.barh(["Asked normally", "Offered 'I don't know'"], [normal, offered], color=C["kenya"])
ax.barh(["Asked normally", "Offered 'I don't know'"], [1 - normal, 1 - offered],
        left=[normal, offered], color=C["bad"])
for i, v in enumerate([normal, offered]):
    ax.text(0.01, i, f"declined {100 * v:.0f}%", va="center", color="white", fontweight="bold")
    ax.text(0.99, i, f"invented an answer {100 * (1 - v):.0f}%", va="center", ha="right",
            color="white", fontweight="bold")
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title(f"{len(fakes)} questions about things that don't exist")
save(fig, "fakes_invented_answers")
plt.show()
display(fakes[fakes.open_abstained == 0][["question", "model_answer"]].head(8)
        .rename(columns={"model_answer": "the model's invented answer"}))
''')

md("""
---
## H4. Retrieval (RAG): do Kenyan documents help, or create false confidence?

M1 is the same model given the 3 most relevant passages from a Kenyan document collection (the
Constitution and 63 Wikipedia articles) before each question. On **trivia** those passages are
irrelevant (the "useless material" test); on **Kenya** questions they can contain the answer.
""")

code('''
if m1 is None:
    print("M1 results not downloaded yet.")
else:
    both = m0.merge(m1[["qid", "y_wrong", "abstained", "open_abstained"]], on="qid",
                    suffixes=("", "_m1"))
    rows = []
    for label, mask in [("Trivia (irrelevant documents)",
                         (both.pool == "trivia") & (both.split == "test")),
                        ("Kenya (relevant documents)", both.pool == "kenya")]:
        d_ = both[mask]
        if len(d_) < 30:
            rows.append({"set": label, "n": len(d_), "note": "pending: run run_m1_rag.ipynb"})
            continue
        w = metrics.mcnemar(d_.y_wrong.values, d_.y_wrong_m1.values)
        a = metrics.mcnemar(d_.abstained.values, d_.abstained_m1.values)
        rows.append({"set": label, "n": len(d_),
                     "wrong M0": f"{100 * d_.y_wrong.mean():.0f}%",
                     "wrong M1": f"{100 * d_.y_wrong_m1.mean():.0f}%",
                     "wrong: p": round(w["p_value"], 4),
                     "'I don't know' M0": f"{100 * d_.abstained.mean():.0f}%",
                     "'I don't know' M1": f"{100 * d_.abstained_m1.mean():.0f}%",
                     "abstain: p": float(f"{a['p_value']:.2g}")})
    display(pd.DataFrame(rows).set_index("set"))
''')

md("""
**Reading it.** On trivia, irrelevant passages made the model take the "I don't know" option far
*more* often (cautious, not falsely confident), with no significant change in accuracy. The Kenya
row fills in once M1 has answered the Kenya questions (`notebooks/run_m1_rag.ipynb`, then
re-download and Run All).
""")

md("""
---
## 7. The same results without the AI-written or AI-edited items

Because AI-written test items could bias a test of an AI model, the key Kenya numbers again,
using only questions written **and not edited** by the students and Trevor. (The UK twins were
all AI-written, so the twin comparison has no human-only version.)
""")

code('''
human = kenya[~kenya.author.eq("Claude (AI)") & ~kenya.reviewer.eq("Claude (AI)")]
rows = []
for label, d_ in [("all Kenya questions", kenya), ("human-written, unedited only", human)]:
    a = d_.abstained.values
    rows.append({"set": label, "n": len(d_), "wrong rate": f"{100 * d_.y_wrong.mean():.1f}%",
                 "probe AUROC (H5)": round(metrics.auroc(d_.y_wrong, d_.p_probe_linear), 3),
                 "gap (H1)": round(metrics.verbalisation_gap(
                     d_.y_wrong.values, d_.p_probe_linear.values, a)["gap"], 3)})
display(pd.DataFrame(rows).set_index("set"))
hf = fakes[~fakes.author.eq("Claude (AI)")]
print(f"Fakes, human-written only (n={len(hf)}): declined when asked normally "
      f"{100 * hf.open_abstained.mean():.0f}%, "
      f"when offered the option {100 * hf.abstained.mean():.0f}%")
''')

md("""
---
## 8. What the live guard would do on the Kenya questions

The guard withholds an answer when the probe's P(wrong) is above 0.25 (a confident wrong answer
counts as 3× worse than a needless refusal).
""")

code('''
t_abstain = M["t_abstain"]
blocked = kenya.p_probe_linear > t_abstain
y = kenya.y_wrong.astype(bool)
display(pd.Series({
    "Kenya questions": len(kenya),
    "wrong if it answers everything": f"{100 * y.mean():.0f}%",
    "blocked by the guard": f"{100 * blocked.mean():.0f}%",
    "wrong among the answers it lets through": f"{100 * y[~blocked].mean():.0f}%",
    "wrong answers caught": f"{100 * (blocked & y).sum() / y.sum():.0f}%",
    "right answers lost": f"{100 * (blocked & ~y).sum() / (~y).sum():.0f}%",
}).to_frame("value"))
''')

md("---\n## Summary")

code('''
g_t, g_k = A["trivia_test"]["verbalisation_gap"], A["kenya"]["verbalisation_gap"]
drop = M["detectors"]["probe_linear"]["ood_auroc_drop"]
rows = [
    ("H1 Silent knowing",
     f"gap trivia {g_t['point']:.2f}, Kenya {g_k['point']:.2f} (both CIs above 0)",
     "Supported" if g_t["lo"] > 0 and g_k["lo"] > 0 else "Partly"),
    ("H2 Worse on Kenya", f"wrong {100 * kenya.y_wrong.mean():.0f}% vs trivia "
     f"{100 * trivia.y_wrong.mean():.0f}%; twins {100 * T['wrong_rate']['kenya']:.0f}% vs UK "
     f"{100 * T['wrong_rate']['uk']:.0f}% (p {T['mcnemar_wrong']['p_value']:.2f})",
     H2_VERDICT.split(":")[0].capitalize()),
    ("H5 Detector travels",
     f"probe AUROC {M['detectors']['probe_linear']['sets']['trivia_test']['auroc']:.2f}"
     f" -> {M['detectors']['probe_linear']['sets']['kenya']['auroc']:.2f} (drop {drop:.2f})",
     "Supported" if drop < 0.1 else "Not supported"),
    ("Fake questions",
     f"invents an answer {100 * (1 - fakes.open_abstained.mean()):.0f}% of the time asked "
     f"normally, {100 * (1 - fakes.abstained.mean()):.0f}% when offered 'I don't know'", "—"),
    ("H4 Retrieval", "trivia: irrelevant documents -> far more 'I don't know', accuracy unchanged;"
     " Kenya: see the H4 table", "Not supported (trivia)"),
    ("H3 Fine-tuning", "built, not run", "Future work"),
]
display(pd.DataFrame(rows, columns=["hypothesis", "result", "verdict"]).set_index("hypothesis"))
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "bluff-detector", "display_name": "bluff-detector (.venv)",
                             "language": "python"}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1])
