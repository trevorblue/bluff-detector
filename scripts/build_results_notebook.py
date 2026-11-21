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
| H3 | Fine-tuning | Does extra training make it stop saying "I don't know" while still knowing? |
""")

code('''
import json
import os
import re
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
def in_passages(variant):
    \"\"\"qid -> was the correct answer (or an accepted one) in the passages M1 was given?\"\"\"
    norm = lambda t: " " + " ".join(re.findall(r"\w+", str(t).lower())) + " "  # noqa: E731
    chunks = {c["id"]: c["text"] for c in read_jsonl("data/corpus/retrieval_chunks.jsonl")}
    gens = {g["qid"]: g for g in read_jsonl(f"data/interim/{variant}/generations.jsonl")
            if g["condition"] == "open"}
    qs = {x["id"]: x for x in read_jsonl("data/processed/questions.jsonl")}
    out = {}
    for qid, g in gens.items():
        text = norm(" ".join(chunks.get(i, "") for i in g["retrieved_ids"]))
        answers = [a for a in [qs[qid].get("gold"), *qs[qid]["aliases"]] if a and len(a) > 1]
        out[qid] = any(norm(a) in text for a in answers)
    return out


if m1 is None:
    print("M1 results not downloaded yet.")
else:
    both = m0.merge(m1[["qid", "y_wrong", "abstained"]], on="qid", suffixes=("", "_m1"))
    covered = in_passages("m1_rag")
    both["answer in passages"] = both.qid.map(covered)
    ken = both.pool == "kenya"
    groups = [("Trivia (documents irrelevant)", (both.pool == "trivia") & (both.split == "test")),
              ("Kenya, all", ken),
              ("Kenya, answer IN the 3 passages", ken & both["answer in passages"]),
              ("Kenya, answer NOT in the passages", ken & ~both["answer in passages"])]
    rows = []
    for label, mask in groups:
        d_ = both[mask]
        w = metrics.mcnemar(d_.y_wrong.values, d_.y_wrong_m1.values)
        a = metrics.mcnemar(d_.abstained.values, d_.abstained_m1.values)
        rows.append({"set": label, "n": len(d_),
                     "wrong M0": f"{100 * d_.y_wrong.mean():.0f}%",
                     "wrong M1": f"{100 * d_.y_wrong_m1.mean():.0f}%",
                     "wrong: p": float(f"{w['p_value']:.2g}"),
                     "'I don't know' M0": f"{100 * d_.abstained.mean():.0f}%",
                     "'I don't know' M1": f"{100 * d_.abstained_m1.mean():.0f}%",
                     "abstain: p": float(f"{a['p_value']:.2g}")})
    H4 = pd.DataFrame(rows).set_index("set")
    display(H4)
    print(f"Kenya questions whose answer was in M1's passages: "
          f"{int(both[ken]['answer in passages'].sum())} of {int(ken.sum())}")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = ["Kenya: answer\\nIN the passages", "Kenya: answer\\nNOT in the passages",
              "Trivia\\n(documents irrelevant)"]
    pick = [groups[2][1], groups[3][1], groups[0][1]]
    x = np.arange(3)
    ax.bar(x - 0.18, [both[m].y_wrong.mean() for m in pick], 0.36, color=C["trivia"],
           label="M0: no documents")
    ax.bar(x + 0.18, [both[m].y_wrong_m1.mean() for m in pick], 0.36, color=C["kenya"],
           label="M1: with 3 passages")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("wrong rate")
    ax.legend()
    ax.set_title("H4: retrieval halves errors when the answer is in the passages")
    save(fig, "h4_retrieval")
    plt.show()
''')

md("""
**Reading it.** When the answer **is** in the passages, retrieval roughly **halves the errors**.
When it **isn't** (on Kenya questions the collection doesn't cover, and on trivia), the model
does **not** become falsely confident: it takes the "I don't know" option as often or more often,
and its accuracy doesn't change significantly. So H4's claim (false confidence when the documents
don't contain the answer) is not supported. About half the Kenya questions are covered by the
document collection, a limitation of its size (the Constitution plus 63 Wikipedia articles).
""")

md("""
---
## H3. Does fine-tuning break "saying" but not "knowing"?

M2 is the same model with a small **QLoRA adapter** trained on 26 Kenyan documents (Acts, Kenya
articles; none of them in M1's collection). H3 predicts that after fine-tuning the model **says
"I don't know" less** (expressed detection falls) while its **inside still registers its
uncertainty** (internal detection holds), so the gap between knowing and saying grows.
Compared pair by pair with M0 on the same questions (`bluff compare`).
""")

code('''
cmp2 = Path("reports/compare/m0_base__vs__m2_qlora.json")
if not cmp2.exists():
    print("M2 results not available yet.")
else:
    C2 = json.loads(cmp2.read_text())["sets"]
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    fmt = lambda d: f"{d['point']:+.3f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"  # noqa: E731
    rows = []
    for key, label in [("trivia_test", "Trivia"), ("kenya", "Kenya")]:
        v = C2[key]
        rows.append({"set": label, "n": v["n"],
                     "'I don't know' M0": pct(v["abstain_rate"]["a"]),
                     "'I don't know' M2": pct(v["abstain_rate"]["b"]),
                     "abstain: p": float(f"{v['mcnemar_abstain']['p_value']:.2g}"),
                     "wrong M0": pct(v["wrong_rate"]["a"]), "wrong M2": pct(v["wrong_rate"]["b"]),
                     "change in internal (knowing)": fmt(v["delta_internal"]),
                     "change in expressed (saying)": fmt(v["delta_expressed"]),
                     "change in gap": fmt(v["delta_gap"])})
    display(pd.DataFrame(rows).set_index("set"))

    M2 = json.loads(Path("reports/m2_qlora/metrics.json").read_text())["abstention"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(2)
    a1.bar(x - 0.18, [C2[k]["abstain_rate"]["a"] for k in ("trivia_test", "kenya")], 0.36,
           color=C["trivia"], label="M0: base")
    a1.bar(x + 0.18, [C2[k]["abstain_rate"]["b"] for k in ("trivia_test", "kenya")], 0.36,
           color=C["bad"], label="M2: fine-tuned")
    a1.set_xticks(x, ["Trivia", "Kenya"])
    a1.set_ylim(0, 0.5)
    a1.set_title("Says 'I don't know' (offered the option)")
    a1.legend()
    labels = ["internal (knowing)", "expressed (saying)"]
    keys = ["internal_detection", "expressed_detection"]
    a2.bar(x - 0.18, [A["kenya"][k]["point"] for k in keys], 0.36, color=C["trivia"], label="M0")
    a2.bar(x + 0.18, [M2["kenya"][k]["point"] for k in keys], 0.36, color=C["bad"], label="M2")
    a2.axhline(0.5, ls="--", lw=0.8, color="grey")
    a2.set_xticks(x, labels)
    a2.set_ylim(0.4, 0.85)
    a2.set_title("Kenya: AUROC of knowing vs saying")
    a2.legend()
    fig.suptitle("H3: fine-tuning halves 'I don't know'; the inside still knows")
    save(fig, "h3_finetuning")
    plt.show()

    log_path = Path("reports/m2_qlora/training_log.json")
    if log_path.exists():
        t = json.loads(log_path.read_text())
        print(f"Fine-tuning: {t['documents']} documents, {t['windows']} windows of 1,024 tokens, "
              f"{t['epochs']} epochs, {t['steps']} steps; loss {t['loss_per_step'][0]:.2f} -> "
              f"{t['loss_per_step'][-1]:.2f}")
''')

md("""
**Reading it.** After fine-tuning, the model says "I don't know" **about half as often** (Kenya
28% → 13%, trivia 35% → 20%; McNemar p < 10⁻⁸ after Holm correction), yet it is **not more often
right** (no significant change in the wrong rate). On Kenya, what its **words** reveal about its
errors falls significantly (expressed detection −0.07), while what its **inside** reveals does not
change significantly (internal detection −0.03, CI includes 0): **saying broke, knowing held**,
as H3 predicts. The gap grows in the predicted direction (+0.04 on Kenya, +0.05 on trivia), but
those increases are not significant at 95%. The fine-tuning was light (42 steps), which makes
the drop in abstention more striking, not less.
""")

md("""
---
## M3. Retrieval and fine-tuning together

M3 is M2's fine-tuned model **and** M1's three retrieved passages. Two questions: does retrieval
still help a fine-tuned model (M2 → M3), and does fine-tuning add anything once the documents
are there (M1 → M3)? Each pair is compared on the same questions (`bluff compare`, McNemar's
test with Holm correction, paired bootstrap on the AUROCs). The passages are identical to M1's
(the same retriever on the same questions), so the H4 split by "answer in the passages" carries
over.
""")

code('''
m2, m3 = load("m2_qlora"), load("m3_rag_qlora")
cmp_m2m3 = Path("reports/compare/m2_qlora__vs__m3_rag_qlora.json")
cmp_m1m3 = Path("reports/compare/m1_rag__vs__m3_rag_qlora.json")
if m3 is None or m1 is None or m2 is None or not cmp_m2m3.exists() or not cmp_m1m3.exists():
    print("M3 results (or a comparison) not available yet.")
else:
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    fmt = lambda d: f"{d['point']:+.3f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"  # noqa: E731
    VERS = [("m0_base", "M0 base"), ("m1_rag", "M1 + RAG"), ("m2_qlora", "M2 + QLoRA"),
            ("m3_rag_qlora", "M3 RAG + QLoRA")]
    rows = []
    for v, label in VERS:
        mv = json.loads(Path(f"reports/{v}/metrics.json").read_text())
        for key, name in [("kenya", "Kenya"), ("trivia_test", "Trivia")]:
            a, dets = mv["abstention"][key], mv["detectors"]
            rows.append({"version": label, "set": name,
                         "wrong": pct(dets["probe_linear"]["sets"][key]["wrong_rate"]),
                         "'I don't know'": pct(a["abstain_rate"]),
                         "words AUROC (logreg)": round(dets["logreg"]["sets"][key]["auroc"], 3),
                         "inside AUROC (linear probe)":
                             round(dets["probe_linear"]["sets"][key]["auroc"], 3),
                         "gap (H1)": round(a["verbalisation_gap"]["point"], 3)})
    display(pd.DataFrame(rows).set_index(["set", "version"])
            .sort_index(level=0, sort_remaining=False))

    rows = []
    for path, label in [(cmp_m2m3, "M2 -> M3 (add retrieval to the fine-tuned model)"),
                        (cmp_m1m3, "M1 -> M3 (add fine-tuning to the retrieval model)")]:
        cc = json.loads(path.read_text())["sets"]
        for key, name in [("kenya", "Kenya"), ("trivia_test", "Trivia")]:
            v = cc[key]
            rows.append({"change": label, "set": name,
                         "'I don't know'": f"{pct(v['abstain_rate']['a'])} -> "
                                           f"{pct(v['abstain_rate']['b'])}",
                         "abstain: p": float(f"{v['mcnemar_abstain']['p_value']:.2g}"),
                         "wrong": f"{pct(v['wrong_rate']['a'])} -> {pct(v['wrong_rate']['b'])}",
                         "change in internal (knowing)": fmt(v["delta_internal"]),
                         "change in expressed (saying)": fmt(v["delta_expressed"]),
                         "change in gap": fmt(v["delta_gap"])})
    display(pd.DataFrame(rows).set_index(["change", "set"]))

    # the H4 split again, now with the fine-tuned model: was the answer in the passages?
    four = m0[["qid", "pool", "split", "y_wrong", "abstained"]]
    for v, s in [(m1, "m1"), (m2, "m2"), (m3, "m3")]:
        four = four.merge(v[["qid", "y_wrong", "abstained"]].rename(
            columns={"y_wrong": f"wrong_{s}", "abstained": f"abstain_{s}"}), on="qid")
    four = four.rename(columns={"y_wrong": "wrong_m0", "abstained": "abstain_m0"})
    four["answer in passages"] = four.qid.map(in_passages("m3_rag_qlora"))
    ken = four.pool == "kenya"
    groups3 = [("Trivia (documents irrelevant)", (four.pool == "trivia") & (four.split == "test")),
               ("Kenya, all", ken),
               ("Kenya, answer IN the 3 passages", ken & four["answer in passages"]),
               ("Kenya, answer NOT in the passages", ken & ~four["answer in passages"])]
    rows = []
    for label, mask in groups3:
        d_ = four[mask]
        row = {"set": label, "n": len(d_)}
        row.update({f"wrong {s.upper()}": pct(d_[f"wrong_{s}"].mean())
                    for s in ("m0", "m1", "m2", "m3")})
        row["M2 -> M3 wrong: p"] = float(
            f"{metrics.mcnemar(d_.wrong_m2.values, d_.wrong_m3.values)['p_value']:.2g}")
        row.update({f"'I don't know' {s.upper()}": pct(d_[f"abstain_{s}"].mean())
                    for s in ("m0", "m1", "m2", "m3")})
        rows.append(row)
    M3T = pd.DataFrame(rows).set_index("set")
    display(M3T)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    x, w = np.arange(4), 0.2
    cols = [C["trivia"], C["kenya"], C["bad"], C["uk"]]
    names = ["M0 base", "M1 + RAG", "M2 + QLoRA", "M3 both"]
    pick = [groups3[2][1], groups3[3][1]]
    for i, s in enumerate(("m0", "m1", "m2", "m3")):
        a1.bar(np.arange(2) + (i - 1.5) * w, [four[m][f"wrong_{s}"].mean() for m in pick], w,
               color=cols[i], label=names[i])
        a2.bar(i, four[ken][f"abstain_{s}"].mean(), 0.6, color=cols[i])
    a1.set_xticks(np.arange(2), ["Kenya: answer\\nIN the passages",
                              "Kenya: answer\\nNOT in the passages"])
    a1.set_ylim(0, 1)
    a1.set_ylabel("wrong rate")
    a1.legend(fontsize=8, ncol=2)
    a1.set_title("Wrong answers")
    a2.set_xticks(x, names)
    a2.set_ylim(0, 0.5)
    a2.set_title("Kenya: says 'I don't know' (offered the option)")
    fig.suptitle("M3: retrieval still halves errors and restores 'I don't know' after fine-tuning",
                 y=1.04)
    save(fig, "m3_combined")
    plt.show()
''')

md("""
**Reading it.**

- **Retrieval still works on a fine-tuned model.** When the answer is in the passages, M3 is wrong
  on 28% of those Kenya questions against M2's 62%, the same halving H4 found for M0 → M1. When
  it isn't, nothing changes significantly.
- **Retrieval undoes the damage fine-tuning did to "saying".** On Kenya, M2 said "I don't know"
  13% of the time; with the passages (M3) it is back to 27%, about M0's 28% (p = 0.0001). What its
  words reveal about its errors recovers too (expressed detection +0.16, CI above 0).
- **Fine-tuning adds nothing once the documents are there.** M1 → M3 on Kenya: the same wrong rate
  (62% → 61%) and the same "I don't know" rate (27% → 27%). Its only clear effect is on trivia,
  where it lowers "I don't know" (88% → 63%), as it did in M2.
- **The inside reading travels worst here.** With relevant Kenyan passages in the prompt, the
  linear probe's Kenya AUROC falls to 0.64 (M0 0.73, M1 0.68), while the words detector holds at
  0.79, and the H1 gap on Kenya disappears (−0.06, CI includes 0). A likely reason: each probe was
  trained on trivia, where the passages are irrelevant, so it never saw what the hidden state looks
  like when the passages contain the answer. For a retrieval system, the detector should be trained
  on questions the documents actually cover.

M3 is therefore the most accurate version on Kenya (61% wrong against M0's 73%), but the one where
the inside reading is least reliable.
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
    ("H4 Retrieval",
     "answer in passages: errors halved; not in passages: no false confidence (see H4 table)",
     "Not supported (helps, no false confidence)"),
    ("H3 Fine-tuning", "'I don't know' halved (Kenya 28% -> 13%, p < 1e-8), accuracy unchanged; "
     "saying fell (expressed -0.07), knowing held (internal -0.03, n.s.)",
     "Supported (gap growth not significant)"),
    ("M3 Both together", "retrieval on the fine-tuned model halves errors when the answer is in "
     "the passages and restores 'I don't know' (Kenya 13% -> 27%); fine-tuning adds nothing on "
     "top of retrieval; the probe travels worst (Kenya AUROC 0.64)", "—"),
]
display(pd.DataFrame(rows, columns=["hypothesis", "result", "verdict"]).set_index("hypothesis"))
''')

md("""
### Save the headline numbers

The demo page's "What we found" tab (`GET /v1/results`) shows these numbers. They are written
here, from the variables above, so the page and this notebook can never disagree.
""")

code('''
def num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def ci(c):
    return {k: num(c[k]) for k in ("point", "lo", "hi")}


def variant_row(v, label):
    path = Path(f"reports/{v}/metrics.json")
    if not path.exists():
        return {"id": v, "label": label, "ran": False}
    mv = json.loads(path.read_text())
    out = {"id": v, "label": label, "ran": True}
    # every metric of every detector, and the model's behaviour on every question set
    sets = ["trivia_test", "kenya", "kenya_unanswerable", "western"]
    out["detectors"] = {
        det: {"ood_drop": num(dm.get("ood_auroc_drop")), **{
            st: {"n": x["n"], "wrong": num(x["wrong_rate"]), "auroc": num(x["auroc"]),
                 "auroc_lo": num(x["auroc_ci"]["lo"]), "auroc_hi": num(x["auroc_ci"]["hi"]),
                 "pr_auc": num(x["pr_auc"]), "ece": num(x["ece"]), "aurc": num(x["aurc"]),
                 "risk80": num(x["risk@0.8"])}
            for st, x in dm["sets"].items() if st in ("trivia_test", "kenya")}}
        for det, dm in mv["detectors"].items()}
    out["behaviour"] = {
        st: {"n": a["n"], "abstain": num(a["abstain_rate"]),
             "hallucination": num(a["hallucination_rate"]),
             "inflation": num(a["inflation_rate"]),
             "silent_knowing": num((a.get("silent_knowing_rate") or {}).get("point")),
             "internal": num((a.get("internal_detection") or {}).get("point")),
             "expressed": num((a.get("expressed_detection") or {}).get("point")),
             "gap": num((a.get("verbalisation_gap") or {}).get("point")),
             "cells": {k: c["n"] for k, c in a["cells"].items()}}
        for st, a in mv["abstention"].items() if st in sets}
    out["latency_ms"] = {k: num(x) for k, x in mv.get("generation_latency_ms", {}).items()}
    for key, name in [("trivia_test", "trivia"), ("kenya", "kenya")]:
        a, d = mv["abstention"][key], mv["detectors"]["probe_linear"]["sets"][key]
        out[name] = {"n": a["n"], "wrong": num(d["wrong_rate"]),
                     "abstain": num(a["abstain_rate"]), "probe_auroc": num(d["auroc"]),
                     "gap": num(a["verbalisation_gap"]["point"])}
    return out


summary = {
    "about": {"model": "Qwen2.5-1.5B-Instruct", "n_trivia": int(len(m0[m0.pool == "trivia"])),
              "n_trivia_test": int(len(trivia)), "n_kenya": int(len(kenya)),
              "n_fakes": int(len(fakes)), "n_twins": int(len(twins))},
    "variants": [variant_row("m0_base", "M0 base"), variant_row("m1_rag", "M1 + RAG"),
                 variant_row("m2_qlora", "M2 + QLoRA"),
                 variant_row("m3_rag_qlora", "M3 RAG + QLoRA")],
    "h1": {"verdict": "supported" if g_t["lo"] > 0 and g_k["lo"] > 0 else "partly",
           **{name: {"internal": ci(A[key]["internal_detection"]),
                     "expressed": ci(A[key]["expressed_detection"]),
                     "gap": ci(A[key]["verbalisation_gap"]),
                     "silent_knowing": ci(A[key]["silent_knowing_rate"])}
              for key, name in [("trivia_test", "trivia"), ("kenya", "kenya")]}},
    "h2": {"verdict": "supported" if p_h2 < 0.05 else "not supported",
           "wrong_kenya": num(kenya.y_wrong.mean()), "wrong_trivia": num(trivia.y_wrong.mean()),
           "fisher_p": num(p_h2), "twins_n": T["n_pairs"],
           "twins_wrong_kenya": num(T["wrong_rate"]["kenya"]),
           "twins_wrong_uk": num(T["wrong_rate"]["uk"]),
           "twins_p": num(T["mcnemar_wrong"]["p_value"])},
    "h5": {"verdict": "supported" if drop < 0.1 else "not supported",
           "auroc_trivia": num(M["detectors"]["probe_linear"]["sets"]["trivia_test"]["auroc"]),
           "auroc_kenya": num(M["detectors"]["probe_linear"]["sets"]["kenya"]["auroc"]),
           "drop": num(drop)},
    "fakes": {"n": int(len(fakes)), "invents_normal": num(1 - fakes.open_abstained.mean()),
              "invents_offered": num(1 - fakes.abstained.mean())},
    "guard": {"t_abstain": num(M["t_abstain"]), "wrong_all": num(y.mean()),
              "blocked": num(blocked.mean()), "wrong_let_through": num(y[~blocked].mean()),
              "caught": num((blocked & y).sum() / y.sum()),
              "right_lost": num((blocked & ~y).sum() / (~y).sum())},
}
if "both" in globals():
    h4 = []
    for label, mask in groups:
        d_ = both[mask]
        h4.append({"set": label, "n": int(len(d_)),
                   "wrong_m0": num(d_.y_wrong.mean()), "wrong_m1": num(d_.y_wrong_m1.mean()),
                   "wrong_p": num(metrics.mcnemar(d_.y_wrong.values,
                                                  d_.y_wrong_m1.values)["p_value"]),
                   "abstain_m0": num(d_.abstained.mean()),
                   "abstain_m1": num(d_.abstained_m1.mean())})
    summary["h4"] = {"verdict": "not supported", "groups": h4}
if cmp2.exists():
    summary["h3"] = {"verdict": "supported", **{
        name: {"abstain_m0": num(C2[key]["abstain_rate"]["a"]),
               "abstain_m2": num(C2[key]["abstain_rate"]["b"]),
               "abstain_p": num(C2[key]["mcnemar_abstain"]["p_value"]),
               "wrong_m0": num(C2[key]["wrong_rate"]["a"]),
               "wrong_m2": num(C2[key]["wrong_rate"]["b"]),
               "delta_internal": ci(C2[key]["delta_internal"]),
               "delta_expressed": ci(C2[key]["delta_expressed"]),
               "delta_gap": ci(C2[key]["delta_gap"])}
        for key, name in [("trivia_test", "trivia"), ("kenya", "kenya")]}}
if "M3T" in globals():
    def change(path):
        cc = json.loads(path.read_text())["sets"]
        return {name: {"abstain_a": num(cc[key]["abstain_rate"]["a"]),
                       "abstain_b": num(cc[key]["abstain_rate"]["b"]),
                       "abstain_p": num(cc[key]["mcnemar_abstain"]["p_value"]),
                       "wrong_a": num(cc[key]["wrong_rate"]["a"]),
                       "wrong_b": num(cc[key]["wrong_rate"]["b"]),
                       "delta_internal": ci(cc[key]["delta_internal"]),
                       "delta_expressed": ci(cc[key]["delta_expressed"]),
                       "delta_gap": ci(cc[key]["delta_gap"])}
                for key, name in [("trivia_test", "trivia"), ("kenya", "kenya")]}
    m3g = []
    for label, mask in groups3:
        d_ = four[mask]
        m3g.append({"set": label, "n": int(len(d_)),
                    **{f"wrong_{s}": num(d_[f"wrong_{s}"].mean())
                       for s in ("m0", "m1", "m2", "m3")},
                    **{f"abstain_{s}": num(d_[f"abstain_{s}"].mean())
                       for s in ("m0", "m1", "m2", "m3")},
                    "wrong_p_m2_m3": num(metrics.mcnemar(d_.wrong_m2.values,
                                                         d_.wrong_m3.values)["p_value"])})
    summary["m3"] = {"m2_to_m3": change(cmp_m2m3), "m1_to_m3": change(cmp_m1m3), "groups": m3g}
Path("reports/results_summary.json").write_text(json.dumps(summary, indent=1))
print("wrote reports/results_summary.json with", ", ".join(summary))
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "bluff-detector", "display_name": "bluff-detector (.venv)",
                             "language": "python"}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1])
