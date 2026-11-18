"""Make one runner notebook per model version from notebooks/colab_run.ipynb (the template).

    .venv\\Scripts\\python scripts\\make_run_notebooks.py

writes notebooks/run_m0_base.ipynb, run_m1_rag.ipynb, run_m2_qlora.ipynb, run_m3_rag_qlora.ipynb.
Each is the template with its version fixed (nothing to edit: open it, connect to a Colab T4,
Run All). M2 and M3 also install the fine-tuning libraries and get a fine-tuning step before
the questions are asked: M2 trains the QLoRA adapter once and saves it to Hugging Face; every
later M2 or M3 run reuses that saved adapter, so a resumed run never mixes two models.

Never edit the run_*.ipynb files by hand: fix colab_run.ipynb, then run this script again.
"""

from __future__ import annotations

import copy
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "colab_run.ipynb"

VERSIONS = {
    "m0_base": ("M0, the base model", "the model exactly as downloaded",
                "H1, H2, H5", "~1 hour (new questions only, after a first run: minutes)"),
    "m1_rag": ("M1, RAG", "the base model, with the 3 most relevant passages from the Kenyan "
               "document collection pasted into every prompt", "H4 (vs M0)", "~1–1.5 hours"),
    "m2_qlora": ("M2, QLoRA fine-tuned", "the base model plus a small adapter trained on the "
                 "Kenyan fine-tuning documents", "H3 (vs M0)",
                 "fine-tuning a few minutes + ~1 hour of questions"),
    "m3_rag_qlora": ("M3, QLoRA + RAG", "M2's adapter plus M1's retrieval",
                     "how RAG and fine-tuning combine (cut from the 5-day plan)", "~1–1.5 hours"),
}

FINETUNE_MD = """## 4b. The fine-tuned adapter (M2 and M3)

M2 trains its QLoRA adapter here **once** and saves it to Hugging Face. Every later run (a
resume after a disconnect, Wednesday's new questions, or M3) downloads and reuses that same
adapter instead of training again, so all of a version's answers come from one model."""

FINETUNE_CODE = '''ADAPTER = "artefacts/adapters/m2_qlora"
snapshot_download(HF_REPO, repo_type="dataset", token=HF_TOKEN, local_dir=".",
                  allow_patterns=[f"{ADAPTER}/*"])
if os.path.exists(f"{ADAPTER}/adapter_config.json"):
    print("Reusing the saved adapter:", ADAPTER)
else:
    assert VARIANT == "m2_qlora", "No adapter yet: run run_m2_qlora.ipynb first"
    for stage in ("validate", "ingest", "audit", "finetune_qlora"):  # audit before training
        r = subprocess.run(["bluff", stage, "--config", CONFIG], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-6000:])
            raise RuntimeError(f"{stage} failed: the reason is printed above")
        print(stage, "done")
    save([ADAPTER])
    print("Adapter trained and saved to Hugging Face:", ADAPTER)'''


def title_cell(variant: str) -> str:
    name, what, tests, time = VERSIONS[variant]
    return f"""# Bluff Detector: run {name}

Runs **{name}** ({what}) on a free Colab T4 GPU and saves everything to your private Hugging
Face dataset repo, so nothing is lost when the Colab session ends. Tests: **{tests}**.
Time: {time}.

**Nothing to edit.** Top right: **Select Kernel → Colab → New Colab Server → GPU → T4**, then
**Run All**. Paste the GitHub token, then the Hugging Face token, when asked.

**If the session drops:** Ctrl+Shift+P → Reload Window → reconnect → **Run All** again. It picks
up where it stopped and only asks the questions not yet answered.

*Made by `scripts/make_run_notebooks.py` from `colab_run.ipynb`: don't edit this file by hand.*"""


def build(variant: str) -> nbformat.NotebookNode:
    nb = copy.deepcopy(nbformat.read(TEMPLATE, as_version=4))
    cells = nb.cells
    cells[0].source = title_cell(variant)
    config_cell = next(c for c in cells if c.cell_type == "code" and "CONFIG =" in c.source)
    config_cell.source = (
        f'CONFIG = "configs/variants/{variant}.yaml"  # fixed: this notebook only runs {variant}\n'
        'HF_REPO = "vuhyeahhh/bluff-detector-runs"  # your private Hugging Face dataset repo'
    )
    if variant in ("m2_qlora", "m3_rag_qlora"):
        install = next(c for c in cells if c.cell_type == "code" and "pip install" in c.source)
        install.source = install.source.replace("'.[llm,data]'", "'.[llm,data,qlora]'")
        restore = next(i for i, c in enumerate(cells)
                       if c.cell_type == "code" and "snapshot_download(" in c.source)
        cells[restore + 1:restore + 1] = [nbformat.v4.new_markdown_cell(FINETUNE_MD),
                                         nbformat.v4.new_code_cell(FINETUNE_CODE)]
    for c in cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    return nb


if __name__ == "__main__":
    for variant in VERSIONS:
        out = ROOT / "notebooks" / f"run_{variant}.ipynb"
        nbformat.write(build(variant), out)
        print("wrote", out.relative_to(ROOT))
