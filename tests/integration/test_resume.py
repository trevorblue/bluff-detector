"""Resume: a run cut off part-way and finished later must equal one uninterrupted run."""

import numpy as np
import pytest

from bluff.config import paths
from bluff.pipeline import generate, ingest
from bluff.utils import read_jsonl, write_json
from tests.conftest import fake_config

pytestmark = pytest.mark.integration


def _outputs(cfg: dict) -> dict:
    p = paths(cfg)
    hidden = np.load(p.interim / "hidden.npz")["hidden"]
    out = {}
    for g in read_jsonl(p.interim / "generations.jsonl"):
        key = (g["qid"], g["condition"], g["template"])
        out[key] = (g["answer"], g["token_probs"],
                    None if g["hidden_index"] is None else hidden[g["hidden_index"]].tolist())
    return out


def test_resumed_run_equals_one_full_run(tmp_path):
    whole = fake_config(tmp_path / "whole")
    ingest.run(whole)
    generate.run(whole)

    split = fake_config(tmp_path / "split", generation={"resume": True, "limit": 50})
    ingest.run(split)
    first = generate.run(split)
    assert first["new"] == 50
    split["generation"]["limit"] = None  # e.g. new questions arrive, or the session restarts
    second = generate.run(split)
    assert second["new"] == second["questions"] - 50

    assert _outputs(split) == _outputs(whole)


def test_resume_survives_files_from_different_saves(tmp_path):
    """An upload caught a save half-way: the answers file is newer than the hidden-state file.
    Questions past the end of the older file are asked again, and the result is unchanged."""
    whole = fake_config(tmp_path / "whole")
    ingest.run(whole)
    generate.run(whole)

    cfg = fake_config(tmp_path / "split", generation={"resume": True, "limit": 40})
    ingest.run(cfg)
    generate.run(cfg)
    p = paths(cfg)
    with np.load(p.interim / "hidden.npz") as f:
        older = {k: f[k] for k in ("hidden", "layers", "positions")}
    np.savez_compressed(p.interim / "hidden.npz", hidden=older["hidden"][:30],
                        layers=older["layers"], positions=older["positions"])
    cfg["generation"]["limit"] = None
    second = generate.run(cfg)
    assert second["new"] == second["questions"] - 30
    assert _outputs(cfg) == _outputs(whole)


def test_resume_drops_answers_to_removed_questions(tmp_path):
    """A source taken out of the question set (e.g. TruthfulQA) leaves no stale answers behind,
    and nothing is asked again."""
    cfg = fake_config(tmp_path, generation={"resume": True})
    ingest.run(cfg)
    generate.run(cfg)
    cfg["generation"]["limit"] = 30
    again = generate.run(cfg)
    assert again["new"] == 0
    p = paths(cfg)
    assert len({g["qid"] for g in read_jsonl(p.interim / "generations.jsonl")}) == 30
    out = _outputs(cfg)
    assert all(h is None or len(h) > 0 for _, _, h in out.values())


def test_resume_refuses_a_different_machine(tmp_path):
    cfg = fake_config(tmp_path, generation={"resume": True, "limit": 20})
    ingest.run(cfg)
    generate.run(cfg)
    write_json(paths(cfg).interim / "run_info.json", {"model": "fake-llm", "device": "Tesla P100"})
    cfg["generation"]["limit"] = None
    with pytest.raises(ValueError, match="cannot resume"):
        generate.run(cfg)
