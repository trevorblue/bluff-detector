"""End to end on the fake backend. Checks the plumbing, not science: the fake model is
built so that hidden states carry signal and Kenya questions are known less often."""

import pytest

from bluff.config import paths
from bluff.pipeline import compare
from bluff.utils import read_json, read_jsonl
from tests.conftest import fake_config, run_all

pytestmark = pytest.mark.integration


def test_pipeline_outputs_and_splits(trained):
    p = paths(trained)
    questions = read_jsonl(p.questions)
    for q in questions:
        if q["pool"].startswith("kenya"):
            assert q["split"] == "test", "Kenya pools must never reach training"
    m = read_json(p.reports / "metrics.json")
    probe = m["detectors"]["probe_linear"]["sets"]
    assert probe["trivia_test"]["auroc"] > 0.7
    assert {"trivia_test", "kenya", "kenya_unanswerable"} <= set(probe)
    for set_name, a in m["abstention"].items():
        assert set(a["cells"]) == {"honest_uncertainty", "silent_knowing",
                                   "abstention_inflation", "earned_confidence"}
        assert sum(c["n"] for c in a["cells"].values()) == a["n"], set_name
    assert (p.serving / "detector.joblib").exists()
    meta = read_json(p.serving / "metadata.json")
    for key in ("git_sha", "data_sha256", "t_abstain", "tau_flag", "version", "detector"):
        assert key in meta


def test_first_registration_is_champion(trained):
    reg = read_json(paths(trained).registry / "registry.json")
    model = reg["models"]["bluff-detector-dev_fake"]
    assert model["aliases"]["champion"] == 1


def test_compare_detects_suppressed_abstention(trained, tmp_path_factory):
    # Same questions, a fake "fine-tuned" model that abstains less (H3 in miniature).
    base_dirs = trained["paths"]
    other = fake_config(tmp_path_factory.mktemp("m2"), "dev_fake_m2.yaml", paths=base_dirs)
    run_all(other)
    out = compare.run(trained, other)
    kenya = out["sets"]["kenya"]
    assert kenya["abstain_rate"]["b"] < kenya["abstain_rate"]["a"]
    assert kenya["delta_expressed"]["point"] < 0
    assert kenya["delta_internal"]["point"] == pytest.approx(0, abs=0.05)
