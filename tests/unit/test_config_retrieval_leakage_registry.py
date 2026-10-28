"""Unit tests: config inheritance and checks, BM25 retrieval, the leakage audit and the
registry's aliases and rollback.
"""

import pytest

from bluff.config import deep_merge, load_config
from bluff.leakage import find_overlaps
from bluff.registry import LocalRegistry
from bluff.retrieval import BM25Retriever
from tests.conftest import ROOT


def test_deep_merge_keeps_unrelated_keys():
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}


def test_variant_inherits_base_and_overrides():
    cfg = load_config(ROOT / "configs" / "variants" / "m1_rag.yaml")
    assert cfg["variant"]["id"] == "m1_rag"
    assert cfg["retrieval"]["enabled"] is True
    assert cfg["model"]["name"] == load_config(ROOT / "configs" / "variants" / "m0_base.yaml")[
        "model"]["name"]


def test_two_level_inheritance():
    cfg = load_config(ROOT / "configs" / "variants" / "dev_fake_m2.yaml")
    assert cfg["model"]["backend"] == "fake" and cfg["model"]["fake_abstain_scale"] == 0.3


def test_config_rejects_probe_layer_not_saved():
    with pytest.raises(ValueError):
        load_config(ROOT / "configs" / "variants" / "m0_base.yaml",
                    overrides={"detectors": {"probe_layer": 3}})


def test_bm25_ranks_relevant_chunk_first():
    r = BM25Retriever([
        {"id": "a", "text": "The Social Health Authority replaced the NHIF in 2024."},
        {"id": "b", "text": "Maize planting during the long rains."},
    ])
    assert r.search("Which body replaced NHIF?", k=1)[0].id == "a"


def test_leakage_finds_copied_question():
    qs = [{"id": "q1", "question": "Which public body replaced the NHIF as Kenya's insurer?"}]
    docs = [{"id": "d1", "text": "Quiz: which public body replaced the NHIF as Kenya's insurer?"},
            {"id": "d2", "text": "Unrelated text about counties."}]
    assert find_overlaps(qs, docs, n=6) == [("q1", "d1")]


def test_registry_aliases_and_rollback(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    (art / "detector.joblib").write_text("x")
    reg = LocalRegistry(tmp_path / "reg")
    v1 = reg.register("m", art, {"summary": {}})
    v2 = reg.register("m", art, {"summary": {}})
    reg.set_alias("m", "champion", v2)
    assert reg.get("m", "champion")["version"] == 2
    reg.set_alias("m", "champion", v1)  # rollback is a pointer move
    assert reg.get("m", "champion")["version"] == 1
    with pytest.raises(ValueError):
        reg.set_alias("m", "champion", 9)
