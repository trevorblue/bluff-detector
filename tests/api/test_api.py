"""API tests: TestClient runs the lifespan (model loads), validation and the endpoint."""

import json

import pytest
from fastapi.testclient import TestClient

from bluff.config import paths
from tests.conftest import DEV_FAKE


@pytest.fixture()
def client(trained, monkeypatch, tmp_path):
    monkeypatch.setenv("BLUFF_CONFIG", str(DEV_FAKE))
    monkeypatch.setenv("BLUFF_ARTEFACT_DIR", str(paths(trained).serving))
    monkeypatch.setenv("BLUFF_REQUEST_LOG", str(tmp_path / "requests.jsonl"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"ready": True}


def test_metadata_carries_provenance(client):
    meta = client.get("/metadata").json()
    assert meta["detector"] == "probe_linear"
    assert "git_sha" in meta and "data_sha256" in meta


def test_ask_contract(client):
    r = client.post("/v1/ask", json={"question": "How many counties does Kenya have?"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in {"answer", "answer_with_citation", "abstain"}
    assert 0.0 <= body["bluff_score"] <= 1.0
    assert (body["answer"] is None) == (body["verdict"] == "abstain")
    assert body["threshold"] == 0.25
    assert set(body["all_detectors"]) == {"logreg", "xgboost", "probe_linear", "probe_mlp"}
    assert body["all_detectors"]["probe_linear"] == pytest.approx(body["bluff_score"])
    assert "min_logprob" in body["signals"]
    if body["verdict"] != "abstain":
        assert body["tokens"] and all(0 <= t["prob"] <= 1 for t in body["tokens"])
    else:
        assert body["tokens"] == []  # word-level detail would reveal the blocked answer


def test_model_card(client):
    card = client.get("/v1/model-card").json()
    assert card["detector"] == "probe_linear" and card["t_abstain"] == 0.25
    assert card["test"]["probe_linear"]["trivia_test"]["auroc"] is not None
    assert "silent_knowing_rate" in card["behaviour"]["trivia_test"]
    ref = card["signal_reference"]["min_logprob"]
    assert set(ref) == {"right", "wrong"}


def test_withheld_answer_only_on_request(client, monkeypatch):
    from app import main

    monkeypatch.setitem(main.STATE["meta"], "t_abstain", -1.0)  # force the gate to abstain
    q = {"question": "How many counties does Kenya have?"}
    hidden = client.post("/v1/ask", json=q).json()
    shown = client.post("/v1/ask", json={**q, "reveal": True}).json()
    assert hidden["verdict"] == shown["verdict"] == "abstain"
    assert hidden["answer"] is None and hidden["withheld_answer"] is None
    assert shown["answer"] is None and shown["withheld_answer"]


def test_every_request_is_logged_for_monitoring(client, tmp_path):
    for q in ["How many counties does Kenya have?", "What is the capital of Kenya?"]:
        client.post("/v1/ask", json={"question": q})
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [r["question"] for r in rows][-2:] == ["How many counties does Kenya have?",
                                                  "What is the capital of Kenya?"]
    assert {"ts", "verdict", "bluff_score", "latency_ms", "features"} <= set(rows[-1])


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/v1/ask" in r.text


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "ok?", "extra": 1}, {}])
def test_rejects_bad_input(client, payload):
    assert client.post("/v1/ask", json=payload).status_code == 422


def test_not_ready_without_artefact(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUFF_CONFIG", str(DEV_FAKE))
    monkeypatch.setenv("BLUFF_ARTEFACT_DIR", str(tmp_path / "missing"))
    from app.main import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/v1/ask", json={"question": "Any question?"}).status_code == 503


def test_results_served_from_the_saved_summary(client, monkeypatch, tmp_path):
    summary = {"h1": {"verdict": "supported"}, "variants": [{"id": "m0_base", "ran": True}]}
    path = tmp_path / "results_summary.json"
    path.write_text(json.dumps(summary))
    monkeypatch.setenv("BLUFF_RESULTS", str(path))
    assert client.get("/v1/results").json() == summary
    monkeypatch.setenv("BLUFF_RESULTS", str(tmp_path / "missing.json"))
    assert client.get("/v1/results").status_code == 404


def test_committed_results_summary_has_every_hypothesis():
    from pathlib import Path

    summary = json.loads(Path("reports/results_summary.json").read_text())
    assert {"h1", "h2", "h3", "h4", "h5", "fakes", "guard", "variants"} <= set(summary)
    assert [v["id"] for v in summary["variants"]][:3] == ["m0_base", "m1_rag", "m2_qlora"]


def test_model_card_says_what_is_running(client):
    card = client.get("/v1/model-card").json()
    assert card["backend"] == "fake" and card["retrieval"] is False and card["adapter"] is None
    assert card["description"]


def test_passages_empty_without_retrieval(client):
    body = client.post("/v1/ask", json={"question": "What is the capital of Kenya?"}).json()
    assert body["passages"] == [] and body["sources"] == []


def test_variants_lists_the_default(client):
    vs = client.get("/v1/variants").json()
    assert [v["id"] for v in vs] == ["dev_fake"] and vs[0]["default"] is True
    assert set(vs[0]["detectors"]) == {"logreg", "xgboost", "probe_linear", "probe_mlp"}
    assert client.get("/v1/model-card", params={"variant": "dev_fake"}).json()["variant"]


def test_any_detector_can_decide(client):
    q = {"question": "What is the capital of Kenya?"}
    for name in ["logreg", "xgboost", "probe_mlp"]:
        body = client.post("/v1/ask", json={**q, "detector": name}).json()
        assert body["detector"] == name
        assert body["bluff_score"] == pytest.approx(body["all_detectors"][name])
        assert (body["verdict"] == "abstain") == (body["bluff_score"] > body["threshold"])


def test_unknown_variant_or_detector_is_refused(client):
    q = {"question": "What is the capital of Kenya?"}
    assert client.post("/v1/ask", json={**q, "variant": "m9_none"}).status_code == 400
    assert client.post("/v1/ask", json={**q, "detector": "oracle"}).status_code == 422
    assert client.get("/v1/model-card", params={"variant": "m9_none"}).status_code == 400
