"""API tests: TestClient runs the lifespan (model loads), validation and the endpoint."""

import pytest
from fastapi.testclient import TestClient

from bluff.config import paths
from tests.conftest import DEV_FAKE


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("BLUFF_CONFIG", str(DEV_FAKE))
    monkeypatch.setenv("BLUFF_ARTEFACT_DIR", str(paths(trained).serving))
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
