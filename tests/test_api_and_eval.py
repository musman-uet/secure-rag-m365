from fastapi.testclient import TestClient

from securerag import api
from securerag.config import Settings


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "local")
    monkeypatch.setenv("RAG_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    for f in (api.get_settings, api.get_pipeline, api.get_directory):
        f.cache_clear()
    return TestClient(api.app)


def test_health(tmp_path, monkeypatch):
    assert _client(tmp_path, monkeypatch).get("/health").json() == {"status": "ok"}


def test_ask_requires_identity(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/ask", json={"question": "annual leave?"}).status_code == 401
    assert c.post("/ask", json={"question": "annual leave?"}, headers={"X-User-Id": "mallory"}).status_code == 401


def test_ask_returns_labelled_cited_answer(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/ask", json={"question": "How long must passwords be?"}, headers={"X-User-Id": "zain"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered" and body["ai_generated"] is True
    assert body["citations"][0]["title"] == "IT Security Policy"
    assert (tmp_path / "audit.jsonl").exists()


def test_ask_validates_input(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/ask", json={"question": ""}, headers={"X-User-Id": "zain"}).status_code == 422


def test_evaluation_runs_with_zero_leaks(tmp_path, monkeypatch):
    from eval.run_eval import run

    monkeypatch.setenv("RAG_BACKEND", "local")
    assert run(Settings.from_env(), k=5, out=None) == 0
