import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from conftest import completion
from rag_chat.api import create_app
from rag_chat.chat import Answer, Source
from rag_chat.workflows import DEFAULT_WORKFLOW, INITIAL_STATE_KEYS


@pytest.fixture
def api_client(manager, tokenizer, tmp_path):
    runtime = SimpleNamespace(manager=manager, tokenizer=lambda: tokenizer)
    model = MagicMock()
    model.__enter__.return_value = model
    model.chat.completions.create.side_effect = [
        completion(json.dumps({"searches": [{"query": "launch", "purpose": "Find the launch date"}]})),
        completion(json.dumps({"decision": "sufficient", "confidence": 0.95, "missing_evidence": []})),
        completion("The launch is in June. [1]"),
    ]
    app = create_app(lambda: runtime, lambda: model, workflow_db_path=tmp_path / "workflows.sqlite3")
    with TestClient(app) as client:
        yield client, manager, model


def setup_session(client):
    created = client.post("/api/sessions")
    assert created.status_code == 201
    headers = {"X-Session-ID": created.json()["id"]}
    response = client.post("/api/documents", headers=headers,
                           files={"file": ("launch.txt", b"The launch is in June.", "text/plain")})
    assert response.json()["status"] == "indexed"
    return headers


def parse_events(response):
    return [(block.splitlines()[0][7:], json.loads(block.splitlines()[1][6:]))
            for block in response.text.strip().split("\n\n") if block.startswith("event:")]


def test_session_restore_chat_events_and_clear(api_client):
    client, manager, model = api_client
    headers = setup_session(client)
    response = client.post("/api/chat", headers=headers, json={"question": "When is launch?"})
    events = parse_events(response)
    assert response.status_code == 200
    assert events[0] == ("progress", {"event": "node_start", "node": "planner", "round": 1})
    assert events[-1][0] == "answer"
    assert any(event[1].get("event") == "search" for event in events)
    state = client.get("/api/session", headers=headers).json()
    assert state["messages"][1]["content"] == "The launch is in June. [1]"
    assert state["messages"][1]["sources"][0]["filename"] == "launch.txt"
    assert state["messages"][1]["trace"] == [item[1] for item in events[:-1]]
    assert state["operation"]["status"] == "complete"
    assert client.delete("/api/session", headers=headers).status_code == 204
    assert client.get("/api/session", headers=headers).status_code == 410


def test_poll_does_not_extend_idle_expiry(api_client):
    client, manager, _ = api_client
    now = [0.0]
    manager.clock = lambda: now[0]
    headers = setup_session(client)
    now[0] = 3599
    assert client.get("/api/session", headers=headers).status_code == 200
    now[0] = 3600
    assert client.get("/api/session", headers=headers).status_code == 410


def test_concurrent_operations_rejected(api_client):
    client, manager, _ = api_client
    headers = setup_session(client)
    library = manager.begin(headers["X-Session-ID"], "chat", "test")
    try:
        assert client.get("/api/session", headers=headers).json()["operation"]["status"] == "running"
        assert client.post("/api/chat", headers=headers, json={"question": "test"}).status_code == 409
        assert client.post("/api/documents", headers=headers, files={"file": ("test.txt", b"test")}).status_code == 409
        assert client.delete("/api/session", headers=headers).status_code == 409
    finally:
        manager.finish(library)


def test_upload_failures_limits_and_recovery(api_client, monkeypatch):
    client, manager, _ = api_client
    headers = setup_session(client)
    duplicate = client.post("/api/documents", headers=headers, files={"file": ("renamed.txt", b"The launch is in June.")})
    assert duplicate.json()["status"] == "duplicate"
    bad = client.post("/api/documents", headers=headers, files={"file": ("bad.pdf", b"broken")})
    assert bad.json()["status"] == "failed"
    assert len(client.get("/api/session", headers=headers).json()["documents"]) == 1
    monkeypatch.setattr("rag_chat.api.MAX_FILE_BYTES", 10)
    assert client.post("/api/documents", headers=headers, files={"file": ("big.txt", b"x" * 11)}).status_code == 413
    assert client.post("/api/documents", headers=headers, files={"file": ("big.txt", b"x" * (1024 * 1024 + 11))}).status_code == 413
    assert client.get("/api/session", headers=headers).json()["operation"]["status"] == "failed"


def test_api_rollback_preserves_existing_document(api_client, monkeypatch):
    client, manager, _ = api_client
    headers = setup_session(client)
    library = manager._libraries[headers["X-Session-ID"]]
    def fail(**kwargs):
        raise RuntimeError("embedding failure")
    monkeypatch.setattr(library.collection, "add", fail)
    response = client.post("/api/documents", headers=headers, files={"file": ("new.txt", b"New content")})
    assert "rolled back" in response.json()["detail"]
    state = client.get("/api/session", headers=headers).json()
    assert state["healthy"] and len(state["documents"]) == 1


def test_history_owned_and_bounded_by_server(api_client, monkeypatch):
    client, manager, _ = api_client
    headers = setup_session(client)
    library = manager._libraries[headers["X-Session-ID"]]
    library.messages = [{"role": role, "content": str(i)} for i in range(8) for role in ("user", "assistant")]
    seen = []
    def answer(library, question, history, client, on_event, workflow):
        seen.extend(history)
        return Answer("June. [1]", [Source(1, "launch.txt", "Text", "June")])
    monkeypatch.setattr("rag_chat.api.answer_question", answer)
    assert client.post("/api/chat", headers=headers, json={"question": "test", "history": []}).status_code == 422
    assert client.post("/api/chat", headers=headers, json={"question": "test"}).status_code == 200
    assert len(seen) == 10 and seen[0]["content"] == "3"


def test_safe_failure_does_not_commit_history(api_client, monkeypatch):
    client, manager, _ = api_client
    headers = setup_session(client)
    def fail(*args, **kwargs):
        raise RuntimeError("secret provider response")
    monkeypatch.setattr("rag_chat.api.answer_question", fail)
    response = client.post("/api/chat", headers=headers, json={"question": "test"})
    assert "secret" not in response.text
    assert parse_events(response)[-1][0] == "error"
    state = client.get("/api/session", headers=headers).json()
    assert state["messages"] == [] and state["operation"]["status"] == "failed"


def test_disconnect_finishes_and_recovers_result(api_client, monkeypatch):
    """Drive ASGI directly: disconnect after the first body, while the worker is blocked."""
    client, manager, _ = api_client
    headers = setup_session(client)
    released = threading.Event()
    def answer(*args, on_event, **kwargs):
        on_event({"event": "node_start", "node": "planner"})
        assert released.wait(5)
        return Answer("Recovered answer. [1]", [Source(1, "launch.txt", "Text", "June")])
    monkeypatch.setattr("rag_chat.api.answer_question", answer)

    async def disconnect():
        body_sent = False
        first_chunk = asyncio.Event()
        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": b'{"question":"test"}', "more_body": False}
            await first_chunk.wait()
            return {"type": "http.disconnect"}
        async def send(message):
            if message["type"] == "http.response.body":
                first_chunk.set()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
                 "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/api/chat",
                 "raw_path": b"/api/chat", "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json"), (b"x-session-id", headers["X-Session-ID"].encode())],
                 "server": ("testserver", 80), "client": ("test", 1)}
        await client.app(scope, receive, send)
        assert manager.snapshot(headers["X-Session-ID"])["operation"]["status"] == "running"
        released.set()
        await asyncio.gather(*client.app.state.tasks)
    client.portal.call(disconnect)
    state = client.get("/api/session", headers=headers).json()
    assert state["messages"][-1]["content"] == "Recovered answer. [1]"
    assert state["operation"]["status"] == "complete"


def test_missing_key_still_allows_uploads(manager, tokenizer, monkeypatch, tmp_path):
    runtime = SimpleNamespace(manager=manager, tokenizer=lambda: tokenizer)
    with TestClient(create_app(lambda: runtime, workflow_db_path=tmp_path / "workflows.sqlite3")) as client:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert client.get("/api/health").json() == {"status": "ok", "answering_configured": False}
        headers = setup_session(client)
        assert client.post("/api/chat", headers=headers, json={"question": "When?"}).status_code == 503
        assert client.get("/api/session").status_code == 400
        assert client.post("/api/chat", headers=headers, json={"question": "   "}).status_code == 422


def test_static_export_and_api_routes_share_one_server(manager, tokenizer, tmp_path, monkeypatch):
    export = tmp_path / "out"
    asset = export / "_next" / "static"
    asset.mkdir(parents=True)
    (export / "index.html").write_text("<main>Folio workspace</main>")
    (asset / "app.js").write_text("window.folio = true")
    runtime = SimpleNamespace(manager=manager, tokenizer=lambda: tokenizer)
    monkeypatch.setenv("GROQ_API_KEY", "")

    with TestClient(create_app(lambda: runtime, frontend_dir=export,
                               workflow_db_path=tmp_path / "workflows.sqlite3")) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Folio workspace" in page.text
        assert client.get("/_next/static/app.js").text == "window.folio = true"
        assert client.get("/api/health").json() == {"status": "ok", "answering_configured": False}


def test_api_starts_without_a_static_export(manager, tokenizer, tmp_path, monkeypatch):
    runtime = SimpleNamespace(manager=manager, tokenizer=lambda: tokenizer)
    monkeypatch.setenv("GROQ_API_KEY", "")
    with TestClient(create_app(lambda: runtime, frontend_dir=tmp_path / "missing",
                               workflow_db_path=tmp_path / "workflows.sqlite3")) as client:
        assert client.get("/").status_code == 404
        assert client.get("/api/health").status_code == 200


def test_saved_workflow_settings_control_chat_execution(api_client):
    client, _, model = api_client
    headers = setup_session(client)
    types = client.get("/api/node-types").json()
    assert {item["type"] for item in types} >= {"planner", "retrieve", "validate", "generate", "need_upload"}
    retriever = next(item for item in types if item["type"] == "retrieve")
    assert retriever["kind"] == "tool"
    assert retriever["requires"] == ["library", "searches", "round"]
    assert retriever["provides"] == ["evidence", "new_evidence_count"]
    assert set(retriever["initial_inputs"]) == INITIAL_STATE_KEYS

    draft = DEFAULT_WORKFLOW.model_copy(deep=True)
    draft.name = "One pass research"
    draft.nodes[2].config = {"max_rounds": 1, "final_confidence": 0.9}
    saved = client.post("/api/workflows", headers=headers, json=draft.model_dump())
    assert saved.status_code == 201
    workflow = saved.json()
    assert workflow["version"] == 1
    model.chat.completions.create.side_effect = [
        completion(json.dumps({"searches": [{"query": "launch", "purpose": "Find launch."}]})),
        completion(json.dumps({"decision": "needs_more_evidence", "confidence": 0.5,
                               "missing_evidence": ["schedule"]})),
    ]
    response = client.post("/api/chat", headers=headers,
                           json={"question": "When?", "workflow_id": workflow["id"]})
    assert response.status_code == 200
    assert "Please upload" in parse_events(response)[-1][1]["content"]
    assert model.chat.completions.create.call_count == 2
    state = client.get("/api/session", headers=headers).json()
    assert state["operation"]["workflow"] == {"id": workflow["id"], "name": workflow["name"], "version": 1}

    draft.nodes[2].config["final_confidence"] = 0.5
    updated = client.put(f"/api/workflows/{workflow['id']}", headers=headers,
                         json={"workflow": draft.model_dump(), "version": 1})
    assert updated.status_code == 200 and updated.json()["version"] == 2
    assert client.put(f"/api/workflows/{workflow['id']}", headers=headers,
                      json={"workflow": draft.model_dump(), "version": 1}).status_code == 409
    model.chat.completions.create.side_effect = [
        completion(json.dumps({"searches": [{"query": "launch", "purpose": "Find launch."}]})),
        completion(json.dumps({"decision": "needs_more_evidence", "confidence": 0.5,
                               "missing_evidence": ["schedule"]})),
        completion("Launch in June. [1]"),
    ]
    second = client.post("/api/chat", headers=headers,
                         json={"question": "When?", "workflow_id": workflow["id"]})
    assert parse_events(second)[-1][1]["content"] == "Launch in June. [1]"
    assert client.get("/api/session", headers=headers).json()["operation"]["workflow"]["version"] == 2
    assert client.post("/api/chat", headers=headers,
                       json={"question": "When?", "workflow_id": "unknown"}).status_code == 404


def test_saved_workflows_are_visible_and_mutable_only_in_their_session(api_client):
    client, _, _ = api_client
    first = setup_session(client)
    second = setup_session(client)
    draft = DEFAULT_WORKFLOW.model_copy(deep=True)
    draft.name = "Private research"
    created = client.post("/api/workflows", headers=first, json=draft.model_dump())
    assert created.status_code == 201
    workflow_id = created.json()["id"]
    url = f"/api/workflows/{workflow_id}"

    assert {item["id"] for item in client.get("/api/workflows", headers=first).json()} == {"default", workflow_id}
    assert [item["id"] for item in client.get("/api/workflows", headers=second).json()] == ["default"]
    assert client.get("/api/workflows/default", headers=second).status_code == 200
    assert client.get(url, headers=second).status_code == 404
    assert client.put(url, headers=second,
                      json={"workflow": draft.model_dump(), "version": 1}).status_code == 404
    assert client.delete(url, headers=second).status_code == 404
    assert client.post("/api/chat", headers=second,
                       json={"question": "When?", "workflow_id": workflow_id}).status_code == 404
    assert client.get(url, headers=first).status_code == 200

    assert client.get("/api/workflows").status_code == 400
    assert client.post("/api/workflows", json=draft.model_dump()).status_code == 400
    assert client.get("/api/workflows", headers={"X-Session-ID": "unknown"}).status_code == 410
    assert client.delete("/api/session", headers=first).status_code == 204
    assert client.get(url, headers=first).status_code == 410
