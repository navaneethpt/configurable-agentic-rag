"""Offline HTTP transport and workflow integration for OpenRouter."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from fastapi.testclient import TestClient
import httpx
import pytest

from test_api import parse_events, setup_session
from rag_chat.agent_contracts import ChatError
from rag_chat.agents.common import GenerationOutput, PlannerOutput, ValidatorOutput
from rag_chat.api import create_app
from rag_chat.chat import answer_question
from rag_chat.indexing import ingest
from rag_chat.model_client import _structured
from rag_chat.providers import (DEFAULT_MODEL, GEMINI_MODEL, GROQ_MODEL, OPENROUTER_MODEL, OpenRouterClient,
                                create_model_client, openrouter_retry_delay, provider_settings,
                                strict_json_schema)
from rag_chat.workflows import DEFAULT_WORKFLOW, WorkflowDraft, WorkflowNode

MODEL = "vendor/model"
ENV = {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "test-secret", "LLM_MODEL": MODEL}


def success(text="answer", reason="stop", **message):
    return httpx.Response(200, json={"choices": [{"finish_reason": reason,
        "message": {"role": "assistant", "content": text, **message}}]})


def adapter(respond):
    transport = httpx.Client(transport=httpx.MockTransport(respond))
    return OpenRouterClient(api_key="test-secret", default_model=MODEL, sdk=transport)


@pytest.mark.parametrize("changes,error", [
    ({}, None), ({"LLM_PROVIDER": " OpenRouter ", "LLM_MODEL": " vendor/model "}, None),
    ({"LLM_MODEL": ""}, None), ({"LLM_MODEL": "default"}, "LLM_MODEL"),
    ({"LLM_MODEL": "model with spaces"}, "LLM_MODEL"), ({"LLM_MODEL": "x" * 101}, "100 characters"),
    ({"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "other", "GROQ_API_KEY": "other"}, "OPENROUTER_API_KEY"),
])
def test_settings(changes, error):
    settings = provider_settings(ENV | changes)
    assert settings.provider == "openrouter"
    if error:
        assert error in settings.configuration_error
        with pytest.raises(ChatError):
            create_model_client(settings)
    else:
        assert settings.configuration_error is None
        expected = OPENROUTER_MODEL if changes.get("LLM_MODEL") == "" else MODEL
        assert settings.default_model == expected and settings.api_key == "test-secret"
    assert "test-secret" not in repr(settings)


def test_defaults_key_isolation_and_environment_precedence(tmp_path, monkeypatch):
    settings = provider_settings({"OPENROUTER_API_KEY": "other"})
    assert settings.provider == "openrouter" and settings.default_model == OPENROUTER_MODEL
    assert settings.configuration_error is None
    assert "OPENROUTER_API_KEY" in provider_settings({"GEMINI_API_KEY": "other"}).configuration_error
    assert provider_settings({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "other"}).default_model == GEMINI_MODEL
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    file = tmp_path / ".env"
    file.write_text("LLM_PROVIDER=gemini\nLLM_MODEL=gemini-other\nOPENROUTER_API_KEY=file-key\n")
    load_dotenv(file, override=False)
    assert provider_settings().default_model == MODEL
    assert provider_settings().api_key == "test-secret"


def test_factory_timeout_and_cleanup():
    with patch("rag_chat.providers.httpx.Client") as constructor:
        with create_model_client(provider_settings({"OPENROUTER_API_KEY": "test-secret"})) as client:
            assert isinstance(client, OpenRouterClient)
            assert client.default_model == OPENROUTER_MODEL
            assert client.resolve_model(DEFAULT_MODEL) == OPENROUTER_MODEL
            assert client.resolve_model(GROQ_MODEL) == OPENROUTER_MODEL
        assert constructor.call_args.kwargs["timeout"] == 45.0
        constructor.return_value.close.assert_called_once()


def test_strict_schema_preserves_references_constraints_and_literal_data():
    schema = {"type": "object", "default": {}, "properties": {
        "default": {"type": "string", "default": "advisory", "minLength": 1,
                    "examples": [{"default": "literal", "properties": {"untouched": 1}}]},
        "nested": {"anyOf": [{"$ref": "#/$defs/Child"}, {"type": "null"}], "default": None},
        "many": {"type": "array", "items": {"$ref": "#/$defs/Child"}, "maxItems": 3}},
        "$defs": {"Child": {"type": "object", "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1, "default": 0}},
            "required": [], "additionalProperties": True}}, "required": ["default"]}
    original = deepcopy(schema)
    result = strict_json_schema(schema)
    assert schema == original and result is not schema
    assert result["required"] == ["default", "nested", "many"]
    assert result["additionalProperties"] is False and "default" not in result
    assert "default" not in result["properties"]["default"]
    assert result["properties"]["default"]["minLength"] == 1
    assert result["properties"]["default"]["examples"] == original["properties"]["default"]["examples"]
    assert result["properties"]["nested"]["anyOf"] == original["properties"]["nested"]["anyOf"]
    assert result["properties"]["many"] == original["properties"]["many"]
    child = result["$defs"]["Child"]
    assert child["required"] == ["score"] and child["additionalProperties"] is False
    assert child["properties"]["score"] == {"type": "number", "minimum": 0, "maximum": 1}
    validator = strict_json_schema(ValidatorOutput.model_json_schema())
    assert "missing_evidence" in validator["required"]
    planner = strict_json_schema(PlannerOutput.model_json_schema())
    assert planner["$defs"]["SearchTask"]["additionalProperties"] is False


def test_serialization_aliases_and_native_repair():
    captured = []
    def respond(request):
        assert str(request.url) == OpenRouterClient.endpoint
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert request.extensions["timeout"] == dict.fromkeys(["connect", "read", "write", "pool"], 45.0)
        captured.append(json.loads(request.content))
        return success("not JSON" if len(captured) == 1 else '{"status":"missing_context","text":"Upload context"}')
    messages = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Earlier response"}]
    client = adapter(respond)
    with client:
        for selected in [DEFAULT_MODEL, GROQ_MODEL, "other/model"]:
            result = _structured(client, GenerationOutput, messages, "generator", model=selected)
            assert result.text == "Upload context"
    assert client.sdk.is_closed
    assert len(captured) == 4  # One repair, not a provider/schema fallback.
    for request in captured:
        assert request["stream"] is False and request["max_tokens"] == 4096
        assert request["provider"] == {"require_parameters": True}
        assert request["response_format"] == {"type": "json_schema", "json_schema": {
            "name": "agent_output", "strict": True, "schema": strict_json_schema(GenerationOutput.model_json_schema())}}
    assert captured[0]["messages"][:3] == messages
    assert [item["model"] for item in captured] == [MODEL, MODEL, MODEL, "other/model"]


def test_unstructured_request_does_not_add_schema_and_invalid_override_does_not_call():
    captured = []
    def respond(request):
        captured.append(json.loads(request.content))
        return success(" answer ")
    with adapter(respond) as client:
        assert client.complete([], model="other/model", max_tokens=250) == "answer"
        for model in ["", "bad model", "x" * 101]:
            with pytest.raises(ChatError, match="valid OpenRouter model"):
                client.complete([], model=model)
    assert captured == [{"model": "other/model", "messages": [], "max_tokens": 250, "stream": False}]


@pytest.mark.parametrize("status,match,attempts", [
    (400, "model or schema", 1), (401, "API key", 1), (402, "credits", 1), (403, "refused", 1),
    (404, "model or schema", 1), (422, "model or schema", 1), (408, "timed out", 2),
    (429, "quota", 2), (500, "could not complete", 2), (502, "could not complete", 2),
    (503, "native JSON-schema", 2),
])
def test_safe_http_errors_and_bounded_retries(status, match, attempts):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"code": status, "message": "private-secret"}},
                              headers={"Retry-After": "0"})
    with patch("rag_chat.providers.sleep") as pause:
        with pytest.raises(ChatError, match=match) as caught:
            with adapter(respond) as client:
                client.complete([], schema=GenerationOutput.model_json_schema())
        assert pause.call_count == attempts - 1
    assert client.sdk.is_closed and len(requests) == attempts
    assert all(request["response_format"]["type"] == "json_schema" for request in requests)
    assert "private-secret" not in str(caught.value)


@pytest.mark.parametrize("code,match", [(401, "API key"), (402, "credits"), (429, "quota"),
                                       (503, "endpoint"), (None, "could not complete")])
def test_200_error_body_never_replays_generation(code, match):
    respond = Mock(return_value=httpx.Response(200, json={"error": {"code": code, "message": "private-secret"}}))
    with patch("rag_chat.providers.sleep") as pause:
        with pytest.raises(ChatError, match=match) as caught:
            with adapter(respond) as client:
                client.complete([])
        pause.assert_not_called()
    assert respond.call_count == 1 and "private-secret" not in str(caught.value)


@pytest.mark.parametrize("response,match", [
    (success("partial", "length"), "truncated"), (success("text", "content_filter"), "refused"),
    (success("text", refusal="private-secret"), "refused"), (success("", "stop"), "empty"),
    (success(None), "empty"), (success("text", "error"), "could not finish"),
    (success({"private-secret": 1}), "malformed"), (httpx.Response(200, text="private-secret"), "malformed"),
    (httpx.Response(200, json={"choices": []}), "malformed"), (httpx.Response(200, json=[]), "malformed"),
    (httpx.Response(200, json={"choices": [{"error": {"message": "private-secret"}}]}), "could not complete"),
])
def test_unusable_responses_are_safe_and_not_retried(response, match):
    respond = Mock(return_value=response)
    with pytest.raises(ChatError, match=match) as caught:
        with adapter(respond) as client:
            client.complete([])
    assert respond.call_count == 1 and "private-secret" not in str(caught.value)


@pytest.mark.parametrize("error", [httpx.ConnectError("private-secret"), httpx.ReadTimeout("private-secret")])
def test_connection_retry_bound(error):
    respond = Mock(side_effect=error)
    with patch("rag_chat.providers.sleep") as pause:
        with pytest.raises(ChatError, match="Cannot reach OpenRouter"):
            with adapter(respond) as client:
                client.complete([])
        pause.assert_called_once_with(1)
    assert respond.call_count == 2 and client.sdk.is_closed


@pytest.mark.parametrize("first", [httpx.ConnectError("secret"), httpx.Response(429), httpx.Response(500)])
def test_transient_error_then_success(first):
    respond = Mock(side_effect=[first, success()])
    with patch("rag_chat.providers.sleep") as pause:
        with adapter(respond) as client:
            assert client.complete([]) == "answer"
        pause.assert_called_once_with(1.0)
    assert respond.call_count == 2


@pytest.mark.parametrize("header,expected", [(None, 1), ("invalid", 1), ("-1", 1), ("nan", 1),
                                            ("inf", 1), ("0", 0), ("60", 60)])
def test_retry_after(header, expected):
    assert openrouter_retry_delay(header) == expected


def test_retry_after_http_date_and_excessive_wait():
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    assert 28 <= openrouter_retry_delay(future) <= 30
    past = format_datetime(datetime.now(timezone.utc) - timedelta(seconds=30), usegmt=True)
    assert openrouter_retry_delay(past) == 0
    for header in ["61", format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)]:
        with pytest.raises(ChatError, match="longer than 60 seconds"):
            openrouter_retry_delay(header)
    respond = Mock(return_value=httpx.Response(429, headers={"Retry-After": "61"}))
    with patch("rag_chat.providers.sleep") as pause:
        with pytest.raises(ChatError, match="Wait before trying again"):
            with adapter(respond) as client:
                client.complete([])
        pause.assert_not_called()
    assert respond.call_count == 1


def test_repair_limit_keeps_native_schema_and_closes():
    respond = Mock(return_value=success("invalid JSON"))
    with pytest.raises(ChatError, match="invalid structured response"):
        with adapter(respond) as client:
            _structured(client, GenerationOutput, [], "generator")
    assert respond.call_count == 2 and client.sdk.is_closed
    bodies = [json.loads(call.args[0].content) for call in respond.call_args_list]
    assert bodies[0]["response_format"] == bodies[1]["response_format"]


@pytest.mark.parametrize("graph", ["default", "retrieval", "generator"])
def test_workflows_through_http_transport(manager, tokenizer, graph):
    library, _ = manager.ensure(None)
    ingest(library, "launch.txt", b"Launch in June.", lambda: tokenizer)
    workflow = DEFAULT_WORKFLOW.model_copy(deep=True) if graph == "default" else WorkflowDraft(
        name=graph, entry="retrieve" if graph == "retrieval" else "generate", nodes=[
            *([WorkflowNode(id="retrieve", type="retrieve", transitions={"next": "generate"})] if graph == "retrieval" else []),
            WorkflowNode(id="generate", type="generate")])
    for node in workflow.nodes:
        if node.type in {"planner", "validate", "generate"}:
            node.config["model"] = GROQ_MODEL
    results = [json.dumps({"searches": [{"query": "launch", "purpose": "Find timing"}]}),
               json.dumps({"decision": "sufficient", "confidence": 0.9, "missing_evidence": []})] if graph == "default" else []
    text = "Please supply context" if graph == "generator" else "Launch in June. [1]"
    results.append(json.dumps({"status": "missing_context" if graph == "generator" else "answered", "text": text}))
    requests, events = [], []
    def respond(request):
        requests.append(json.loads(request.content))
        return success(results[len(requests) - 1])
    with adapter(respond) as client:
        answer = answer_question(library, "When?", [], client, workflow=workflow, on_event=events.append)
    assert answer.text == text and bool(answer.sources) == (graph != "generator")
    assert all(request["model"] == MODEL for request in requests)
    expected = ["planner", "retrieve", "validate", "generate"] if graph == "default" else (
        ["retrieve", "generate"] if graph == "retrieval" else ["generate"])
    assert [event["node_type"] for event in events if event["event"] == "node_start"] == expected


@pytest.mark.parametrize("setting,value", [("OPENROUTER_API_KEY", ""), ("LLM_MODEL", "default")])
def test_api_config_error_preserves_uploads_and_does_not_start_research(manager, tokenizer, monkeypatch, tmp_path, setting, value):
    for name, env_value in ENV.items():
        monkeypatch.setenv(name, env_value)
    # Defaults supply the model; an invalid explicit override still rejects chat.
    monkeypatch.setenv(setting, value)
    monkeypatch.setenv("GEMINI_API_KEY", "other-secret")
    runtime = SimpleNamespace(manager=manager, tokenizer=lambda: tokenizer)
    with TestClient(create_app(lambda: runtime, workflow_db_path=tmp_path / "workflows.sqlite3")) as client:
        health = client.get("/api/health").json()
        assert health["provider"] == "openrouter" and health["answering_configured"] is False
        assert setting in health["configuration_error"] and "secret" not in json.dumps(health)
        headers = setup_session(client)
        before = client.get("/api/session", headers=headers).json()["operation"]
        response = client.post("/api/chat", headers=headers, json={"question": "When?"})
        assert response.status_code == 503 and response.json()["detail"] == health["configuration_error"]
        assert client.get("/api/session", headers=headers).json()["operation"] == before


def test_api_completed_answer_recovery_and_isolation(manager, tokenizer, monkeypatch, tmp_path):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("LLM_PROVIDER")
    monkeypatch.delenv("LLM_MODEL")
    # Exercise startup defaults without letting the developer's .env supply overrides.
    monkeypatch.setattr("rag_chat.api.load_dotenv", lambda *args, **kwargs: None)
    respond = Mock(side_effect=[success(json.dumps(value)) for value in [
        {"searches": [{"query": "launch", "purpose": "Find timing"}]},
        {"decision": "sufficient", "confidence": 0.95, "missing_evidence": []},
        {"status": "answered", "text": "Launch in June. [1]"}]])
    model = OpenRouterClient(api_key="test-secret", sdk=httpx.Client(transport=httpx.MockTransport(respond)))
    factory = Mock(return_value=model)
    monkeypatch.setattr("rag_chat.api.create_model_client", factory)
    runtime = SimpleNamespace(manager=manager, tokenizer=lambda: tokenizer)
    with TestClient(create_app(lambda: runtime, workflow_db_path=tmp_path / "workflows.sqlite3")) as client:
        health = client.get("/api/health").json()
        assert health["provider"] == "openrouter" and health["default_model"] == OPENROUTER_MODEL
        assert health["answering_configured"] is True
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        assert client.get("/api/health").json() == health
        headers = setup_session(client)
        response = client.post("/api/chat", headers=headers, json={"question": "When?"})
        assert parse_events(response)[-1][0] == "answer"
        state = client.get("/api/session", headers=headers).json()
        assert state["operation"]["status"] == "complete"
        assert state["messages"][-1]["content"] == "Launch in June. [1]"
        assert state["messages"][-1]["sources"][0]["filename"] == "launch.txt"
        other = setup_session(client)
        assert client.get("/api/session", headers=other).json()["messages"] == []
    assert factory.call_args.args[0].provider == "openrouter" and model.sdk.is_closed
    assert all(json.loads(call.args[0].content)["model"] == OPENROUTER_MODEL for call in respond.call_args_list)


@pytest.mark.parametrize("finish", ["answered", "needs_upload"])
def test_conditional_retry_routes_keep_native_schema(manager, tokenizer, finish):
    library, _ = manager.ensure(None)
    ingest(library, "launch.txt", b"Launch in June.", lambda: tokenizer)
    plan = {"searches": [{"query": "launch", "purpose": "Find timing"}]}
    insufficient = {"decision": "needs_more_evidence", "confidence": 0.1,
                    "missing_evidence": ["annual budget document"]}
    enough = {"decision": "sufficient", "confidence": 0.95, "missing_evidence": []}
    results = [plan, insufficient, plan, enough if finish == "answered" else insufficient]
    if finish == "answered":
        results += [{"status": "answered", "text": "Launch in June. [999]"},
                    {"status": "answered", "text": "Launch in June. [1]"}]
    else:
        results += [plan, insufficient]
    requests, events = [], []
    def respond(request):
        requests.append(json.loads(request.content))
        return success(json.dumps(results[len(requests) - 1]))
    with adapter(respond) as client:
        answer = answer_question(library, "When?", [], client, on_event=events.append)
    nodes = [event["node_type"] for event in events if event["event"] == "node_start"]
    if finish == "answered":
        assert nodes == ["planner", "retrieve", "validate"] * 2 + ["generate"]
        assert answer.text == "Launch in June. [1]" and answer.sources[0].number == 1
        assert requests[-2]["response_format"] == requests[-1]["response_format"]
    else:
        assert nodes == ["planner", "retrieve", "validate"] * 3 + ["need_upload"]
        assert "annual budget document" in answer.text and answer.sources == []
    assert len(requests) == 6
    assert all(request["provider"] == {"require_parameters": True} for request in requests)
