"""Offline SDK and workflow integration checks for both model providers."""
import json
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import httpx
import pytest

from conftest import completion
from rag_chat.agent_contracts import ChatError
from rag_chat.agents.common import GenerationOutput
from rag_chat.chat import answer_question
from rag_chat.indexing import ingest
from rag_chat.model_client import _structured
from rag_chat.providers import (DEFAULT_MODEL, GEMINI_MODEL, GROQ_MODEL, GeminiClient,
                                GroqClient, create_model_client, provider_settings)
from rag_chat.workflows import DEFAULT_WORKFLOW, WorkflowDraft, WorkflowNode


def gemini_response(text, reason=types.FinishReason.STOP):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason=reason, content=types.Content(role="model", parts=[types.Part(text=text)]))])


@pytest.mark.parametrize("env,provider,model,key,error", [
    ({"GEMINI_API_KEY": " gemini-secret "}, "gemini", GEMINI_MODEL, "gemini-secret", None),
    ({"GROQ_API_KEY": "other-provider-secret"}, "gemini", GEMINI_MODEL, "", "GEMINI_API_KEY"),
    ({"LLM_PROVIDER": "groq", "GROQ_API_KEY": " groq-secret "}, "groq", GROQ_MODEL, "groq-secret", None),
    ({"LLM_PROVIDER": " Gemini ", "GEMINI_API_KEY": "gemini-secret", "LLM_MODEL": " gemini-custom "},
     "gemini", "gemini-custom", "gemini-secret", None),
    ({"LLM_PROVIDER": "gemini", "GROQ_API_KEY": "wrong-provider-secret"}, "gemini", GEMINI_MODEL, "", "GEMINI_API_KEY"),
    ({"LLM_PROVIDER": "groq", "GEMINI_API_KEY": "wrong-provider-secret"}, "groq", GROQ_MODEL, "", "GROQ_API_KEY"),
    ({"LLM_PROVIDER": "private-secret"}, None, None, "", "LLM_PROVIDER"),
    ({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "key", "LLM_MODEL": "llama-model"},
     "gemini", "llama-model", "key", "LLM_MODEL"),
    ({"LLM_MODEL": "x" * 101}, "gemini", "x" * 101, "", "100 characters"),
])
def test_settings(env, provider, model, key, error):
    settings = provider_settings(env)
    assert (settings.provider, settings.default_model, settings.api_key) == (provider, model, key)
    assert (error in settings.configuration_error) if error else settings.configuration_error is None
    assert "secret" not in repr(settings)
    if error:
        with pytest.raises(ChatError):
            create_model_client(settings)


def test_environment_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    file = tmp_path / ".env"
    file.write_text("LLM_PROVIDER=groq\nLLM_MODEL=file-model\nGROQ_API_KEY=file-key\n")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-environment")
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    load_dotenv(file, override=False)
    settings = provider_settings()
    assert settings.provider == "gemini" and settings.default_model == "gemini-environment"
    assert settings.api_key == "env-key"


@pytest.mark.parametrize("provider,key,adapter,path", [
    ("groq", "GROQ_API_KEY", GroqClient, "rag_chat.providers.Groq"),
    ("gemini", "GEMINI_API_KEY", GeminiClient, "rag_chat.providers.genai.Client"),
])
def test_factory_uses_selected_credentials_and_closes_sdk(provider, key, adapter, path):
    settings = provider_settings({"LLM_PROVIDER": provider, key: "chosen-key"})
    with patch(path) as constructor:
        with create_model_client(settings) as client:
            assert isinstance(client, adapter)
        kwargs = constructor.call_args.kwargs
        assert kwargs["api_key"] == "chosen-key"
        if provider == "groq":
            assert kwargs["timeout"] == 45.0 and kwargs["max_retries"] == 1
        else:
            assert kwargs["vertexai"] is False
            assert kwargs["http_options"].timeout == 45000
            assert kwargs["http_options"].retry_options.attempts == 2
        constructor.return_value.close.assert_called_once()


def test_gemini_translation_native_schema_and_overrides():
    sdk = Mock()
    sdk.models.generate_content.return_value = gemini_response('{"status":"missing_context","text":"Upload context"}')
    adapter = GeminiClient(default_model="gemini-env-model", sdk=sdk)
    messages = [{"role": "system", "content": "Primary rules"},
                {"role": "user", "content": "Original question"},
                {"role": "assistant", "content": "Earlier response"},
                {"role": "system", "content": "Repair rules"},
                {"role": "user", "content": "Correction"}]
    for selected, expected in [(DEFAULT_MODEL, "gemini-env-model"), (GROQ_MODEL, "gemini-env-model"),
                               ("gemini-override", "gemini-override")]:
        result = _structured(adapter, GenerationOutput, messages, "generator", model=selected)
        assert result.status == "missing_context"
        request = sdk.models.generate_content.call_args.kwargs
        assert request["model"] == expected
        assert [item.role for item in request["contents"]] == ["user", "model", "user"]
        assert [item.parts[0].text for item in request["contents"]] == ["Original question", "Earlier response", "Correction"]
        assert "Primary rules\n\nRepair rules" in request["config"].system_instruction
        assert request["config"].response_mime_type == "application/json"
        assert request["config"].response_json_schema == GenerationOutput.model_json_schema()
        assert request["config"].max_output_tokens == 4096


def test_groq_default_and_explicit_model_overrides():
    sdk = Mock()
    sdk.chat.completions.create.return_value = completion("answer")
    client = GroqClient(default_model="groq-env-model", sdk=sdk)
    for model, expected in [(DEFAULT_MODEL, "groq-env-model"), (GROQ_MODEL, "groq-env-model"),
                             ("groq-custom-model", "groq-custom-model")]:
        assert client.complete([], model=model) == "answer"
        assert sdk.chat.completions.create.call_args.kwargs["model"] == expected


def test_native_gemini_request_serializes_schema_roles_and_thinking():
    captured = []
    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
            "role": "model", "parts": [{"text": '{"status":"missing_context","text":"Upload context"}'}]}}]})
    transport = httpx.Client(transport=httpx.MockTransport(respond))
    sdk = genai.Client(api_key="dummy", vertexai=False, http_options=types.HttpOptions(httpx_client=transport))
    try:
        with GeminiClient(sdk=sdk) as client:
            result = _structured(client, GenerationOutput, [
                {"role": "system", "content": "Grounded answers only"},
                {"role": "user", "content": "Question"}], "generator")
        assert result.text == "Upload context"
        assert captured[0]["generationConfig"]["responseMimeType"] == "application/json"
        assert captured[0]["generationConfig"]["responseJsonSchema"] == GenerationOutput.model_json_schema()
        assert captured[0]["generationConfig"]["thinkingConfig"]["thinking_level"] == "LOW"
        assert captured[0]["generationConfig"]["maxOutputTokens"] == 4096
        assert captured[0]["contents"][0]["role"] == "user"
        assert "Grounded answers only" in captured[0]["systemInstruction"]["parts"][0]["text"]
    finally:
        transport.close()


def test_gemini_repairs_once_and_closes_on_failure():
    sdk = Mock()
    sdk.models.generate_content.side_effect = [gemini_response("not JSON"), gemini_response("still not JSON")]
    with pytest.raises(ChatError, match="invalid structured response"):
        with GeminiClient(sdk=sdk) as client:
            _structured(client, GenerationOutput, [{"role": "user", "content": "Question"}], "generator")
    assert sdk.models.generate_content.call_count == 2
    assert sdk.models.generate_content.call_args.kwargs["config"].response_json_schema
    sdk.close.assert_called_once()


@pytest.mark.parametrize("code,match", [(400, "model or request"), (401, "API key"), (403, "access"),
                                          (404, "model"), (408, "reach Gemini"), (429, "quota"), (503, "could not complete")])
def test_gemini_api_errors_are_safe(code, match):
    sdk = Mock()
    sdk.models.generate_content.side_effect = errors.APIError(code, {"error": {"message": "private-secret"}})
    with pytest.raises(ChatError, match=match) as caught:
        GeminiClient(sdk=sdk).complete([{"role": "user", "content": "Question"}])
    assert "private-secret" not in str(caught.value)


@pytest.mark.parametrize("response,match", [
    (gemini_response("", types.FinishReason.STOP), "empty"),
    (gemini_response("partial", types.FinishReason.MAX_TOKENS), "truncated"),
    (gemini_response("blocked", types.FinishReason.SAFETY), "blocked"),
    (types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(
        block_reason=types.BlockedReason.SAFETY)), "blocked"),
])
def test_gemini_unusable_responses(response, match):
    sdk = Mock()
    sdk.models.generate_content.return_value = response
    with pytest.raises(ChatError, match=match):
        GeminiClient(sdk=sdk).complete([{"role": "user", "content": "Question"}])


@pytest.mark.parametrize("error", [httpx.ReadTimeout("private-secret"), httpx.ConnectError("private-secret"),
                                    RuntimeError("private-secret")])
def test_gemini_transport_errors_do_not_expose_details(error):
    sdk = Mock()
    sdk.models.generate_content.side_effect = error
    with pytest.raises(ChatError) as caught:
        GeminiClient(sdk=sdk).complete([{"role": "user", "content": "Question"}])
    assert "private-secret" not in str(caught.value)


@pytest.mark.parametrize("adapter,model", [(GroqClient, "gemini-other"), (GeminiClient, "llama-other")])
def test_incompatible_models_never_call_provider(adapter, model):
    sdk = Mock()
    with pytest.raises(ChatError, match="incompatible"):
        adapter(sdk=sdk).complete([], model=model)
    assert sdk.mock_calls == []


@pytest.mark.parametrize("reason,match", [("length", "truncated"), ("content_filter", "blocked")])
def test_groq_unusable_responses(reason, match):
    sdk = Mock()
    response = completion("partial")
    response.choices[0].finish_reason = reason
    sdk.chat.completions.create.return_value = response
    with pytest.raises(ChatError, match=match):
        GroqClient(sdk=sdk).complete([])


@pytest.mark.parametrize("provider", ["groq", "gemini"])
@pytest.mark.parametrize("graph", ["default", "retrieval", "generator"])
def test_workflows_use_each_adapter_with_legacy_defaults(manager, tokenizer, provider, graph):
    library, _ = manager.ensure(None)
    ingest(library, "launch.txt", b"Launch in June.", lambda: tokenizer)
    workflow = DEFAULT_WORKFLOW.model_copy(deep=True) if graph == "default" else WorkflowDraft(
        name=graph, entry="retrieve" if graph == "retrieval" else "generate", nodes=[
            *([WorkflowNode(id="retrieve", type="retrieve", transitions={"next": "generate"})] if graph == "retrieval" else []),
            WorkflowNode(id="generate", type="generate")])
    for node in workflow.nodes:
        if node.type in {"planner", "validate", "generate"}:
            node.config["model"] = GROQ_MODEL
    responses = [json.dumps({"searches": [{"query": "launch", "purpose": "Find timing"}]}),
                 json.dumps({"decision": "sufficient", "confidence": 0.9, "missing_evidence": []})] if graph == "default" else []
    responses.append(json.dumps({"status": "missing_context" if graph == "generator" else "answered",
                                "text": "Please supply context" if graph == "generator" else "Launch in June. [1]"}))
    sdk = Mock()
    adapter = GroqClient(sdk=sdk) if provider == "groq" else GeminiClient(sdk=sdk)
    call = sdk.chat.completions.create if provider == "groq" else sdk.models.generate_content
    call.side_effect = [completion(text) if provider == "groq" else gemini_response(text) for text in responses]
    result = answer_question(library, "When?", [], adapter, workflow=workflow)
    assert result.text == ("Please supply context" if graph == "generator" else "Launch in June. [1]")
    assert bool(result.sources) == (graph != "generator")
    assert all(item.kwargs["model"] == adapter.default_model for item in call.call_args_list)


@pytest.mark.parametrize("provider,status,attempts", [("gemini", 429, 2), ("gemini", 403, 1),
                                                        ("groq", 429, 2), ("groq", 401, 1)])
def test_real_sdk_transport_retry_bound(provider, status, attempts):
    count = 0
    def respond(request):
        nonlocal count
        count += 1
        return httpx.Response(status, json={"error": {"message": "private-secret", "code": status}},
                              headers={"retry-after": "0.01"})
    transport = httpx.Client(transport=httpx.MockTransport(respond))
    if provider == "gemini":
        sdk = genai.Client(api_key="dummy", vertexai=False, http_options=types.HttpOptions(
            timeout=45000, httpx_client=transport, retry_options=types.HttpRetryOptions(
                attempts=2, initial_delay=0.01, max_delay=0.01)))
        client = GeminiClient(sdk=sdk)
    else:
        from groq import Groq
        client = GroqClient(sdk=Groq(api_key="dummy", http_client=transport, timeout=45, max_retries=1))
    with pytest.raises(ChatError):
        with client:
            client.complete([{"role": "user", "content": "Question"}])
    # The Google SDK deliberately leaves caller-owned HTTP clients open.
    transport.close()
    assert count == attempts
