import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from groq import APITimeoutError, AuthenticationError, RateLimitError
import pytest

from conftest import completion, answer_completion
from rag_chat.chat import ChatError, MODEL, NO_EVIDENCE, answer_question
from rag_chat.indexing import ingest


def populated(manager, tokenizer):
    library, _ = manager.ensure(None)
    ingest(library, "schedule.txt", b"The launch is in June.", lambda: tokenizer)
    return library


def plan(query="launch"):
    return completion(json.dumps({"searches": [{"query": query, "purpose": "Find launch timing."}]}))


def sufficient(confidence=0.95):
    return completion(json.dumps({"decision": "sufficient", "confidence": confidence, "missing_evidence": []}))


def needs_more(*gaps, confidence=0.2):
    return completion(json.dumps({"decision": "needs_more_evidence", "confidence": confidence, "missing_evidence": list(gaps)}))


def answered(text="The launch is in June. [1]"):
    return answer_completion(text)


def retrieval_result(chunk_id, text):
    return {"ids": [[chunk_id]], "documents": [[text]], "metadatas": [[{
        "filename": "evidence.txt", "location": "Text", "document_id": "doc", "chunk": 0,
    }]]}


def mock_library(*results):
    collection = Mock()
    collection.count.return_value = max(1, len(results))
    collection.query.side_effect = list(results)
    return SimpleNamespace(healthy=True, collection=collection)


def test_grounded_answer_and_citation_mapping(manager, tokenizer, client):
    client.sdk.chat.completions.create.side_effect = [plan(), sufficient(), answered()]
    result = answer_question(populated(manager, tokenizer), "When is launch?", [], client)
    assert result.text.endswith("[1]")
    assert result.sources[0].filename == "schedule.txt"
    assert [event["event"] for event in result.trace] == ["planner", "retrieve", "validate", "generate"]
    calls = client.sdk.chat.completions.create.call_args_list
    assert len(calls) == 3
    assert calls[-1].kwargs["model"] == MODEL and calls[-1].kwargs["reasoning_effort"] == "low"
    payload = json.loads(calls[-1].kwargs["messages"][1]["content"])
    assert payload["excerpts"][0]["text"] == "The launch is in June."


def test_planner_resolves_bounded_followup_history(manager, tokenizer, client):
    client.sdk.chat.completions.create.side_effect = [plan("launch date"), sufficient(), answered("June. [1]")]
    history = [{"role": role, "content": f"turn {i}"} for i in range(7) for role in ["user", "assistant"]]
    answer_question(populated(manager, tokenizer), "When is it?", history, client)
    calls = client.sdk.chat.completions.create.call_args_list
    assert len(calls) == 3
    payload = json.loads(calls[0].kwargs["messages"][1]["content"])
    assert len(payload["conversation"]) == 10
    assert payload["conversation"][0]["content"] == "turn 2"


def test_validator_requests_a_second_retrieval_round(manager, tokenizer, client):
    client.sdk.chat.completions.create.side_effect = [
        plan("launch"), needs_more("the release schedule"),
        plan("release schedule"), sufficient(), answered(),
    ]
    library = mock_library(
        retrieval_result("launch", "The launch is in June."),
        retrieval_result("schedule", "The release schedule starts in July."),
    )
    result = answer_question(library, "Compare the launch and release schedule.", [], client)
    assert result.text.endswith("[1]")
    assert [event["event"] for event in result.trace].count("planner") == 2
    assert [event["event"] for event in result.trace].count("validate") == 2
    second_plan = result.trace[3]
    assert second_plan["event"] == "planner"
    assert second_plan["round"] == 2


def test_duplicate_chunks_are_not_sent_twice_to_generator(manager, tokenizer, client):
    duplicate_plan = completion(json.dumps({"searches": [
        {"query": "launch", "purpose": "Find launch timing."},
        {"query": "launch", "purpose": "Find launch timing again."},
    ]}))
    client.sdk.chat.completions.create.side_effect = [duplicate_plan, sufficient(), answered()]
    answer_question(populated(manager, tokenizer), "When is launch?", [], client)
    payload = json.loads(client.sdk.chat.completions.create.call_args_list[-1].kwargs["messages"][1]["content"])
    assert len(payload["excerpts"]) == 1


def test_no_new_evidence_requests_named_upload(manager, tokenizer, client):
    client.sdk.chat.completions.create.side_effect = [plan("moon composition"), needs_more("a lunar-science document")]
    library = mock_library({"ids": [[]], "documents": [[]], "metadatas": [[]]})
    result = answer_question(library, "What is the moon made of?", [], client)
    assert "lunar-science document" in result.text
    assert result.sources == []
    assert result.trace[-1]["event"] == "need_upload"
    assert client.sdk.chat.completions.create.call_count == 2


def test_round_limit_requests_validator_named_upload(manager, tokenizer, client):
    client.sdk.chat.completions.create.side_effect = [
        plan("launch"), needs_more("the signed contract"),
        plan("contract"), needs_more("the signed contract"),
        plan("signed contract"), needs_more("the signed contract"),
    ]
    library = mock_library(
        retrieval_result("first", "The launch is in June."),
        retrieval_result("second", "The contract has a termination clause."),
        retrieval_result("third", "The contract has a renewal clause."),
    )
    result = answer_question(library, "What does the contract say?", [], client)
    assert "signed contract" in result.text
    assert result.trace[-1]["reason"] == "budget_exhausted"


@pytest.mark.parametrize("confidence,passes", [(0.0, False), (0.4999, False), (0.5, True), (0.51, True), (1.0, True)])
def test_third_attempt_confidence_boundary_with_duplicate_results(client, confidence, passes):
    client.sdk.chat.completions.create.side_effect = [
        plan(), needs_more("standard agreement", confidence=0.9),
        plan(), needs_more("standard agreement", confidence=0.9),
        plan(), needs_more("standard agreement", confidence=confidence),
        *([answered()] if passes else []),
    ]
    library = mock_library(*[retrieval_result("same", "Launch in June")] * 3)
    result = answer_question(library, "Compare the agreements", [], client)
    checks = [item for item in result.trace if item["event"] == "validate"]
    assert [item["round"] for item in checks] == [1, 2, 3]
    assert [item["accepted"] for item in checks] == [False, False, passes]
    assert checks[-1]["confidence"] == confidence
    assert library.collection.query.call_count == 3
    assert client.sdk.chat.completions.create.call_count == (7 if passes else 6)
    if passes:
        assert result.text.endswith("[1]")
        assert checks[-1]["acceptance_reason"] == "third_attempt_confidence"
        request = client.sdk.chat.completions.create.call_args.kwargs["messages"]
        assert "explicitly identify what remains unknown" in request[0]["content"]
        assert json.loads(request[1]["content"])["missing_evidence"] == ["standard agreement"]
    else:
        assert "Please upload" in result.text and result.sources == []


def test_evidence_cap_does_not_skip_third_validation(client, monkeypatch):
    from rag_chat.workflows import DEFAULT_WORKFLOW
    workflow = DEFAULT_WORKFLOW.model_copy(deep=True)
    workflow.nodes[1].config = {"max_evidence": 1}
    client.sdk.chat.completions.create.side_effect = [
        plan(), needs_more("standard agreement"), plan(), needs_more("standard agreement"),
        plan(), needs_more("standard agreement", confidence=0.5), answered(),
    ]
    library = mock_library(retrieval_result("same", "Launch in June"))
    result = answer_question(library, "Compare agreements", [], client, workflow=workflow)
    assert result.text.endswith("[1]")
    assert library.collection.query.call_count == 1
    assert len([item for item in result.trace if item["event"] == "validate"]) == 3


def test_third_attempt_low_confidence_does_not_pass_sufficient_label(client):
    client.sdk.chat.completions.create.side_effect = [
        plan(), needs_more("standard agreement"), plan(), needs_more("standard agreement"),
        plan(), sufficient(confidence=0.49),
    ]
    library = mock_library(*[retrieval_result("same", "Launch in June")] * 3)
    result = answer_question(library, "Compare agreements", [], client)
    assert "Please upload" in result.text
    assert not result.sources


@pytest.mark.parametrize("confidence", [-0.1, 1.1, "50%", True, None])
def test_invalid_confidence_gets_one_repair_then_fails(client, confidence):
    invalid = needs_more("standard agreement", confidence=confidence)
    client.sdk.chat.completions.create.side_effect = [plan(), invalid, invalid]
    library = mock_library(retrieval_result("same", "Launch in June"))
    with pytest.raises(ChatError, match="evidence validator returned an invalid structured response"):
        answer_question(library, "Compare agreements", [], client)
    assert client.sdk.chat.completions.create.call_count == 3


def test_repaired_response_does_not_count_as_another_validation_attempt(client):
    missing_confidence = completion(json.dumps({"decision": "needs_more_evidence", "missing_evidence": []}))
    client.sdk.chat.completions.create.side_effect = [
        plan(), needs_more("standard agreement"), plan(), needs_more("standard agreement"),
        plan(), missing_confidence, needs_more("standard agreement", confidence=0.5), answered(),
    ]
    library = mock_library(*[retrieval_result("same", "Launch in June")] * 3)
    result = answer_question(library, "Compare agreements", [], client)
    assert result.text.endswith("[1]")
    assert len([item for item in result.trace if item["event"] == "validate"]) == 3
    assert client.sdk.chat.completions.create.call_count == 8


def test_empty_evidence_never_passes_with_high_confidence(client):
    client.sdk.chat.completions.create.side_effect = [plan(), sufficient(confidence=1.0)]
    library = mock_library({"ids": [[]], "documents": [[]], "metadatas": [[]]})
    result = answer_question(library, "Compare agreements", [], client)
    assert "Please upload" in result.text
    assert result.sources == []
    assert result.trace[-2]["accepted"] is False


def test_structured_output_is_repaired_once(manager, tokenizer, client):
    client.sdk.chat.completions.create.side_effect = [
        completion("not JSON"), plan(), sufficient(), answered(),
    ]
    result = answer_question(populated(manager, tokenizer), "When is launch?", [], client)
    assert result.text.endswith("[1]")
    assert client.sdk.chat.completions.create.call_count == 4


def test_second_invalid_structured_output_is_safe(manager, tokenizer, client):
    client.sdk.chat.completions.create.side_effect = [completion("not JSON"), completion("still not JSON")]
    with pytest.raises(ChatError, match="planner returned an invalid structured response"):
        answer_question(populated(manager, tokenizer), "When is launch?", [], client)


def test_no_evidence_and_empty_library(manager, tokenizer, client):
    empty, _ = manager.ensure(None)
    assert answer_question(empty, "What?", [], client).text == NO_EVIDENCE
    client.sdk.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("text", ["Unsupported answer.", "Made up. [9]", "Mixed. [1] [8]", ""])
def test_rejects_missing_or_invalid_citations(manager, tokenizer, client, text):
    client.sdk.chat.completions.create.side_effect = [plan(), sufficient(), answer_completion(text), answer_completion(text)]
    with pytest.raises(ChatError):
        answer_question(populated(manager, tokenizer), "When?", [], client)


@pytest.mark.parametrize("error,match", [
    (AuthenticationError("secret", response=httpx.Response(401, request=httpx.Request("POST", "https://example.org")), body=None), "API key"),
    (RateLimitError("secret", response=httpx.Response(429, request=httpx.Request("POST", "https://example.org")), body=None), "rate limit"),
    (APITimeoutError(request=httpx.Request("POST", "https://example.org")), "reach Groq"),
    (RuntimeError("secret"), "could not complete"),
])
def test_provider_errors_are_safe(manager, tokenizer, client, error, match):
    client.sdk.chat.completions.create.side_effect = error
    with pytest.raises(ChatError, match=match) as caught:
        answer_question(populated(manager, tokenizer), "When?", [], client)
    assert "secret" not in str(caught.value)
