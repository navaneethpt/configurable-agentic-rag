import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import Field
import pytest

from conftest import completion, answer_completion
from rag_chat.chat import Answer, ChatError, NO_EVIDENCE, answer_question, build_agentic_graph
from rag_chat.agent_contracts import AgentOutput, NodeInput, NodeResult
from rag_chat.agents.common import ResponseOutput
from pydantic import TypeAdapter
from rag_chat.indexing import ingest
from rag_chat.workflows import (DEFAULT_WORKFLOW, NodeConfig, NodeType, WorkflowConflict,
                                WorkflowDraft, WorkflowError, WorkflowNode, WorkflowNotFound, WorkflowStore,
                                node_catalog, register_node, unregister_node, validate_workflow, initial_workflow_state)


def configured_default(**validator_settings):
    draft = DEFAULT_WORKFLOW.model_copy(deep=True)
    draft.name = "Focused research"
    next(node for node in draft.nodes if node.id == "validate").config = validator_settings
    return draft


def test_workflows_persist_and_reject_stale_updates(tmp_path):
    path = tmp_path / "workflows.sqlite3"
    first = WorkflowStore(path)
    created = first.create("session-a", configured_default(max_rounds=1, final_confidence=0.9))
    assert created.version == 1
    assert WorkflowStore(path).get("session-a", created.id).nodes[2].config == {"max_rounds": 1, "final_confidence": 0.9}
    assert [item.id for item in first.list("session-b")] == ["default"]
    with pytest.raises(WorkflowNotFound):
        first.get("session-b", created.id)
    updated = first.update("session-a", created.id, configured_default(max_rounds=2), created.version)
    assert updated.version == 2
    with pytest.raises(WorkflowConflict):
        first.update("session-a", created.id, configured_default(max_rounds=3), created.version)
    with pytest.raises(WorkflowNotFound):
        first.update("session-b", created.id, configured_default(max_rounds=3), updated.version)
    with pytest.raises(WorkflowNotFound):
        first.delete("session-b", created.id)
    first.delete("session-a", created.id)
    assert [item.id for item in first.list("session-a")] == ["default"]


def test_old_shared_workflows_are_hidden_after_schema_migration(tmp_path):
    path = tmp_path / "workflows.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE workflows (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
            body TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        db.execute("INSERT INTO workflows VALUES (?, ?, 1, ?, ?)",
                   ("old-shared", "Old shared", DEFAULT_WORKFLOW.model_dump_json(), "2026-01-01"))
    store = WorkflowStore(path)
    assert [item.id for item in store.list("session-a")] == ["default"]
    with pytest.raises(WorkflowNotFound):
        store.get("session-a", "old-shared")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT session_id FROM workflows WHERE id = 'old-shared'").fetchone() == (None,)


def test_graph_validation_keeps_structural_checks_without_agent_dependencies():
    incomplete = DEFAULT_WORKFLOW.model_copy(deep=True)
    incomplete.nodes[0].transitions = {}
    with pytest.raises(WorkflowError, match="connect these outputs"):
        validate_workflow(incomplete)
    disconnected = DEFAULT_WORKFLOW.model_copy(deep=True)
    disconnected.nodes.append(WorkflowNode(id="generate_1", type="generate"))
    with pytest.raises(WorkflowError, match="Nodes not connected to start .*generate_1"):
        validate_workflow(disconnected)
    with pytest.raises(WorkflowError, match="Invalid settings"):
        validate_workflow(configured_default(shell_command="echo never"))
    too_few_steps = DEFAULT_WORKFLOW.model_copy(deep=True)
    too_few_steps.max_steps = 2
    with pytest.raises(WorkflowError, match="Maximum steps"):
        validate_workflow(too_few_steps)
    nonterminal = WorkflowDraft(name="No terminal", entry="retrieve", nodes=[
        WorkflowNode(id="retrieve", type="retrieve"),
    ])
    with pytest.raises(WorkflowError, match="connect these outputs"):
        validate_workflow(nonterminal)


def short_workflow(start="generate"):
    nodes = [] if start == "generate" else [WorkflowNode(id=start, type=start, transitions={
        port: "generate" for port in node_catalog_by_type(start)["outputs"]})]
    return WorkflowDraft(name="Independent agents", entry=start, nodes=[*nodes, WorkflowNode(id="generate", type="generate")])


def node_catalog_by_type(key):
    return next(item for item in node_catalog() if item["type"] == key)


@pytest.mark.parametrize("start", ["generate", "planner", "retrieve", "validate", "need_upload"])
def test_every_builtin_can_start_and_save_without_upstream_dependencies(tmp_path, start):
    saved = WorkflowStore(tmp_path / "workflows.sqlite3").create("session-a", short_workflow(start))
    assert saved.entry == start


@pytest.mark.parametrize("start", ["generate", "planner", "retrieve", "validate"])
def test_independent_workflows_run_only_configured_nodes(client, start):
    collection = Mock()
    collection.count.return_value = 1
    collection.query.return_value = {"ids": [["one"]], "documents": [["Launch in June."]],
        "metadatas": [[{"filename": "launch.txt", "location": "Text"}]]}
    prefix = [completion(json.dumps({"searches": [{"query": "launch", "purpose": "Find date"}]}))] if start == "planner" else [
        completion(json.dumps({"decision": "sufficient", "confidence": 0.9}))] if start == "validate" else []
    client.sdk.chat.completions.create.side_effect = [*prefix, answer_completion("Launch in June. [1]") if start == "retrieve"
        else answer_completion("Please upload the release schedule.", "missing_context")]
    result = answer_question(SimpleNamespace(healthy=True, collection=collection), "When is launch?", [], client,
        workflow=short_workflow(start))
    assert [item["event"] for item in result.trace] == ([start] if start != "generate" else []) + ["generate"]
    if start == "retrieve":
        assert result.sources[0].filename == "launch.txt"
        assert collection.query.call_args.kwargs["query_texts"] == ["When is launch?"]
    else:
        assert result.text == "Please upload the release schedule." and result.sources == []
        collection.query.assert_not_called()
    if start == "validate":
        assert result.trace[0]["accepted"] is False and result.trace[0]["confidence"] == 0
    assert client.sdk.chat.completions.create.call_count == len(prefix) + 1
    payload = json.loads(client.sdk.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert payload["question"] == "When is launch?"


def test_custom_start_receives_original_context_and_empty_prior_results(client):
    received = []
    def factory(model, on_event, config):
        def run(inputs):
            received.append(inputs)
            data = {"status": "missing_context", "text": inputs.question, "sources": []}
            return NodeResult(AgentOutput("answer", data), None, Answer(inputs.question, []))
        return run
    register_node(NodeType("context_answer", "Context answer", "agent", "Test entry context.", (), NodeConfig,
        factory, output_kind="answer", output_model=ResponseOutput, terminal_role="generator"))
    try:
        workflow = WorkflowDraft(name="Context", entry="custom", nodes=[WorkflowNode(id="custom", type="context_answer")])
        library = SimpleNamespace(healthy=True, collection=Mock())
        library.collection.count.return_value = 1
        history = [{"role": "user", "content": "Which launch?"}]
        assert answer_question(library, "When?", history, client, workflow=workflow).text == "When?"
        inputs = received[0]
        assert isinstance(inputs, NodeInput)
        assert inputs.question == "When?" and inputs.history == history
        assert inputs.services.library is library and inputs.prior_results == ()
        assert inputs.step == 1 and inputs.is_terminal
        history.append({"role": "user", "content": "Another turn"})
        assert len(inputs.history) == 1
        library.collection.count.return_value = 0
        assert answer_question(library, "When?", history, client, workflow=workflow).text == NO_EVIDENCE
        assert len(received) == 1
        client.sdk.chat.completions.create.assert_not_called()
    finally:
        unregister_node("context_answer")


def test_intermediate_results_are_retained_but_do_not_become_user_responses(client):
    workflow = WorkflowDraft(name="Draft then request", entry="generate", nodes=[
        WorkflowNode(id="generate", type="generate", transitions={"next": "need_upload"}),
        WorkflowNode(id="need_upload", type="need_upload", transitions={"next": "final"}),
        WorkflowNode(id="final", type="generate"),
    ])
    client.sdk.chat.completions.create.side_effect = [answer_completion("Draft context request", "missing_context"),
        answer_completion("Final context request", "missing_context")]
    library = SimpleNamespace(healthy=True, collection=Mock())
    library.collection.count.return_value = 1
    result = answer_question(library, "When?", [], client, workflow=workflow)
    assert result.text == "Final context request"
    assert [item["terminal"] for item in result.trace] == [False, False, True]
    payload = json.loads(client.sdk.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert [item["node_id"] for item in payload["prior_outputs"]] == ["generate", "need_upload"]
    assert payload["excerpts"] == []  # Draft text cannot become document evidence.
    assert payload["prior_outputs"][0]["data"]["text"] == "Draft context request"


@pytest.mark.parametrize("status,text", [("answered", "Launch in June. [1]"),
                                         ("missing_context", "Please provide more information. [1]")])
def test_generator_without_passages_rejects_fabricated_citations(client, status, text):
    library = SimpleNamespace(healthy=True, collection=Mock())
    library.collection.count.return_value = 1
    client.sdk.chat.completions.create.side_effect = [answer_completion(text, status), answer_completion(text, status)]
    with pytest.raises(ChatError, match="source references"):
        answer_question(library, "When?", [], client, workflow=short_workflow())
    library.collection.query.assert_not_called()


def test_generator_repairs_missing_citations_once_using_available_passages(client):
    library = SimpleNamespace(healthy=True, collection=Mock())
    library.collection.count.return_value = 1
    library.collection.query.return_value = {
        "ids": [["launch"]], "documents": [["Launch is in June."]],
        "metadatas": [[{"filename": "launch.txt", "location": "page 1"}]], "distances": [[0.1]],
    }
    client.sdk.chat.completions.create.side_effect = [
        answer_completion("Launch in June."), answer_completion("Launch in June. [1]"),
    ]
    result = answer_question(library, "When?", [], client, workflow=short_workflow("retrieve"))
    assert result.text == "Launch in June. [1]" and len(result.sources) == 1
    assert client.sdk.chat.completions.create.call_count == 2
    correction = json.loads(client.sdk.chat.completions.create.call_args.kwargs["messages"][-2]["content"])
    assert correction["available_source_numbers"] == [1]
    assert len(result.trace) == 2


def test_repeated_nodes_keep_ordered_outputs_without_overwrites(client):
    seen = []
    def factory(model, on_event, config):
        def run(inputs):
            seen.append(inputs)
            return NodeResult(AgentOutput("numbers", [inputs.step]), "again" if inputs.step < 3 else "done")
        return run
    register_node(NodeType("counter_agent", "Counter", "tool", "Count executions.", ("again", "done"), NodeConfig,
        factory, output_kind="numbers", output_model=TypeAdapter(list[int])))
    try:
        workflow = WorkflowDraft(name="Repeat", entry="counter", nodes=[
            WorkflowNode(id="counter", type="counter_agent", transitions={"again": "counter", "done": "need_upload"}),
            WorkflowNode(id="need_upload", type="need_upload"),
        ])
        library = SimpleNamespace(healthy=True, collection=Mock())
        library.collection.count.return_value = 1
        result = build_agentic_graph(client, workflow=workflow).invoke(initial_workflow_state(library, "When?", []))
        assert [item.node_id for item in result["results"]] == ["counter", "counter", "counter", "need_upload"]
        assert [item.output.data for item in result["results"][:3]] == [[1], [2], [3]]
        assert [len(inputs.prior_results) for inputs in seen] == [0, 1, 2]
        assert [item.step for item in result["results"]] == [1, 2, 3, 4]
        assert all(inputs.question == "When?" for inputs in seen)
    finally:
        unregister_node("counter_agent")


def test_validator_retry_rounds_work_without_a_planner(client):
    workflow = short_workflow("retrieve")
    workflow.nodes[0].transitions = {"next": "validate"}
    workflow.nodes.extend([
        WorkflowNode(id="validate", type="validate", transitions={"sufficient": "generate", "retry": "retrieve",
                                                                  "needs_upload": "need_upload"}),
        WorkflowNode(id="need_upload", type="need_upload"),
    ])
    collection = Mock()
    collection.count.return_value = 1
    collection.query.return_value = {"ids": [["one"]], "documents": [["Launch in June."]],
        "metadatas": [[{"filename": "launch.txt", "location": "Text"}]]}
    client.sdk.chat.completions.create.side_effect = [completion(json.dumps({"decision": "needs_more_evidence",
        "confidence": 0.2, "missing_evidence": ["the signed contract"]}))] * 3
    result = answer_question(SimpleNamespace(healthy=True, collection=collection), "When?", [], client, workflow=workflow)
    assert "signed contract" in result.text
    assert [item["round"] for item in result.trace if item["event"] == "validate"] == [1, 2, 3]
    assert collection.query.call_count == 3 and client.sdk.chat.completions.create.call_count == 3
    assert all(call.kwargs["query_texts"] == ["When?"] for call in collection.query.call_args_list)


def test_new_retrieval_does_not_reuse_an_earlier_validation(client):
    workflow = WorkflowDraft(name="New evidence", entry="retrieve", nodes=[
        WorkflowNode(id="retrieve", type="retrieve", transitions={"next": "validate"}),
        WorkflowNode(id="validate", type="validate", transitions={port: "second" for port in
            ("sufficient", "retry", "needs_upload")}),
        WorkflowNode(id="second", type="retrieve", transitions={"next": "generate"}),
        WorkflowNode(id="generate", type="generate"),
    ])
    collection = Mock()
    collection.count.return_value = 2
    collection.query.side_effect = [{"ids": [[identity]], "documents": [[text]],
        "metadatas": [[{"filename": "launch.txt", "location": "Text"}]]}
        for identity, text in [("one", "Launch in June."), ("two", "Contract not signed.")]]
    client.sdk.chat.completions.create.side_effect = [completion(json.dumps({"decision": "sufficient", "confidence": 0.9})),
        answer_completion("Launch in June. [1]")]
    answer_question(SimpleNamespace(healthy=True, collection=collection), "When?", [], client, workflow=workflow)
    payload = json.loads(client.sdk.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert payload["validation"] is None
    assert len(payload["excerpts"]) == 2
    assert [item["node_id"] for item in payload["prior_outputs"]] == ["retrieve", "validate", "second"]


def test_invalid_output_schema_and_mutated_input_do_not_corrupt_prior_results(client):
    def factory(model, on_event, config):
        def run(inputs):
            if inputs.prior_results:
                inputs.prior_results[0].output.data[0] = 99
                inputs.history[0]["content"] = "changed"
            return NodeResult(AgentOutput("numbers", [inputs.step]), "next")
        return run
    register_node(NodeType("isolated_agent", "Isolated", "tool", "Test isolation.", ("next",), NodeConfig, factory,
                           output_kind="numbers", output_model=TypeAdapter(list[int])))
    try:
        workflow = WorkflowDraft(name="Isolated", entry="first", nodes=[
            WorkflowNode(id="first", type="isolated_agent", transitions={"next": "second"}),
            WorkflowNode(id="second", type="isolated_agent", transitions={"next": "need_upload"}),
            WorkflowNode(id="need_upload", type="need_upload"),
        ])
        history = [{"role": "user", "content": "original"}]
        result = build_agentic_graph(client, workflow=workflow).invoke(initial_workflow_state(None, "When?", history))
        assert [item.output.data for item in result["results"][:2]] == [[1], [2]]
        assert result["history"] == history == [{"role": "user", "content": "original"}]
    finally:
        unregister_node("isolated_agent")
    register_node(NodeType("invalid_agent", "Invalid", "tool", "Test schema.", ("next",), NodeConfig,
        lambda *_: lambda inputs: NodeResult(AgentOutput("numbers", ["bad"])),
        output_kind="numbers", output_model=TypeAdapter(list[int])))
    try:
        workflow.nodes[0].type = workflow.nodes[1].type = "invalid_agent"
        with pytest.raises(ChatError, match="does not match its schema"):
            build_agentic_graph(client, workflow=workflow).invoke(initial_workflow_state(None, "When?", []))
    finally:
        unregister_node("invalid_agent")


def test_configured_planner_and_retriever_control_searches(client):
    draft = DEFAULT_WORKFLOW.model_copy(deep=True)
    draft.nodes[0].config = {"max_searches": 2}
    draft.nodes[1].config = {"max_results_per_search": 1}
    client.sdk.chat.completions.create.side_effect = [
        completion(json.dumps({"searches": [
            {"query": query, "purpose": "Find launch date"} for query in ("a", "b", "c")
        ]})),
        completion(json.dumps({"decision": "sufficient", "confidence": 0.9, "missing_evidence": []})),
        answer_completion("Launch in June. [1]"),
    ]
    collection = Mock()
    collection.count.return_value = 3
    collection.query.return_value = {"ids": [["one"]], "documents": [["Launch in June."]],
                                     "metadatas": [[{"filename": "launch.txt", "location": "Text"}]]}
    library = SimpleNamespace(healthy=True, collection=collection)
    assert answer_question(library, "When?", [], client, workflow=draft).text == "Launch in June. [1]"
    assert collection.query.call_count == 2
    assert all(call.kwargs["n_results"] == 1 for call in collection.query.call_args_list)


def test_registered_agent_appears_in_catalog_and_runs(manager, tokenizer, client):
    class FixedAnswerConfig(NodeConfig):
        response: str = Field(default="Configured answer", min_length=1)

    def factory(model, on_event, config):
        def run(state):
            text = config["response"]
            return NodeResult(AgentOutput("answer", {"status": "missing_context", "text": text, "sources": []}), None, Answer(text, []))
        return run

    register_node(NodeType("fixed_answer", "Fixed answer", "agent", "Example extension.",
                           (), FixedAnswerConfig, factory, output_kind="answer", output_model=ResponseOutput, terminal_role="generator"))
    try:
        assert any(item["type"] == "fixed_answer" and "response" in item["config_schema"]["properties"]
                   for item in node_catalog())
        workflow = WorkflowDraft(name="Extension", entry="custom", nodes=[
            WorkflowNode(id="custom", type="fixed_answer", config={"response": "From extension"}),
        ])
        library, _ = manager.ensure(None)
        ingest(library, "launch.txt", b"Launch in June.", lambda: tokenizer)
        result = answer_question(library, "When?", [], client, workflow=workflow)
        assert result.text == "From extension"
        client.sdk.chat.completions.create.assert_not_called()
    finally:
        unregister_node("fixed_answer")


def test_bundled_clarification_extension_uses_the_new_contract(client):
    import importlib
    import sys
    name = "rag_chat.extensions.clarify"
    if name in sys.modules:
        importlib.reload(sys.modules[name])
    else:
        importlib.import_module(name)
    try:
        catalog = node_catalog_by_type("clarify")
        assert catalog["terminal_role"] == "evidence_request"
        assert catalog["output_kind"] == "evidence_request"
        workflow = WorkflowDraft(name="Clarify", entry="custom", nodes=[
            WorkflowNode(id="custom", type="clarify", config={"question": "Please upload the contract."}),
        ])
        library = SimpleNamespace(healthy=True, collection=Mock())
        library.collection.count.return_value = 1
        assert answer_question(library, "When?", [], client, workflow=workflow).text == "Please upload the contract."
        client.sdk.chat.completions.create.assert_not_called()
    finally:
        unregister_node("clarify")


def test_step_limit_stops_repeating_extension(manager, tokenizer, client):
    class EmptyConfig(NodeConfig):
        pass

    def factory(model, on_event, config):
        return lambda inputs: NodeResult(AgentOutput("custom", {}), "again")

    register_node(NodeType("loop_agent", "Loop agent", "agent", "Test a bounded loop.",
                           ("again", "done"), EmptyConfig, factory))
    try:
        workflow = WorkflowDraft(name="Loop", entry="loop", max_steps=3, nodes=[
            WorkflowNode(id="loop", type="loop_agent", transitions={"again": "loop", "done": "need_upload"}),
            WorkflowNode(id="need_upload", type="need_upload"),
        ])
        library, _ = manager.ensure(None)
        ingest(library, "launch.txt", b"Launch in June.", lambda: tokenizer)
        with pytest.raises(ChatError, match="step limit"):
            answer_question(library, "When?", [], client, workflow=workflow)
    finally:
        unregister_node("loop_agent")
