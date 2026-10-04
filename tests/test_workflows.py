import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import Field
import pytest

from conftest import completion
from rag_chat.chat import Answer, ChatError, answer_question
from rag_chat.indexing import ingest
from rag_chat.workflows import (DEFAULT_WORKFLOW, NodeConfig, NodeType, WorkflowConflict,
                                WorkflowDraft, WorkflowError, WorkflowNode, WorkflowNotFound, WorkflowStore,
                                node_catalog, register_node, unregister_node, validate_workflow)


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


def test_graph_validation_rejects_broken_connections_and_missing_state():
    incomplete = DEFAULT_WORKFLOW.model_copy(deep=True)
    incomplete.nodes[0].transitions = {}
    with pytest.raises(WorkflowError, match="connect these outputs"):
        validate_workflow(incomplete)

    disconnected = DEFAULT_WORKFLOW.model_copy(deep=True)
    disconnected.nodes.append(WorkflowNode(id="generate_1", type="generate"))
    with pytest.raises(WorkflowError, match="Nodes not connected to start .*generate_1"):
        validate_workflow(disconnected)

    too_early = WorkflowDraft(name="Invalid", entry="generate", nodes=[
        WorkflowNode(id="generate", type="generate"),
    ])
    with pytest.raises(WorkflowError, match="earlier node"):
        validate_workflow(too_early)

    unknown_setting = configured_default(max_rounds=1, shell_command="echo never")
    with pytest.raises(WorkflowError, match="Invalid settings"):
        validate_workflow(unknown_setting)

    too_few_steps = DEFAULT_WORKFLOW.model_copy(deep=True)
    too_few_steps.max_steps = 2
    with pytest.raises(WorkflowError, match="Maximum steps"):
        validate_workflow(too_few_steps)


def test_configured_planner_and_retriever_control_searches(client):
    draft = DEFAULT_WORKFLOW.model_copy(deep=True)
    draft.nodes[0].config = {"max_searches": 2}
    draft.nodes[1].config = {"max_results_per_search": 1}
    client.chat.completions.create.side_effect = [
        completion(json.dumps({"searches": [
            {"query": query, "purpose": "Find launch date"} for query in ("a", "b", "c")
        ]})),
        completion(json.dumps({"decision": "sufficient", "confidence": 0.9, "missing_evidence": []})),
        completion("Launch in June. [1]"),
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
            return {"answer": Answer(config.get("response", "Configured answer"), [])}, None
        return run

    register_node(NodeType("fixed_answer", "Fixed answer", "agent", "Example extension.",
                           (), FixedAnswerConfig, factory, provides=("answer",)))
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
        client.chat.completions.create.assert_not_called()
    finally:
        unregister_node("fixed_answer")


def test_step_limit_stops_repeating_extension(manager, tokenizer, client):
    class EmptyConfig(NodeConfig):
        pass

    def factory(model, on_event, config):
        return lambda state: ({}, "again")

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
