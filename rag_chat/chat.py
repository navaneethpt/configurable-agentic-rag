"""Generic sequential workflow execution with independent registered handlers."""
from copy import deepcopy
import json
from typing import Any, Callable, TypedDict
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError
from .agent_contracts import (AgentOutput, Answer, ChatError, NodeExecution, NodeInput, NodeResult,
                              SessionServices, Source)
from .workflows import DEFAULT_WORKFLOW, WorkflowDraft, initial_workflow_state, node_type, validate_workflow
from .model_client import MODEL
from .agents import common  # registers all built-in factories
from .agents.common import NO_EVIDENCE


class AgentState(TypedDict, total=False):
    library: Any
    question: str
    history: list[dict[str, str]]
    results: list[NodeExecution]
    answer: Answer
    trace: list[dict[str, Any]]
    steps: int
    route: str


def _context_history(history: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": turn["role"], "content": turn["content"][:8000]}
        for turn in history[-10:]
        if turn.get("role") in {"user", "assistant"} and isinstance(turn.get("content"), str)
    ]


def build_agentic_graph(client, on_event: Callable[[dict[str, Any]], None] | None = None,
                        workflow: WorkflowDraft | None = None):
    workflow = validate_workflow(workflow or DEFAULT_WORKFLOW)
    graph = StateGraph(AgentState)
    for instance in workflow.nodes:
        definition = node_type(instance.type)
        settings = definition.config_model.model_validate(instance.config).model_dump()
        handler = definition.factory(client, on_event, settings)

        def run(state, *, handler=handler, instance=instance, definition=definition):
            step = state.get("steps", 0) + 1
            if step > workflow.max_steps:
                raise ChatError("The workflow exceeded its step limit. Edit the workflow and try again.")
            inputs = NodeInput(state["question"], deepcopy(state["history"]),
                tuple(deepcopy(state.get("results", []))), SessionServices(state["library"]),
                instance.id, step, not instance.transitions)
            if on_event:
                on_event({"event": "node_start", "node": instance.id, "node_type": definition.key,
                          "label": definition.label, "step": step})
            result = handler(inputs)
            if not isinstance(result, NodeResult) or not isinstance(result.output, AgentOutput):
                raise ChatError(f"Workflow node {instance.id} returned an invalid result.")
            if result.output.kind != definition.output_kind:
                raise ChatError(f"Workflow node {instance.id} returned an invalid output kind.")
            try:
                data = json.loads(json.dumps(result.output.data, allow_nan=False))
                if not isinstance(data, (dict, list)):
                    raise ValueError("Output must be an object or array")
                if definition.output_model:
                    model = definition.output_model
                    validated = model.validate_python(data) if hasattr(model, "validate_python") else model.model_validate(data)
                    data = model.dump_python(validated, mode="json") if hasattr(model, "dump_python") else validated.model_dump(mode="json")
            except (ValueError, TypeError, ValidationError):
                raise ChatError(f"Workflow node {instance.id} returned output that does not match its schema.") from None
            if instance.transitions and result.route not in instance.transitions:
                raise ChatError(f"Workflow node {instance.id} returned an invalid transition.")
            if inputs.is_terminal and result.route is not None:
                raise ChatError(f"Terminal workflow node {instance.id} returned a transition.")
            if inputs.is_terminal and not isinstance(result.response, Answer):
                raise ChatError(f"Terminal workflow node {instance.id} did not return an answer.")
            event = {**result.details, "event": definition.key, "node": instance.id, "node_type": definition.key,
                     "label": definition.label, "step": step, "terminal": inputs.is_terminal,
                     "output_kind": result.output.kind}
            if on_event:
                on_event(event)
            trace = [*state.get("trace", []), event]
            update = {"results": [*state.get("results", []),
                NodeExecution(instance.id, definition.key, step, AgentOutput(result.output.kind, data))],
                "steps": step, "route": result.route or "", "trace": trace}
            if inputs.is_terminal:
                update["answer"] = Answer(result.response.text, deepcopy(result.response.sources), trace)
            return update

        graph.add_node(instance.id, run)
        if instance.transitions:
            graph.add_conditional_edges(instance.id, lambda state: state["route"], instance.transitions)
        else:
            graph.add_edge(instance.id, END)
    graph.add_edge(START, workflow.entry)
    return graph.compile()


def answer_question(library, question: str, history: list[dict], client,
                    on_event: Callable[[dict[str, Any]], None] | None = None,
                    workflow: WorkflowDraft | None = None) -> Answer:
    question = question.strip()
    if not question or len(question) > 2000:
        raise ChatError("Enter a question between 1 and 2,000 characters.")
    if not library.healthy:
        raise ChatError("This library needs to be reset. Select Clear session and upload again.")
    try:
        if library.collection.count() == 0:
            return Answer(NO_EVIDENCE, [])
    except Exception:
        raise ChatError("Cannot read the document library. Clear the session and upload again.") from None
    selected = workflow or DEFAULT_WORKFLOW
    result = build_agentic_graph(client, on_event, selected).invoke(
        initial_workflow_state(library, question, _context_history(history)),
        config={"recursion_limit": selected.max_steps * 2 + 10})
    answer = result.get("answer")
    if not isinstance(answer, Answer):
        raise ChatError("The workflow ended without an answer. Edit the workflow and try again.")
    return answer
