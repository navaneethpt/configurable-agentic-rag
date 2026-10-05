# Workflow authoring

Folio executes the saved graph selected for each question. The server captures the
workflow ID, version, and full definition before starting work, so editing a saved
workflow cannot change an in-flight answer. The assistant message and operation
snapshot record that version. Custom workflows are saved in SQLite under the current
session ID. Other sessions cannot list, edit, delete, or run them. The built-in
default template is available to every session. Because sessions live in memory,
ending a session or restarting Python makes its saved workflows inaccessible, even
though their database rows remain. Older shared custom workflows are retained but
hidden because they cannot safely be assigned to a session.

## Build in the UI

Open **Configure agents** in the header. The catalog lists every registered agent and tool.
Choose a saved workflow, or duplicate one to create a new version of the graph.
Select a node card to edit its settings and choose a destination for every named
output. Set the start node, name, and maximum number of node executions, then save.
Saving activates the workflow for new questions in this browser tab. **Use selected**
switches to an existing saved workflow in the same session without editing it.

The default graph contains these types:

| Type | Kind | Outputs | Settings |
|---|---|---|---|
| `planner` | agent | `next` | Groq model, additional instructions, searches per round |
| `retrieve` | tool | `next` | Results per search, total evidence cap |
| `validate` | agent | `sufficient`, `retry`, `needs_upload` | Groq model, additional instructions, rounds, final confidence |
| `generate` | agent | terminal | Groq model, additional instructions |
| `need_upload` | agent | terminal | none |

The server checks node types, primitive settings, every connection, reachability,
paths to terminal nodes, and required state before saving. Cycles are allowed and
stopped by `max_steps`. A terminal node must return an `Answer`. These checks
prevent many wiring mistakes; an extension's Python handler can still fail at run
time, in which case the chat stream reports an error without committing a reply.

## Question and node dependencies

For a healthy session with uploaded documents, every workflow's selected start
node receives the user's question as `state["question"]`, along with the bounded
conversation context in `state["history"]`. This applies to built-in and custom
registered nodes. When no documents have been uploaded, Folio returns its immediate
no-document response without running any workflow nodes.

The editor shows **Required inputs** and **Produced outputs** for the selected
node. Inputs marked **Provided when the workflow starts** come from the shared
initial-state contract. Other required inputs must be produced by an earlier node
on every path to the selected node. Save errors identify missing inputs and the
registered node types that produce them. These are state outputs, distinct from
the named connections such as `next` and `retry`.

The question does not replace results from earlier agents. For example, retrieval
requires `searches` from a search planner; the answer generator requires
`validation` from an evidence validator. A `retrieve → generate` graph remains
invalid without those results. To omit a node, remove it and reconnect the graph
while preserving the required inputs of every remaining node. No searches or
validation results are fabricated to fill missing dependencies.

## Add an agent or tool in Python

Agent and tool code is trusted server code. A browser user can connect and configure
registered types, but cannot submit Python or shell commands as a node. To add a
type, create a module and register its metadata and factory. For example:

```python
# rag_chat/extensions/clarify.py
from pydantic import Field

from rag_chat.chat import Answer
from rag_chat.workflows import NodeConfig, NodeType, register_node


class ClarifyConfig(NodeConfig):
    question: str = Field(default="Which document should I use?", min_length=1)


def build_clarify(client, on_event, config):
    def run(state):
        question = config.get("question", "Which document should I use?")
        if on_event:
            on_event({"event": "clarify", "question": question})
        return {"answer": Answer(question, [])}, None

    return run


register_node(NodeType(
    key="clarify", label="Ask for clarification", kind="agent",
    description="Ask the user for a missing detail.", outputs=(),
    config_model=ClarifyConfig, factory=build_clarify,
    provides=("answer",),
))
```

Set `FOLIO_NODE_MODULES=rag_chat.extensions.clarify` in `.env`, restart the
single-worker backend, and reopen the builder. The new type and its editable
`question` field come from `/api/node-types`; no frontend code change is needed.
The module name must be importable by Python. Multiple modules can be separated
by commas.

A factory receives the request's Groq client, an optional event callback, and the
validated settings dictionary. It returns a handler that receives graph state and
returns `(state_updates, output_name)`. Use `None` for terminal nodes. Declare
`requires` for state keys the handler reads and `provides` for keys it writes.
The shared initial state contains `library`, `question`, `history`, `round`,
`feedback`, `evidence`, `trace`, `steps`, and `route`. Validation and execution use
the same contract, also exposed as `initial_inputs` in each catalog entry. Empty
evidence and feedback lists, round and step counters at zero, and an empty route
are initialization values rather than results from an earlier agent.
The default agents add `searches`, `validation`, and
`answer` as they run. Settings must be flat strings, integers, numbers, or booleans
so the generic UI can render them.

Existing saved workflows that reference an extension need that module loaded when
they run. The default workflow remains available if the extension is removed.

## API shape

`GET /api/node-types` returns the catalog and JSON schemas, including `requires`,
`provides`, and the additive `initial_inputs` field. Saved graph schemas and
extension handler signatures are unchanged. Every workflow endpoint
requires the current `X-Session-ID` header; `GET /api/workflows` returns only that
session's saved definitions and the default template. A new workflow is posted to
`/api/workflows` as:

```json
{
  "name": "One-pass research",
  "entry": "planner",
  "max_steps": 20,
  "nodes": [
    {"id": "planner", "type": "planner", "config": {}, "transitions": {"next": "retrieve"}},
    {"id": "retrieve", "type": "retrieve", "config": {}, "transitions": {"next": "validate"}},
    {"id": "validate", "type": "validate", "config": {"max_rounds": 1},
     "transitions": {"sufficient": "generate", "retry": "planner", "needs_upload": "need_upload"}},
    {"id": "generate", "type": "generate", "config": {}, "transitions": {}},
    {"id": "need_upload", "type": "need_upload", "config": {}, "transitions": {}}
  ]
}
```

`PUT /api/workflows/{id}` takes `{ "version": 1, "workflow": { ... } }`.
A stale version returns 409. Chat requests include the saved ID:
`{ "question": "When is launch?", "workflow_id": "..." }`.
The built-in `default` workflow can be copied but not edited or deleted.
