# Workflow authoring

Folio executes the saved graph selected for each question. The server captures its
ID, version, and definition before execution, so edits cannot change an in-flight
answer. Workflows are saved in SQLite under the current session ID. Other sessions
cannot list, edit, delete, or execute them. The default template is shared.
Sessions live in memory: expiry or restarting Python makes their saved workflows
inaccessible, although their database rows remain. No authentication is required.

## Compose agents in the UI

Open **Configure agents**, copy a workflow, and add or remove registered nodes.
Choose a start node, connect every named routing outcome, and save. Saving activates
that version for new questions in this browser tab. **Use selected** activates a
previously saved version from the same session.

Every node receives the original `question`, bounded conversation `history`, and
all earlier outputs along the executed path. Missing upstream outputs are handled
by the node; saving does not require a planner or validator to precede another
node. The inspector shows accepted inputs, missing-input behavior, and produced
output fields. A healthy session with processed documents is still required to ask
questions; the API rejects requests without an uploaded document.

Only an **Answer generator** or **Request more evidence** may finish a workflow.
Their **After this node** control chooses whether to finish or continue. When they
continue, their output is available to later nodes; their response is not committed
to the conversation. Only the terminal response becomes the assistant message.

| Node | Available inputs it uses | Output | Behavior without earlier outputs |
|---|---|---|---|
| Search planner | Question, history, feedback and earlier results | `search_queries` | Plan searches from the question |
| Document retrieval | Search queries, question, earlier passages | `passages` with source metadata and queries used | Search with the original question |
| Evidence validator | Question, passages and queries | `validation`, confidence and missing evidence | Report insufficient evidence when passages are absent |
| Answer generator | Question, history, passages, optional validation and earlier results | `answer` with `answered` or `missing_context` status | Ask for the context needed to answer |
| Request more evidence | Question and available evidence gaps | `evidence_request` | Request relevant documents |

Examples after document upload:

- **Generator alone:** the LLM explains missing context; uploaded documents have not been retrieved automatically.
- **Retrieval → Generator:** search directly using the question and generate a cited answer from passages, without a planner or validator.
- **Planner → Generator:** plan searches, then explain missing context because no passages were retrieved.
- **Validator → Generator:** identify insufficient evidence, then request missing context.

The server validates node types, settings, connections, reachability, a path to an
eligible terminal, and execution limits. Sequential conditional routing and retry
loops are supported; parallel branches are not executed. `max_steps` bounds loops.
Nodes are never silently added. Operational failures, invalid results, and exhausted
step limits produce errors without committing a partial assistant reply.

## Results and grounding

Each execution adds a result containing its node ID, type, step, and a tagged
structured output. The payload can be an object or an array. Results remain ordered;
repeating a node does not overwrite previous executions. Each handler receives a
copy of earlier results, so modifying its input cannot rewrite prior output.

Retrieval accumulates unique passages within the configured evidence cap. Research
rounds count retrieval executions, including attempts that find no new passages.
The default validator retries up to three rounds. Its final confidence threshold
can permit a partial answer; generation must identify evidence gaps. A validation
result predating a later retrieval is not applied to that newer evidence.

Generators use passages as factual evidence. Generated drafts are not source
material. Supported answers must cite available passage numbers. Missing-context
responses contain no factual answer or citations. A citation identifies a passage;
it does not independently prove every claim is correct. The generator makes one
correction attempt when citation checks fail; an invalid corrected response is
still rejected.

## Register an agent or tool

Agents are trusted Python code registered on the server. Browser users can configure
and connect registered types but cannot submit Python or shell commands. Built-ins
use the same registry and contracts as extensions; the runner contains no built-in
agent dispatch logic.

```python
# rag_chat/extensions/clarify.py
from pydantic import Field
from rag_chat.agent_contracts import AgentOutput, Answer, NodeResult
from rag_chat.agents.common import EvidenceRequestOutput
from rag_chat.workflows import NodeConfig, NodeType, register_node

class ClarifyConfig(NodeConfig):
    request: str = Field(default="Please upload the relevant contract.", min_length=1)

def build_clarify(client, on_event, config):
    def run(inputs):
        # inputs.question, inputs.history, inputs.prior_results are always available.
        # inputs.services.library is the session library, not an LLM input.
        text = config["request"]
        return NodeResult(
            output=AgentOutput("evidence_request", {"text": text, "missing_evidence": []}),
            route=None if inputs.is_terminal else "next",
            response=Answer(text, []),
        )
    return run

register_node(NodeType(
    key="clarify", label="Request contract", kind="agent",
    description="Request the contract needed for this task.", outputs=("next",),
    config_model=ClarifyConfig, factory=build_clarify,
    accepted_inputs=("question", "earlier outputs"),
    output_kind="evidence_request", output_model=EvidenceRequestOutput,
    missing_input_behavior="Request the relevant contract.",
    terminal_role="evidence_request",
))
```

Set `FOLIO_NODE_MODULES=rag_chat.extensions.clarify`, restart the backend, and reopen
the builder. Multiple importable modules can be separated by commas. Registered
nodes and flat primitive configuration fields appear automatically.

Factory arguments remain `(client, on_event, config)`. The client is now a
provider-neutral, context-managed `ModelClient`, rather than the raw Groq SDK.
Use `client.complete(messages, model="default", max_tokens=1024, schema=None)`
for text, or the shared `rag_chat.model_client._structured` helper for Pydantic
outputs and bounded repair. `schema` is a JSON Schema dictionary; Gemini uses
native structured output, while Groq retains the shared schema prompt. The
client returns text and raises safe `ChatError` on provider failures. Extensions
that used `client.chat.completions.create` must adopt this interface; the runner
owns client cleanup. For example:

```python
from rag_chat.model_client import _structured
from rag_chat.agents.common import GenerationOutput

output = _structured(client, GenerationOutput, messages, "custom generator",
                     model=config.get("model", "default"))
```

`LLM_PROVIDER` selects Groq or Gemini application-wide; Gemini is the default
when unset, using `gemini-3.8-flash`. `LLM_MODEL` supplies
the default; agents may override it with a model ID from that provider. The
legacy `openai/gpt-oss-20b` value follows the selected default. Other overrides
are preserved and must be updated when switching providers. No graph/database
migration is needed. See [Model providers](../README.md#model-providers) for keys,
timeouts, configuration errors, and deployment settings.

**The handler contract has
changed:** extensions must accept `NodeInput` and return `NodeResult`, replacing the
old shared-state dictionary and `(state_updates, output_name)` tuple. Existing
extensions must be updated before use. `requires`/`provides` declarations are
replaced by accepted inputs and output schemas; accepted inputs do not create
save-time dependencies.

`NodeInput` contains `question`, `history`, `prior_results`, `services`, `node_id`,
`step`, and `is_terminal`. `NodeExecution` contains `node_id`, `node_type`, `step`,
and `output`. An `AgentOutput` contains `kind` and object-or-array `data`.
`NodeResult` contains the output, route, optional response, and optional trace details.
Use a Pydantic model or `TypeAdapter` for `output_model`; the runner validates the
payload before recording it. Reuse the standard output schemas when producing
queries, passages, validation, answers, or evidence requests consumed by built-ins.
Only nodes declared as generators or evidence requests may end a workflow; terminal
handlers must return an `Answer` and `route=None`. Continuing handlers choose a
registered routing outcome. Any `response` from a continuing handler is ignored.

## API and compatibility

`GET /api/node-types` exposes configuration schemas, accepted inputs, common input
schema, output kind/schema, missing-input behavior, routing outcomes, and terminal
role. Session-owned workflow endpoints still require `X-Session-ID`.

The saved graph shape is unchanged: `name`, `entry`, `max_steps`, and `nodes` with
`id`, `type`, `config`, and `transitions`. A terminal generator or evidence request
uses empty transitions; to continue, use `{"next": "target_id"}`. The default graph
and existing built-in workflows need no database migration. Custom workflows still
need referenced extensions loaded. Stale updates return 409. Chat requests select
`{"question": "When is launch?", "workflow_id": "..."}`.
