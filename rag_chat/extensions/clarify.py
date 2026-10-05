"""Example optional agent that requests a missing detail from the user."""

from pydantic import Field

from rag_chat.agent_contracts import AgentOutput, Answer, NodeResult
from rag_chat.agents.common import EvidenceRequestOutput
from rag_chat.workflows import NodeConfig, NodeType, register_node


class ClarifyConfig(NodeConfig):
    question: str = Field(default="Which document should I use?", min_length=1)


def build_clarify(client, on_event, config):
    def run(inputs):
        question = config["question"]
        return NodeResult(
            AgentOutput("evidence_request", {"text": question, "missing_evidence": []}),
            None if inputs.is_terminal else "next", Answer(question, []),
        )

    return run


register_node(NodeType(
    key="clarify", label="Ask for clarification", kind="agent",
    description="Ask the user for a missing detail.", outputs=("next",),
    config_model=ClarifyConfig, factory=build_clarify,
    accepted_inputs=("question", "earlier outputs"),
    output_kind="evidence_request", output_model=EvidenceRequestOutput,
    missing_input_behavior="Ask for the missing document or detail.",
    terminal_role="evidence_request",
))
