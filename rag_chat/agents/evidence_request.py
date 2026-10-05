from ..agent_contracts import AgentOutput, Answer, NodeResult
from ..workflows import NodeConfig, NodeType, register_node
from .common import EvidenceRequestOutput, rounds, validation


def build(client, on_event, config):
    def run(inputs):
        previous = validation(inputs)
        missing = previous["missing_evidence"] if previous else []
        earlier = inputs.latest("evidence_request")
        if not missing and earlier:
            missing = earlier.output.data["missing_evidence"]
        text = ("I need more evidence to answer reliably. Please upload documents or sections covering: "
                + "; ".join(missing[:3]) + ".") if missing else (
                "I need more relevant evidence to answer reliably. Please upload the document or section that covers this question.")
        reason = "budget_exhausted" if previous and rounds(inputs) >= previous["round_limit"] else "no_new_evidence"
        return NodeResult(AgentOutput("evidence_request", {"text": text, "missing_evidence": missing}),
            None if inputs.is_terminal else "next", Answer(text, []),
            details={"round": rounds(inputs), "reason": reason, "missing_evidence": missing})
    return run


register_node(NodeType("need_upload", "Request more evidence", "agent", "Describe missing documents or context.",
    ("next",), NodeConfig, build, accepted_inputs=("question", "validation", "evidence_request"),
    output_kind="evidence_request", output_model=EvidenceRequestOutput, terminal_role="evidence_request",
    missing_input_behavior="Ask for relevant documents when no specific evidence gaps are available."))
