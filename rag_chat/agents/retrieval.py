from ..agent_contracts import AgentOutput, ChatError, NodeResult
from ..workflows import NodeType, register_node
from .common import MAX_EVIDENCE, RetrieveConfig, RetrievalOutput, passages, rounds, searches


def build(client, on_event, config):
    def run(inputs):
        round_number = rounds(inputs) + 1
        collection = inputs.services.library.collection
        try:
            count = collection.count()
        except Exception:
            raise ChatError("Cannot read the document library. Clear the session and upload again.") from None
        previous = passages(inputs)
        existing = {item["chunk_id"] for item in previous}
        added_passages, traces = [], []
        tasks = searches(inputs)
        cap = config.get("max_evidence", MAX_EVIDENCE)
        for task in tasks:
            if not count or len(existing) >= cap:
                break
            try:
                result = collection.query(query_texts=[task["query"]],
                    n_results=min(config["max_results_per_search"], count), include=["documents", "metadatas"])
                ids, documents, metadatas = (result.get(key, [[]])[0] for key in ("ids", "documents", "metadatas"))
                added = 0
                for index, (identity, text, metadata) in enumerate(zip(ids, documents, metadatas)):
                    identity = str(identity or f"{metadata.get('document_id', '')}:{metadata.get('chunk', index)}")
                    if identity in existing:
                        continue
                    existing.add(identity)
                    added_passages.append({"chunk_id": identity, "filename": metadata["filename"],
                                          "location": metadata["location"], "text": text})
                    added += 1
                    if len(existing) >= cap:
                        break
            except Exception:
                raise ChatError("Document retrieval failed. Check the embedding model and try again.") from None
            traces.append({**task, "new_sources": added})
            if on_event:
                on_event({"event": "search", "round": round_number, "query": task["query"],
                          "results": len(ids), "new_sources": added})
        return NodeResult(AgentOutput("passages", {"passages": added_passages, "searches": tasks, "round": round_number}),
                          details={"round": round_number, "searches": traces, "total_sources": len(existing)})
    return run


register_node(NodeType("retrieve", "Document retrieval", "tool", "Search this session's document library.",
    ("next",), RetrieveConfig, build, accepted_inputs=("question", "search_queries", "passages"),
    output_kind="passages", output_model=RetrievalOutput,
    missing_input_behavior="Search with the original question when no search-query output is available."))
