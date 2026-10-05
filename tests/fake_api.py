"""Offline browser-test server. Never used by the normal API entry point."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import chromadb
from chromadb.config import Settings
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from conftest import TestEmbedding, completion
from rag_chat.api import create_app
from rag_chat.sessions import SessionManager

clock = [0.0]


def runtime():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return SimpleNamespace(
        manager=SessionManager(chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False)),
                               TestEmbedding(), clock=lambda: clock[0], start_worker=False),
        tokenizer=lambda: tokenizer,
    )


def model():
    client = MagicMock()
    client.__enter__.return_value = client
    def complete(**kwargs):
        time.sleep(0.4)
        messages = kwargs["messages"]
        system = messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        question = payload["question"].lower()
        if "retrieval planner" in system:
            return completion(json.dumps({"searches": [{"query": "launch date", "purpose": "Find relevant dates"}]}))
        if "You verify" in system:
            partial = "partial" in question
            missing = "budget" in question or partial
            return completion(json.dumps({"decision": "needs_more_evidence" if missing else "sufficient",
                                          "confidence": 0.5 if partial else 0.2 if missing else 0.95,
                                          "missing_evidence": ["the annual budget document"] if missing else []}))
        return completion(json.dumps({"status": "answered" if payload.get("excerpts") else "missing_context",
                                     "text": "Project Cedar launches in June. [1]" if payload.get("excerpts")
                                     else "Please upload the document or section that covers this question."}))
    client.chat.completions.create.side_effect = complete
    return client


workflow_directory = TemporaryDirectory(prefix="folio-browser-tests-")
app = create_app(runtime, model, workflow_db_path=Path(workflow_directory.name) / "workflows.sqlite3")


@app.post("/test/expire")
def expire():
    clock[0] += 3601
    app.state.runtime.manager.cleanup()
    return {"ok": True}
