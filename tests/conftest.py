import json
from types import SimpleNamespace
from unittest.mock import Mock

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from rag_chat.sessions import SessionManager


class TestEmbedding(EmbeddingFunction):
    """Small deterministic semantic buckets; exercises real Chroma without downloads."""
    def __init__(self):
        pass

    def __call__(self, input):
        return [np.array([1.0 + text.lower().count("apple"),
                          1.0 + text.lower().count("ocean"),
                          1.0 + text.lower().count("launch")], dtype=np.float32) for text in input]

    @staticmethod
    def name():
        return "test-semantic-buckets"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return TestEmbedding()


@pytest.fixture
def tokenizer():
    result = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    result.pre_tokenizer = Whitespace()
    return result


@pytest.fixture
def manager():
    result = SessionManager(
        chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False)),
        TestEmbedding(), start_worker=False,
    )
    yield result
    result.close()
    for identity in list(result._libraries):
        result.clear(identity)


@pytest.fixture
def client():
    result = Mock()
    result.chat.completions.create.return_value = completion("The launch is in June. [1]")
    return result


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def answer_completion(text="The launch is in June. [1]", status="answered"):
    return completion(json.dumps({"status": status, "text": text}))
