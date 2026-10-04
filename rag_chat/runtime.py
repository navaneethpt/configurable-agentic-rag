"""Process-wide model services and isolated, expiring document collections."""

import atexit
from threading import Lock

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from tokenizers import Tokenizer

from .embedding import SharedMiniLMEmbedding
from .sessions import SessionManager


class Runtime:
    def __init__(self):
        self.embedding = SharedMiniLMEmbedding()
        self.manager = SessionManager(
            chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False)),
            self.embedding,
        )
        self._tokenizer = None
        self._model_lock = Lock()
        atexit.register(self.manager.close)

    def tokenizer(self):
        with self._model_lock:
            if self._tokenizer is None:
                # The public embedding call downloads/verifies model files on first use.
                self.embedding(["Initialize local document embeddings."])
                path = ONNXMiniLM_L6_V2.DOWNLOAD_PATH / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME / "tokenizer.json"
                self._tokenizer = Tokenizer.from_file(str(path))
                # A separate tokenizer prevents long documents being silently truncated
                # and avoids changing the embedding function's padding/truncation settings.
                self._tokenizer.no_truncation()
                self._tokenizer.no_padding()
            return self._tokenizer


_runtime = None
_runtime_lock = Lock()


def get_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = Runtime()
        return _runtime
