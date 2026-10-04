"""Atomic-per-document ingestion; call while holding a session operation."""

from dataclasses import dataclass
from hashlib import sha256

from .documents import DocumentError, MAX_DOCUMENTS, chunk_sections, extract_sections, display_filename


@dataclass(frozen=True)
class IndexedDocument:
    filename: str
    chunks: int
    duplicate: bool = False


def ingest(library, filename: str, data: bytes, tokenizer_factory, batch_size=8) -> IndexedDocument:
    if not library.healthy:
        raise DocumentError("This library needs to be reset. Select Clear session and upload again.")
    identity = sha256(data).hexdigest()
    if identity in library.documents:
        previous = library.documents[identity]
        return IndexedDocument(previous.filename, previous.chunks, duplicate=True)
    if len(library.documents) >= MAX_DOCUMENTS:
        raise DocumentError("The session already contains 10 documents. Clear the session to start again.")
    sections = extract_sections(filename, data)
    try:
        chunks = chunk_sections(sections, filename, tokenizer_factory())
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("Cannot load the embedding tokenizer. Check internet access for the initial model download.") from None

    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            library.collection.add(
                ids=[f"{identity}:{chunk.index}" for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[{"document_id": identity, "filename": chunk.filename,
                            "location": chunk.location, "chunk": chunk.index} for chunk in batch],
            )
    except Exception:
        try:
            library.collection.delete(where={"document_id": identity})
        except Exception:
            library.healthy = False
            raise DocumentError("Indexing and cleanup failed. Select Clear session before continuing.") from None
        raise DocumentError("Embedding or indexing failed; this file was rolled back. Check the model download and retry.") from None
    result = IndexedDocument(display_filename(filename), len(chunks))
    library.documents[identity] = result
    return result
