"""Local, single-worker API. Session data and in-flight work live only in memory."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.datastructures import UploadFile

from .chat import ChatError, answer_question
from .documents import DocumentError, MAX_FILE_BYTES
from .indexing import ingest
from .runtime import Runtime
from .providers import create_model_client, provider_settings
from .sessions import SessionBusy, SessionExpired
from .workflows import (WorkflowConflict, WorkflowDraft, WorkflowError, WorkflowNotFound,
                        WorkflowStore, load_extensions, node_catalog, validate_workflow)


class BodyTooLarge(Exception):
    pass


class UploadLimit:
    """Bound multipart input before the parser can spool an arbitrary request."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/documents":
            return await self.app(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > MAX_FILE_BYTES + 1024 * 1024:
                raise BodyTooLarge()
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except BodyTooLarge:
            await JSONResponse({"detail": "Upload exceeds the 20 MB file limit."}, 413)(scope, receive, send)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    workflow_id: str = Field(default="default", min_length=1, max_length=64)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()


class WorkflowUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1)
    workflow: WorkflowDraft


def create_app(runtime_factory=Runtime, client_factory=None, frontend_dir: Path | None = None,
               workflow_db_path: Path | None = None) -> FastAPI:
    static_dir = frontend_dir or Path(__file__).resolve().parents[1] / "frontend" / "out"

    @asynccontextmanager
    async def lifespan(app):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        app.state.provider_settings = provider_settings()
        load_extensions()
        app.state.runtime = runtime_factory()
        database = workflow_db_path or Path(os.getenv(
            "FOLIO_WORKFLOWS_DB", str(Path(__file__).resolve().parents[1] / ".data" / "workflows.sqlite3")))
        app.state.workflows = WorkflowStore(database)
        app.state.tasks = set()
        if static_dir.is_dir():
            app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
        yield
        if app.state.tasks:
            await asyncio.gather(*app.state.tasks, return_exceptions=True)
        app.state.runtime.manager.close()

    api = FastAPI(title="Document Research API", lifespan=lifespan)
    api.add_middleware(UploadLimit)

    @api.exception_handler(SessionExpired)
    async def expired(request, error):
        return JSONResponse({"detail": str(error)}, status_code=410)

    @api.exception_handler(SessionBusy)
    async def busy(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @api.exception_handler(WorkflowNotFound)
    async def missing_workflow(request, error):
        return JSONResponse({"detail": "Workflow not found."}, status_code=404)

    @api.exception_handler(WorkflowConflict)
    async def workflow_conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @api.exception_handler(WorkflowError)
    async def invalid_workflow(request, error):
        return JSONResponse({"detail": str(error)}, status_code=422)

    def manager():
        return api.state.runtime.manager

    def identity(value):
        if not value:
            raise HTTPException(400, "A session is required.")
        return value

    def workflow_session(value):
        session_id = identity(value)
        manager().snapshot(session_id)
        return session_id

    def spawn(coroutine):
        task = asyncio.create_task(coroutine)
        api.state.tasks.add(task)
        task.add_done_callback(api.state.tasks.discard)
        return task

    def configured():
        return client_factory is not None or api.state.provider_settings.configuration_error is None

    @api.get("/api/health")
    def health():
        settings = api.state.provider_settings
        return {"status": "ok", "answering_configured": configured(), "provider": settings.provider,
                "default_model": settings.default_model,
                "configuration_error": None if client_factory else settings.configuration_error}

    @api.get("/api/node-types")
    def catalog():
        return node_catalog()

    @api.get("/api/workflows")
    def workflows(x_session_id: str | None = Header(default=None)):
        return api.state.workflows.list(workflow_session(x_session_id))

    @api.get("/api/workflows/{workflow_id}")
    def get_workflow(workflow_id: str, x_session_id: str | None = Header(default=None)):
        return api.state.workflows.get(workflow_session(x_session_id), workflow_id)

    @api.post("/api/workflows", status_code=201)
    def create_workflow(draft: WorkflowDraft, x_session_id: str | None = Header(default=None)):
        return api.state.workflows.create(workflow_session(x_session_id), draft)

    @api.put("/api/workflows/{workflow_id}")
    def update_workflow(workflow_id: str, request: WorkflowUpdate,
                        x_session_id: str | None = Header(default=None)):
        return api.state.workflows.update(workflow_session(x_session_id), workflow_id,
                                          request.workflow, request.version)

    @api.delete("/api/workflows/{workflow_id}", status_code=204)
    def delete_workflow(workflow_id: str, x_session_id: str | None = Header(default=None)):
        api.state.workflows.delete(workflow_session(x_session_id), workflow_id)

    @api.post("/api/sessions", status_code=201)
    def create_session():
        library, _ = manager().ensure(None)
        return {"id": library.id}

    @api.get("/api/session")
    def session(x_session_id: str | None = Header(default=None)):
        return manager().snapshot(identity(x_session_id))

    @api.delete("/api/session", status_code=204)
    def clear(x_session_id: str | None = Header(default=None)):
        manager().snapshot(identity(x_session_id))
        manager().clear(x_session_id)

    @api.post("/api/documents")
    async def upload(request: Request, x_session_id: str | None = Header(default=None)):
        library = manager().begin(identity(x_session_id), "upload")
        dispatched = False
        try:
            async with request.form(max_files=1, max_fields=0) as form:
                file = form.get("file")
                if not isinstance(file, UploadFile):
                    raise HTTPException(422, "Choose one file to upload.")
                data = await file.read(MAX_FILE_BYTES + 1)
                filename = file.filename or "document"
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, "The file exceeds the 20 MB limit.")

            def work():
                error = None
                try:
                    with library.lock:
                        result = ingest(library, filename, data, api.state.runtime.tokenizer)
                    return {"status": "duplicate" if result.duplicate else "indexed", **asdict(result)}
                except DocumentError as exc:
                    error = str(exc)
                    return {"status": "failed", "filename": filename, "detail": error}
                except Exception:
                    error = "Document processing failed. Please try again."
                    return {"status": "failed", "filename": filename, "detail": error}
                finally:
                    manager().finish(library, error=error)

            task = spawn(asyncio.to_thread(work))
            dispatched = True
            return await asyncio.shield(task)
        finally:
            if not dispatched:
                manager().finish(library, error="Upload was interrupted or rejected. Please try again.")

    @api.post("/api/chat")
    async def chat(body: ChatRequest, x_session_id: str | None = Header(default=None)):
        session_id = identity(x_session_id)
        snapshot = manager().snapshot(session_id)
        workflow = api.state.workflows.get(session_id, body.workflow_id)
        validate_workflow(workflow)
        if not configured():
            raise HTTPException(503, api.state.provider_settings.configuration_error)
        if not snapshot["healthy"]:
            raise HTTPException(409, "The document library needs to be cleared before continuing.")
        if not snapshot["documents"]:
            raise HTTPException(409, "Upload and process documents before asking a question.")
        library = manager().begin(session_id, "chat", body.question, workflow={
            "id": workflow.id, "name": workflow.name, "version": workflow.version,
        })
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=128)
        connected = True

        def enqueue(event, payload):
            if connected:
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait((event, payload))

        def progress(event):
            manager().record_event(library, event)
            loop.call_soon_threadsafe(enqueue, "progress", event)

        def work():
            error = None
            messages = None
            result = None
            try:
                factory = client_factory or (lambda: create_model_client(api.state.provider_settings))
                with library.lock, factory() as client:
                    answer = answer_question(library, body.question, library.messages[-10:], client,
                                             on_event=progress, workflow=workflow)
                result = {"id": uuid4().hex, "role": "assistant", "content": answer.text,
                          "sources": [asdict(source) for source in answer.sources],
                          "trace": list(library.latest_operation["events"]),
                          "workflow": {"id": workflow.id, "name": workflow.name, "version": workflow.version}}
                messages = [{"id": uuid4().hex, "role": "user", "content": body.question,
                             "sources": [], "trace": []}, result]
            except ChatError as exc:
                error = str(exc)
            except Exception:
                error = "Research could not complete. Please try again."
            finally:
                manager().finish(library, messages=messages, error=error)
            loop.call_soon_threadsafe(enqueue, "error" if error else "answer",
                                     {"detail": error} if error else result)

        spawn(asyncio.to_thread(work))

        async def events():
            nonlocal connected
            try:
                while True:
                    try:
                        event, payload = await asyncio.wait_for(queue.get(), timeout=10)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
                    if event in {"answer", "error"}:
                        break
            finally:
                connected = False
                while not queue.empty():
                    queue.get_nowait()

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        })

    return api


app = create_app()
