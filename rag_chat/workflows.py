"""Validated, persistent workflow definitions and a registry of executable nodes."""

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class WorkflowError(ValueError):
    """A workflow definition is invalid or cannot be changed."""


class WorkflowNotFound(KeyError):
    pass


class WorkflowConflict(RuntimeError):
    pass


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


NodeFactory = Callable[[Any, Callable[[dict[str, Any]], None] | None, dict[str, Any]], Callable]


@dataclass(frozen=True)
class NodeType:
    key: str
    label: str
    kind: Literal["agent", "tool"]
    description: str
    outputs: tuple[str, ...]
    config_model: type[NodeConfig]
    factory: NodeFactory | None = None
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def catalog(self) -> dict[str, Any]:
        schema = self.config_model.model_json_schema()
        return {
            "type": self.key, "label": self.label, "kind": self.kind,
            "description": self.description, "outputs": list(self.outputs),
            "requires": list(self.requires), "provides": list(self.provides),
            "config_schema": schema,
        }


_nodes: dict[str, NodeType] = {}
_registry_lock = RLock()
_loaded_modules: set[str] = set()


def register_node(node: NodeType) -> None:
    """Register trusted Python code. The UI can configure nodes but cannot supply code."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", node.key):
        raise ValueError("Node type must use a lowercase identifier.")
    if len(set(node.outputs)) != len(node.outputs) or any(not output for output in node.outputs):
        raise ValueError("Node outputs must be nonempty and unique.")
    if node.factory is None and node.key not in {"planner", "retrieve", "validate", "generate", "need_upload"}:
        raise ValueError("Extension nodes need an execution factory.")
    for name, field in node.config_model.model_json_schema().get("properties", {}).items():
        if field.get("type") not in {"string", "integer", "number", "boolean"}:
            raise ValueError(f"Setting {name} must have a simple type the workflow editor can render.")
    with _registry_lock:
        if node.key in _nodes:
            raise ValueError(f"Node type {node.key!r} is already registered.")
        _nodes[node.key] = node


def unregister_node(key: str) -> None:
    """Remove a node registration, primarily for isolated extension tests."""
    with _registry_lock:
        _nodes.pop(key, None)


def node_type(key: str) -> NodeType:
    with _registry_lock:
        try:
            return _nodes[key]
        except KeyError:
            raise WorkflowError(f"Unknown node type: {key}.") from None


def node_catalog() -> list[dict[str, Any]]:
    with _registry_lock:
        return [node.catalog() for node in sorted(_nodes.values(), key=lambda item: (item.kind, item.label))]


def load_extensions() -> None:
    """Import explicitly configured, trusted Python modules that call register_node."""
    for name in filter(None, (item.strip() for item in os.getenv("FOLIO_NODE_MODULES", "").split(","))):
        with _registry_lock:
            if name in _loaded_modules:
                continue
            importlib.import_module(name)
            _loaded_modules.add(name)


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    type: str = Field(min_length=2, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict)
    transitions: dict[str, str] = Field(default_factory=dict)


class WorkflowDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    entry: str
    nodes: list[WorkflowNode] = Field(min_length=1, max_length=30)
    max_steps: int = Field(default=40, ge=1, le=120)

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Workflow name cannot be blank.")
        return value.strip()


class SavedWorkflow(WorkflowDraft):
    id: str
    version: int
    updated_at: str


def validate_workflow(draft: WorkflowDraft) -> WorkflowDraft:
    ids = [node.id for node in draft.nodes]
    if len(set(ids)) != len(ids):
        raise WorkflowError("Node IDs must be unique.")
    if draft.entry not in ids:
        raise WorkflowError("The entry node does not exist.")
    nodes = {node.id: node for node in draft.nodes}
    for node in draft.nodes:
        definition = node_type(node.type)
        try:
            definition.config_model.model_validate(node.config)
        except ValidationError as exc:
            raise WorkflowError(f"Invalid settings for {node.id}: {exc.errors()[0]['msg']}.") from None
        if set(node.transitions) != set(definition.outputs):
            raise WorkflowError(f"{node.id} must connect these outputs: {', '.join(definition.outputs) or '(none)'}.")
        if any(target not in nodes for target in node.transitions.values()):
            raise WorkflowError(f"{node.id} has a transition to a missing node.")

    reached: set[str] = set()
    pending = [draft.entry]
    while pending:
        current = pending.pop()
        if current not in reached:
            reached.add(current)
            pending.extend(nodes[current].transitions.values())
    if reached != set(nodes):
        missing = [identity for identity in ids if identity not in reached]
        raise WorkflowError(
            f"Nodes not connected to start ({draft.entry}): {', '.join(missing)}. "
            "Select a connected node and point one of its outputs to each new node."
        )

    terminals = {node.id for node in draft.nodes if not node.transitions}
    if not terminals:
        raise WorkflowError("The workflow needs an answer node.")
    can_finish = set(terminals)
    while True:
        expanded = can_finish | {node.id for node in draft.nodes
                                 if any(target in can_finish for target in node.transitions.values())}
        if expanded == can_finish:
            break
        can_finish = expanded
    if reached != can_finish:
        raise WorkflowError("Every node must have a path to an answer node.")
    frontier = [(draft.entry, 1)]
    seen = set()
    while frontier:
        identity, distance = frontier.pop(0)
        if identity in seen:
            continue
        seen.add(identity)
        if identity in terminals:
            if draft.max_steps < distance:
                raise WorkflowError(f"Maximum steps must be at least {distance} to reach an answer.")
            break
        frontier.extend((target, distance + 1) for target in nodes[identity].transitions.values())

    # Every possible path to a node must supply the state it reads. This also
    # rejects graphs that could reach a generator without validating evidence.
    base = {"library", "question", "history", "round", "feedback", "evidence", "trace", "steps", "route"}
    definitions = {node.id: node_type(node.type) for node in draft.nodes}
    universe = base | {key for definition in definitions.values()
                       for key in (*definition.requires, *definition.provides)}
    predecessors: dict[str, set[str]] = {identity: set() for identity in nodes}
    for node in draft.nodes:
        for target in node.transitions.values():
            predecessors[target].add(node.id)
    available = {identity: set(universe) for identity in nodes}
    while True:
        changed = False
        for identity in ids:
            incoming = [available[source] | set(definitions[source].provides)
                        for source in predecessors[identity]]
            if identity == draft.entry:
                incoming.append(base)
            guaranteed = set.intersection(*incoming) if incoming else set()
            if guaranteed != available[identity]:
                available[identity] = guaranteed
                changed = True
        if not changed:
            break
    for identity in ids:
        missing = set(definitions[identity].requires) - available[identity]
        if missing:
            raise WorkflowError(f"{identity} needs state from an earlier node: {', '.join(sorted(missing))}.")
    for identity in terminals:
        if "answer" not in definitions[identity].provides:
            raise WorkflowError(f"Terminal node {identity} must produce an answer.")
    return draft


DEFAULT_WORKFLOW = WorkflowDraft(
    name="Default research", entry="planner", nodes=[
        WorkflowNode(id="planner", type="planner", transitions={"next": "retrieve"}),
        WorkflowNode(id="retrieve", type="retrieve", transitions={"next": "validate"}),
        WorkflowNode(id="validate", type="validate", transitions={
            "sufficient": "generate", "retry": "planner", "needs_upload": "need_upload",
        }),
        WorkflowNode(id="generate", type="generate"),
        WorkflowNode(id="need_upload", type="need_upload"),
    ],
)


class WorkflowStore:
    """SQLite storage for session-owned workflows and a shared default template."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
                body TEXT NOT NULL, updated_at TEXT NOT NULL, session_id TEXT)""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(workflows)")}
            if "session_id" not in columns:
                # Old shared custom workflows cannot be assigned to a session safely.
                db.execute("ALTER TABLE workflows ADD COLUMN session_id TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS workflows_session_id ON workflows(session_id)")
            if db.execute("SELECT 1 FROM workflows WHERE id = 'default'").fetchone() is None:
                self._insert(db, "default", DEFAULT_WORKFLOW, None)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _insert(db, workflow_id: str, draft: WorkflowDraft, session_id: str | None) -> None:
        db.execute("""INSERT INTO workflows (id, name, version, body, updated_at, session_id)
                      VALUES (?, ?, 1, ?, ?, ?)""", (
            workflow_id, draft.name, draft.model_dump_json(), datetime.now(timezone.utc).isoformat(), session_id,
        ))

    @staticmethod
    def _saved(row) -> SavedWorkflow:
        body = json.loads(row[3])
        return SavedWorkflow(**body, id=row[0], version=row[2], updated_at=row[4])

    def list(self, session_id: str) -> list[SavedWorkflow]:
        with self._connect() as db:
            rows = db.execute("""SELECT id, name, version, body, updated_at FROM workflows
                                 WHERE id = 'default' OR session_id = ? ORDER BY name, id""",
                              (session_id,)).fetchall()
        return [self._saved(row) for row in rows]

    def get(self, session_id: str, workflow_id: str) -> SavedWorkflow:
        with self._connect() as db:
            row = db.execute("""SELECT id, name, version, body, updated_at FROM workflows
                                WHERE id = ? AND (id = 'default' OR session_id = ?)""",
                             (workflow_id, session_id)).fetchone()
        if row is None:
            raise WorkflowNotFound(workflow_id)
        return self._saved(row)

    def create(self, session_id: str, draft: WorkflowDraft) -> SavedWorkflow:
        validate_workflow(draft)
        identity = uuid4().hex
        with self._lock, self._connect() as db:
            self._insert(db, identity, draft, session_id)
        return self.get(session_id, identity)

    def update(self, session_id: str, workflow_id: str, draft: WorkflowDraft, version: int) -> SavedWorkflow:
        if workflow_id == "default":
            raise WorkflowError("Copy the default workflow to create an editable version.")
        validate_workflow(draft)
        with self._lock, self._connect() as db:
            current = db.execute("SELECT version FROM workflows WHERE id = ? AND session_id = ?",
                                 (workflow_id, session_id)).fetchone()
            if current is None:
                raise WorkflowNotFound(workflow_id)
            if current[0] != version:
                raise WorkflowConflict("This workflow changed elsewhere. Reload it before saving.")
            db.execute("""UPDATE workflows SET name = ?, body = ?, version = ?, updated_at = ?
                          WHERE id = ? AND session_id = ?""", (
                draft.name, draft.model_dump_json(), version + 1,
                datetime.now(timezone.utc).isoformat(), workflow_id, session_id,
            ))
        return self.get(session_id, workflow_id)

    def delete(self, session_id: str, workflow_id: str) -> None:
        if workflow_id == "default":
            raise WorkflowError("The default workflow cannot be deleted.")
        with self._lock, self._connect() as db:
            deleted = db.execute("DELETE FROM workflows WHERE id = ? AND session_id = ?",
                                 (workflow_id, session_id)).rowcount
        if not deleted:
            raise WorkflowNotFound(workflow_id)
