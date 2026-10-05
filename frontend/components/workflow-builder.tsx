"use client";

import { useEffect, useState } from "react";
import { ArrowRight, Copy, Plus, Save, Trash2, X } from "lucide-react";
import { api } from "../lib/api";
import { ConfigProperty, copyDraft, NodeType, SavedWorkflow, unreachableNodeIds, WorkflowDraft, WorkflowNode } from "../lib/workflows";

type Props = {
  open: boolean;
  session?: string;
  activeId: string;
  onClose: () => void;
  onActivate: (workflow: SavedWorkflow) => void;
};

function detail(error: unknown) {
  return error instanceof Error ? error.message : "The workflow could not be saved.";
}

export default function WorkflowBuilder({ open, session, activeId, onClose, onActivate }: Props) {
  const [catalog, setCatalog] = useState<NodeType[]>([]);
  const [workflows, setWorkflows] = useState<SavedWorkflow[]>([]);
  const [draft, setDraft] = useState<WorkflowDraft>();
  const [selectedId, setSelectedId] = useState("default");
  const [editingId, setEditingId] = useState<string>();
  const [version, setVersion] = useState(1);
  const [selectedNode, setSelectedNode] = useState<string>();
  const [addType, setAddType] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    if (!open || !session) return;
    let cancelled = false;
    setBusy(true); setError(""); setNotice("");
    Promise.all([api<NodeType[]>("/node-types"), api<SavedWorkflow[]>("/workflows", session)])
      .then(([types, saved]) => {
        if (cancelled) return;
        setCatalog(types); setWorkflows(saved);
        const chosen = saved.find(item => item.id === activeId) || saved.find(item => item.id === "default");
        if (chosen) load(chosen);
      })
      .catch(reason => { if (!cancelled) setError(detail(reason)); })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  // Saving activates the workflow while this dialog is open. Keep the edited
  // draft and confirmation visible; refresh from the server on the next open.
  }, [open, session]);

  function load(workflow: SavedWorkflow) {
    setDraft(copyDraft(workflow));
    setSelectedId(workflow.id);
    setEditingId(workflow.id === "default" ? undefined : workflow.id);
    setVersion(workflow.version);
    setSelectedNode(workflow.entry);
    setError(""); setNotice("");
  }

  function change(update: (current: WorkflowDraft) => WorkflowDraft) {
    setDraft(current => current ? update(current) : current);
    setNotice("");
  }

  function changeNode(id: string, update: (node: WorkflowNode) => WorkflowNode) {
    change(current => ({ ...current, nodes: current.nodes.map(node => node.id === id ? update(node) : node) }));
  }

  function addNode() {
    const definition = catalog.find(item => item.type === addType);
    if (!definition || !draft) return;
    let number = 1;
    let id = `${definition.type}_${number}`;
    while (draft.nodes.some(node => node.id === id)) id = `${definition.type}_${++number}`;
    change(current => ({ ...current, nodes: [...current.nodes, {
      id, type: definition.type, config: {},
      transitions: Object.fromEntries(definition.outputs.map(output => [output, ""])),
    }] }));
    setSelectedNode(id);
    setAddType("");
  }

  function removeNode(id: string) {
    change(current => {
      const nodes = current.nodes.filter(node => node.id !== id).map(node => ({
        ...node,
        transitions: Object.fromEntries(Object.entries(node.transitions).map(([output, target]) =>
          [output, target === id ? "" : target])),
      }));
      return { ...current, nodes, entry: current.entry === id ? nodes[0]?.id || "" : current.entry };
    });
    if (selectedNode === id) setSelectedNode(draft?.nodes.find(node => node.id !== id)?.id);
  }

  function duplicate() {
    if (!draft) return;
    setDraft({ ...draft, name: `${draft.name} copy`, nodes: draft.nodes.map(node => ({
      ...node, config: { ...node.config }, transitions: { ...node.transitions },
    })) });
    setEditingId(undefined); setNotice("This copy will be saved as a new workflow.");
  }

  async function save() {
    if (!draft || !session) return;
    setBusy(true); setError(""); setNotice("");
    try {
      const saved = editingId
        ? await api<SavedWorkflow>(`/workflows/${editingId}`, session, {
          method: "PUT", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workflow: draft, version }),
        })
        : await api<SavedWorkflow>("/workflows", session, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(draft),
        });
      setWorkflows(current => [...current.filter(item => item.id !== saved.id), saved].sort((a, b) => a.name.localeCompare(b.name)));
      load(saved);
      onActivate(saved);
      setNotice(`Saved version ${saved.version}. New questions will use this workflow.`);
    } catch (reason) { setError(detail(reason)); }
    finally { setBusy(false); }
  }

  async function removeWorkflow() {
    if (!session || !editingId || busy || !window.confirm("Delete this saved workflow?")) return;
    setBusy(true); setError("");
    try {
      await api(`/workflows/${editingId}`, session, { method: "DELETE" });
      const remaining = workflows.filter(item => item.id !== editingId);
      setWorkflows(remaining);
      const fallback = remaining.find(item => item.id === "default");
      if (fallback) { load(fallback); if (activeId === editingId) onActivate(fallback); }
    } catch (reason) { setError(detail(reason)); }
    finally { setBusy(false); }
  }

  if (!open) return null;
  const node = draft?.nodes.find(item => item.id === selectedNode);
  const nodeDefinition = catalog.find(item => item.type === node?.type);
  const fields = Object.entries(nodeDefinition?.config_schema.properties || {});
  const savedDraft = workflows.find(item => item.id === selectedId);
  const isDirty = draft && (!savedDraft || JSON.stringify(copyDraft(savedDraft)) !== JSON.stringify(draft));
  const unreachable = draft ? unreachableNodeIds(draft) : [];
  const unconnected = draft?.nodes.flatMap(item => Object.entries(item.transitions)
    .filter(([, target]) => !target).map(([output]) => `${item.id}.${output}`)) || [];

  function renderField(key: string, property: ConfigProperty) {
    if (!node) return null;
    const value = node.config[key] ?? property.default ?? "";
    const label = property.title || key.replaceAll("_", " ");
    const setValue = (next: string | number | boolean) => changeNode(node.id, current => ({
      ...current, config: { ...current.config, [key]: next },
    }));
    return <label className="workflow-field" key={key}>
      <span>{label}</span>
      {property.type === "boolean"
        ? <input type="checkbox" checked={Boolean(value)} onChange={event => setValue(event.target.checked)} />
        : property.type === "integer" || property.type === "number"
          ? <input type="number" value={String(value)} min={property.minimum} max={property.maximum}
              step={property.type === "integer" ? 1 : 0.05} onChange={event => setValue(Number(event.target.value))} />
          : key === "instructions"
            ? <textarea value={String(value)} maxLength={property.maxLength} rows={5}
                placeholder="Optional instructions for this agent" onChange={event => setValue(event.target.value)} />
            : <input value={String(value)} maxLength={property.maxLength} onChange={event => setValue(event.target.value)} />}
      {property.description && <small>{property.description}</small>}
    </label>;
  }

  return <div className="workflow-overlay" onKeyDown={event => { if (event.key === "Escape") onClose(); }}>
    <section className="workflow-dialog" role="dialog" aria-modal="true" aria-label="Configure agents">
      <header className="workflow-header">
        <div><span className="eyebrow">YOUR RESEARCH WORKFLOW</span><h2>Configure agents</h2><p>Choose the agents and tools, adjust their settings, and connect them for your task. <a href="/agents/">See how the default agents work together</a></p></div>
        <button className="icon-button" aria-label="Close workflow builder" onClick={onClose}><X size={21} /></button>
      </header>
      {error && <div className="workflow-error" role="alert">{error}</div>}
      {notice && <div className="workflow-notice" role="status">{notice}</div>}
      <div className="workflow-body">
        <aside className="workflow-sidebar">
          <label className="workflow-field"><span>This session’s workflows</span>
            <select aria-label="Saved workflow" value={selectedId} onChange={event => {
              const saved = workflows.find(item => item.id === event.target.value); if (saved) load(saved);
            }}>{workflows.map(item => <option key={item.id} value={item.id}>{item.name}{item.id === activeId ? " · active" : ""}</option>)}</select>
          </label>
          <div className="workflow-actions">
            <button onClick={duplicate} disabled={!draft || busy}><Copy size={14} /> Duplicate</button>
            <button onClick={removeWorkflow} disabled={!editingId || busy}><Trash2 size={14} /> Delete</button>
          </div>
          <h3>Available nodes</h3>
          <p>Registered server nodes appear here automatically. Saved workflows are visible only in this session.</p>
          <div className="workflow-palette">{catalog.map(item => <button key={item.type}
            onClick={() => { setAddType(item.type); }} className={addType === item.type ? "selected" : ""}>
              <small>{item.kind}</small><strong>{item.label}</strong><span>{item.description}</span>
            </button>)}</div>
          <button className="workflow-add" onClick={addNode} disabled={!addType || busy}><Plus size={15} /> Add selected node</button>
        </aside>
        <div className="workflow-main">
          {draft && <>
            <div className="workflow-basics">
              <label className="workflow-field"><span>Workflow name</span><input aria-label="Workflow name" value={draft.name}
                maxLength={100} onChange={event => change(current => ({ ...current, name: event.target.value }))} /></label>
              <label className="workflow-field"><span>Start at</span><select aria-label="Start node" value={draft.entry}
                onChange={event => change(current => ({ ...current, entry: event.target.value }))}>
                {draft.nodes.map(item => <option key={item.id} value={item.id}>{item.id}</option>)}</select></label>
              <label className="workflow-field"><span>Maximum steps</span><input type="number" min={1} max={120}
                value={draft.max_steps} onChange={event => change(current => ({ ...current, max_steps: Number(event.target.value) }))} /></label>
            </div>
            <p className="workflow-hint">{editingId ? "Save edits as a new version for future questions." : "The default is a starting point. Saving creates your own workflow and activates it for future questions."} Choose a node to edit its settings and connect each output. Loops are allowed within the step limit.</p>
            <p className="workflow-hint">Your question and conversation history are passed to the selected start node. Additional required inputs must come from earlier nodes. With no uploaded documents, Folio responds immediately without running the workflow.</p>
            {(unreachable.length > 0 || unconnected.length > 0) && <div className="workflow-connectivity" role="status">
              {unreachable.length > 0 && <p><strong>Not connected to Start:</strong> {unreachable.join(", ")}. Select a node already on the path and set one of its Connections to the new node.</p>}
              {unconnected.length > 0 && <p><strong>Outputs needing a destination:</strong> {unconnected.join(", ")}.</p>}
            </div>}
            <div className="workflow-editor">
              <div className="workflow-node-list">{draft.nodes.map(item => {
                const definition = catalog.find(type => type.type === item.type);
                return <button key={item.id} className={`workflow-node-card ${selectedNode === item.id ? "selected" : ""} ${unreachable.includes(item.id) ? "unreachable" : ""}`}
                  onClick={() => setSelectedNode(item.id)}>
                  <small>{definition?.kind || "missing type"} · {item.id}{draft.entry === item.id ? " · start" : ""}{unreachable.includes(item.id) ? " · not connected" : ""}</small>
                  <strong>{definition?.label || item.type}</strong>
                  <span>{Object.entries(item.transitions).length ? Object.entries(item.transitions).map(([port, target]) =>
                    `${port} → ${target || "unconnected"}`).join(" · ") : "Ends workflow"}</span>
                </button>;
              })}</div>
              <div className="workflow-inspector">{node && <>
                <div className="workflow-inspector-head"><div><small>{nodeDefinition?.kind}</small><h3>{nodeDefinition?.label || node.type}</h3><p>{nodeDefinition?.description}</p></div>
                  <button aria-label={`Remove ${node.id}`} onClick={() => removeNode(node.id)} disabled={draft.nodes.length === 1}><Trash2 size={16} /></button></div>
                <div className="workflow-id">Node ID: <code>{node.id}</code></div>
                <section className="workflow-inputs" aria-label="Node inputs and outputs">
                  <h4>Required inputs</h4>
                  {nodeDefinition?.requires.length ? <ul>{nodeDefinition.requires.map(input => {
                    const initial = nodeDefinition.initial_inputs.includes(input);
                    const producers = catalog.filter(type => type.provides.includes(input));
                    return <li key={input}><code>{input}</code> — {initial ? "Provided when the workflow starts" :
                      producers.length ? `From an earlier node: ${producers.map(type => type.label).join(", ")}` : "Needs an earlier producer; none is registered"}</li>;
                  })}</ul> : <p>No additional inputs required.</p>}
                  <h4>Produced outputs</h4>
                  <p>{nodeDefinition?.provides.length ? nodeDefinition.provides.join(", ") : "No state outputs declared."}</p>
                </section>
                {fields.length > 0 && <div className="workflow-fields"><h4>Settings</h4>{fields.map(([key, property]) => renderField(key, property))}</div>}
                <div className="workflow-transitions"><h4>Connections</h4>{nodeDefinition?.outputs.length
                  ? nodeDefinition.outputs.map(output => <label className="workflow-field" key={output}><span>{output.replaceAll("_", " ")} <ArrowRight size={13} /></span>
                    <select aria-label={`${node.id} ${output} target`} value={node.transitions[output] || ""}
                      onChange={event => changeNode(node.id, current => ({ ...current,
                        transitions: { ...current.transitions, [output]: event.target.value },
                      }))}><option value="">Select destination</option>{draft.nodes.map(target =>
                        <option key={target.id} value={target.id}>{target.id}</option>)}</select></label>)
                  : <p>This node ends the workflow with an answer.</p>}</div>
              </>}</div>
            </div>
          </>}
        </div>
      </div>
      <footer className="workflow-footer"><span>{draft?.nodes.length || 0} nodes · {catalog.length} available types{isDirty ? " · unsaved edits" : ""}</span>
        <div><button className="workflow-secondary" disabled={busy || !draft || selectedId === activeId}
          onClick={() => { const saved = workflows.find(item => item.id === selectedId); if (saved) { onActivate(saved); setNotice("New questions will use this workflow."); } }}>Use selected</button>
          <button className="workflow-save" disabled={busy || !draft || !session || unreachable.length > 0 || unconnected.length > 0} onClick={save}><Save size={15} /> {editingId ? "Save changes and use" : "Save new workflow and use"}</button></div>
      </footer>
    </section>
  </div>;
}
