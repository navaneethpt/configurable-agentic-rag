"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowUp, BookOpen, Check, ChevronRight, FileText, FlaskConical, FolderOpen, Layers, LoaderCircle, Plus, Search, Settings2, Trash2, Upload, X } from "lucide-react";
import { api, ApiError, Message, research, Snapshot, Trace } from "../lib/api";
import { forgetSession, getSession, WORKFLOW_KEY } from "../lib/session";
import { SavedWorkflow } from "../lib/workflows";
import WorkflowBuilder from "./workflow-builder";
const stageNames: Record<string, string> = { planner: "Planning searches", retrieve: "Searching documents", validate: "Checking evidence", generate: "Preparing answer", need_upload: "More evidence needed" };
const completedNames: Record<string, string> = { planner: "Search plan ready", retrieve: "Retrieval complete", validate: "Evidence checked", generate: "Answer ready", need_upload: "More evidence needed", search: "Search complete" };
function label(event: Trace) {
  if (event.event === "node_start") return stageNames[event.node_type || event.node || ""] || event.label || event.node;
  if (event.terminal === false && event.node_type === "generate") return "Intermediate draft ready";
  if (event.terminal === false && event.node_type === "need_upload") return "Evidence request prepared";
  return completedNames[event.event] || event.label || event.event;
}

function ResearchTimeline({ events }: { events: Trace[] }) {
  return <ol className="timeline">{events.map((event, index) => <li key={index}>
    <span className={`timeline-dot ${event.event === "node_start" ? "start" : ""}`} />
    <div className="timeline-label">{label(event)}{event.round && <span>Round {event.round}</span>}</div>
    {event.searches?.map((search, i) => <div className="search-card" key={i}><strong>{search.query}</strong><p>{search.purpose}</p>{search.new_sources !== undefined && <small>{search.new_sources} new sources</small>}</div>)}
    {event.query && <p className="query">“{event.query}”</p>}
    {event.results !== undefined && <small>{event.results} results · {event.new_sources} new</small>}
    {event.total_sources !== undefined && <small>{event.total_sources} unique sources gathered</small>}
    {event.decision && <p className="decision">{event.acceptance_reason === "third_attempt_confidence" ? "Passed third-attempt confidence threshold" : (event.accepted ?? event.decision === "sufficient") ? "Evidence is sufficient" : "Additional evidence needed"}</p>}
    {event.confidence !== undefined && <small>Validator confidence: {Number((event.confidence * 100).toFixed(2))}%</small>}
    {event.missing_evidence?.map((gap, i) => <p key={i}>{gap}</p>)}
    {event.outcome && <p>{event.outcome === "answered" ? event.terminal === false ? "Draft available to later nodes" : "Answer checked and ready" : "Missing context"}</p>}
    {event.reason && <small>{event.reason.replaceAll("_", " ")}</small>}
  </li>)}</ol>;
}

export default function Workspace() {
  const [session, setSession] = useState<string>();
  const [snapshot, setSnapshot] = useState<Snapshot>();
  const [configured, setConfigured] = useState(false);
  const [activeWorkflow, setActiveWorkflow] = useState<SavedWorkflow>();
  const [workflowBuilderOpen, setWorkflowBuilderOpen] = useState(false);
  const [error, setError] = useState("");
  const [expired, setExpired] = useState(false);
  const [loading, setLoading] = useState(true);
  const [localBusy, setLocalBusy] = useState(false);
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [uploads, setUploads] = useState<{ filename: string; status: string; detail?: string }[]>([]);
  const [events, setEvents] = useState<Trace[]>([]);
  const [selected, setSelected] = useState<string>();
  const [sourceNumber, setSourceNumber] = useState<number>();
  const [panel, setPanel] = useState<"sources" | "research">("research");
  const [drawer, setDrawer] = useState<"documents" | "research" | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const end = useRef<HTMLDivElement>(null);
  const busy = localBusy || snapshot?.operation?.status === "running";

  const handleError = useCallback((reason: unknown) => {
    if (reason instanceof ApiError && reason.status === 410) {
      forgetSession(); setSession(undefined); setActiveWorkflow(undefined); setWorkflowBuilderOpen(false);
      setExpired(true); setSnapshot(undefined);
      setEvents([]); setPending(""); setSelected(undefined); setFiles([]); setUploads([]);
    }
    setError(reason instanceof Error ? reason.message : "Connection unavailable. Please retry.");
  }, []);

  const refresh = useCallback(async (id: string, recoverDraft = false) => {
    const state = await api<Snapshot>("/session", id);
    setSnapshot(state);
    const operation = state.operation;
    setPending(operation?.kind === "chat" && operation.status === "running" ? operation.question : "");
    if (operation?.kind === "chat") {
      setEvents(operation.events);
      if (recoverDraft && operation.status === "failed") { setError(operation.error || "Research failed."); setQuestion(operation.question); }
      if (recoverDraft && operation.status === "complete") setQuestion("");
    }
    return state;
  }, []);

  const initialize = useCallback(async () => {
    setLoading(true); setError("");
    try {
      const id = await getSession();
      const [health, workflows] = await Promise.all([
        api<{ answering_configured: boolean }>("/health"), api<SavedWorkflow[]>("/workflows", id),
      ]);
      setConfigured(health.answering_configured);
      const chosen = workflows.find(item => item.id === sessionStorage.getItem(WORKFLOW_KEY))
        || workflows.find(item => item.id === "default");
      setActiveWorkflow(chosen);
      setSession(id);
      const state = await refresh(id, true);
      setSelected(state.messages.filter(m => m.role === "assistant").at(-1)?.id);
    } catch (reason) { handleError(reason); }
    finally { setLoading(false); }
  }, [handleError, refresh]);

  useEffect(() => { void initialize(); }, [initialize]);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("configure") !== "agents") return;
    setWorkflowBuilderOpen(true);
    params.delete("configure");
    window.history.replaceState(null, "", `${window.location.pathname}${params.size ? `?${params}` : ""}${window.location.hash}`);
  }, []);
  useEffect(() => {
    if (!session || snapshot?.operation?.status !== "running" || localBusy || expired) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try { if (!cancelled) await refresh(session, true); } catch (reason) { if (!cancelled) handleError(reason); }
      if (!cancelled) timer = setTimeout(poll, 1000);
    };
    timer = setTimeout(poll, 1000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [session, snapshot?.operation?.status, localBusy, expired, refresh, handleError]);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [snapshot?.messages.length, pending]);
  useEffect(() => { if (drawer) dialog.current?.showModal(); else dialog.current?.close(); }, [drawer]);

  async function reset() {
    if (busy) return;
    if (!expired && snapshot && !window.confirm("Start a new session? This ends access to its documents, conversation, and saved workflows.")) return;
    setLocalBusy(true);
    try {
      if (!expired && session && snapshot) await api("/session", session, { method: "DELETE" });
      forgetSession(); setActiveWorkflow(undefined);
      setSnapshot(undefined); setExpired(false); setEvents([]); setSelected(undefined);
      setSourceNumber(undefined); setQuestion(""); setPending(""); setUploads([]); setFiles([]);
      await initialize();
    } catch (reason) { handleError(reason); }
    finally { setLocalBusy(false); }
  }

  function chooseFiles(incoming: File[]) {
    if (busy) return;
    setFiles(previous => [...previous, ...incoming]);
  }

  async function processFiles() {
    if (!session || busy) return;
    setLocalBusy(true); setError(""); setUploads([]);
    try {
      for (const file of files) {
        setUploads(items => [...items, { filename: file.name, status: "processing" }]);
        const form = new FormData(); form.append("file", file);
        let result;
        try {
          if (file.size > 20 * 1024 * 1024) throw new Error("The file exceeds the 20 MB limit.");
          result = await api<{ filename: string; status: string; detail?: string }>("/documents", session, { method: "POST", body: form });
        } catch (reason) {
          if (reason instanceof ApiError && [409, 410].includes(reason.status)) throw reason;
          result = { filename: file.name, status: "failed", detail: reason instanceof Error ? reason.message : "Upload failed" };
        }
        setUploads(items => [...items.slice(0, -1), result]);
      }
      setFiles([]);
      await refresh(session);
    } catch (reason) { handleError(reason); }
    finally { setLocalBusy(false); }
  }

  async function send() {
    if (!session || busy || !question.trim()) return;
    const sent = question.trim();
    const previousOperation = snapshot?.operation?.id;
    setLocalBusy(true); setPending(sent); setError(""); setEvents([]); setSelected(undefined); setPanel("research");
    let failed = false;
    try {
      await research(session, sent, ({ event, data }) => {
        if (event === "progress") setEvents(items => [...items, data as Trace]);
        if (event === "answer") { setSelected((data as Message).id); setQuestion(""); }
        if (event === "error") { failed = true; setError((data as { detail: string }).detail); }
      }, activeWorkflow?.id || "default", previousOperation);
    } catch (reason) { failed = true; handleError(reason); }
    finally {
      try {
        const state = await refresh(session);
        if (state.operation?.kind === "chat" && state.operation.id !== previousOperation && state.operation.status === "complete") { failed = false; setError(""); setQuestion(""); setSelected(state.messages.at(-1)?.id); }
      } catch (reason) { handleError(reason); }
      if (failed) setQuestion(sent);
      setLocalBusy(false);
    }
  }

  const activeMessage = snapshot?.messages.find(message => message.id === selected);
  const shownEvents = activeMessage?.trace || events;
  const runningEvent = events.at(-1);
  function inspect(message: Message, number?: number) {
    setSelected(message.id); setSourceNumber(number); setPanel(number ? "sources" : "research");
    if (window.innerWidth < 1200) setDrawer("research");
  }

  const documents = <>
    <div className="panel-heading"><span>Your library</span><span className="count">{snapshot?.documents.length || 0} / 10</span></div>
    <div className="upload-zone" onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); chooseFiles(Array.from(event.dataTransfer.files)); }}>
      <Upload size={23} /><strong>Add your documents</strong><span>Drop files here or browse</span>
      <label className={`secondary-button ${busy ? "disabled" : ""}`}><Plus size={15} /> Choose files<input aria-label="Choose documents" type="file" multiple accept=".pdf,.docx,.txt,.md" disabled={busy || !snapshot} onChange={event => { chooseFiles(Array.from(event.target.files || [])); event.target.value = ""; }} /></label>
      <small>PDF, DOCX, TXT, MD · up to 20 MB</small>
    </div>
    {files.length > 0 && <div className="selected-files"><p>{files.length} file{files.length !== 1 ? "s" : ""} selected</p>{files.map((file, index) => <div key={index}><span>{file.name}</span><button aria-label={`Remove ${file.name}`} disabled={busy} onClick={() => setFiles(items => items.filter((_, i) => i !== index))}><X size={14} /></button></div>)}<button className="primary-button" disabled={busy || !snapshot?.healthy} onClick={processFiles}>{busy ? "Processing…" : "Process documents"}</button></div>}
    <div aria-live="polite" className="upload-results">{uploads.map((item, i) => <p key={i} className={item.status === "failed" ? "failure" : ""}><strong>{item.filename}</strong><span>{item.status}{item.detail ? ` · ${item.detail}` : ""}</span></p>)}</div>
    <div className="document-list">{snapshot?.documents.map(doc => <div className="document" key={doc.id}><FileText size={19} /><div><strong title={doc.filename}>{doc.filename}</strong><small>{doc.chunks} searchable passages</small></div><Check size={13} className="teal" /></div>)}</div>
    {!snapshot?.documents.length && <p className="sidebar-hint">Your documents stay together in this temporary workspace.</p>}
    <div className="sidebar-footer"><span className="eyebrow">SESSION STORAGE</span><p>Documents expire after 1 hour of inactivity. Refreshing keeps your workspace.</p><button className="text-button" disabled={busy || loading} onClick={reset}><Trash2 size={14} /> Clear session</button></div>
  </>;

  const evidence = <>
    <div className="panel-heading"><span>Behind the answer</span><FlaskConical size={16} /></div>
    <div className="tabs" role="tablist" aria-label="Evidence panel"><button role="tab" aria-selected={panel === "sources"} onClick={() => setPanel("sources")}>Sources {activeMessage?.sources.length ? `(${activeMessage.sources.length})` : ""}</button><button role="tab" aria-selected={panel === "research"} onClick={() => setPanel("research")}>Research</button></div>
    <div className="evidence-content">{panel === "sources" ? activeMessage?.sources.length ? activeMessage.sources.map(source => <article className={`source-card ${source.number === sourceNumber ? "selected" : ""}`} key={source.number}><div><span className="source-number">{source.number}</span><strong>{source.filename}</strong></div><small>{source.location}</small><p>{source.text}</p></article>) : <div className="panel-empty"><BookOpen size={28} /><h3>Every answer has a source.</h3><p>Select a citation in an answer to explore the supporting passage.</p></div> : shownEvents.length ? <ResearchTimeline events={shownEvents} /> : <div className="panel-empty"><Layers size={28} /><h3>Follow the research.</h3><p>Watch focused searches, evidence checks, and follow-up research take shape here.</p><div className="steps"><span>Plan</span><ChevronRight size={12} /><span>Search</span><ChevronRight size={12} /><span>Verify</span></div></div>}</div>
  </>;

  return <div className="workspace">
    <header className="topbar"><a className="brand" href="/"><span className="brand-mark"><Layers size={20} /></span>folio<span className="brand-divider" /> <small>Document research</small></a><div className="topbar-actions"><span className="session-badge"><span /> Temporary workspace</span><button className="workflow-open" aria-label="Configure agents" onClick={() => setWorkflowBuilderOpen(true)}><Settings2 size={16} /><span className="workflow-full-label">Configure agents</span><span className="workflow-short-label">Agents</span></button><button className="mobile-documents icon-button" aria-label="Open documents" onClick={() => setDrawer("documents")}><FolderOpen size={20} /></button><button className="mobile-research icon-button" aria-label="Open research" onClick={() => setDrawer("research")}><FlaskConical size={20} /></button></div></header>
    <aside className="left-panel">{documents}</aside>
    <main className="chat-panel"><div className="chat-header"><div><span className="eyebrow">YOUR RESEARCH SPACE</span><h1>Ask. Explore. Understand.</h1><small className="active-workflow">Workflow: {activeWorkflow?.name || "Default research"}{activeWorkflow ? ` · v${activeWorkflow.version}` : ""}</small></div><span className="grounded"><BookOpen size={14} /> Grounded in your files</span></div>
      <section className="agent-callout" aria-label="Configurable agents">
        <span className="agent-callout-icon"><Settings2 size={18} /></span>
        <div className="agent-callout-copy"><strong>Make the research agents work your way.</strong><p>Current: {activeWorkflow?.name || "Default research"}{activeWorkflow ? ` · version ${activeWorkflow.version}` : ""}. The default plans searches, retrieves your documents, checks evidence, and writes a cited answer. Configure their settings and connections for your task.</p></div>
        <div className="agent-callout-actions"><a href="/agents/">How the agents work <ChevronRight size={14} /></a><button onClick={() => setWorkflowBuilderOpen(true)}>Configure agents</button></div>
      </section>
      {error && <div className="alert" role="alert">{error}<button onClick={expired ? reset : initialize} disabled={localBusy}>{expired ? "Start new session" : "Reconnect"}</button></div>}
      {!loading && snapshot && !configured && <div className="alert">Answering is not configured. Set GROQ_API_KEY in the backend .env and restart. You can still upload documents.</div>}
      {snapshot && !snapshot.healthy && <div className="alert">The document library needs to be cleared before continuing.</div>}
      <div className="conversation">
        {loading ? <div className="welcome"><LoaderCircle className="spin" /><p>Opening your workspace…</p></div> : !snapshot?.messages.length && !pending ? <div className="welcome"><div className="welcome-symbol"><BookOpen size={30} /></div><span className="eyebrow">LESS SEARCHING. MORE UNDERSTANDING.</span><h2>Your documents,<br /><em>a clearer picture.</em></h2><p>Bring your files. Ask a question. Follow the evidence<br className="desktop-break" /> as your research assistant connects the dots.</p><div className="suggestions">{["Summarize the key points", "Compare the agreements", "Find the important dates"].map(text => <button key={text} onClick={() => setQuestion(text)}><Search size={14} />{text}<ChevronRight size={14} /></button>)}</div><small><span className="step-badge">1</span> Add documents <span className="step-line" /> <span className="step-badge">2</span> Start a conversation</small></div> : snapshot?.messages.map(message => <article key={message.id} className={`message ${message.role}`}><div className="message-author">{message.role === "assistant" ? <><span className="mini-mark"><Layers size={13} /></span> Folio {message.workflow && <small>· {message.workflow.name} v{message.workflow.version}</small>}</> : "You"}</div><div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ({ href, children }) => href?.startsWith("#source-") ? <button className="citation" aria-label={`Source ${href.slice(8)}`} onClick={() => inspect(message, Number(href.slice(8)))}>{children}</button> : <a href={href} target="_blank" rel="noreferrer">{children}</a> }}>{message.content.replace(/\[(\d+)\]/g, (match, number) => message.sources.some(source => source.number === Number(number)) ? `[${number}](#source-${number})` : match)}</ReactMarkdown></div>{message.role === "assistant" && <button className="inspect-button" onClick={() => inspect(message)}><FlaskConical size={13} /> View research <span>· {message.sources.length} sources</span><ChevronRight size={12} /></button>}</article>)}
        {pending && <><article className="message user"><div className="message-author">You</div><p>{pending}</p></article><div className="researching" role="status"><LoaderCircle className="spin" size={16} />{runningEvent ? label(runningEvent) : "Starting research…"}</div></>}
        <div ref={end} />
      </div>
      <div className="composer-area"><form className="composer" onSubmit={event => { event.preventDefault(); void send(); }}><textarea aria-label="Ask about your documents" placeholder="What would you like to understand?" rows={2} maxLength={2000} value={question} disabled={busy || expired} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (configured && snapshot?.healthy && snapshot.documents.length) void send(); } }} /><div className="composer-bottom"><span><FileText size={13} />{snapshot?.documents.length || 0} documents in context</span><button className="send-button" aria-label="Send question" disabled={busy || !question.trim() || !configured || !snapshot?.healthy || !snapshot.documents.length || expired}><ArrowUp size={18} /></button></div></form><p>Answers are based on your documents. Check sources for important decisions.</p></div>
    </main><aside className="right-panel">{evidence}</aside>
    <dialog ref={dialog} className="drawer" aria-label={drawer === "documents" ? "Documents" : "Evidence"} onCancel={() => setDrawer(null)} onClick={event => { if (event.target === event.currentTarget) setDrawer(null); }}>{drawer && <div className="drawer-inner"><button className="drawer-close icon-button" aria-label="Close panel" onClick={() => setDrawer(null)}><X /></button>{drawer === "documents" ? documents : evidence}</div>}</dialog>
    <WorkflowBuilder open={workflowBuilderOpen} session={session} activeId={activeWorkflow?.id || "default"}
      onClose={() => setWorkflowBuilderOpen(false)} onActivate={saved => {
        setActiveWorkflow(saved); sessionStorage.setItem(WORKFLOW_KEY, saved.id);
      }} />
  </div>;
}
