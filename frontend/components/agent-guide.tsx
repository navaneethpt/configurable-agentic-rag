"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, BookOpen, CheckCircle2, GitBranch, Layers, Settings2 } from "lucide-react";
import { api, ApiError } from "../lib/api";
import { forgetSession, getSession, WORKFLOW_KEY } from "../lib/session";
import { NodeType, SavedWorkflow } from "../lib/workflows";

const explanations: Record<string, string> = {
  planner: "Turns your question and recent conversation into focused search queries. Validator feedback can send it back to plan another search.",
  retrieve: "Searches your session documents with earlier search queries or directly with your original question. It returns unique passages with file and location details.",
  validate: "Checks whether those passages cover the material parts of the question. It decides whether to answer, search again, or request missing documents.",
  generate: "Writes from available passages, with or without a validator. If passages are missing, it explains what context is needed. It can produce an intermediate draft or finish with a cited answer.",
  need_upload: "Describes missing evidence, or requests relevant documents when no specific gaps exist. It can finish the workflow or pass the request to another node.",
};

const outcomeLabels: Record<string, string> = {
  next: "Continue", sufficient: "Enough evidence", retry: "Search again", needs_upload: "Need documents",
};

export default function AgentGuide() {
  const [types, setTypes] = useState<NodeType[]>([]);
  const [defaultFlow, setDefaultFlow] = useState<SavedWorkflow>();
  const [activeFlow, setActiveFlow] = useState<SavedWorkflow>();
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load(session: string) {
      const [catalog, workflow] = await Promise.all([
        api<NodeType[]>("/node-types"), api<SavedWorkflow>("/workflows/default", session),
      ]);
      if (cancelled) return;
      setTypes(catalog); setDefaultFlow(workflow);
      const activeId = sessionStorage.getItem(WORKFLOW_KEY);
      if (activeId && activeId !== "default") {
        try {
          const active = await api<SavedWorkflow>(`/workflows/${activeId}`, session);
          if (!cancelled) setActiveFlow(active);
        } catch { if (!cancelled) { sessionStorage.removeItem(WORKFLOW_KEY); setActiveFlow(workflow); } }
      } else setActiveFlow(workflow);
    }
    getSession().then(async session => {
      try { await load(session); }
      catch (reason) {
        if (!(reason instanceof ApiError) || reason.status !== 410) throw reason;
        forgetSession();
        await load(await getSession());
      }
    })
      .catch(reason => { if (!cancelled) setError(reason instanceof Error ? reason.message : "Could not load the agent configuration."); });
    return () => { cancelled = true; };
  }, []);

  function label(type: string) { return types.find(item => item.type === type)?.label || type; }
  function setting(type: string, key: string) {
    const node = defaultFlow?.nodes.find(item => item.type === type);
    const definition = types.find(item => item.type === type);
    return node?.config[key] ?? definition?.config_schema.properties?.[key]?.default ?? "—";
  }

  return <main className="agent-guide">
    <header className="guide-topbar"><a className="guide-brand" href="/"><span className="brand-mark"><Layers size={20} /></span>folio</a>
      <a className="guide-return" href="/"><ArrowLeft size={15} /> Back to research</a></header>
    <div className="guide-content">
      <section className="guide-hero"><span className="eyebrow">CONFIGURABLE DOCUMENT RESEARCH</span>
        <h1>How Folio’s agents work together</h1>
        <p>Each agent has its own function and handles the inputs available to it. Configure which agents and tools run, how they connect, and which one returns the final response. Your documents form a temporary searchable library.</p>
        <div className="guide-hero-actions"><a className="guide-primary" href="/?configure=agents"><Settings2 size={16} /> Configure agents for your task</a>
          <a className="guide-secondary" href="/">Ask your documents <ArrowRight size={15} /></a></div>
        {activeFlow && <p className="guide-active"><CheckCircle2 size={15} /> Currently selected in this tab: <strong>{activeFlow.name} · version {activeFlow.version}</strong></p>}
      </section>

      <section className="guide-section" aria-labelledby="independent-agents-title">
        <span className="eyebrow">INDEPENDENT AGENTS</span><h2 id="independent-agents-title">Each agent works independently</h2>
        <p>Every node receives your original question, recent conversation, and the ordered outputs of earlier steps on the executed path. It produces its own structured result and chooses a routing outcome. Missing earlier results are handled when the agent runs; no planner or validator is required before another agent.</p>
        <p><strong>Generator alone:</strong> asks for missing context; it does not automatically search uploaded documents. <strong>Retrieval → Generator:</strong> searches directly with your question, then writes from supporting passages without a planner or validator.</p>
        <p><strong>Planner → Generator:</strong> creates search queries, then asks for context because no passages were retrieved. <strong>Validator → Generator:</strong> reports insufficient evidence, then requests context. Upload a document before running any workflow.</p>
        <p>Answer generator and Request more evidence can finish or continue. Intermediate drafts and requests go to later nodes; only the terminal response becomes the assistant message. Drafts are not document evidence, and validation from before a later retrieval does not validate the newer evidence. Repeated nodes keep separate results. Folio executes exactly your saved graph, sequentially, with bounded retry loops.</p>
      </section>

      <section className="guide-section" aria-labelledby="model-provider-title">
        <span className="eyebrow">MODEL PROVIDER</span><h2 id="model-provider-title">Groq or Google Gemini</h2>
        <p>Gemini is the default provider, using <code>gemini-3.8-flash</code> when no model override is configured. The server administrator selects one provider for all workflows using <code>LLM_PROVIDER=gemini</code> or <code>LLM_PROVIDER=groq</code>. Gemini uses a Google AI Studio key in <code>GEMINI_API_KEY</code>; Groq uses <code>GROQ_API_KEY</code>. Keys stay on the server.</p>
        <p>Set an agent’s model to <code>default</code> to use the server’s model, configured through <code>LLM_MODEL</code>. You can override individual agents with another model ID from the selected provider. Switching providers requires restarting the backend; explicit model overrides must match the new provider.</p>
        <p>Older saved model settings of <code>openai/gpt-oss-20b</code> follow the selected provider’s default for compatibility. Your workflow connections and the agents’ individual functions stay the same. Document search and embeddings run locally; questions, conversation, and retrieved excerpts go to the selected provider.</p>
      </section>

      <section className="guide-section" aria-labelledby="default-flow-title">
        <div className="guide-heading"><div><span className="eyebrow">STARTING CONFIGURATION</span><h2 id="default-flow-title">The default research workflow</h2></div>
          <span className="guide-pill"><GitBranch size={14} /> {defaultFlow?.nodes.length || 5} connected nodes</span></div>
        <p>Folio starts with three reasoning agents, one document-search tool, and an agent that requests missing evidence. These are the actual connections in the saved default workflow:</p>
        {error && <p className="guide-error" role="alert">{error}</p>}
        {!defaultFlow && !error && <p role="status">Loading the default configuration…</p>}
        {defaultFlow && <div className="guide-flow">{defaultFlow.nodes.map((node, index) => {
          const definition = types.find(item => item.type === node.type);
          return <article className="guide-node" key={node.id}>
            <div className="guide-node-head"><span className="guide-number">{index + 1}</span><div><small>{definition?.kind || "node"}</small><h3>{definition?.label || node.type}</h3></div></div>
            <p>{explanations[node.type] || definition?.description}</p>
            <div className="guide-edges">{Object.entries(node.transitions).length
              ? Object.entries(node.transitions).map(([outcome, target]) => {
                const next = defaultFlow.nodes.find(item => item.id === target);
                return <span key={outcome}>{outcomeLabels[outcome] || outcome} <ArrowRight size={12} /> {label(next?.type || target)}</span>;
              })
              : <span>{node.type === "generate" ? "Cited answer" : "Request the missing evidence"}</span>}</div>
          </article>;
        })}</div>}
      </section>

      <section className="guide-section guide-two-column" aria-label="Default behavior and configuration">
        <div className="guide-panel"><span className="eyebrow">WHAT HAPPENS BY DEFAULT</span><h2>Bounded research, with an evidence check</h2>
          <ul>
            <li>Planner: up to <strong>{String(setting("planner", "max_searches"))} searches</strong> per round.</li>
            <li>Retriever: up to <strong>{String(setting("retrieve", "max_results_per_search"))} passages</strong> per search and <strong>{String(setting("retrieve", "max_evidence"))} unique passages</strong> overall.</li>
            <li>Validator: up to <strong>{String(setting("validate", "max_rounds"))} rounds</strong>. If evidence is still incomplete in the final round, confidence must reach <strong>{Math.round(Number(setting("validate", "final_confidence")) * 100)}%</strong> to generate a partial answer.</li>
            <li>The answer generator cites retrieved passages. If evidence is missing, the workflow asks for relevant documents.</li>
          </ul>
          <p className="guide-note"><BookOpen size={15} /> A citation points to a passage; it does not independently prove every claim is correct.</p>
        </div>
        <div className="guide-panel guide-customize"><span className="eyebrow">MAKE THE AGENTS YOURS</span><h2>Configure the workflow for your requirement</h2>
          <ol>
            <li>Open <strong>Configure agents</strong> and copy the default workflow.</li>
            <li>Add or remove registered agents and tools, change their settings, and arrange them for your task. Every node receives your original question and all earlier outputs.</li>
            <li>Connect routing outcomes, choose the start node, and finish each path with Answer generator or Request more evidence. These two types can also continue to another node. Save to activate your workflow.</li>
          </ol>
          <p>A generator alone asks for missing context. Retrieval followed by generation searches directly with your question; planner and validator are optional. Upload a document before asking questions.</p>
          <p>Saved workflows are visible only in this session. Earlier answers keep the workflow version that produced them. New agent and tool types registered on the server appear in the builder automatically.</p>
          <a className="guide-primary" href="/?configure=agents"><Settings2 size={16} /> Configure agents</a>
        </div>
      </section>
    </div>
  </main>;
}
