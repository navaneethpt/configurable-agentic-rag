export type Source = { number: number; filename: string; location: string; text: string };
export type Trace = {
  event: string; node?: string; node_type?: string; label?: string; step?: number; terminal?: boolean; output_kind?: string; round?: number; query?: string; results?: number;
  new_sources?: number; total_sources?: number; decision?: string; outcome?: string;
  reason?: string; missing_evidence?: string[];
  confidence?: number; accepted?: boolean; acceptance_reason?: string;
  searches?: { query: string; purpose: string; new_sources?: number }[];
};
export type WorkflowRef = { id: string; name: string; version: number };
export type Message = { id: string; role: "user" | "assistant"; content: string; sources: Source[]; trace: Trace[]; workflow?: WorkflowRef };
export type Snapshot = {
  id: string; healthy: boolean;
  documents: { id: string; filename: string; chunks: number }[];
  messages: Message[];
  operation: null | { id: string; kind: string; status: string; question: string; events: Trace[]; error: string | null; workflow?: WorkflowRef };
};
export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}
export async function api<T>(path: string, session?: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init, cache: "no-store",
    headers: { ...(session ? { "X-Session-ID": session } : {}), ...init.headers },
  });
  await check(response);
  return response.status === 204 ? undefined as T : response.json();
}
async function check(response: Response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(typeof body.detail === "string" ? body.detail : "The request could not complete. Please try again.", response.status);
  }
}
export type StreamEvent = { event: string; data: unknown };
export async function readEvents(body: ReadableStream<Uint8Array>, receive: (event: StreamEvent) => void) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;
  function consume() {
    buffer = buffer.replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const lines = block.split("\n");
      const event = lines.find(line => line.startsWith("event:"))?.slice(6).trim();
      const data = lines.filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
      if (event && data) {
        receive({ event, data: JSON.parse(data) });
        if (event === "answer" || event === "error") { terminal = true; return; }
      }
    }
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      consume();
      if (terminal || done) break;
    }
    if (!terminal) throw new Error("Connection interrupted. Checking research status…");
  } finally {
    if (terminal) void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
export async function research(session: string, question: string, receive: (event: StreamEvent) => void,
                               workflowId = "default", previousOperationId?: string) {
  const controller = new AbortController();
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const streamed = async () => {
    const response = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json", "X-Session-ID": session },
      body: JSON.stringify({ question, workflow_id: workflowId }), signal: controller.signal,
    });
    await check(response);
    if (!response.body) throw new Error("The research stream was unavailable.");
    await readEvents(response.body, receive);
  };
  // A proxy can keep SSE open or lose its final frame. Recover the saved result
  // while streaming, without submitting the question a second time.
  const recovered = new Promise<void>((resolve, reject) => {
    const poll = async () => {
      try {
        const state = await api<Snapshot>("/session", session, { signal: controller.signal });
        if (stopped) return;
        const operation = state.operation;
        if (operation?.kind === "chat" && operation.id !== previousOperationId && operation.question === question) {
          if (operation.status === "complete") {
            const answer = state.messages.filter(message => message.role === "assistant").at(-1);
            if (answer) { receive({ event: "answer", data: answer }); resolve(); return; }
          }
          if (operation.status === "failed") {
            receive({ event: "error", data: { detail: operation.error || "Research failed." } });
            resolve(); return;
          }
        }
      } catch (reason) {
        if (stopped) return;
        if (reason instanceof ApiError && reason.status === 410) { reject(reason); return; }
        // A temporary polling failure does not discard the live stream.
      }
      if (!stopped) timer = setTimeout(poll, 1000);
    };
    timer = setTimeout(poll, 1000);
  });
  try { await Promise.race([streamed(), recovered]); }
  finally { stopped = true; clearTimeout(timer); controller.abort(); }
}
