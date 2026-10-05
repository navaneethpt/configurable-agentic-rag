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
        if (event === "answer" || event === "error") terminal = true;
      }
    }
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      consume();
      if (done) break;
    }
    if (!terminal) throw new Error("Connection interrupted. Checking research status…");
  } finally { reader.releaseLock(); }
}
export async function research(session: string, question: string, receive: (event: StreamEvent) => void, workflowId = "default") {
  const response = await fetch("/api/chat", {
    method: "POST", headers: { "Content-Type": "application/json", "X-Session-ID": session },
    body: JSON.stringify({ question, workflow_id: workflowId }),
  });
  await check(response);
  if (!response.body) throw new Error("The research stream was unavailable.");
  await readEvents(response.body, receive);
}
