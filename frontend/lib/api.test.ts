import { afterEach, describe, expect, it, vi } from "vitest";
import { Message, readEvents, research, Snapshot } from "./api";

function stream(text: string, stride = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({ start(controller) {
    for (let i = 0; i < bytes.length; i += stride) controller.enqueue(bytes.slice(i, i + stride));
    controller.close();
  } });
}
describe("SSE parser", () => {
  it("handles split frames, UTF-8 and keepalives", async () => {
    const events: unknown[] = [];
    await readEvents(stream(': keepalive\n\nevent: progress\ndata: {"query":"café"}\n\nevent: answer\ndata: {"content":"June"}\n\n'), event => events.push(event));
    expect(events).toEqual([{ event: "progress", data: { query: "café" } }, { event: "answer", data: { content: "June" } }]);
  });
  it("supports CRLF framing and error terminal events", async () => {
    const events: unknown[] = [];
    await readEvents(stream('event: error\r\ndata: {"detail":"Retry"}\r\n\r\n'), event => events.push(event));
    expect(events).toEqual([{ event: "error", data: { detail: "Retry" } }]);
  });
  it("rejects a truncated stream instead of submitting again", async () => {
    await expect(readEvents(stream('event: progress\ndata: {}\n\n'), () => {})).rejects.toThrow("interrupted");
  });
  it.each(["answer", "error"])("finishes on %s without waiting for the connection to close", async event => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new TextEncoder().encode(`event: ${event}\ndata: {}\n\n`)); },
      cancel() { cancelled = true; },
    });
    const events: unknown[] = [];
    await readEvents(body, event => events.push(event));
    expect(events).toEqual([{ event, data: {} }]);
    expect(cancelled).toBe(true);
  }, 500);
});

describe("research recovery while streaming", () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
  const answer: Message = { id: "answer-new", role: "assistant", content: "June. [1]", sources: [], trace: [] };
  function snapshot(status = "complete", id = "operation-new"): Snapshot {
    return { id: "session", healthy: true, documents: [], messages: status === "complete" ? [answer] : [],
      operation: { id, kind: "chat", status, question: "When?", events: [], error: status === "failed" ? "Provider unavailable" : null } };
  }
  function mockFetch(states: (Snapshot | number)[]) {
    const body = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new TextEncoder().encode('event: progress\ndata: {"event":"generate"}\n\n')); },
    });
    const fetch = vi.fn(async (path: string) => {
      if (path === "/api/chat") return new Response(body);
      const state = states.shift();
      return typeof state === "number" ? new Response("{}", { status: state }) : new Response(JSON.stringify(state));
    });
    vi.stubGlobal("fetch", fetch);
    return fetch;
  }
  it("recovers a completed answer when only progress arrives and the stream stays open", async () => {
    vi.useFakeTimers();
    const fetch = mockFetch([snapshot()]);
    const receive = vi.fn();
    const pending = research("session", "When?", receive, "default", "operation-old");
    await vi.advanceTimersByTimeAsync(1000);
    await pending;
    expect(receive).toHaveBeenLastCalledWith({ event: "answer", data: answer });
    expect(fetch.mock.calls.filter(([path]) => path === "/api/chat")).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(3000);
    expect(fetch).toHaveBeenCalledTimes(2);
  });
  it("ignores the previous answer to the same question and waits for the new operation", async () => {
    vi.useFakeTimers();
    mockFetch([snapshot("complete", "operation-old"), snapshot("running"), snapshot()]);
    const receive = vi.fn();
    const pending = research("session", "When?", receive, "default", "operation-old");
    await vi.advanceTimersByTimeAsync(2000);
    expect(receive.mock.calls.filter(([event]) => event.event === "answer")).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1000);
    await pending;
    expect(receive).toHaveBeenLastCalledWith({ event: "answer", data: answer });
  });
  it("recovers the saved failure instead of waiting forever for an error frame", async () => {
    vi.useFakeTimers();
    mockFetch([snapshot("failed")]);
    const receive = vi.fn();
    const pending = research("session", "When?", receive);
    await vi.advanceTimersByTimeAsync(1000);
    await pending;
    expect(receive).toHaveBeenLastCalledWith({ event: "error", data: { detail: "Provider unavailable" } });
  });
  it("retries a temporary polling failure without resubmitting chat", async () => {
    vi.useFakeTimers();
    const fetch = mockFetch([503, snapshot()]);
    const receive = vi.fn();
    const pending = research("session", "When?", receive);
    await vi.advanceTimersByTimeAsync(2000);
    await pending;
    expect(receive).toHaveBeenLastCalledWith({ event: "answer", data: answer });
    expect(fetch.mock.calls.filter(([path]) => path === "/api/chat")).toHaveLength(1);
  });
  it("stops polling once the normal answer frame arrives", async () => {
    vi.useFakeTimers();
    const fetch = vi.fn(async () => new Response(stream(`event: answer\ndata: ${JSON.stringify(answer)}\n\n`)));
    vi.stubGlobal("fetch", fetch);
    await research("session", "When?", () => {});
    await vi.advanceTimersByTimeAsync(3000);
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
