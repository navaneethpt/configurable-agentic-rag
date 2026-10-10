import { expect, test, Page } from "@playwright/test";

async function upload(page: Page) {
  await page.goto("/");
  const input = page.getByLabel("Choose documents");
  await expect(input).toBeEnabled();
  await input.setInputFiles({ name: "launch.md", mimeType: "text/markdown", buffer: Buffer.from("Project Cedar launches in June.") });
  await page.getByRole("button", { name: "Process documents" }).click();
  await expect(page.getByText("1 searchable passages")).toBeVisible();
}
async function ask(page: Page, question: string) {
  await page.getByRole("textbox", { name: "Ask about your documents" }).fill(question);
  await page.getByRole("button", { name: "Send question" }).click();
}

test("upload, cited answer, historical evidence, refresh and clear", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await upload(page);
  await ask(page, "When does it launch?");
  await expect(page.getByRole("button", { name: "Source 1", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Source 1", exact: true }).click();
  await expect(page.locator(".source-card.selected")).toContainText("Project Cedar launches in June.");
  await ask(page, "What is the annual budget?");
  await expect(page.locator(".message.assistant").last()).toContainText("Please upload");
  await expect(page.locator(".message.assistant").last()).toContainText("annual budget document");
  await page.locator(".message.assistant").first().getByRole("button", { name: /View research/ }).click();
  await expect(page.locator(".right-panel")).toContainText("Answer checked and ready");
  await page.reload();
  await expect(page.locator(".message.assistant")).toHaveCount(2);
  await expect(page.locator(".right-panel")).toContainText("Round 3");
  await page.screenshot({ path: "test-results/workspace-desktop.png", fullPage: true });
  page.on("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Clear session" }).click();
  await expect(page.getByRole("heading", { name: /Your documents,/ })).toBeVisible();
  await expect(page.locator(".document")).toHaveCount(0);
});

test("failed file does not prevent good upload and session can expire", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  await page.getByLabel("Choose documents").setInputFiles([
    { name: "bad.pdf", mimeType: "application/pdf", buffer: Buffer.from("bad") },
    { name: "good.txt", mimeType: "text/plain", buffer: Buffer.from("June launch") },
  ]);
  await page.getByRole("button", { name: "Process documents" }).click();
  await expect(page.locator(".failure")).toContainText("bad.pdf");
  await expect(page.getByText("1 searchable passages")).toBeVisible();
  await request.post("http://127.0.0.1:8100/test/expire");
  await page.reload();
  await expect(page.locator("main").getByRole("alert")).toContainText("expired");
  await page.getByRole("button", { name: "Start new session" }).click();
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  await expect(page.locator(".document")).toHaveCount(0);
});

test("reload during research recovers without submitting twice", async ({ page }) => {
  await upload(page);
  await ask(page, "When is launch?");
  await expect(page.getByRole("status")).toContainText(/Planning|Searching|Checking|Preparing/);
  await page.reload();
  await expect(page.locator(".message.assistant")).toHaveCount(1);
  await expect(page.locator(".message.user")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "Source 1", exact: true })).toBeVisible();
});

for (const dropTerminal of [false, true]) {
  test(`shows the answer when the proxy ${dropTerminal ? "drops the final event" : "keeps the completed stream open"}`, async ({ page }) => {
    await page.addInitScript(dropTerminal => {
      const fetch = window.fetch.bind(window);
      window.fetch = async (input, init) => {
        const response = await fetch(input, init);
        const url = input instanceof Request ? input.url : String(input);
        if (new URL(url, window.location.origin).pathname !== "/api/chat" || !response.ok) return response;
        const text = await response.text();
        const forwarded = dropTerminal ? text.split("\n\n").filter(frame => !/^event: (answer|error)\n/.test(frame)).join("\n\n") : text;
        const abort = () => controller?.error(new DOMException("Aborted", "AbortError"));
        let controller: ReadableStreamDefaultController<Uint8Array>;
        const body = new ReadableStream<Uint8Array>({
          start(value) {
            controller = value;
            controller.enqueue(new TextEncoder().encode(forwarded));
            if (init?.signal?.aborted) abort();
            else init?.signal?.addEventListener("abort", abort, { once: true });
            // Deliberately leave the response open to reproduce a stalled proxy.
          },
          cancel() { init?.signal?.removeEventListener("abort", abort); },
        });
        return new Response(body, { status: response.status, headers: response.headers });
      };
    }, dropTerminal);
    let submissions = 0;
    page.on("request", request => { if (new URL(request.url()).pathname === "/api/chat") submissions++; });
    await upload(page);
    await ask(page, "When does it launch?");
    await expect(page.locator(".right-panel")).toContainText("Answer checked and ready");
    await expect(page.locator(".message.assistant")).toContainText("Project Cedar launches in June.");
    await expect(page.getByRole("button", { name: "Source 1", exact: true })).toBeVisible();
    await expect(page.getByRole("textbox", { name: "Ask about your documents" })).toBeEnabled();
    await expect(page.getByRole("status")).toHaveCount(0);
    expect(submissions).toBe(1);
    await page.reload();
    await expect(page.locator(".message.assistant")).toHaveCount(1);
    await expect(page.locator(".message.user")).toHaveCount(1);
  });
}

test("failed chat keeps the question available for manual retry", async ({ page }) => {
  await upload(page);
  await page.route("**/api/chat", route => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Provider temporarily unavailable" }) }));
  await ask(page, "When is launch?");
  await expect(page.locator("main").getByRole("alert")).toContainText("Provider temporarily unavailable");
  await expect(page.getByRole("textbox", { name: "Ask about your documents" })).toHaveValue("When is launch?");
  await expect(page.getByRole("button", { name: "Send question" })).toBeEnabled();
  await expect(page.locator(".message.assistant")).toHaveCount(0);
  await page.unroute("**/api/chat");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".message.assistant")).toHaveCount(1);
});

test("mobile drawers, citation and layout", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "Open documents" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("dialog").getByLabel("Choose documents").setInputFiles({ name: "launch.txt", mimeType: "text/plain", buffer: Buffer.from("Launch in June") });
  await page.getByRole("dialog").getByRole("button", { name: "Process documents" }).click();
  await expect(page.getByRole("dialog")).toContainText("1 searchable passages");
  await page.getByRole("button", { name: "Close panel" }).click();
  await ask(page, "When is launch?");
  await page.getByRole("button", { name: "Source 1", exact: true }).click();
  await expect(page.getByRole("dialog").locator(".source-card")).toContainText("June");
  await page.screenshot({ path: "test-results/workspace-mobile-sources.png", fullPage: true });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.screenshot({ path: "test-results/workspace-mobile.png", fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});

test("proxy forwards files larger than the default Next.js 10 MB limit", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  const id = await page.evaluate(() => sessionStorage.getItem("folio-session"));
  const response = await request.post("/api/documents", {
    headers: { "X-Session-ID": id! },
    multipart: { file: { name: "large.txt", mimeType: "text/plain", buffer: Buffer.alloc(11 * 1024 * 1024, "x") } },
  });
  expect(response.status()).toBe(200);
  expect(await response.json()).toMatchObject({ status: "failed", detail: "Too much extracted text. Split the document into smaller files." });
});

test("third-attempt 50 percent acceptance is visible in research", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await upload(page);
  await ask(page, "Give a partial answer about launch");
  await expect(page.getByRole("button", { name: "Source 1", exact: true })).toBeVisible();
  await expect(page.locator(".right-panel")).toContainText("Passed third-attempt confidence threshold");
  await expect(page.locator(".right-panel")).toContainText("Validator confidence: 50%");
  await expect(page.locator(".right-panel")).toContainText("Round 3");
});

test("workflow builder saves and activates settings used by research", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await upload(page);
  await page.locator(".topbar").getByRole("button", { name: "Configure agents" }).click();
  const builder = page.getByRole("dialog", { name: "Configure agents" });
  await expect(builder).toBeVisible();
  await expect(builder).toContainText("Document retrieval");
  await expect(builder).toContainText("agent");
  await expect(builder).toContainText("tool");
  await expect(builder).toContainText("Every node receives your original question, conversation history, and all earlier outputs");
  const inputs = builder.getByRole("region", { name: "Node inputs and outputs" });
  await expect(inputs).toContainText("question, history, validation, earlier outputs");
  await expect(inputs).toContainText("Plan searches from the original question");
  await builder.locator(".workflow-node-card").nth(1).click();
  await expect(inputs).toContainText("Search with the original question when no search-query output is available.");
  await expect(inputs).toContainText("passages — passages, searches, round");
  await builder.getByRole("textbox", { name: "Workflow name" }).fill("One pass research");
  await builder.getByRole("button", { name: /validate.*Evidence validator/i }).click();
  await expect(inputs).toContainText("Report insufficient evidence when no passages are available; queries are optional.");
  await expect(inputs).toContainText("validation — decision, confidence, missing_evidence");
  await builder.getByLabel("Max Rounds").fill("1");
  await builder.getByLabel("Final Confidence").fill("0.9");
  await page.screenshot({ path: "test-results/workflow-builder.png", fullPage: true });
  await builder.getByRole("button", { name: "Save new workflow and use" }).click();
  await expect(builder).toContainText("Saved version 1");
  await page.getByRole("button", { name: "Close workflow builder" }).click();
  await expect(page.locator(".active-workflow")).toContainText("One pass research · v1");
  await ask(page, "Give a partial answer about launch");
  await expect(page.locator(".message.assistant")).toContainText("Please upload");
  await expect(page.locator(".right-panel")).toContainText("Round 1");
  await expect(page.locator(".right-panel")).not.toContainText("Round 3");
  await page.reload();
  await expect(page.locator(".active-workflow")).toContainText("One pass research · v1");
});

test("retrieval and generation save and execute without planner or validator", async ({ page }) => {
  await upload(page);
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  await page.getByRole("button", { name: "Configure agents" }).first().click();
  const builder = page.getByRole("dialog", { name: "Configure agents" });
  await builder.getByRole("textbox", { name: "Workflow name" }).fill("Direct retrieval");
  await expect(builder.getByLabel("Start node")).toHaveValue("planner");
  await builder.getByRole("button", { name: "Remove planner", exact: true }).click();
  await builder.getByRole("button", { name: /validate.*Evidence validator/ }).click();
  await builder.getByRole("button", { name: "Remove validate", exact: true }).click();
  await builder.getByRole("button", { name: /need_upload.*Request more evidence/ }).click();
  await builder.getByRole("button", { name: "Remove need_upload", exact: true }).click();
  await builder.getByRole("button", { name: /retrieve.*Document retrieval/ }).click();
  await builder.getByLabel("retrieve next target").selectOption("generate");
  await builder.getByRole("button", { name: "Save new workflow and use" }).click();
  await expect(builder).toContainText("Saved version 1");
  await builder.getByRole("button", { name: "Close workflow builder" }).click();
  await ask(page, "When does Project Cedar launch?");
  await expect(page.locator(".message.assistant")).toContainText("Project Cedar launches in June.");
  await expect(page.getByRole("button", { name: "Source 1", exact: true })).toBeVisible();
  await expect(page.locator(".right-panel")).not.toContainText("Search plan ready");
  await expect(page.locator(".right-panel")).not.toContainText("Evidence checked");
});

test("generator alone asks for context without hidden retrieval", async ({ page }) => {
  await upload(page);
  await page.locator(".topbar").getByRole("button", { name: "Configure agents" }).click();
  const builder = page.getByRole("dialog", { name: "Configure agents" });
  await expect(builder.getByLabel("Start node")).toHaveValue("planner");
  await builder.getByRole("button", { name: "Remove planner", exact: true }).click();
  await builder.getByRole("button", { name: "Remove retrieve", exact: true }).click();
  await builder.getByRole("button", { name: "Remove validate", exact: true }).click();
  await builder.getByRole("button", { name: /need_upload.*Request more evidence/ }).click();
  await builder.getByRole("button", { name: "Remove need_upload", exact: true }).click();
  await expect(builder.getByLabel("Start node")).toHaveValue("generate");
  await builder.getByRole("button", { name: "Save new workflow and use" }).click();
  await expect(builder).toContainText("Saved version 1");
  await builder.getByRole("button", { name: "Close workflow builder" }).click();
  await ask(page, "When does Project Cedar launch?");
  await expect(page.locator(".message.assistant")).toContainText("Please upload");
  await expect(page.getByRole("button", { name: "Source 1", exact: true })).toHaveCount(0);
  await expect(page.locator(".right-panel")).not.toContainText("Retrieval complete");
});

test("generator and evidence request can continue with only the terminal response shown", async ({ page }) => {
  await upload(page);
  await page.locator(".topbar").getByRole("button", { name: "Configure agents" }).click();
  const builder = page.getByRole("dialog", { name: "Configure agents" });
  await expect(builder.getByLabel("Start node")).toHaveValue("planner");
  await builder.getByRole("button", { name: "Remove planner", exact: true }).click();
  await builder.getByRole("button", { name: "Remove retrieve", exact: true }).click();
  await builder.getByRole("button", { name: "Remove validate", exact: true }).click();
  await builder.getByRole("button", { name: /generate.*Answer generator/ }).click();
  await builder.getByLabel("generate behavior").selectOption("continue");
  await builder.getByLabel("generate next target").selectOption("need_upload");
  await builder.locator(".workflow-palette").getByRole("button", { name: /Answer generator/ }).click();
  await builder.getByRole("button", { name: "Add selected node" }).click();
  await builder.getByRole("button", { name: /need_upload.*Request more evidence/ }).click();
  await builder.getByLabel("need_upload behavior").selectOption("continue");
  await builder.getByLabel("need_upload next target").selectOption("generate_1");
  await builder.getByRole("button", { name: "Save new workflow and use" }).click();
  await expect(builder).toContainText("Saved version 1");
  await builder.getByRole("button", { name: "Close workflow builder" }).click();
  await ask(page, "When?");
  await expect(page.locator(".message.assistant")).toHaveCount(1);
  await expect(page.locator(".message.assistant")).toContainText("Please upload");
  await expect(page.locator(".right-panel")).toContainText("Intermediate draft ready");
  await expect(page.locator(".right-panel")).toContainText("Evidence request prepared");
});

test("workflow builder identifies disconnected nodes and guides their connections", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  await page.getByRole("button", { name: "Configure agents" }).first().click();
  const builder = page.getByRole("dialog", { name: "Configure agents" });
  await builder.getByRole("textbox", { name: "Workflow name" }).fill("Extra retrieval pass");
  await builder.locator(".workflow-palette").getByRole("button", { name: /Document retrieval/ }).click();
  await builder.getByRole("button", { name: "Add selected node" }).click();
  await expect(builder.getByRole("status")).toContainText("Not connected to Start: retrieve_1");
  await expect(builder.getByRole("status")).toContainText("retrieve_1.next");
  await expect(builder.getByRole("button", { name: "Save new workflow and use" })).toBeDisabled();
  await builder.getByLabel("retrieve_1 next target").selectOption("validate");
  await builder.locator(".workflow-node-card").nth(1).click();
  await builder.getByLabel("retrieve next target").selectOption("retrieve_1");
  await expect(builder.getByRole("status")).toHaveCount(0);
  await builder.getByRole("button", { name: "Save new workflow and use" }).click();
  await expect(builder).toContainText("Saved version 1");
});

test("saved workflows stay in the browser session that created them", async ({ page, browser }) => {
  await page.goto("/");
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  await page.getByRole("button", { name: "Configure agents" }).first().click();
  const builder = page.getByRole("dialog", { name: "Configure agents" });
  await builder.getByRole("textbox", { name: "Workflow name" }).fill("Private research");
  await builder.getByRole("button", { name: "Save new workflow and use" }).click();
  await expect(builder).toContainText("Saved version 1");
  await page.getByRole("button", { name: "Close workflow builder" }).click();

  const otherContext = await browser.newContext();
  try {
    const other = await otherContext.newPage();
    await other.goto("/agents/");
    await expect(other.locator(".guide-node")).toHaveCount(5);
    await other.goto("/");
    await expect(other.getByLabel("Choose documents")).toBeEnabled();
    await other.getByRole("button", { name: "Configure agents" }).first().click();
    const otherBuilder = other.getByRole("dialog", { name: "Configure agents" });
    await expect(otherBuilder.getByRole("option", { name: "Default research" })).toHaveCount(1);
    await expect(otherBuilder.getByRole("option", { name: "Private research" })).toHaveCount(0);
    await expect(other.locator(".active-workflow")).toContainText("Default research");
  } finally { await otherContext.close(); }

  page.on("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Clear session" }).click();
  await expect(page.locator(".active-workflow")).toContainText("Default research");
  await page.getByRole("button", { name: "Configure agents" }).first().click();
  await expect(page.getByRole("dialog", { name: "Configure agents" })
    .getByRole("option", { name: "Private research" })).toHaveCount(0);
});

test("agent guide explains the default flow and opens configuration", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/");
  await expect(page.getByText("Make the research agents work your way.")).toBeVisible();
  await page.getByRole("link", { name: "How the agents work" }).click();
  await expect(page).toHaveURL(/\/agents\/$/);
  await expect(page.getByRole("heading", { name: "How Folio’s agents work together" })).toBeVisible();
  await expect(page.locator(".guide-node")).toHaveCount(5);
  await expect(page.locator(".guide-flow")).toContainText("Search again");
  await expect(page.locator(".guide-panel").first()).toContainText("3 rounds");
  await page.screenshot({ path: "test-results/agent-guide.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("heading", { name: "How Folio’s agents work together" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  await page.screenshot({ path: "test-results/agent-guide-mobile.png", fullPage: true });
  await page.getByRole("link", { name: "Configure agents for your task" }).click();
  await expect(page.getByRole("dialog", { name: "Configure agents" })).toBeVisible();
  await expect(page.getByRole("dialog", { name: "Configure agents" })).toContainText("The default is a starting point");
});
