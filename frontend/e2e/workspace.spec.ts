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
  await expect(builder).toContainText("Your question and conversation history are passed to the selected start node");
  const inputs = builder.getByRole("region", { name: "Node inputs and outputs" });
  await expect(inputs).toContainText("question — Provided when the workflow starts");
  await expect(inputs).toContainText("history — Provided when the workflow starts");
  await builder.locator(".workflow-node-card").nth(1).click();
  await expect(inputs).toContainText("searches — From an earlier node: Search planner");
  await expect(inputs).toContainText("evidence, new_evidence_count");
  await builder.getByRole("textbox", { name: "Workflow name" }).fill("One pass research");
  await builder.getByRole("button", { name: /validate.*Evidence validator/i }).click();
  await expect(inputs).toContainText("searches — From an earlier node: Search planner");
  await expect(inputs).toContainText("validation, feedback, round_limit");
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

test("workflow save explains missing required inputs and their producers", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  await page.getByRole("button", { name: "Configure agents" }).first().click();
  const builder = page.getByRole("dialog", { name: "Configure agents" });
  await expect(builder.getByLabel("Start node")).toHaveValue("planner");
  await builder.getByRole("button", { name: "Remove planner", exact: true }).click();
  await builder.getByRole("button", { name: /validate.*Evidence validator/ }).click();
  await builder.getByRole("button", { name: "Remove validate", exact: true }).click();
  await builder.getByRole("button", { name: /need_upload.*Request more evidence/ }).click();
  await builder.getByRole("button", { name: "Remove need_upload", exact: true }).click();
  await builder.getByRole("button", { name: /retrieve.*Document retrieval/ }).click();
  await builder.getByLabel("retrieve next target").selectOption("generate");
  await builder.getByRole("button", { name: "Save new workflow and use" }).click();
  await expect(builder.getByRole("alert")).toContainText("retrieve needs state from an earlier node: searches");
  await expect(builder.getByRole("alert")).toContainText("Search planner (planner)");
  await expect(builder.getByRole("alert")).toContainText("question is already available");
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
