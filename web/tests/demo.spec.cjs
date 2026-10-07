const fs = require("node:fs");
const path = require("node:path");

const { expect, test } = require("@playwright/test");

const stages = [
  ["forage", "Forage", false],
  ["survival", "Survival", false],
  ["shelter", "Shelter", false],
  ["competition", "Competition", false],
  ["predator-prey", "Predator-prey", true],
  ["obstacles", "Obstacles", true],
];

function releaseManifest() {
  const manifestPath = path.resolve("web/checkpoints/manifest.json");
  return JSON.parse(fs.readFileSync(manifestPath, "utf8"));
}

function assetPath(asset, field = "path") {
  return path.resolve("web", asset[field]);
}

function collectPageFailures(page) {
  const failures = [];
  page.on("pageerror", (error) => failures.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") failures.push(`console: ${message.text()}`);
  });
  page.on("requestfailed", (request) => {
    failures.push(`request: ${request.url()} ${request.failure()?.errorText}`);
  });
  return failures;
}

async function waitForReady(page) {
  await expect(page.locator("#checkpoint-status")).toHaveText("Ready");
  await expect(page.locator("#runtime-status")).toHaveText("WASM inference running");
}

async function performanceMetric(page, attribute) {
  return page.evaluate((name) => {
    const value = document.documentElement.getAttribute(name);
    return value === null ? Number.NaN : Number(value);
  }, attribute);
}

test("all curated ecosystem routes visibly run for five seconds", async ({ page }) => {
  const failures = collectPageFailures(page);
  await page.goto("/");
  await waitForReady(page);
  await expect(page.locator("#environment-title")).toHaveText("Survival");

  for (const [key, title, needsFox] of stages) {
    await page.locator(`a[data-stage="${key}"]`).click();
    await waitForReady(page);
    await expect(page.locator("#environment-title")).toHaveText(title);
    await expect(page.locator("#checkpoint-qualification")).toHaveText(
      "best compatible available",
    );
    await expect(page.locator("#checkpoint-record")).toContainText(`${key}-bunny-`);
    await expect(page.locator("#bunny-upload-state")).toHaveText("Curated bunny policy ready.");
    if (needsFox) {
      await expect(page.locator("#fox-upload-group")).toBeVisible();
      await expect(page.locator("#fox-upload-state")).toHaveText("Curated fox policy ready.");
    } else {
      await expect(page.locator("#fox-upload-group")).toBeHidden();
    }

    const initialFrame = await page.locator("#ecosystem-canvas").screenshot();
    const initialEpisode = Number(await page.locator("#metric-episode").textContent());
    const initialStep = Number(await page.locator("#metric-step").textContent());
    await page.waitForTimeout(5_100);
    const finalFrame = await page.locator("#ecosystem-canvas").screenshot();
    const finalEpisode = Number(await page.locator("#metric-episode").textContent());
    const finalStep = Number(await page.locator("#metric-step").textContent());
    expect([finalEpisode, finalStep]).not.toEqual([initialEpisode, initialStep]);
    expect(Buffer.compare(initialFrame, finalFrame)).not.toBe(0);
    await expect.poll(() => performanceMetric(page, "data-checkpoint-decode-ms")).toBeGreaterThanOrEqual(0);
    await expect.poll(() => performanceMetric(page, "data-first-render-ms")).toBeGreaterThan(0);
    await expect.poll(() => performanceMetric(page, "data-steady-frame-ms")).toBeGreaterThan(0);

    if (key === "survival" || key === "predator-prey") {
      await page.screenshot({ path: `test-results/wasm-demo-${key}.png`, fullPage: true });
    }
  }

  expect(failures).toEqual([]);
});

test("checkpoint upload is atomic, recoverable, and resettable", async ({ page }) => {
  const manifest = releaseManifest();
  const survival = manifest.stages.find((entry) => entry.stage === "survival");
  const failures = collectPageFailures(page);
  await page.goto("/#/ecosystem/survival");
  await waitForReady(page);

  await page.locator("#bunny-checkpoint").setInputFiles(assetPath(survival.bunny));
  await expect(page.locator("#metric-policy")).toHaveText("uploaded, metadata unverified");
  await expect(page.locator("#checkpoint-qualification")).toHaveText("unverified");

  await page.locator("#reset-checkpoints").click();
  await waitForReady(page);
  await expect(page.locator("#metric-policy")).toHaveText("curated default");
  await expect(page.locator("#checkpoint-qualification")).toHaveText(
    "best compatible available",
  );

  const beforeMalformed = Number(await page.locator("#metric-step").textContent());
  await page.locator("#bunny-checkpoint").setInputFiles({
    name: "malformed.mpk",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("not a named MessagePack checkpoint"),
  });
  await expect(page.locator("#checkpoint-status")).toHaveText("Upload rejected");
  await expect(page.locator("#runtime-status")).toHaveText("WASM inference running");
  await expect
    .poll(async () => Number(await page.locator("#metric-step").textContent()))
    .not.toBe(beforeMalformed);
  await expect(page.locator("#metric-policy")).toHaveText("curated default");
  expect(failures).toEqual([]);
});

test("matching sidecars verify a paired predator upload", async ({ page }) => {
  const manifest = releaseManifest();
  const predator = manifest.stages.find((entry) => entry.stage === "predator-prey");
  await page.goto("/#/ecosystem/predator-prey");
  await waitForReady(page);

  await page
    .locator("#bunny-checkpoint-config")
    .setInputFiles(assetPath(predator.bunny, "config_path"));
  await page.locator("#bunny-checkpoint").setInputFiles(assetPath(predator.bunny));
  await expect(page.locator("#bunny-upload-state")).toContainText("waiting for fox");
  await page
    .locator("#fox-checkpoint-config")
    .setInputFiles(assetPath(predator.fox, "config_path"));
  await page.locator("#fox-checkpoint").setInputFiles(assetPath(predator.fox));

  await expect(page.locator("#metric-policy")).toHaveText("uploaded, sidecars verified");
  await expect(page.locator("#checkpoint-qualification")).toHaveText("local sidecars verified");
  await expect(page.locator("#bunny-upload-state")).toHaveText("Uploaded policy is active.");
  await expect(page.locator("#fox-upload-state")).toHaveText("Uploaded policy is active.");
});

test("pause, restart, and speed controls change inference progression", async ({ page }) => {
  await page.goto("/#/ecosystem/survival");
  await waitForReady(page);

  await page.locator("#pause-playback").click();
  await expect(page.locator("#runtime-status")).toHaveText("Inference paused");
  const pausedStep = await page.locator("#metric-step").textContent();
  await page.waitForTimeout(700);
  await expect(page.locator("#metric-step")).toHaveText(pausedStep);

  const episode = Number(await page.locator("#metric-episode").textContent());
  await page.locator("#restart-environment").click();
  await expect(page.locator("#metric-episode")).toHaveText(String(episode + 1));
  await expect(page.locator("#metric-step")).toHaveText("0");

  await page.locator("#playback-speed").selectOption("4");
  await page.locator("#pause-playback").click();
  await expect(page.locator("#runtime-status")).toHaveText("WASM inference running");
  await expect
    .poll(async () => Number(await page.locator("#metric-step").textContent()))
    .toBeGreaterThan(4);
});
