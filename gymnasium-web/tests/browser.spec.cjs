const { expect, test } = require("@playwright/test");

async function paused(page) {
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused");
}

// Exercise the shipped UI and actual Rust worker, including the repository URL prefix.
test("training, frozen inference, pause, step, speed, and policy round trip", async ({ page }) => {
  const failures = [];
  page.on("pageerror", error => failures.push(error.message));
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await page.setViewportSize({ width: 1280, height: 800 });
  const chart = await page.locator("#returns-chart").boundingBox();
  expect(chart.y + chart.height).toBeLessThanOrEqual(800);
  await page.locator("#speed").selectOption("16");
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(500);
  await expect(page.locator("#updates")).toHaveText("0");
  await paused(page);
  const frozen = await page.locator("#transitions").innerText();
  await page.locator("#seed").fill("-1");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect(page.locator("#error")).toBeVisible();
  await expect(page.locator("#mode")).toHaveText("Inference");
  await expect(page.locator("#status")).toHaveText("Paused");
  await page.locator("#seed").fill("42");
  await page.waitForTimeout(300);
  await expect(page.locator("#transitions")).toHaveText(frozen);
  await page.getByRole("button", { name: "Step", exact: true }).click();
  await expect(page.locator("#transitions")).toHaveText((Number(frozen.replaceAll(",", "")) + 1).toLocaleString("en-US"));
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Training in this browser");
  await expect.poll(async () => Number((await page.locator("#updates").innerText()).replaceAll(",", ""))).toBeGreaterThan(20);
  await expect(page.locator("#rate")).toHaveText("3.0e-4");
  await paused(page);
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download policy", exact: true }).click();
  const download = await downloadEvent;
  await page.locator("#import").setInputFiles(await download.path());
  await expect(page.locator("#policy-source")).toContainText("Uploaded policy:");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await expect(page.locator("#updates")).toHaveText("0");
  await expect(page.locator("#rate")).toHaveText("None");
  await expect(page.locator("#error")).toBeHidden();
  await page.locator("#import").setInputFiles({ name: "invalid.mpk", mimeType: "application/octet-stream", buffer: Buffer.from([0, 1, 2]) });
  await expect(page.locator("#error")).toBeVisible();
  await page.getByRole("button", { name: "Run bundled model", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await expect(page.locator("#error")).toBeHidden();
  await paused(page);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(failures).toEqual([]);
});

test("MountainCar selection trains and loads its qualified policy", async ({ page }) => {
  await page.goto("./?env=mountain-car");
  await expect(page.locator("h1")).toHaveText("MountainCar-v0");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await page.locator("#speed").selectOption("16");
  await expect.poll(async () => Number((await page.locator("#episodes").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await expect(page.locator("#return-summary")).toContainText("Mean of last");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect.poll(async () => Number((await page.locator("#updates").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await expect(page.locator("#rate")).toHaveText("1.0e-3");
  await paused(page);
  await page.locator("#environment").selectOption("cartpole");
  await expect(page.locator("h1")).toHaveText("CartPole-v1");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
});

test("late bundled-model loading cannot replace a newer training session", async ({ page }) => {
  let pending;
  await page.route("**/models/cartpole.mpk", route => { pending = route; });
  await page.goto("./", { waitUntil: "commit" });
  await expect.poll(() => Boolean(pending)).toBe(true);
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect(page.locator("#mode")).toHaveText("Training · DQN");
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await pending.continue();
  await expect(page.locator("#model-evidence")).toContainText("mean 500");
  await expect(page.locator("#mode")).toHaveText("Training · DQN");
});

// A separate worker can consume the full transition budget without display throttling.
for (const qualification of [
  { task: "cartpole", target: 475, selection: 475, validationCount: 20, minimumSuccesses: 180, checkpoints: 20, improvement: 400, testStart: 100000 },
  { task: "mountain-car", target: -110, selection: -105, validationCount: 100, minimumSuccesses: 190, checkpoints: 100, improvement: 80, testStart: 300000 },
]) test(`${qualification.task} browser DQN learns and passes held-out scores`, async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium", "Full score qualification runs in Chromium; all engines run optimizer and inference checks.");
  test.setTimeout(1_200_000);
  await page.goto(`./?env=${qualification.task}`);
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await paused(page);
  const evaluation = page.evaluate(async qualification => {
    async function client() {
      const worker = new Worker(new URL("gymnasium-worker_loader.js", document.baseURI), { type: "module", name: qualification.task });
      await new Promise((resolve, reject) => {
        worker.onerror = event => reject(new Error(event.message));
        worker.onmessage = event => { const message = JSON.parse(event.data); if (message.event === "ready" && message.protocol === 1) resolve(); else reject(new Error("worker protocol mismatch")); };
      });
      return {
        worker,
        send(command) {
          return new Promise((resolve, reject) => {
            worker.onerror = event => reject(new Error(event.message));
            worker.onmessage = event => { const message = JSON.parse(event.data); if (message.event === "error") reject(new Error(message.message)); else resolve(message); };
            worker.postMessage(JSON.stringify(command));
          });
        },
      };
    }
    const trainer = await client(), evaluator = await client();
    await new Promise(resolve => { window.startOfflineQualification = resolve; window.offlineQualificationReady = true; });
    async function evaluate(bytes, start, count) {
      let sum = 0, full = 0;
      for (let seed = start; seed < start + count; seed++) {
        await evaluator.send({ command: "start_inference", seed, bytes });
        for (;;) {
          const { snapshot } = await evaluator.send({ command: "advance", steps: 256 });
          if (snapshot.completed.length) { const score = snapshot.completed[0].reward; sum += score; if (qualification.task === "cartpole" ? score === 500 : score > -200) full++; break; }
        }
      }
      return { mean: sum / count, full, count };
    }
    try {
      await trainer.send({ command: "start_training", seed: 42 });
      const initial = await trainer.send({ command: "export" });
      const baseline = await evaluate(initial.bytes, 10000, qualification.validationCount);
      const history = [];
      let selected;
      for (let checkpoint = 1; checkpoint <= qualification.checkpoints; checkpoint++) {
        for (let batch = 0; batch < 40; batch++) await trainer.send({ command: "advance", steps: 250 });
        const candidate = await trainer.send({ command: "export" });
        const validation = await evaluate(candidate.bytes, 10000, qualification.validationCount);
        history.push({ transitions: checkpoint * 10000, ...validation });
        if (validation.mean >= qualification.selection && validation.full / validation.count >= qualification.minimumSuccesses / 200) { selected = candidate.bytes; break; }
      }
      return { baseline, history, test: selected ? await evaluate(selected, qualification.testStart, 200) : null };
    } finally { trainer.worker.terminate(); evaluator.worker.terminate(); }
  }, qualification);
  await expect.poll(() => page.evaluate(() => Boolean(window.offlineQualificationReady))).toBe(true);
  await page.context().setOffline(true);
  await page.evaluate(() => window.startOfflineQualification());
  const result = await evaluation.finally(() => page.context().setOffline(false));
  await testInfo.attach("browser-learning.json", { body: JSON.stringify(result, null, 2), contentType: "application/json" });
  console.log(JSON.stringify(result));
  expect(result.test).not.toBeNull();
  expect(result.test.mean).toBeGreaterThanOrEqual(qualification.target);
  expect(result.test.full).toBeGreaterThanOrEqual(qualification.minimumSuccesses);
  expect(result.test.mean - result.baseline.mean).toBeGreaterThanOrEqual(qualification.improvement);
});
