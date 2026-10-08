const { expect, test } = require("@playwright/test");
const { mkdir, writeFile } = require("node:fs/promises");

async function paused(page) {
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused");
}

test("pause survives snapshots between pointer down and up", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await page.locator("#speed").selectOption("16");
  const button = page.getByRole("button", { name: "Pause", exact: true });
  await button.hover();
  await page.mouse.down();
  const before = await page.locator("#transitions").innerText();
  await expect(page.locator("#transitions")).not.toHaveText(before);
  await page.mouse.up();
  await expect(page.locator("#status")).toHaveText("Paused");
});

for (const task of ["cartpole", "mountain-car", "mountain-car-continuous"]) test(`${task}: every speed advances inference without optimizer updates`, async ({ page }) => {
  await page.goto(`./?env=${task}`);
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  for (const speed of ["1", "2", "4", "8", "16"]) {
    await page.locator("#speed").selectOption(speed);
    const before = await page.locator("#transitions").innerText();
    await expect(page.locator("#transitions")).not.toHaveText(before);
    await expect(page.locator("#updates")).toHaveText("0");
  }
  await paused(page);
  const before = await page.locator("#transitions").innerText();
  for (const speed of ["1", "2", "4", "8", "16"]) {
    await page.locator("#speed").selectOption(speed);
    await expect(page.locator("#status")).toHaveText("Paused");
    await expect(page.locator("#transitions")).toHaveText(before);
  }
  await page.getByRole("button", { name: "Step", exact: true }).click();
  await expect(page.locator("#transitions")).toHaveText((Number(before.replaceAll(",", "")) + 1).toLocaleString("en-US"));
  await expect(page.locator("#updates")).toHaveText("0");
  await expect(page.locator("#status")).toHaveText("Paused");
});

test("invalid uploads preserve the paused inference session", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await paused(page);
  const before = await page.locator("#transitions").innerText();
  for (const buffer of [Buffer.from([0, 1, 2]), Buffer.alloc(131073)]) {
    await page.locator("#import").setInputFiles({ name: "invalid.mpk", mimeType: "application/octet-stream", buffer });
    await expect(page.locator("#error")).toBeVisible();
    await expect(page.locator("#status")).toHaveText("Paused");
    await expect(page.locator("#transitions")).toHaveText(before);
    await expect(page.locator("#policy-source")).toHaveText("Bundled model");
  }
  await page.getByRole("button", { name: "Step", exact: true }).click();
  await expect(page.locator("#transitions")).toHaveText((Number(before.replaceAll(",", "")) + 1).toLocaleString("en-US"));
});

test("late upload validation cannot replace a newer training session", async ({ page }) => {
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener("message", event => {
          if (window.holdPolicyValidation && JSON.parse(event.data).event === "started") {
            event.stopImmediatePropagation();
            window.releasePolicyValidation = () => this.dispatchEvent(new MessageEvent("message", { data: event.data }));
          }
        });
      }
    };
  });
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await paused(page);
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download policy", exact: true }).click();
  const download = await downloadEvent;
  await page.evaluate(() => { window.holdPolicyValidation = true; });
  await page.locator("#import").setInputFiles(await download.path());
  await expect.poll(() => page.evaluate(() => Boolean(window.releasePolicyValidation))).toBe(true);
  await page.evaluate(() => { window.holdPolicyValidation = false; });
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect(page.locator("#mode")).toHaveText("Training · DQN");
  await expect.poll(async () => Number(await page.locator("#transitions").innerText())).toBeGreaterThan(0);
  await page.evaluate(() => window.releasePolicyValidation());
  await expect(page.locator("#mode")).toHaveText("Training · DQN");
  await expect(page.locator("#error")).toBeHidden();
});

test("validator failures preserve the active policy", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await paused(page);
  const before = await page.locator("#transitions").innerText();
  for (const failure of ["constructor", "worker", "protocol", "json", "timeout"]) {
    await page.evaluate(failure => {
      document.getElementById("error").hidden = true;
      window.Worker = class {
        constructor() {
          if (failure === "constructor") throw new Error("Constructor failed");
          queueMicrotask(() => {
            if (failure === "worker") this.onerror({ message: "Worker failed" });
            if (failure === "protocol") this.onmessage({ data: '{"event":"ready","protocol":99}' });
            if (failure === "json") this.onmessage({ data: "invalid JSON" });
          });
        }
        terminate() {}
      };
    }, failure);
    await page.locator("#import").setInputFiles({ name: `${failure}.mpk`, mimeType: "application/octet-stream", buffer: Buffer.from([0]) });
    await expect(page.locator("#error")).toBeVisible();
    await expect(page.locator("#status")).toHaveText("Paused");
    await expect(page.locator("#transitions")).toHaveText(before);
    await expect(page.locator("#policy-source")).toHaveText("Bundled model");
  }
});

for (const mode of ["inference", "training"]) test(`hidden ${mode} pauses after the in-flight batch without automatic resume`, async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(document, "hidden", { get: () => Boolean(window.testHidden) });
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener("message", event => {
          if (window.holdSnapshot && JSON.parse(event.data).event === "snapshot") {
            event.stopImmediatePropagation();
            window.releaseSnapshot = () => this.dispatchEvent(new MessageEvent("message", { data: event.data }));
          }
        });
      }
    };
  });
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await page.locator("#speed").selectOption("16");
  if (mode === "training") {
    await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
    await expect.poll(async () => Number((await page.locator("#updates").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  }
  await page.evaluate(() => { window.holdSnapshot = true; });
  await expect.poll(() => page.evaluate(() => Boolean(window.releaseSnapshot))).toBe(true);
  await page.evaluate(() => { window.testHidden = true; document.dispatchEvent(new Event("visibilitychange")); });
  await expect(page.locator("#status")).toHaveText("Pausing after the current batch");
  await page.evaluate(() => { window.holdSnapshot = false; window.releaseSnapshot(); });
  await expect(page.locator("#status")).toHaveText("Paused · tab hidden");
  const transitions = await page.locator("#transitions").innerText();
  const updates = await page.locator("#updates").innerText();
  await page.evaluate(() => { window.testHidden = false; document.dispatchEvent(new Event("visibilitychange")); });
  await expect(page.locator("#status")).toHaveText("Paused");
  await page.waitForTimeout(200);
  await expect(page.locator("#transitions")).toHaveText(transitions);
  await expect(page.locator("#updates")).toHaveText(updates);
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await expect(page.locator("#transitions")).not.toHaveText(transitions);
});

test("a session started in a hidden page waits for explicit resume", async ({ page }) => {
  await page.addInitScript(() => { Object.defineProperty(document, "hidden", { get: () => true }); });
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Paused · tab hidden");
  await expect(page.locator("#transitions")).toHaveText("0");
});

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
  { task: "mountain-car-continuous", algorithm: "PPO", target: 90, selection: 90, validationCount: 20, minimumSuccesses: 190, checkpoints: 195, batchSteps: 256, improvement: 0, testStart: 500000 },
]) test(`${qualification.task} browser ${qualification.algorithm ?? "DQN"} learns and passes held-out scores`, async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium", "Full score qualification runs in Chromium; all engines run optimizer and inference checks.");
  test.setTimeout(1_200_000);
  await page.goto(`./?env=${qualification.task}`);
  if (qualification.algorithm === "PPO") {
    await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
    await expect(page.locator("#pause")).toBeEnabled();
  } else await expect(page.locator("#status")).toHaveText("Running frozen policy");
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
          if (snapshot.completed.length) { const score = snapshot.completed[0].reward; sum += score; if (qualification.task === "cartpole" ? score === 500 : score > (qualification.algorithm === "PPO" ? 0 : -200)) full++; break; }
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
        for (let batch = 0; batch < 40; batch++) await trainer.send({ command: "advance", steps: qualification.batchSteps ?? 250 });
        const candidate = await trainer.send({ command: "export" });
        const validation = await evaluate(candidate.bytes, 10000, qualification.validationCount);
        history.push({ transitions: checkpoint * 40 * (qualification.batchSteps ?? 250), ...validation });
        if (validation.mean >= qualification.selection && validation.full / validation.count >= qualification.minimumSuccesses / 200) { selected = candidate.bytes; break; }
      }
      return { baseline, history, test: selected ? await evaluate(selected, qualification.testStart, 200) : null, policy: selected };
    } finally { trainer.worker.terminate(); evaluator.worker.terminate(); }
  }, qualification);
  await expect.poll(() => page.evaluate(() => Boolean(window.offlineQualificationReady))).toBe(true);
  await page.context().setOffline(true);
  await page.evaluate(() => window.startOfflineQualification());
  const result = await evaluation.finally(() => page.context().setOffline(false));
  const { policy, ...metrics } = result;
  await mkdir(testInfo.outputPath(), { recursive: true });
  const reportPath = testInfo.outputPath(`${qualification.task}-learning.json`);
  await writeFile(reportPath, JSON.stringify(metrics, null, 2));
  await testInfo.attach(`${qualification.task}-learning.json`, { path: reportPath, contentType: "application/json" });
  if (policy) {
    const policyPath = testInfo.outputPath(`${qualification.task}-policy.mpk`);
    await writeFile(policyPath, Buffer.from(policy));
    await testInfo.attach(`${qualification.task}-policy.mpk`, { path: policyPath, contentType: "application/octet-stream" });
  }
  console.log(JSON.stringify(metrics));
  expect(result.test).not.toBeNull();
  expect(result.test.mean).toBeGreaterThanOrEqual(qualification.target);
  expect(result.test.full).toBeGreaterThanOrEqual(qualification.minimumSuccesses);
  if (qualification.algorithm === "PPO") expect(result.test.mean - result.baseline.mean).toBeGreaterThan(0);
  else expect(result.test.mean - result.baseline.mean).toBeGreaterThanOrEqual(qualification.improvement);
});

test("learning rate uses optimizer updates on its horizontal axis", async ({ page }) => {
  await page.addInitScript(() => {
    const original = CanvasRenderingContext2D.prototype.fillText;
    window.chartLabels = {};
    CanvasRenderingContext2D.prototype.fillText = function (text, ...args) {
      const labels = window.chartLabels[this.canvas.id] ?? [];
      window.chartLabels[this.canvas.id] = [...labels, String(text)].slice(-6);
      return original.call(this, text, ...args);
    };
  });
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await page.locator("#speed").selectOption("16");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect.poll(async () => Number((await page.locator("#updates").innerText()).replaceAll(",", ""))).toBeGreaterThan(20);
  await paused(page);
  const updates = await page.locator("#updates").innerText();
  const transitions = await page.locator("#transitions").innerText();
  expect(updates).not.toBe(transitions);
  expect(await page.evaluate(() => window.chartLabels["rate-chart"].slice(-2))).toEqual([updates, "Optimizer updates"]);
  expect(await page.evaluate(() => window.chartLabels["returns-chart"].at(-1))).toBe("Transitions");
});
