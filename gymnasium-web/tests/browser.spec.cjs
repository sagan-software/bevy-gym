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
  await page.locator("#speed").selectOption("16");
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(500);
  await expect(page.locator("#updates")).toHaveText("0");
  await paused(page);
  const frozen = await page.locator("#transitions").innerText();
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

// A separate worker can consume the full transition budget without display throttling.
test("browser DQN learns and passes held-out scores", async ({ page }, testInfo) => {
  test.setTimeout(1_200_000);
  await page.goto("./");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await paused(page);
  const result = await page.evaluate(async () => {
    async function client() {
      const worker = new Worker(new URL("gymnasium-worker_loader.js", document.baseURI), { type: "module" });
      await new Promise((resolve, reject) => {
        worker.onerror = event => reject(new Error(event.message));
        worker.onmessage = event => { if (JSON.parse(event.data).event === "ready") resolve(); };
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
    async function evaluate(bytes, start, count) {
      let sum = 0, full = 0;
      for (let seed = start; seed < start + count; seed++) {
        await evaluator.send({ command: "start_inference", seed, bytes });
        for (;;) {
          const { snapshot } = await evaluator.send({ command: "advance", steps: 256 });
          if (snapshot.completed.length) { const score = snapshot.completed[0].reward; sum += score; if (score === 500) full++; break; }
        }
      }
      return { mean: sum / count, full, count };
    }
    try {
      await trainer.send({ command: "start_training", seed: 42 });
      const initial = await trainer.send({ command: "export" });
      const baseline = await evaluate(initial.bytes, 10000, 20);
      const history = [];
      let selected;
      for (let checkpoint = 1; checkpoint <= 20; checkpoint++) {
        for (let batch = 0; batch < 40; batch++) await trainer.send({ command: "advance", steps: 250 });
        const candidate = await trainer.send({ command: "export" });
        const validation = await evaluate(candidate.bytes, 10000, 20);
        history.push({ transitions: checkpoint * 10000, ...validation });
        if (validation.mean >= 475 && validation.full >= 18) { selected = candidate.bytes; break; }
      }
      return { baseline, history, test: selected ? await evaluate(selected, 100000, 200) : null };
    } finally { trainer.worker.terminate(); evaluator.worker.terminate(); }
  });
  await testInfo.attach("browser-learning.json", { body: JSON.stringify(result, null, 2), contentType: "application/json" });
  console.log(JSON.stringify(result));
  expect(result.test).not.toBeNull();
  expect(result.test.mean).toBeGreaterThanOrEqual(475);
  expect(result.test.full).toBeGreaterThanOrEqual(180);
  expect(result.test.mean - result.baseline.mean).toBeGreaterThanOrEqual(400);
});
