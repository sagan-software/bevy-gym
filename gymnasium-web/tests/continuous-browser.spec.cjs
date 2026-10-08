const { test, expect } = require("@playwright/test");
test.use({ video: { mode: "on", size: { width: 1280, height: 800 } }, viewport: { width: 1280, height: 800 } });

test("continuous MountainCar trains both PPO optimizers and freezes an exported policy", async ({ page }, testInfo) => {
  await page.goto("./?env=mountain-car-continuous");
  await expect(page.locator("h1")).toHaveText("MountainCarContinuous-v0");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await page.locator("#speed").selectOption("16");
  await expect(page.locator("#mode")).toHaveText("Training · PPO");
  await expect.poll(async () => Number((await page.locator("#updates").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await expect(page.locator("#rate")).toHaveText("3.0e-3");
  await expect(page.locator("#critic-rate")).toHaveText("1.0e-3");
  await expect(page.locator("#rate-summary")).toContainText("Actor 0.003 · critic 0.001");
  await expect(page.locator("#status")).toContainText("8 environments");
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused");
  await page.screenshot({ path: testInfo.outputPath("continuous-training-desktop.png"), fullPage: true });
  const chart = await page.locator("#returns-chart").boundingBox();
  expect(chart.y + chart.height).toBeLessThanOrEqual(800);
  const counters = await page.locator("#transitions").innerText();
  await page.waitForTimeout(300);
  await expect(page.locator("#transitions")).toHaveText(counters);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download policy", exact: true }).click();
  const policy = await download;
  await page.locator("#import").setInputFiles(await policy.path());
  await expect(page.locator("#mode")).toHaveText("Inference");
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await expect(page.locator("#updates")).toHaveText("0");
  await expect(page.locator("#rate")).toHaveText("None");
  await expect(page.locator("#critic-rate")).toHaveText("None");
  await expect(page.locator("#error")).toBeHidden();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: testInfo.outputPath("continuous-inference-mobile.png"), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("bundled continuous policy passes held-out scores without optimizer updates", async ({ page }) => {
  await page.goto("./?env=mountain-car-continuous");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await expect(page.locator("#model-evidence")).toContainText("200 held-out episodes");
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused");
  const result = await page.evaluate(async () => {
    const bytes = Array.from(new Uint8Array(await (await fetch("models/mountain-car-continuous.mpk")).arrayBuffer()));
    const worker = new Worker(new URL("gymnasium-worker_loader.js", document.baseURI), { type: "module", name: "mountain-car-continuous" });
    await new Promise((resolve, reject) => {
      worker.onerror = event => reject(new Error(event.message));
      worker.onmessage = event => JSON.parse(event.data).event === "ready" ? resolve() : reject(new Error("worker initialization failed"));
    });
    function send(command) {
      return new Promise((resolve, reject) => {
        worker.onerror = event => reject(new Error(event.message));
        worker.onmessage = event => {
          const response = JSON.parse(event.data);
          if (response.event === "error") reject(new Error(response.message)); else resolve(response);
        };
        worker.postMessage(JSON.stringify(command));
      });
    }
    let total = 0, successes = 0;
    try {
      for (let seed = 400000; seed < 400200; seed++) {
        await send({ command: "start_inference", seed, bytes });
        for (;;) {
          const { snapshot } = await send({ command: "advance", steps: 256 });
          if (snapshot.optimizer_steps !== 0 || snapshot.learning_rate !== null || snapshot.critic_learning_rate !== null) throw new Error("inference retained optimizer state");
          if (snapshot.completed.length) {
            const score = snapshot.completed[0].reward;
            total += score; if (score > 0) successes++;
            break;
          }
        }
      }
    } finally { worker.terminate(); }
    return { mean: total / 200, successes };
  });
  console.log("continuous bundle", result);
  expect(result.mean).toBeGreaterThanOrEqual(90);
  expect(result.successes).toBeGreaterThanOrEqual(190);
});
