const { test, expect } = require("@playwright/test");
const { writeFile } = require("node:fs/promises");

test("bundled Acrobot passes held-out scores with frozen parameters", async ({ page }, testInfo) => {
  await page.goto("./?env=acrobot");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await expect(page.locator("#model-evidence")).toContainText("200 held-out episodes");
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused");
  const result = await page.evaluate(async () => {
    const assetBase = new URL(".", document.querySelector('script[type="module"]').src);
    const bytes = [...new Uint8Array(await (await fetch(new URL("models/acrobot.mpk", assetBase))).arrayBuffer())];
    const worker = new Worker(new URL("gymnasium-worker_loader.js", assetBase), { type: "module", name: "acrobot" });
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
    const episodes = [];
    try {
      await send({ command: "start_inference", seed: 700000, bytes });
      const canonical = (await send({ command: "export" })).bytes;
      for (let seed = 700000; seed < 700200; seed++) {
        await send({ command: "start_inference", seed, bytes });
        for (;;) {
          const { snapshot } = await send({ command: "advance", steps: 256 });
          if (snapshot.optimizer_steps !== 0 || snapshot.learning_rate !== null || snapshot.epsilon !== null) throw new Error("inference retained optimizer state");
          if (snapshot.completed.length) {
            const reward = snapshot.completed[0].reward;
            // Natural termination earns zero on the last step; truncation returns -500.
            episodes.push({ seed, reward, reached: reward > -500 });
            break;
          }
        }
      }
      const after = (await send({ command: "export" })).bytes;
      if (JSON.stringify(after) !== JSON.stringify(canonical)) throw new Error("inference changed policy parameters");
      return {
        mean: episodes.reduce((sum, episode) => sum + episode.reward, 0) / episodes.length,
        successes: episodes.filter(episode => episode.reached).length,
        episodes,
      };
    } finally { worker.terminate(); }
  });
  await writeFile(testInfo.outputPath("acrobot-bundle.json"), JSON.stringify(result, null, 2));
  console.log(`${testInfo.project.name} Acrobot bundle: mean ${result.mean}, goals ${result.successes}/200`);
  expect(result.mean).toBeGreaterThanOrEqual(-100);
  expect(result.successes).toBeGreaterThanOrEqual(190);
});
