const { test, expect } = require("@playwright/test");

test("Pendulum evaluation retains all 200 rewards and 150 dwell samples", async ({ page }) => {
  await page.goto("./?env=pendulum");
  const result = await page.evaluate(async () => {
    const assetBase = new URL(".", document.querySelector('script[type="module"]').src);
    const worker = new Worker(new URL("gymnasium-worker_loader.js", assetBase), { type: "module", name: "pendulum" });
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
    try {
      await send({ command: "start_training", seed: 42 });
      const { bytes } = await send({ command: "export" });
      const evaluation = await send({ command: "evaluate_pendulum", bytes, seed: 17 });
      const unchanged = await send({ command: "advance", steps: 1 });
      await send({ command: "start_inference", bytes, seed: 17 });
      const { snapshot } = await send({ command: "advance", steps: 200 });
      return { evaluation, unchanged: unchanged.snapshot, snapshot };
    } finally { worker.terminate(); }
  });
  expect(result.evaluation.event).toBe("evaluation");
  expect(result.evaluation.score.reward).toBe(result.snapshot.completed[0].reward);
  expect(result.evaluation.score.upright_steps).toBeGreaterThanOrEqual(0);
  expect(result.evaluation.score.upright_steps).toBeLessThanOrEqual(150);
  expect(result.unchanged.transitions).toBe(1);
  expect(result.unchanged.completed).toEqual([]);
  expect(result.snapshot.optimizer_steps).toBe(0);
});
