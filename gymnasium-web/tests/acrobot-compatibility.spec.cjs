const { test, expect } = require("@playwright/test");
const { readFile } = require("node:fs/promises");
const path = require("node:path");

for (const [name, policyFile] of [
  ["native", "native-v2/seed-43.mpk"],
  ["browser", "browser-v1/acrobot-policy.mpk"],
]) test(`Acrobot ${name} record agrees with native transition probes`, async ({ page }, testInfo) => {
  test.skip(process.env.BEVY_GYM_ACROBOT_COMPATIBILITY !== "1", "Requires the recorded qualification artifacts.");
  const root = path.join(__dirname, "../../runs/browser-acrobot");
  const record = JSON.parse(await readFile(path.join(root, `${name}-transition-probes.json`), "utf8"));
  const bytes = [...await readFile(path.join(root, policyFile))];
  const canonical = [...await readFile(path.join(root, `${name}-canonical.mpk`))];
  await page.goto("./?env=acrobot");
  const actual = await page.evaluate(async ({ bytes, record }) => {
    const assetBase = new URL(".", document.querySelector('script[type="module"]').src);
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
    try {
      const traces = [];
      for (const trace of record.traces) {
        await send({ command: "start_inference", seed: trace.seed, bytes });
        const states = [];
        for (const _state of trace.states) {
          const { snapshot } = await send({ command: "advance", steps: 1 });
          if (snapshot.optimizer_steps !== 0) throw new Error("Frozen inference performed an update");
          states.push(snapshot.state);
        }
        traces.push(states);
      }
      return { traces, bytes: (await send({ command: "export" })).bytes };
    } finally { worker.terminate(); }
  }, { bytes, record });
  expect(actual.bytes).toEqual(canonical);
  let maximumError = 0;
  for (let trace = 0; trace < record.traces.length; trace++) {
    for (let step = 0; step < record.traces[trace].states.length; step++) {
      for (let coordinate = 0; coordinate < 4; coordinate++) {
        maximumError = Math.max(maximumError, Math.abs(actual.traces[trace][step][coordinate] - record.traces[trace].states[step][coordinate]));
      }
    }
  }
  console.log(`${name} ${testInfo.project.name}: maximum native/browser state difference ${maximumError}`);
  expect(maximumError).toBeLessThanOrEqual(record.maximum_absolute_state_error);
});
