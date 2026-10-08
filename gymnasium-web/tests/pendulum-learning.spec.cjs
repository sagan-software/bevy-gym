const { test, expect } = require("@playwright/test");
const { mkdir, writeFile } = require("node:fs/promises");
const { createHash } = require("node:crypto");
const os = require("node:os");

test("Pendulum browser PPO passes offline return and upright-dwell qualification", async ({ page, browser }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium" || process.env.BEVY_GYM_QUALIFY_PENDULUM !== "1",
    "Full qualification is an explicit release gate; all engines run functional and frozen-policy checks.");
  test.setTimeout(28_800_000);
  await mkdir(testInfo.outputPath(), { recursive: true });
  const started = performance.now();
  const hardware = { cpu: os.cpus()[0].model, logicalCpus: os.cpus().length, platform: os.platform(), architecture: os.arch(), browser: browser.version() };
  await page.exposeFunction("retainPendulumProgress", async report => {
    await writeFile(testInfo.outputPath("pendulum-progress.json"), JSON.stringify({ ...report, elapsedSeconds: (performance.now() - started) / 1000, hardware }, null, 2));
    console.log(`Pendulum ${report.transitions} transitions: validation mean ${report.validation.mean.toFixed(3)}, dwell ${report.validation.dwell_rate.toFixed(4)}`);
  });
  await page.goto("./?env=pendulum");
  const evaluation = page.evaluate(async () => {
    const assetBase = new URL(".", document.querySelector('script[type="module"]').src);
    async function client() {
      const worker = new Worker(new URL("gymnasium-worker_loader.js", assetBase), { type: "module", name: "pendulum" });
      await new Promise((resolve, reject) => {
        worker.onerror = event => reject(new Error(event.message));
        worker.onmessage = event => JSON.parse(event.data).event === "ready" ? resolve() : reject(new Error("worker initialization failed"));
      });
      return {
        worker,
        send(command) {
          return new Promise((resolve, reject) => {
            worker.onerror = event => reject(new Error(event.message));
            worker.onmessage = event => {
              const response = JSON.parse(event.data);
              if (response.event === "error") reject(new Error(response.message)); else resolve(response);
            };
            worker.postMessage(JSON.stringify(command));
          });
        },
      };
    }
    const trainer = await client(), evaluator = await client();
    await new Promise(resolve => { window.startOfflineQualification = resolve; window.offlineQualificationReady = true; });
    async function score(bytes, first, count) {
      const episodes = [];
      for (let seed = first; seed < first + count; seed++) {
        const response = await evaluator.send({ command: "evaluate_pendulum", bytes, seed });
        episodes.push(response.score);
      }
      return {
        mean: episodes.reduce((sum, episode) => sum + episode.reward, 0) / count,
        dwell_rate: episodes.reduce((sum, episode) => sum + episode.upright_steps, 0) / (count * 150),
        episodes,
      };
    }
    try {
      await trainer.send({ command: "start_training", seed: 42 });
      const initial = await trainer.send({ command: "export" });
      const baseline = await score(initial.bytes, 10000, 100);
      const history = [];
      let transitions = 0, selected = null, best = initial.bytes, bestMean = baseline.mean;
      await window.retainPendulumProgress({ transitions, validation: baseline, history });
      while (transitions < 2000000) {
        const checkpoint = Math.min(transitions + 10240, 2000000);
        while (transitions < checkpoint) {
          const steps = Math.min(256, checkpoint - transitions);
          await trainer.send({ command: "advance", steps });
          transitions += steps;
        }
        const candidate = await trainer.send({ command: "export" });
        const validation = await score(candidate.bytes, 10000, 100);
        history.push({ transitions, mean: validation.mean, dwell_rate: validation.dwell_rate });
        await window.retainPendulumProgress({ transitions, validation, history });
        if (validation.mean > bestMean) { best = candidate.bytes; bestMean = validation.mean; }
        if (validation.mean >= -200 && validation.dwell_rate >= 0.70) {
          selected = { policy: candidate.bytes, validation };
          break;
        }
      }
      return {
        profile: "pendulum-browser-ppo-v1", seed: 42, transitions, baseline, history,
        validationSeeds: [10000, 10100], testSeeds: [600000, 600200],
        validation: selected?.validation ?? null,
        test: selected ? await score(selected.policy, 600000, 200) : null,
        policy: selected?.policy ?? best,
      };
    } finally { trainer.worker.terminate(); evaluator.worker.terminate(); }
  });
  await expect.poll(() => page.evaluate(() => Boolean(window.offlineQualificationReady))).toBe(true);
  await page.context().setOffline(true);
  await page.evaluate(() => window.startOfflineQualification());
  const result = await evaluation.finally(() => page.context().setOffline(false));
  const { policy, ...scores } = result;
  const bytes = Buffer.from(policy);
  const report = { ...scores, elapsedSeconds: (performance.now() - started) / 1000, hardware, sha256: createHash("sha256").update(bytes).digest("hex") };
  await writeFile(testInfo.outputPath("pendulum-learning.json"), JSON.stringify(report, null, 2));
  await writeFile(testInfo.outputPath("pendulum-policy.mpk"), bytes);
  await testInfo.attach("pendulum-learning.json", { path: testInfo.outputPath("pendulum-learning.json"), contentType: "application/json" });
  await testInfo.attach("pendulum-policy.mpk", { path: testInfo.outputPath("pendulum-policy.mpk"), contentType: "application/octet-stream" });
  expect(result.history.some(row => row.transitions <= 500000 && row.mean > result.baseline.mean)).toBe(true);
  expect(result.test).not.toBeNull();
  expect(result.test.mean).toBeGreaterThanOrEqual(-200);
  expect(result.test.dwell_rate).toBeGreaterThanOrEqual(0.70);
});
