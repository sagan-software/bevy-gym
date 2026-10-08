const { test, expect } = require("@playwright/test");
const { mkdir, writeFile } = require("node:fs/promises");
const { createHash } = require("node:crypto");
const os = require("node:os");

test("Acrobot browser DQN passes offline return and goal qualification", async ({ page, browser }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium" || process.env.BEVY_GYM_QUALIFY_ACROBOT !== "1",
    "Full qualification is an explicit release gate; every engine runs functional checks.");
  test.setTimeout(10_800_000);
  await mkdir(testInfo.outputPath(), { recursive: true });
  const started = performance.now();
  const hardware = { cpu: os.cpus()[0].model, logicalCpus: os.cpus().length, platform: os.platform(), architecture: os.arch(), browser: browser.version() };
  await page.exposeFunction("retainAcrobotProgress", async report => {
    await writeFile(testInfo.outputPath("acrobot-progress.json"), JSON.stringify({ ...report, elapsedSeconds: (performance.now() - started) / 1000, hardware }, null, 2));
    console.log(`Acrobot ${report.transitions} transitions: validation mean ${report.validation.mean.toFixed(3)}, goals ${report.validation.goal_rate}`);
  });
  await page.goto("./?env=acrobot");
  const evaluation = page.evaluate(async () => {
    const assetBase = new URL(".", document.querySelector('script[type="module"]').src);
    async function client() {
      const worker = new Worker(new URL("gymnasium-worker_loader.js", assetBase), { type: "module", name: "acrobot" });
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
        await evaluator.send({ command: "start_inference", bytes, seed });
        let episode;
        while (!episode) {
          const response = await evaluator.send({ command: "advance", steps: 256 });
          episode = response.snapshot.completed[0];
        }
        // A 500-step truncation returns -500; a goal on step 500 returns -499.
        episodes.push({ reward: episode.reward, reached: episode.reward > -500 });
      }
      return {
        mean: episodes.reduce((sum, episode) => sum + episode.reward, 0) / count,
        goal_rate: episodes.filter(episode => episode.reached).length / count,
        episodes,
      };
    }
    try {
      await trainer.send({ command: "start_training", seed: 42 });
      const initial = await trainer.send({ command: "export" });
      const baseline = await score(initial.bytes, 10000, 100);
      const history = [];
      let transitions = 0, selected = null, best = initial.bytes, bestMean = baseline.mean;
      await window.retainAcrobotProgress({ transitions, validation: baseline, history });
      while (transitions < 1000000) {
        const checkpoint = transitions + 10000;
        while (transitions < checkpoint) {
          const steps = Math.min(256, checkpoint - transitions);
          await trainer.send({ command: "advance", steps });
          transitions += steps;
        }
        const candidate = await trainer.send({ command: "export" });
        const validation = await score(candidate.bytes, 10000, 100);
        history.push({ transitions, mean: validation.mean, goal_rate: validation.goal_rate });
        await window.retainAcrobotProgress({ transitions, validation, history });
        if (validation.mean > bestMean) { best = candidate.bytes; bestMean = validation.mean; }
        if (validation.mean >= -90 && validation.goal_rate >= 0.95) {
          selected = { policy: candidate.bytes, validation };
          break;
        }
      }
      return {
        profile: "acrobot-browser-dqn-v1", seed: 42, transitions, baseline, history,
        validationSeeds: [10000, 10100], testSeeds: [800000, 800200], validationStopMean: -90,
        validation: selected?.validation ?? null,
        test: selected ? await score(selected.policy, 800000, 200) : null,
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
  await writeFile(testInfo.outputPath("acrobot-learning.json"), JSON.stringify(report, null, 2));
  await writeFile(testInfo.outputPath("acrobot-policy.mpk"), bytes);
  await testInfo.attach("acrobot-learning.json", { path: testInfo.outputPath("acrobot-learning.json"), contentType: "application/json" });
  await testInfo.attach("acrobot-policy.mpk", { path: testInfo.outputPath("acrobot-policy.mpk"), contentType: "application/octet-stream" });
  expect(result.history.some(row => row.transitions <= 250000 && row.mean > result.baseline.mean)).toBe(true);
  expect(result.test).not.toBeNull();
  expect(result.test.mean).toBeGreaterThanOrEqual(-100);
  expect(result.test.goal_rate).toBeGreaterThanOrEqual(0.95);
});
