const { test, expect } = require("@playwright/test");

const tasks = { cartpole: 1024, "mountain-car": 2048, "mountain-car-continuous": 512, pendulum: 512, acrobot: 5008 };

for (const [task, trainingBudget] of Object.entries(tasks)) test(`${task}: all playback speeds preserve training and frozen replay`, async ({ page }) => {
  test.setTimeout(360000);
  // Only presentation time is controlled; the worker executes every real transition and update.
  // https://playwright.dev/docs/clock#tick-through-time-manually-firing-all-the-timers-consistently
  await page.clock.install({ time: new Date("2026-10-08T00:00:00Z") });
  await page.clock.pauseAt(new Date("2026-10-08T00:00:01Z"));
  await page.addInitScript(task => {
    // Acrobot's 5 Hz physics needs fewer presentation frames during virtual-time checks.
    // A 100 ms cadence stays within the page's long-frame clamp and preserves real worker work.
    if (task === "acrobot") window.requestAnimationFrame = callback => setTimeout(() => callback(performance.now()), 100);
    const NativeWorker = Worker;
    window.transitionBudget = 0;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.transitions = 0;
        this.episodes = [];
        window.boundedWorker = this;
        this.addEventListener("message", ({ data }) => {
          const message = JSON.parse(data);
          if (message.event !== "snapshot" || this !== window.boundedWorker) return;
          this.transitions = message.snapshot.transitions;
          this.episodes.push(...message.snapshot.completed);
          window.lastSnapshot = { ...message.snapshot, completed: this.episodes };
          // Pause before the page receives its final snapshot, exactly at the common budget.
          if (window.transitionBudget && this.transitions === window.transitionBudget) document.getElementById("pause").click();
        });
      }
      postMessage(raw) {
        const command = JSON.parse(raw);
        if (command.command === "advance" && window.transitionBudget) {
          command.steps = Math.min(command.steps, window.transitionBudget - this.transitions);
        }
        super.postMessage(JSON.stringify(command));
      }
    };
  }, task);
  await page.goto(`./?env=${task}`);

  async function finishBudget(budget) {
    await expect(page.locator("#pause")).toBeEnabled();
    for (let attempt = 0; attempt < 300; attempt++) {
      const before = await page.evaluate(() => window.lastSnapshot?.transitions ?? 0);
      if (before === budget) break;
      await page.clock.runFor(task === "acrobot" ? 10000 : 1000);
      // This checks deterministic work, not latency; CPU-only PPO updates can be slow in Firefox.
      await expect.poll(() => page.evaluate(() => window.lastSnapshot?.transitions ?? 0), { timeout: 90000 }).toBeGreaterThan(before);
    }
    await expect(page.locator("#status")).toHaveText("Paused");
    const snapshot = await page.evaluate(() => window.lastSnapshot);
    expect(snapshot.transitions).toBe(budget);
    return snapshot;
  }

  let trainingReference, inferencePolicy;
  for (const speed of ["1", "2", "4", "8", "16"]) {
    await page.locator("#speed").selectOption(speed);
    await page.evaluate(budget => { window.transitionBudget = budget; window.lastSnapshot = undefined; }, trainingBudget);
    await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
    const snapshot = await finishBudget(trainingBudget);
    expect(snapshot.optimizer_steps).toBeGreaterThan(0);
    if (trainingReference) expect(snapshot).toEqual(trainingReference);
    else {
      trainingReference = snapshot;
      const download = page.waitForEvent("download");
      await page.getByRole("button", { name: "Download policy", exact: true }).click();
      inferencePolicy = await (await download).path();
    }
  }

  let inferenceReference;
  for (const speed of ["1", "2", "4", "8", "16"]) {
    await page.locator("#speed").selectOption(speed);
    await page.evaluate(() => { window.transitionBudget = 256; window.lastSnapshot = undefined; });
    await page.locator("#import").setInputFiles(inferencePolicy);
    await expect(page.locator("#mode")).toHaveText("Inference");
    await expect(page.locator("#status")).toHaveText("Running frozen policy");
    const snapshot = await finishBudget(256);
    expect(snapshot.optimizer_steps).toBe(0);
    expect(snapshot.learning_rate).toBeNull();
    expect(snapshot.critic_learning_rate).toBeNull();
    if (inferenceReference) expect(snapshot).toEqual(inferenceReference);
    else inferenceReference = snapshot;
  }
});
