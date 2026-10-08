const { test, expect } = require("@playwright/test");

const tasks = ["cartpole", "mountain-car", "mountain-car-continuous", "pendulum", "acrobot"];

for (const task of tasks) test(`${task}: an actual worker exception stops training and restart recovers`, async ({ page }) => {
  await page.addInitScript(() => {
    const NativeWorker = Worker;
    window.workerInstances = [];
    window.Worker = class extends NativeWorker {
      constructor(url, options) {
        // Throw inside the real worker after it has loaded and started training.
        const source = `
          addEventListener("message", event => {
            if (event.data === "test-worker-crash") {
              event.stopImmediatePropagation();
              throw new Error("Injected worker exception");
            }
          });
          await import(${JSON.stringify(String(url))});
        `;
        const blob = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
        super(blob, options);
        URL.revokeObjectURL(blob);
        window.workerInstances.push(this);
      }
      terminate() { this.wasTerminated = true; super.terminate(); }
    };
  });
  await page.goto(`./?env=${task}`);
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await page.evaluate(() => window.workerInstances.at(-1).postMessage("test-worker-crash"));
  await expect(page.locator("#error")).toContainText("Injected worker exception");
  await expect(page.locator("#status")).toHaveText("Session stopped");
  await expect(page.locator("#pause")).toBeDisabled();
  await expect(page.locator("#step")).toBeDisabled();
  await expect(page.locator("#export")).toBeDisabled();
  expect(await page.evaluate(() => window.workerInstances.at(-1).wasTerminated)).toBe(true);
  const frozen = await page.locator("#transitions").innerText();
  await page.waitForTimeout(200);
  await expect(page.locator("#transitions")).toHaveText(frozen);
  await page.getByRole("button", { name: "Restart", exact: true }).click();
  await expect(page.locator("#error")).toBeHidden();
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  expect(await page.evaluate(() => window.workerInstances.at(-1).wasTerminated ?? false)).toBe(false);
});
