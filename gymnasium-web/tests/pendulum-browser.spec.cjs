const { test, expect } = require("@playwright/test");
test.use({ video: { mode: "on", size: { width: 1280, height: 800 } }, viewport: { width: 1280, height: 800 } });

test("Pendulum trains locally and replays its exported policy without optimization", async ({ page }, testInfo) => {
  await page.goto("./?env=pendulum");
  await expect(page.locator("h1")).toHaveText("Pendulum-v1");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await page.locator("#speed").selectOption("16");
  await expect(page.locator("#mode")).toHaveText("Training · PPO");
  await expect.poll(async () => Number((await page.locator("#updates").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await expect(page.locator("#rate")).toHaveText("3.0e-3");
  await expect(page.locator("#critic-rate")).toHaveText("1.0e-3");
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused");
  expect(await page.locator("#scene").evaluate(canvas => [canvas.width, canvas.height])).toEqual([500, 500]);
  const scene = await page.locator("#scene").boundingBox();
  expect(Math.abs(scene.width - scene.height)).toBeLessThan(1);
  const chart = await page.locator("#returns-chart").boundingBox();
  expect(chart.y + chart.height).toBeLessThanOrEqual(800);
  await page.screenshot({ path: testInfo.outputPath("pendulum-training-desktop.png"), fullPage: true });
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download policy", exact: true }).click();
  await page.locator("#import").setInputFiles(await (await download).path());
  await expect(page.locator("#mode")).toHaveText("Inference");
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused");
  const before = Number((await page.locator("#transitions").innerText()).replaceAll(",", ""));
  await page.getByRole("button", { name: "Step", exact: true }).click();
  await expect(page.locator("#transitions")).toHaveText((before + 1).toLocaleString("en-US"));
  await expect(page.locator("#updates")).toHaveText("0");
  await expect(page.locator("#rate")).toHaveText("None");
  await expect(page.locator("#critic-rate")).toHaveText("None");
  await expect(page.locator("#error")).toBeHidden();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: testInfo.outputPath("pendulum-inference-mobile.png"), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

for (const outcome of ["success", "failure"]) test(`a delayed Pendulum image ${outcome} cannot restart a newer training session`, async ({ page }) => {
  await page.route("**/models/pendulum.mpk", route => route.abort());
  await page.addInitScript(outcome => {
    const NativeImage = Image, NativeWorker = Worker;
    let images = 0;
    window.createdWorkers = 0;
    window.Image = class extends NativeImage {
      async decode() {
        const index = ++images;
        await super.decode();
        if (index === 1) await new Promise((resolve, reject) => {
          window.releaseFirstImage = outcome === "success" ? resolve : () => reject(new Error("delayed image failure"));
        });
      }
    };
    window.Worker = class extends NativeWorker {
      constructor(...args) { super(...args); window.createdWorkers++; }
    };
  }, outcome);
  await page.goto("./?env=pendulum");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect.poll(() => page.evaluate(() => typeof window.releaseFirstImage)).toBe("function");
  await page.locator("#seed").fill("17");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect(page.locator("#pause")).toBeEnabled();
  await expect(page.locator("#policy-source")).toHaveText("Fresh model · seed 17");
  await expect(page.locator("#error")).toBeHidden();
  await page.evaluate(() => window.releaseFirstImage());
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await expect(page.locator("#policy-source")).toHaveText("Fresh model · seed 17");
  expect(await page.evaluate(() => window.createdWorkers)).toBe(1);
  await expect(page.locator("#error")).toBeHidden();
});

test("Pendulum reports an image failure and retries on a new session", async ({ page }) => {
  await page.route("**/assets/clockwise.png", route => route.abort());
  await page.goto("./?env=pendulum");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect(page.locator("#error")).toContainText("Pendulum rendering asset failed to load.");
  await expect(page.locator("#pause")).toBeDisabled();
  await page.unroute("**/assets/clockwise.png");
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await expect.poll(async () => Number((await page.locator("#transitions").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
  await expect(page.locator("#error")).toBeHidden();
});
