const { test, expect } = require("@playwright/test");
test.use({ video: { mode: "on", size: { width: 1280, height: 800 } }, viewport: { width: 1280, height: 800 } });

test("Acrobot 1x follows its 0.2-second physical timestep", async ({ page }) => {
  await page.clock.install({ time: new Date("2026-10-08T00:00:00Z") });
  await page.clock.pauseAt(new Date("2026-10-08T00:00:01Z"));
  await page.addInitScript(() => {
    window.requestAnimationFrame = callback => { window.nextFrame = callback; return 1; };
  });
  await page.goto("./?env=acrobot");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await page.evaluate(() => { window.presentationTime = performance.now(); });
  for (let transition = 1; transition <= 10; transition++) {
    // Two 100 ms frames avoid the page's 100 ms long-frame clamp.
    for (let frame = 0; frame < 2; frame++) {
      await page.evaluate(() => { window.presentationTime += 100; window.nextFrame(window.presentationTime); });
    }
    await expect(page.locator("#transitions")).toHaveText(String(transition));
  }
  await expect(page.locator("#updates")).toHaveText("0");
});

test("Acrobot trains locally and replays its exported policy without optimization", async ({ page }, testInfo) => {
  test.setTimeout(180000);
  await page.goto("./?env=acrobot");
  await expect(page.locator("h1")).toHaveText("Acrobot-v1");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await page.waitForTimeout(20000);
  await expect(page.locator("#updates")).toHaveText("0");
  await page.screenshot({ path: testInfo.outputPath("acrobot-inference-desktop.png"), fullPage: true });
  await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
  await page.locator("#speed").selectOption("16");
  await expect(page.locator("#mode")).toHaveText("Training · DQN");
  await expect.poll(async () => Number((await page.locator("#updates").innerText()).replaceAll(",", "")), { timeout: 120000 }).toBeGreaterThan(0);
  await expect(page.locator("#rate")).toHaveText("3.0e-4");
  await expect(page.locator("#critic-rate-row")).toBeHidden();
  await expect(page.locator("#epsilon-row")).toBeVisible();
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused");
  expect(await page.locator("#scene").evaluate(canvas => [canvas.width, canvas.height])).toEqual([500, 500]);
  const scene = await page.locator("#scene").boundingBox();
  expect(Math.abs(scene.width - scene.height)).toBeLessThan(1);
  const chart = await page.locator("#returns-chart").boundingBox();
  expect(chart.y + chart.height).toBeLessThanOrEqual(800);
  await expect(page.locator("#return-summary")).not.toHaveText("No completed episodes");
  await page.screenshot({ path: testInfo.outputPath("acrobot-training-desktop.png"), fullPage: true });
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
  await expect(page.locator("#error")).toBeHidden();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: testInfo.outputPath("acrobot-inference-mobile.png"), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
