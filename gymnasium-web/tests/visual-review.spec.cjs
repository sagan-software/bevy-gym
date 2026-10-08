const { test, expect } = require("@playwright/test");

test.use({ video: { mode: "on", size: { width: 1280, height: 800 } }, viewport: { width: 1280, height: 800 } });

for (const environment of ["cartpole", "mountain-car"]) {
  test(`visual review ${environment}`, async ({ page }, testInfo) => {
    await page.goto(`./?env=${environment}`);
    await expect(page.locator("#status")).toHaveText("Running frozen policy");
    await page.waitForTimeout(6000);
    await page.screenshot({ path: testInfo.outputPath(`${environment}-desktop.png`), fullPage: true });
    const chart = await page.locator("#returns-chart").boundingBox();
    expect(chart.y + chart.height).toBeLessThanOrEqual(800);
    const scene = await page.locator("#scene").boundingBox();
    expect(scene.width / scene.height).toBeCloseTo(1.5, 4);
    await page.locator("#speed").selectOption("16");
    await page.getByRole("button", { name: "Train from scratch", exact: true }).click();
    await expect(page.locator("#status")).toHaveText("Training in this browser");
    await expect.poll(async () => Number((await page.locator("#updates").innerText()).replaceAll(",", ""))).toBeGreaterThan(0);
    await page.waitForTimeout(3000);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: testInfo.outputPath(`${environment}-mobile.png`), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.getByRole("button", { name: "Pause", exact: true }).click();
    await expect(page.locator("#status")).toHaveText("Paused");
  });
}
