const { test, expect } = require("@playwright/test");

test("a new page cannot reuse unversioned application or worker assets", async ({ page }) => {
  const stale = [];
  await page.route(url => /^\/bevy-gym\/(?:app\.js|renderers\.js|gymnasium-worker[^/]*|models\/)/.test(url.pathname), route => {
    stale.push(route.request().url());
    return route.abort();
  });
  await page.goto("./?env=mountain-car-continuous");
  await expect(page.locator("#status")).toHaveText("Running frozen policy");
  await expect(page.locator("h1")).toHaveText("MountainCarContinuous-v0");
  await expect(page.locator("#error")).toBeHidden();
  expect(stale).toEqual([]);
  const script = await page.locator('script[type="module"]').getAttribute("src");
  expect(script).toMatch(/^build-[0-9a-f]{64}\/app\.js$/);
});
