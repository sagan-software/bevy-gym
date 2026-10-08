const { expect, test } = require("@playwright/test");

// Fail before compilation when an installed browser cannot create a page.
test.describe.configure({ timeout: 20000 });
test("browser runtime starts", async ({ page, browser }, testInfo) => {
  await page.goto("data:text/html,<title>Browser runtime ready</title>");
  await expect(page).toHaveTitle("Browser runtime ready");
  console.log(`${testInfo.project.name}: ${browser.version()}`);
});
