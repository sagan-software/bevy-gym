const { defineConfig } = require("@playwright/test");
module.exports = defineConfig({
  testDir: "./tests",
  testMatch: "*.spec.cjs",
  workers: 1,
  outputDir: "../test-results/gymnasium",
  projects: [
    { name: "chromium", use: { browserName: "chromium" } },
    { name: "firefox", use: { browserName: "firefox" } },
    { name: "webkit", use: { browserName: "webkit" } },
  ],
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: process.env.BEVY_GYM_BROWSER_URL ?? "http://127.0.0.1:4174/bevy-gym/",
    headless: true,
    locale: "en-US",
    trace: "retain-on-failure",
  },
});
