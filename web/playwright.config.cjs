const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: process.env.BEVY_GYM_WEB_URL ?? "http://127.0.0.1:4173",
    browserName: "chromium",
    headless: true,
  },
});
