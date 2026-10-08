const { test, expect } = require("@playwright/test");
const { readFile, writeFile, mkdir } = require("node:fs/promises");
const path = require("node:path");

// Compare foreground pixels against the actual pinned Gymnasium renderer.
// A two-pixel neighborhood permits Pygame/Canvas rasterization differences.
for (const [name, renderer, state, goal] of [
  ["cartpole-upright", "drawCartPole", [0, 0, 0, 0]],
  ["cartpole-tilted", "drawCartPole", [-1.25, 0, 0.2, 0]],
  ["mountain-car-valley", "drawMountainCar", [-0.5, 0]],
  ["mountain-car-slope", "drawMountainCar", [0.45, 0.03]],
  ["mountain-car-continuous-valley", "drawMountainCar", [-0.5, 0], 0.45],
  ["mountain-car-continuous-slope", "drawMountainCar", [0.45, 0.03], 0.45],
]) {
  test(`Gymnasium rendering matches ${name}`, async ({ page }, testInfo) => {
    await page.goto("./");
    const reference = await readFile(path.join(__dirname, "fixtures", `${name}.png`));
    const result = await page.evaluate(async ({ renderer, state, goal, reference }) => {
      const renderers = await import("./renderers.js");
      const actual = document.createElement("canvas");
      renderers[renderer](actual, state, goal);
      const expected = document.createElement("canvas");
      expected.width = 600; expected.height = 400;
      const img = new Image(); img.src = reference; await img.decode();
      expected.getContext("2d").drawImage(img, 0, 0);
      const a = actual.getContext("2d").getImageData(0, 0, 600, 400).data;
      const b = expected.getContext("2d").getImageData(0, 0, 600, 400).data;
      function mismatch(source, target) {
        let foreground = 0, missed = 0;
        for (let y = 0; y < 400; y++) for (let x = 0; x < 600; x++) {
          const i = (y * 600 + x) * 4;
          if (Math.min(source[i], source[i + 1], source[i + 2]) > 220) continue;
          foreground++;
          let found = false;
          for (let dy = -2; dy <= 2 && !found; dy++) for (let dx = -2; dx <= 2; dx++) {
            if (x + dx < 0 || x + dx >= 600 || y + dy < 0 || y + dy >= 400) continue;
            const j = ((y + dy) * 600 + x + dx) * 4;
            if ([0, 1, 2].every(c => Math.abs(source[i + c] - target[j + c]) <= 80)) { found = true; break; }
          }
          if (!found) missed++;
        }
        return { foreground, missed, fraction: missed / foreground };
      }
      return { width: actual.width, height: actual.height, actual: mismatch(a, b), reference: mismatch(b, a), png: actual.toDataURL() };
    }, { renderer, state, goal, reference: `data:image/png;base64,${reference.toString("base64")}` });
    const artifact = testInfo.outputPath(`${name}-actual.png`);
    await mkdir(path.dirname(artifact), { recursive: true });
    await writeFile(artifact, Buffer.from(result.png.split(",")[1], "base64"));
    await testInfo.attach(`${name}-actual`, { path: artifact, contentType: "image/png" });
    console.log(name, JSON.stringify({ actual: result.actual, reference: result.reference }));
    expect([result.width, result.height]).toEqual([600, 400]);
    expect(result.actual.foreground).toBeGreaterThan(1000);
    expect(result.reference.fraction).toBeLessThan(0.005);
    expect(result.actual.fraction).toBeLessThan(0.005);
  });
}
