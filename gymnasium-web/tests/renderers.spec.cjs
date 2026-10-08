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
  ["pendulum-upright", "drawPendulum", [0, 0, 0, 0]],
  ["pendulum-down", "drawPendulum", [Math.PI, 0, 0, 0]],
  ["pendulum-positive-torque", "drawPendulum", [-0.8, 2, 2, 0]],
  ["pendulum-negative-torque", "drawPendulum", [1.4, -3, -1, 0]],
]) {
  test(`Gymnasium rendering matches ${name}`, async ({ page }, testInfo) => {
    await page.goto("./");
    const reference = await readFile(path.join(__dirname, "fixtures", `${name}.png`));
    const result = await page.evaluate(async ({ renderer, state, goal, reference }) => {
      const assetBase = new URL(".", document.querySelector('script[type="module"]').src);
      const renderers = await import(new URL("renderers.js", assetBase));
      if (renderer === "drawPendulum") await renderers.loadPendulumAsset();
      const actual = document.createElement("canvas");
      renderers[renderer](actual, state, goal);
      const expected = document.createElement("canvas");
      const width = renderer === "drawPendulum" ? 500 : 600, height = renderer === "drawPendulum" ? 500 : 400;
      expected.width = width; expected.height = height;
      const img = new Image(); img.src = reference; await img.decode();
      expected.getContext("2d").drawImage(img, 0, 0);
      const a = actual.getContext("2d").getImageData(0, 0, width, height).data;
      const b = expected.getContext("2d").getImageData(0, 0, width, height).data;
      function mismatch(source, target) {
        let foreground = 0, missed = 0;
        for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 4;
          if (Math.min(source[i], source[i + 1], source[i + 2]) > 220) continue;
          foreground++;
          let found = false;
          for (let dy = -2; dy <= 2 && !found; dy++) for (let dx = -2; dx <= 2; dx++) {
            if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;
            const j = ((y + dy) * width + x + dx) * 4;
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
    expect([result.width, result.height]).toEqual(renderer === "drawPendulum" ? [500, 500] : [600, 400]);
    expect(result.actual.foreground).toBeGreaterThan(1000);
    expect(result.reference.fraction).toBeLessThan(0.005);
    expect(result.actual.fraction).toBeLessThan(0.005);
  });
}
