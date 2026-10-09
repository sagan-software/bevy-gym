import assert from "node:assert/strict";
import { copyFile, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

// Run the actual entry modules in an isolated process with a browser boundary stub.
async function boot(t, bindings, assertions) {
  const directory = await mkdtemp(join(tmpdir(), "pursuit-start-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  for (const name of ["start.js", "audio.js", "camera.js"]) {
    await copyFile(new URL(`../pursuit/${name}`, import.meta.url), join(directory, name));
  }
  await writeFile(join(directory, "package.json"), '{"type":"module"}');
  await writeFile(join(directory, "drone-pursuit.js"), bindings);
  const program = `
    import assert from "node:assert/strict";
    globalThis.AudioContext = class {
      state = "suspended";
      resume() { this.state = "running"; return Promise.resolve(); }
    };
    const loading = { removed: false, remove() { this.removed = true; } };
    globalThis.document = new EventTarget();
    const canvas = new EventTarget();
    document.querySelector = selector => selector === "#pursuit-canvas" ? canvas : loading;
    const errors = [];
    console.error = error => errors.push(error.message);
    await import("./start.js");
    ${assertions}
  `;
  const result = spawnSync(process.execPath, ["--input-type=module"], {
    cwd: directory, input: program, encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr);
}

test("audio activation precedes the bindings' captured constructor", async t => {
  await boot(t, `
    const CapturedContext = globalThis.AudioContext;
    export default async function init() {
      if (typeof globalThis.droneCameraCapture !== "function") throw new Error("camera adapter missing");
      globalThis.context = new CapturedContext();
    }
  `, `
    assert.equal(loading.removed, true);
    assert.equal(context.state, "suspended");
    document.dispatchEvent(new Event("keydown"));
    await Promise.resolve();
    assert.equal(context.state, "running");
    assert.deepEqual(errors, []);
  `);
});

test("a bindings import failure retains a reload instruction", async t => {
  await boot(t, 'throw new Error("bindings unavailable");', `
    assert.equal(loading.removed, false);
    assert.equal(loading.textContent, "The arena could not start. Reload to retry.");
    assert.deepEqual(errors, ["bindings unavailable"]);
  `);
});
