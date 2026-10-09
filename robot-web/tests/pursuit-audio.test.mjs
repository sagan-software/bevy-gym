import assert from "node:assert/strict";
import test from "node:test";
import { activateAudio } from "../pursuit/audio.js";

test("audio resumes on play gestures and retries after a rejected resume", async () => {
  let attempts = 0;
  class Context {
    state = "suspended";
    resume() {
      attempts += 1;
      if (attempts === 1) return Promise.reject(new Error("gesture rejected"));
      this.state = "running";
      return Promise.resolve();
    }
  }
  const scope = { AudioContext: Context };
  const events = new EventTarget();
  activateAudio(scope, events);
  events.dispatchEvent(new Event("pointerdown"));
  const context = new scope.AudioContext();
  assert.equal(attempts, 0);
  events.dispatchEvent(new Event("keydown"));
  await Promise.resolve();
  assert.equal(context.state, "suspended");
  events.dispatchEvent(new Event("pointerup"));
  await Promise.resolve();
  assert.equal(context.state, "running");
  events.dispatchEvent(new Event("keydown"));
  assert.equal(attempts, 2);
});

test("closed contexts are discarded and missing audio support is harmless", () => {
  const events = new EventTarget();
  activateAudio({}, events);
  let attempts = 0;
  class Context {
    state = "closed";
    resume() { attempts += 1; return Promise.resolve(); }
  }
  const scope = { AudioContext: Context };
  activateAudio(scope, events);
  new scope.AudioContext();
  events.dispatchEvent(new Event("pointerdown"));
  events.dispatchEvent(new Event("keydown"));
  assert.equal(attempts, 0);
});
