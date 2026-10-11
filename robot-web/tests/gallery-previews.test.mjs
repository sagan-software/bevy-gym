import assert from 'node:assert/strict';
import { test } from 'node:test';
import { initializePreviews } from '../../gallery/previews.mjs';

function fixture(reduced = false) {
  const listeners = new Map();
  const document = { hidden: false, addEventListener: (key, fn) => listeners.set(key, fn) };
  const preference = { matches: reduced, addEventListener: (_, fn) => listeners.set('motion', fn) };
  const button = { textContent: '', hidden: true, addEventListener: (_, fn) => listeners.set('click', fn) };
  const videos = Array.from({ length: 2 }, () => ({
    muted: false, plays: 0, pauses: 0, dataset: { src: 'preview.mp4' },
    play() { this.plays++; return Promise.resolve(); },
    pause() { this.pauses++; },
  }));
  let intersect;
  class Observer {
    constructor(callback) { intersect = callback; }
    observe() {}
  }
  initializePreviews({ document, videos, button, preference, Observer });
  return { document, videos, button, preference, listeners, intersect };
}

test('only visible previews play; scrolling and hidden tabs pause them', () => {
  const f = fixture();
  assert.equal(f.videos[0].plays, 0);
  f.intersect([{ target: f.videos[0], isIntersecting: true }]);
  assert.equal(f.videos[0].plays, 1);
  assert.equal(f.videos[1].plays, 0);
  assert.equal(f.videos[0].muted, true);
  assert.equal(f.videos[0].src, 'preview.mp4');
  f.document.hidden = true;
  f.listeners.get('visibilitychange')();
  assert.equal(f.videos[0].pauses, 1);
  f.document.hidden = false;
  f.listeners.get('visibilitychange')();
  assert.equal(f.videos[0].plays, 2);
  f.intersect([{ target: f.videos[0], isIntersecting: false }]);
  assert.equal(f.videos[0].pauses, 2);
});

test('reduced motion starts paused and explicit play/pause remains available', () => {
  const f = fixture(true);
  f.intersect([{ target: f.videos[0], isIntersecting: true }]);
  assert.equal(f.videos[0].plays, 0);
  assert.equal(f.button.textContent, 'Play previews');
  f.listeners.get('click')();
  assert.equal(f.videos[0].plays, 1);
  assert.equal(f.button.textContent, 'Pause previews');
  f.listeners.get('click')();
  assert.equal(f.videos[0].pauses, 2);
  assert.equal(f.button.textContent, 'Play previews');
});

test('motion preference changes pause automatic playback', () => {
  const f = fixture();
  f.intersect([{ target: f.videos[0], isIntersecting: true }]);
  f.preference.matches = true;
  f.listeners.get('motion')();
  assert.equal(f.videos[0].pauses, 1);
  assert.equal(f.button.textContent, 'Play previews');
});

test('blocked autoplay retains a poster and offers an explicit retry', async () => {
  const f = fixture();
  f.videos[0].play = () => Promise.reject(new Error('autoplay blocked'));
  f.intersect([{ target: f.videos[0], isIntersecting: true }]);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.button.textContent, 'Play previews');
  f.videos[0].play = () => { f.videos[0].plays++; return Promise.resolve(); };
  f.listeners.get('click')();
  assert.equal(f.videos[0].plays, 1);
});

test('a stale play failure cannot stop a newer successful request', async () => {
  const f = fixture();
  let rejectA;
  f.videos[0].play = () => new Promise((_, reject) => { rejectA = reject; });
  f.intersect([{ target: f.videos[0], isIntersecting: true }]);
  f.videos[0].play = () => Promise.resolve();
  f.listeners.get('visibilitychange')();
  await new Promise(resolve => setImmediate(resolve));
  rejectA(new Error('older request failed'));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.button.textContent, 'Pause previews');
});

test('a play failure after manual pause cannot change the paused state', async () => {
  const f = fixture();
  let reject;
  f.videos[0].play = () => new Promise((_, fail) => { reject = fail; });
  f.intersect([{ target: f.videos[0], isIntersecting: true }]);
  f.listeners.get('click')();
  reject(new Error('paused request failed'));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.button.textContent, 'Play previews');
});

test('a new reduced-motion preference prevents a pending failure from disabling later playback', async () => {
  const f = fixture();
  let reject;
  f.videos[0].play = () => new Promise((_, fail) => { reject = fail; });
  f.intersect([{ target: f.videos[0], isIntersecting: true }]);
  f.preference.matches = true;
  reject(new Error('preference changed'));
  await new Promise(resolve => setImmediate(resolve));
  f.videos[0].play = () => Promise.resolve();
  f.preference.matches = false;
  f.listeners.get('motion')();
  assert.equal(f.button.textContent, 'Pause previews');
});

test('the browser entry selects the gallery media, control and motion preference', async () => {
  const f = fixture();
  const names = ['document', 'matchMedia', 'IntersectionObserver'];
  const originals = names.map(name => globalThis[name]);
  const selectors = [];
  let query;
  try {
    globalThis.document = {
      ...f.document,
      querySelectorAll(selector) { selectors.push(selector); return f.videos; },
      querySelector(selector) { selectors.push(selector); return f.button; },
    };
    globalThis.matchMedia = value => { query = value; return f.preference; };
    globalThis.IntersectionObserver = class {
      observe() {}
    };
    await import('../../gallery/previews-browser.mjs');
    assert.deepEqual(selectors, ['.example video', '#preview-toggle']);
    assert.equal(query, '(prefers-reduced-motion: reduce)');
    assert.equal(f.button.hidden, false);
    assert.equal(f.button.textContent, 'Pause previews');
  } finally {
    names.forEach((name, index) => {
      if (originals[index] === undefined) delete globalThis[name];
      else globalThis[name] = originals[index];
    });
  }
});
