import assert from 'node:assert/strict';
import test from 'node:test';
import { activateCamera } from '../pursuit/camera.js';

test('pointer lock rejection leaves drag aiming available without an unhandled promise', async () => {
  const listeners = new Map();
  const canvas = {
    addEventListener: (name, callback) => listeners.set(name, callback),
    requestPointerLock: () => Promise.reject(new Error('Embedded preview rejects pointer lock')),
  };
  const document = { querySelector: () => canvas, pointerLockElement: null, exitPointerLock() {} };
  const scope = {};
  activateCamera(scope, document);
  await scope.droneCameraCapture();
  assert.equal(scope.droneCameraCaptured(), false);
  let prevented = false;
  listeners.get('contextmenu')({ preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
});

test('capture follows the actual browser state and release clears it', async () => {
  const document = { pointerLockElement: null };
  const canvas = { addEventListener() {}, requestPointerLock() { document.pointerLockElement = canvas; } };
  document.querySelector = () => canvas;
  document.exitPointerLock = () => { document.pointerLockElement = null; };
  const scope = {};
  activateCamera(scope, document);
  await scope.droneCameraCapture();
  assert.equal(scope.droneCameraCaptured(), true);
  scope.droneCameraRelease();
  assert.equal(scope.droneCameraCaptured(), false);
});
