import assert from 'node:assert/strict';
import test from 'node:test';
import { CheckpointFile } from '../checkpoint-file.mjs';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const file = (name, bytes = [1, 2]) => ({name, size: bytes.length, arrayBuffer: async () => Uint8Array.from(bytes).buffer});
function setup() {
  const calls = [], states = [];
  const input = new CheckpointFile({recovery: bytes => calls.push(['recovery', ...bytes]), damage: bytes => calls.push(['damage', ...bytes])}, state => states.push(state));
  return {input, calls, states};
}

test('the selected kind receives bytes and the filename remains text', async () => {
  const {input, calls, states} = setup();
  await input.watch(file('<hover>.mpk'), 'damage');
  assert.deepEqual(calls, [['damage', 1, 2]]);
  assert.equal(states.at(-1).phase, 'watching');
  assert.equal(states.at(-1).name, '<hover>.mpk');
  await input.watch(file('recovery.mpk', [3]), 'recovery');
  assert.deepEqual(calls.at(-1), ['recovery', 3]);
});

test('missing files, invalid kinds, and oversized files fail before reading', async () => {
  const {input, calls, states} = setup();
  let reads = 0;
  const large = {name:'large.mpk', size:1048577, arrayBuffer:async()=>{reads++;return new ArrayBuffer(0);}};
  await input.watch(null, 'recovery');
  assert.equal(states.at(-1).phase, 'failed');
  await input.watch(large, 'unknown');
  await input.watch(large, 'recovery');
  assert.equal(reads, 0);
  assert.equal(calls.length, 0);
  await input.watch({...large, size:1048576}, 'recovery');
  assert.equal(reads, 1);
  assert.equal(calls.length, 1);
});

test('resolve B, then resolve A last; only B reaches the viewer', async () => {
  const {input, calls, states} = setup();
  const a = deferred(), b = deferred();
  const first = input.watch({name:'A.mpk',size:1,arrayBuffer:()=>a.promise}, 'recovery');
  input.clear();
  const second = input.watch({name:'B.mpk',size:1,arrayBuffer:()=>b.promise}, 'damage');
  b.resolve(Uint8Array.from([2]).buffer); await second;
  a.resolve(Uint8Array.from([1]).buffer); await first;
  assert.deepEqual(calls, [['damage', 2]]);
  assert.equal(states.at(-1).name, 'B.mpk');
});

test('clearing a pending selection ignores its later read error', async () => {
  const {input, calls, states} = setup();
  const pending = deferred();
  const result = input.watch({name:'A.mpk',size:1,arrayBuffer:()=>pending.promise}, 'recovery');
  input.clear();
  pending.reject(new Error('read failed')); await result;
  assert.deepEqual(calls, []);
  assert.equal(states.at(-1).phase, 'idle');
});

test('read failures and rejected checkpoints produce recoverable errors', async () => {
  const states = [];
  const input = new CheckpointFile({recovery:()=>{throw new Error('invalid model');}}, state=>states.push(state));
  await input.watch({name:'unreadable',size:1,arrayBuffer:async()=>{throw new Error('read failed');}}, 'recovery');
  assert.equal(states.at(-1).phase, 'failed');
  await input.watch(file('bad.mpk'), 'recovery');
  assert.equal(states.at(-1).phase, 'failed');
});
