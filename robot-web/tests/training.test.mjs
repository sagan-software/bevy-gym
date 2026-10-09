import assert from 'node:assert/strict';
import test from 'node:test';
import { Training } from '../training.mjs';

class WorkerDouble {
  sent = [];
  terminated = false;
  postMessage(text) { this.sent.push(JSON.parse(text)); }
  terminate() { this.terminated = true; }
  emit(value) { this.onmessage?.({data: JSON.stringify(value)}); }
}
const flush = () => new Promise(resolve => setImmediate(resolve));
const started = seed => ({event:'started',seed,updates:0,transitions:0,update_limit:260,actor_learning_rate:0.0003,critic_learning_rate:0.001});
const progress = updates => ({event:'progress',status:updates===260?'complete':'training',updates,transitions:updates*512,optimizer_steps:updates*4,actor_loss:0.1,critic_loss:2,entropy:-2,approximate_kl:0.01,actor_learning_rate:0.0003,critic_learning_rate:0.001});

function setup() {
  const workers=[]; const changes=[];
  const training=new Training(state=>changes.push(state),()=>{const w=new WorkerDouble();workers.push(w);return w;});
  return {training,workers,changes};
}

async function start(training, workers, seed=7) {
  const promise=training.start(seed); const worker=workers.at(-1);
  worker.emit({event:'ready',protocol:1}); await flush();
  assert.deepEqual(worker.sent.shift(),{command:'start',seed});
  worker.emit(started(seed)); await promise;
  assert.deepEqual(worker.sent.shift(),{command:'advance'});
  return worker;
}

test('pause waits for the pending update; resume preserves the run',async()=>{
  const {training,workers}=setup(); const worker=await start(training,workers);
  training.pause(); assert.equal(training.state.phase,'pausing');
  worker.emit(progress(1)); await flush();
  assert.equal(training.state.phase,'paused'); assert.equal(training.state.updates,1);
  assert.equal(worker.sent.length,0);
  training.resume(); assert.deepEqual(worker.sent.shift(),{command:'advance'});
  training.pause(); worker.emit(progress(2)); await flush();
  assert.equal(training.state.updates,2); assert.equal(workers.length,1);
  training.discard(); assert.equal(worker.terminated,true); assert.equal(training.state.phase,'idle');
});

test('discard A, start B, then resolve A last without changing B',async()=>{
  const {training,workers}=setup(); const a=await start(training,workers,7);
  const stale=a.onmessage;
  training.discard(); const b=await start(training,workers,8);
  training.pause(); b.emit(progress(1)); await flush();
  const before=training.state;
  stale({data:JSON.stringify(progress(99))}); await flush();
  assert.deepEqual(training.state,before); assert.equal(before.seed,8);
  assert.equal(a.terminated,true); assert.equal(b.terminated,false);
  training.discard();
});

const scores=()=>['0','1','2','42','18446744073709551615'].map(seed=>({seed,steps:500,reward:450,final_distance:0.2,survived:true}));
const evaluation=()=>({event:'evaluation',episodes:scores(),baseline:scores().map(s=>({...s,survived:false,steps:90,reward:50}))});
async function paused(setupResult) {
  const worker=await start(setupResult.training,setupResult.workers);
  setupResult.training.pause();worker.emit(progress(1));await flush();return worker;
}

test('seed validation precedes worker creation and valid endpoints start',async()=>{
  const {training,workers}=setup();const before=training.state;
  for(const seed of [-1,0.5,4294967296,NaN,Infinity,'7',null]) await assert.rejects(training.start(seed),RangeError);
  assert.equal(workers.length,0);assert.equal(training.state,before);
  training.pause();training.resume();assert.equal(training.state,before);
  await assert.rejects(training.checkpoint(),/Pause training/);
  for(const seed of [0,4294967295]) {
    const worker=await start(training,workers,seed);
    await assert.rejects(training.start(7),/Discard/);
    await assert.rejects(training.checkpoint(),/Pause training/);
    training.resume();assert.equal(worker.sent.length,0);
    training.discard();
  }
});

test('discard during readiness rejects old initialization and preserves a new run',async()=>{
  const {training,workers}=setup();const pending=training.start(7);const a=workers[0];const late=a.onmessage;
  training.discard();await pending;const b=await start(training,workers,8);training.pause();b.emit(progress(1));await flush();
  late({data:JSON.stringify({event:'ready',protocol:1})});await flush();
  assert.equal(training.state.seed,8);assert.equal(training.state.phase,'paused');training.discard();
});

test('unsupported, malformed, and unexpected replies fail without sending work',async()=>{
  for(const data of [null,'{','null','[]',JSON.stringify({event:'ready',protocol:2}),JSON.stringify({event:'error',message:'Unavailable'}),' '.repeat(1_048_577)]) {
    const {training,workers}=setup();const pending=training.start(7);const worker=workers[0];
    worker.onmessage({data});await pending;
    assert.equal(training.state.phase,'failed');assert.equal(worker.sent.length,0);assert.equal(worker.terminated,true);
    training.discard();assert.equal(training.state.phase,'idle');
  }
});

test('worker creation, transport, and delivery failures expose a retryable state',async()=>{
  const creation=new Training(()=>{},()=>{throw new Error('Unavailable');});
  await creation.start(7);assert.equal(creation.state.error,'Unavailable');
  for(const kind of ['error','empty_error','messageerror','post']) {
    const {training,workers}=setup();const pending=training.start(7);const worker=workers[0];
    if(kind==='error') worker.onerror({message:'Crashed'});
    if(kind==='empty_error') worker.onerror({});
    if(kind==='messageerror') worker.onmessageerror();
    if(kind==='post') {worker.postMessage=()=>{throw new Error('Cannot send');};worker.emit({event:'ready',protocol:1});}
    await pending;assert.equal(training.state.phase,'failed');assert.equal(worker.terminated,true);
    const retry=await start(training,workers,9);assert.equal(training.state.phase,'running');assert.equal(retry.terminated,false);training.discard();
  }
});

test('recipe and progress validation reject incompatible worker output',async()=>{
  for(const change of [{seed:8},{updates:1},{transitions:1},{update_limit:999},{actor_learning_rate:1},{critic_learning_rate:1}]) {
    const {training,workers}=setup();const pending=training.start(7);const worker=workers[0];
    worker.emit({event:'ready',protocol:1});await flush();worker.emit({...started(7),...change});await pending;
    assert.equal(training.state.phase,'failed');assert.equal(worker.sent.length,1);
  }
  for(const change of [{updates:2},{transitions:1},{status:'complete'},{optimizer_steps:0},{optimizer_steps:0.5},{optimizer_steps:Number.MAX_SAFE_INTEGER+1},
    ...['actor_loss','critic_loss','entropy','approximate_kl','actor_learning_rate','critic_learning_rate'].map(key=>({[key]:null}))]) {
    const s=setup();const worker=await start(s.training,s.workers);worker.emit({...progress(1),...change});await flush();
    assert.equal(s.training.state.phase,'failed');assert.equal(s.training.state.updates,0);
  }
});

test('checkpoint export and scoring preserve paused state and exact bytes',async()=>{
  const s=setup();const worker=await paused(s);const pending=s.training.checkpoint();
  assert.equal(s.training.state.phase,'inspecting');assert.deepEqual(worker.sent.shift(),{command:'export'});
  await assert.rejects(s.training.checkpoint(),/Pause training/);
  s.training.pause();s.training.resume();worker.emit({event:'policy',bytes:[0,1,255]});await flush();
  assert.deepEqual(worker.sent.shift(),{command:'evaluate',bytes:[0,1,255]});worker.emit(evaluation());
  const result=await pending;assert.deepEqual(result,{bytes:Uint8Array.of(0,1,255),seed:7,updates:1});
  assert.equal(s.training.state.phase,'paused');assert.equal(s.training.state.evaluation.updates,1);s.training.discard();
});

test('invalid checkpoints and evaluation records cannot reach the viewer',async()=>{
  for(const bytes of [undefined,[],[-1],[256],[0.5],['1'],[null]]) {
    const s=setup();const worker=await paused(s);const pending=s.training.checkpoint();const rejection=assert.rejects(pending,/invalid checkpoint/);
    worker.emit({event:'policy',bytes});await rejection;assert.equal(s.training.state.phase,'failed');
  }
  for(const invalid of [undefined,[],[...scores(),scores()[0]],scores().map((score,index)=>index===0?{...score,seed:'7'}:score),
    ...[{steps:0},{steps:501},{steps:1.5},{reward:null},{final_distance:-1},{final_distance:null},{survived:1}].map(change=>scores().map((score,index)=>index===0?{...score,...change}:score))]) {
    for(const key of ['episodes','baseline']) {
      const s=setup();const worker=await paused(s);const pending=s.training.checkpoint();const rejection=assert.rejects(pending,/invalid evaluation/);
      worker.emit({event:'policy',bytes:[1]});await flush();worker.emit({...evaluation(),[key]:invalid});await rejection;
      assert.equal(s.training.state.phase,'failed');
    }
  }
});

test('discarded checkpoint A cannot affect paused run B when A replies last',async()=>{
  const s=setup();const a=await paused(s);const checkpoint=s.training.checkpoint();const rejected=assert.rejects(checkpoint,{name:'AbortError'});const late=a.onmessage;
  s.training.discard();const b=await start(s.training,s.workers,8);s.training.pause();b.emit(progress(1));await flush();
  const before=s.training.state;late({data:JSON.stringify({event:'policy',bytes:[1]})});await rejected;await flush();
  assert.equal(s.training.state,before);s.training.discard();
});

test('completion stops at update 260 and keeps checkpoint export available',async()=>{
  const s=setup();const worker=await start(s.training,s.workers);
  for(let update=1;update<=260;update++) {
    worker.emit(progress(update));await flush();
    if(update<260) assert.deepEqual(worker.sent.shift(),{command:'advance'});
  }
  assert.equal(s.training.state.phase,'complete');assert.equal(worker.sent.length,0);s.training.resume();s.training.pause();
  const pending=s.training.checkpoint();worker.emit({event:'policy',bytes:[1]});await flush();worker.emit(evaluation());await pending;
  assert.equal(s.training.state.phase,'complete');s.training.discard();
});

test('unsolicited and error replies stop the active run',async()=>{
  for(const response of [{event:'error',message:'Failed update'},{event:'error'},{event:'policy',bytes:[1]}]) {
    const s=setup();const worker=await start(s.training,s.workers);worker.emit(response);await flush();
    assert.equal(s.training.state.phase,'failed');assert.equal(worker.terminated,true);
  }
  const s=setup();const worker=await paused(s);worker.emit(progress(2));await flush();assert.equal(s.training.state.phase,'failed');
});

test('a resolved old response cannot publish after the run is discarded',async()=>{
  const s=setup();const pending=s.training.start(7);const a=s.workers[0];
  a.emit({event:'ready',protocol:1});await flush();a.emit(started(7));s.training.discard();await pending;
  assert.equal(s.training.state.phase,'idle');assert.equal(a.sent.length,1);
  const b=await start(s.training,s.workers,8);b.emit(progress(1));s.training.discard();await flush();
  assert.equal(s.training.state.phase,'idle');assert.equal(s.training.state.updates,0);
});

test('discard between export and evaluation cannot start another request',async()=>{
  const s=setup();const worker=await paused(s);const pending=s.training.checkpoint();const rejected=assert.rejects(pending,{name:'AbortError'});
  worker.emit({event:'policy',bytes:[1]});s.training.discard();await rejected;
  assert.deepEqual(worker.sent,[{command:'export'}]);assert.equal(s.training.state.phase,'idle');
});

test('discard after evaluation resolves prevents checkpoint delivery',async()=>{
  const s=setup();const worker=await paused(s);const pending=s.training.checkpoint();const rejected=assert.rejects(pending,{name:'AbortError'});
  worker.emit({event:'policy',bytes:[1]});await flush();worker.emit(evaluation());s.training.discard();await rejected;
  assert.equal(s.training.state.evaluation,null);assert.equal(s.training.state.phase,'idle');
});
