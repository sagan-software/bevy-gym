import { drawCartPole, drawMountainCar } from "./renderers.js";
const $ = (id) => document.getElementById(id);
const tasks = {
  "cartpole": { title: "CartPole-v1", algorithm: "DQN", hz: 50, minimum: 0, maximum: 500, target: 475, rateMaximum: .0004, replay: 1000, lanes: 1 },
  "mountain-car": { title: "MountainCar-v0", algorithm: "DQN", hz: 30, minimum: -200, maximum: 0, target: -110, rateMaximum: .0014, replay: 2000, lanes: 1, goal: .5 },
  "mountain-car-continuous": { title: "MountainCarContinuous-v0", algorithm: "PPO", hz: 30, minimum: -100, maximum: 100, target: 90, rateMaximum: .0035, replay: 512, lanes: 8, goal: .45 },
};
const requestedTask = new URL(location.href).searchParams.get("env");
const taskName = Object.hasOwn(tasks, requestedTask) ? requestedTask : "cartpole";
const task = tasks[taskName], continuous = task.algorithm === "PPO";
$("updates-label").textContent = continuous ? "Updates per optimizer" : "Optimizer updates";
$("rate-label").textContent = continuous ? "Actor rate" : "Learning rate";
$("loss-label").textContent = continuous ? "Actor loss" : "TD loss";
$("epsilon-row").hidden = continuous;
$("critic-rate-row").hidden = !continuous;
$("critic-loss-row").hidden = !continuous;
$("environment").value = taskName;
$("environment").onchange = () => { location.search = new URLSearchParams({ env: $("environment").value }).toString(); };
document.querySelector("h1").textContent = task.title;
$("score-target").textContent = `Target mean ≥${task.target}`;
$("scene").setAttribute("aria-label", task.goal ? "MountainCar simulation: a car climbs the right hill" : "CartPole simulation: a cart balances an upright pole");
let worker, generation = 0, uploadGeneration = 0, cancelPolicyValidation;
/** @type {"loading" | "running" | "advancing" | "pausing" | "paused" | "failed"} */
let phase = "loading";
let currentMode = "inference", initialBytes, initialSource, bundledBytes, latest, lastTime = 0, debt = 0;
let returns = [], rates = [], criticRates = [], speedWindow = { time: performance.now(), transitions: 0 };

function setPhase(next) {
  phase = next;
  const unavailable = phase === "loading" || phase === "failed";
  $("pause").disabled = unavailable;
  $("step").disabled = unavailable || phase === "advancing" || phase === "pausing";
  $("export").disabled = unavailable;
  $("restart").disabled = phase === "loading";
  const label = phase === "paused" || phase === "pausing" ? "Resume" : "Pause";
  // WebKit can cancel a held click when its text node is replaced by a snapshot.
  if ($("pause").textContent !== label) $("pause").textContent = label;
}
function pausedStatus() { return document.hidden ? "Paused · tab hidden" : "Paused"; }
function syncVisibility() {
  if (document.hidden && (phase === "running" || phase === "advancing")) {
    setPhase(phase === "advancing" ? "pausing" : "paused");
    debt = 0;
    $("achieved").textContent = "0.0× actual";
    $("status").textContent = phase === "pausing" ? "Pausing after the current batch" : pausedStatus();
  } else if (phase === "paused") $("status").textContent = pausedStatus();
}
document.addEventListener("visibilitychange", syncVisibility);
function fail(message) {
  if (worker) worker.onmessage = null;
  worker?.terminate();
  setPhase("failed");
  $("error").textContent = `${message} Restart the session or load another policy.`;
  $("error").hidden = false;
  $("status").textContent = "Session stopped";
}
function seed() {
  if (!$("seed").reportValidity()) throw new Error("Seed must be an integer from 0 to 4294967295.");
  return Number($("seed").value);
}
function post(command) { worker.postMessage(JSON.stringify(command)); }
function start(mode, bytes, source = "Bundled model") {
  let runSeed;
  try { runSeed = seed(); } catch (error) { $("error").textContent = error.message; $("error").hidden = false; return; }
  const run = ++generation;
  ++uploadGeneration; cancelPolicyValidation?.();
  worker?.terminate();
  currentMode = mode; initialBytes = bytes; initialSource = source; latest = undefined;
  setPhase("loading"); debt = 0; returns = []; rates = []; criticRates = [];
  $("error").hidden = true;
  $("mode").textContent = mode === "training" ? `Training · ${task.algorithm}` : "Inference";
  $("policy-source").textContent = mode === "training" ? `Fresh model · seed ${runSeed}` : source;
  $("status").textContent = "Starting Rust worker";
  $("pause").textContent = "Pause";
  ["transitions", "episodes", "return", "updates"].forEach((id) => $(id).textContent = "0");
  ["rate", "epsilon", "loss", "critic-rate", "critic-loss"].forEach((id) => $(id).textContent = "None");
  $("return-summary").textContent = "No completed episodes";
  $("rate-summary").textContent = mode === "training" ? "Waiting for the first optimizer update." : "Inference performs no optimizer updates.";
  drawScene([0, 0, 0, 0]); drawCharts();
  worker = new Worker(new URL("gymnasium-worker_loader.js", document.baseURI), { type: "module", name: taskName });
  worker.onerror = (event) => { if (run === generation) fail(event.message || "The Rust worker failed."); };
  worker.onmessage = ({ data }) => {
    if (run !== generation) return;
    let message;
    try { message = JSON.parse(data); } catch { fail("The worker returned an invalid message."); return; }
    switch (message.event) {
      case "ready":
        if (message.protocol !== 1) { fail("Worker protocol changed. Reload this page."); break; }
        post(mode === "training" ? { command: "start_training", seed: runSeed }
          : { command: "start_inference", seed: runSeed, bytes: Array.from(bytes) });
        break;
      case "started":
        setPhase("running"); lastTime = performance.now();
        speedWindow = { time: lastTime, transitions: 0 };
        $("status").textContent = mode === "training" ? (continuous ? "Collecting PPO rollout" : "Collecting replay transitions") : "Running frozen policy";
        syncVisibility();
        break;
      case "snapshot": setPhase(phase === "pausing" ? "paused" : "running"); update(message.snapshot); break;
      case "policy": {
        const url = URL.createObjectURL(new Blob([new Uint8Array(message.bytes)], { type: "application/octet-stream" }));
        const link = document.createElement("a"); link.href = url; link.download = `${taskName}-${currentMode}-${latest?.transitions ?? 0}.mpk`; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000); break;
      }
      case "error": fail(message.message); break;
      default: fail("Unknown worker response.");
    }
  };
}
function advance(steps) { setPhase(phase === "paused" ? "pausing" : "advancing"); post({ command: "advance", steps }); }
function update(snapshot) {
  latest = snapshot;
  $("transitions").textContent = snapshot.transitions.toLocaleString();
  $("episodes").textContent = snapshot.episode_count.toLocaleString();
  $("return").textContent = snapshot.episode_return.toFixed(0);
  $("updates").textContent = snapshot.optimizer_steps.toLocaleString();
  $("rate").textContent = snapshot.learning_rate?.toExponential(1) ?? "None";
  $("critic-rate").textContent = snapshot.critic_learning_rate?.toExponential(1) ?? "None";
  $("critic-loss").textContent = snapshot.critic_loss?.toFixed(4) ?? "None";
  $("epsilon").textContent = snapshot.epsilon?.toFixed(3) ?? "None";
  $("loss").textContent = snapshot.loss?.toFixed(4) ?? "None";
  returns.push(...snapshot.completed.map((episode) => [episode.transition, episode.reward]));
  if (returns.length > 2000) returns.splice(0, returns.length - 2000);
  if (snapshot.optimizer_steps > 0) {
    rates = [[0, snapshot.learning_rate], [snapshot.optimizer_steps, snapshot.learning_rate]];
    criticRates = continuous ? [[0, snapshot.critic_learning_rate], [snapshot.optimizer_steps, snapshot.critic_learning_rate]] : [];
    $("rate-summary").textContent = continuous
      ? `Actor ${snapshot.learning_rate} · critic ${snapshot.critic_learning_rate} · ${snapshot.optimizer_steps.toLocaleString()} updates`
      : `Adam · constant ${snapshot.learning_rate} · ${snapshot.optimizer_steps.toLocaleString()} updates`;
  }
  if (returns.length) {
    const recent = returns.slice(-20);
    const mean = recent.reduce((sum, point) => sum + point[1], 0) / recent.length;
    $("return-summary").textContent = `Mean of last ${recent.length} episodes: ${mean.toFixed(1)} · raw returns and rolling mean`;
  }
  $("status").textContent = phase === "paused" ? pausedStatus() : currentMode === "inference" ? "Running frozen policy"
    : snapshot.optimizer_steps ? (continuous ? "Training 8 environments · showing environment 1" : "Training in this browser")
      : `Collecting ${continuous ? "PPO rollout" : "replay transitions"} · ${snapshot.transitions}/${task.replay.toLocaleString()}`;
  const now = performance.now();
  if (phase === "running" && now - speedWindow.time >= 1000) {
    const achieved = (snapshot.transitions - speedWindow.transitions) * (1000 / (task.hz * (currentMode === "training" ? task.lanes : 1))) / (now - speedWindow.time);
    $("achieved").textContent = `${achieved.toFixed(1)}× actual`;
    speedWindow = { time: now, transitions: snapshot.transitions };
  }
  drawScene(snapshot.state); drawCharts();
}
function frame(now) {
  const elapsed = Math.min((now - lastTime) / 1000, .1); lastTime = now;
  if (phase === "running" || phase === "advancing") {
    // One in-flight batch bounds control latency. Speed never changes physics or updates per transition.
    debt = Math.min(debt + elapsed * task.hz * (currentMode === "training" ? task.lanes : 1) * Number($("speed").value), 64);
    if (phase === "running" && debt >= 1) { const steps = Math.floor(debt); debt -= steps; advance(steps); }
  } else { debt = 0; }
  requestAnimationFrame(frame);
}
$("pause").onclick = () => {
  const transitions = { running: "paused", advancing: "pausing", paused: "running", pausing: "advancing" };
  const next = transitions[phase];
  if (!next) return;
  setPhase(next); debt = 0;
  $("status").textContent = phase === "pausing" ? "Pausing after the current batch" : phase === "paused" ? pausedStatus() : "Running";
  speedWindow = { time: performance.now(), transitions: latest?.transitions ?? 0 };
  if (phase === "paused" || phase === "pausing") $("achieved").textContent = "0.0× actual";
};
$("step").onclick = () => { if (phase === "running" || phase === "paused") { setPhase("paused"); advance(1); } };
$("restart").onclick = () => start(currentMode, initialBytes, initialSource);
$("train").onclick = () => start("training");
$("bundled").onclick = () => { if (bundledBytes) start("inference", bundledBytes); };
$("export").onclick = () => post({ command: "export" });
// Validate in an isolated worker before replacing the active simulation or optimizer.
function validatePolicy(bytes) {
  return new Promise((resolve, reject) => {
    const candidate = new Worker(new URL("gymnasium-worker_loader.js", document.baseURI), { type: "module", name: taskName });
    const finish = (error) => {
      clearTimeout(timeout); candidate.terminate();
      if (cancelPolicyValidation === cancel) cancelPolicyValidation = undefined;
      if (error) reject(error); else resolve();
    };
    const cancel = () => finish(new Error("Policy validation superseded."));
    const timeout = setTimeout(() => finish(new Error("Policy validation timed out. Try uploading again.")), 10000);
    cancelPolicyValidation = cancel;
    candidate.onerror = event => finish(new Error(event.message || "Policy validation failed."));
    candidate.onmessage = ({ data }) => {
      try {
        const message = JSON.parse(data);
        if (message.event === "ready" && message.protocol === 1) {
          candidate.postMessage(JSON.stringify({ command: "start_inference", seed: 0, bytes: Array.from(bytes) }));
        } else if (message.event === "started") finish();
        else finish(new Error(message.message || "Policy validator returned an invalid response."));
      } catch (error) { finish(error); }
    };
  });
}
$("import").onchange = async (event) => {
  const file = event.target.files[0]; if (!file) return;
  event.target.value = "";
  const run = generation, upload = ++uploadGeneration;
  cancelPolicyValidation?.();
  const current = () => run === generation && upload === uploadGeneration;
  try {
    if (file.size > 131072) throw new Error("Policy exceeds the 128 KiB limit.");
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (!current()) return;
    await validatePolicy(bytes);
    if (current()) start("inference", bytes, `Uploaded policy: ${file.name}`);
  } catch (error) {
    if (current()) { $("error").textContent = error.message; $("error").hidden = false; }
  }
};
function context(id) {
  const canvas = $(id), bounds = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(bounds.width * dpr); canvas.height = Math.round(bounds.height * dpr);
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); return [ctx, bounds.width, bounds.height];
}
function drawScene(state) {
  const canvas = $("scene");
  if (task.goal) drawMountainCar(canvas, state, task.goal);
  else drawCartPole(canvas, state);
}
function chart(id, points, maximum, target, moving, minimum = 0, horizontalLabel = "Transitions", secondary = []) {
  const [ctx, width, height] = context(id), left = 48, right = width - 12, top = 22, bottom = height - 34;
  const first = points[0]?.[0] ?? 0, last = Math.max(first + 1, points.at(-1)?.[0] ?? 1);
  const x = (value) => left + (value - first) / (last - first) * (right - left);
  const y = (value) => bottom - (value - minimum) / (maximum - minimum) * (bottom - top);
  ctx.font = "11px system-ui"; ctx.fillStyle = "#a5b2a8"; ctx.textAlign = "right";
  for (let i = 0; i <= 2; i++) {
    const value = minimum + (maximum - minimum) * i / 2; ctx.fillText(maximum > 0 && maximum < 1 ? value.toExponential(1) : value.toFixed(0), left - 8, y(value) + 4);
    ctx.strokeStyle = "#34453a"; ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(right, y(value)); ctx.stroke();
  }
  if (target != null) { ctx.strokeStyle = "#a5b2a8"; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(left,y(target));ctx.lineTo(right,y(target));ctx.stroke();ctx.setLineDash([]); }
  const line = (series, color) => { ctx.strokeStyle = color;ctx.lineWidth = 1.5;ctx.beginPath();series.forEach((point,index)=>{if(index)ctx.lineTo(x(point[0]),y(point[1]));else ctx.moveTo(x(point[0]),y(point[1]));});ctx.stroke(); };
  line(points, moving ? "#76937d" : "#b6ee63");
  if (secondary.length) {
    line(secondary, "#7bc9f0");
    ctx.textAlign = "left"; ctx.fillStyle = "#b6ee63"; ctx.fillText("Actor", left, 12);
    ctx.fillStyle = "#7bc9f0"; ctx.fillText("Critic", left + 52, 12);
  }
  if (moving) line(points.map((point,index)=>{const batch=points.slice(Math.max(0,index-19),index+1);return [point[0],batch.reduce((sum,p)=>sum+p[1],0)/batch.length];}),"#b6ee63");
  ctx.fillStyle="#a5b2a8";ctx.textAlign="left";ctx.fillText(first.toLocaleString(),left,bottom+18);ctx.textAlign="right";ctx.fillText(last.toLocaleString(),right,bottom+18);ctx.textAlign="center";ctx.fillText(horizontalLabel,(left+right)/2,height-3);
}
function drawCharts() { chart("returns-chart", returns, task.maximum, task.target, true, task.minimum); chart("rate-chart", rates, task.rateMaximum, null, false, 0, continuous ? "Updates per optimizer" : "Optimizer updates", criticRates); }
window.addEventListener("resize",()=>{drawScene(latest?.state ?? [0,0,0,0]);drawCharts();});
drawScene([0,0,0,0]);drawCharts();requestAnimationFrame(frame);
const initialGeneration = generation;
$("bundled").disabled = true;
try {
  const [response, metadataResponse] = await Promise.all([
    fetch(new URL(`models/${taskName}.mpk`, document.baseURI)),
    fetch(new URL(`models/${taskName}.json`, document.baseURI)),
  ]);
  if (!response.ok || !metadataResponse.ok) throw new Error("Bundled policy or qualification report failed to load.");
  const metadata = await metadataResponse.json();
  const bytes = new Uint8Array(await response.arrayBuffer());
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  const hash = Array.from(digest, value => value.toString(16).padStart(2, "0")).join("");
  if (hash !== metadata.sha256) throw new Error("Bundled policy does not match its qualification report.");
  const report = metadata.qualification.find(run => metadata.selected_run
    ? run.run_id === metadata.selected_run : run.seed === metadata.selected_training_seed);
  if (!Number.isFinite(report?.test_mean)) throw new Error("Bundled policy has no qualification score.");
  bundledBytes = bytes;
  $("model-evidence").textContent = `Bundled model · mean ${report.test_mean.toFixed(2)} over ${report.test_episodes} held-out episodes.`;
  $("bundled").disabled = false;
  if (generation === initialGeneration) start("inference", bundledBytes);
} catch (error) {
  $("model-evidence").textContent = "Bundled model unavailable.";
  if (generation === initialGeneration) fail(error.message);
}
