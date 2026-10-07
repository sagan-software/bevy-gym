const $ = (id) => document.getElementById(id);
let worker, generation = 0, running = false, ready = false, busy = false;
let currentMode = "inference", initialBytes, bundledBytes, latest, lastTime = 0, debt = 0;
let returns = [], rates = [], speedWindow = { time: performance.now(), transitions: 0 };
const controlled = ["pause", "step", "restart", "export"];

function fail(message) {
  running = false;
  $("error").textContent = `${message} Restart the session or load another policy.`;
  $("error").hidden = false;
  $("status").textContent = "Session stopped";
  $("pause").textContent = "Resume";
}
function seed() {
  if (!$("seed").reportValidity()) throw new Error("Seed must be an integer from 0 to 4294967295.");
  return Number($("seed").value);
}
function post(command) { worker.postMessage(JSON.stringify(command)); }
function start(mode, bytes) {
  let runSeed;
  try { runSeed = seed(); } catch (error) { fail(error.message); return; }
  const run = ++generation;
  worker?.terminate();
  currentMode = mode; initialBytes = bytes; latest = undefined;
  ready = false; busy = false; running = false; debt = 0; returns = []; rates = [];
  controlled.forEach((id) => $(id).disabled = true);
  $("error").hidden = true;
  $("mode").textContent = mode === "training" ? "Training · DQN" : "Inference";
  $("status").textContent = "Starting Rust worker";
  $("pause").textContent = "Pause";
  ["transitions", "episodes", "return", "updates"].forEach((id) => $(id).textContent = "0");
  ["rate", "epsilon", "loss"].forEach((id) => $(id).textContent = "None");
  $("return-summary").textContent = "No completed episodes";
  $("rate-summary").textContent = mode === "training" ? "Waiting for the first optimizer update." : "Inference performs no optimizer updates.";
  drawScene([0, 0, 0, 0]); drawCharts();
  worker = new Worker(new URL("gymnasium-worker_loader.js", document.baseURI), { type: "module" });
  worker.onerror = (event) => { if (run === generation) fail(event.message || "The Rust worker failed."); };
  worker.onmessage = ({ data }) => {
    if (run !== generation) return;
    let message;
    try { message = JSON.parse(data); } catch { fail("The worker returned an invalid message."); return; }
    switch (message.event) {
      case "ready":
        post(mode === "training" ? { command: "start_training", seed: runSeed }
          : { command: "start_inference", seed: runSeed, bytes: Array.from(bytes) });
        break;
      case "started":
        ready = true; running = true; lastTime = performance.now();
        speedWindow = { time: lastTime, transitions: 0 };
        controlled.forEach((id) => $(id).disabled = false);
        $("status").textContent = mode === "training" ? "Collecting replay transitions" : "Running frozen policy";
        break;
      case "snapshot": busy = false; update(message.snapshot); break;
      case "policy": {
        const url = URL.createObjectURL(new Blob([new Uint8Array(message.bytes)], { type: "application/octet-stream" }));
        const link = document.createElement("a"); link.href = url; link.download = `cartpole-${currentMode}-${latest?.transitions ?? 0}.mpk`; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000); break;
      }
      case "error": busy = false; fail(message.message); break;
      default: fail("Unknown worker response.");
    }
  };
}
function advance(steps) { busy = true; post({ command: "advance", steps }); }
function update(snapshot) {
  latest = snapshot;
  $("transitions").textContent = snapshot.transitions.toLocaleString();
  $("episodes").textContent = snapshot.episode_count.toLocaleString();
  $("return").textContent = snapshot.episode_return.toFixed(0);
  $("updates").textContent = snapshot.optimizer_steps.toLocaleString();
  $("rate").textContent = snapshot.learning_rate?.toExponential(1) ?? "None";
  $("epsilon").textContent = snapshot.epsilon?.toFixed(3) ?? "None";
  $("loss").textContent = snapshot.loss?.toFixed(4) ?? "None";
  returns.push(...snapshot.completed.map((episode) => [episode.transition, episode.reward]));
  if (returns.length > 2000) returns.splice(0, returns.length - 2000);
  if (snapshot.optimizer_steps > 0) {
    rates = [[0, snapshot.learning_rate], [snapshot.transitions, snapshot.learning_rate]];
    $("rate-summary").textContent = `Adam · constant ${snapshot.learning_rate} · ${snapshot.optimizer_steps.toLocaleString()} updates`;
  }
  if (returns.length) {
    const recent = returns.slice(-20);
    const mean = recent.reduce((sum, point) => sum + point[1], 0) / recent.length;
    $("return-summary").textContent = `Mean of last ${recent.length} episodes: ${mean.toFixed(1)} · raw returns and rolling mean`;
  }
  $("status").textContent = !running ? "Paused" : currentMode === "inference" ? "Running frozen policy"
    : snapshot.optimizer_steps ? "Training in this browser" : `Collecting replay transitions · ${snapshot.transitions}/1,000`;
  const now = performance.now();
  if (now - speedWindow.time >= 1000) {
    const achieved = (snapshot.transitions - speedWindow.transitions) * 20 / (now - speedWindow.time);
    $("achieved").textContent = `${achieved.toFixed(1)}× actual`;
    speedWindow = { time: now, transitions: snapshot.transitions };
  }
  drawScene(snapshot.state); drawCharts();
}
function frame(now) {
  const elapsed = Math.min((now - lastTime) / 1000, .1); lastTime = now;
  if (ready && running) {
    // One in-flight batch bounds control latency. Speed never changes physics or updates per transition.
    debt = Math.min(debt + elapsed * 50 * Number($("speed").value), 64);
    if (!busy && debt >= 1) { const steps = Math.floor(debt); debt -= steps; advance(steps); }
  } else { debt = 0; }
  requestAnimationFrame(frame);
}
$("pause").onclick = () => {
  running = !running; debt = 0;
  $("pause").textContent = running ? "Pause" : "Resume";
  $("status").textContent = running ? "Running" : busy ? "Pausing after the current batch" : "Paused";
  speedWindow = { time: performance.now(), transitions: latest?.transitions ?? 0 };
  if (!running) $("achieved").textContent = "0.0× actual";
};
$("step").onclick = () => { if (ready && !busy) { running = false; $("pause").textContent = "Resume"; advance(1); } };
$("restart").onclick = () => start(currentMode, initialBytes);
$("train").onclick = () => start("training");
$("bundled").onclick = () => { if (bundledBytes) start("inference", bundledBytes); };
$("export").onclick = () => post({ command: "export" });
$("import").onchange = async (event) => {
  const file = event.target.files[0]; if (!file) return;
  const run = generation;
  try {
    if (file.size > 131072) throw new Error("Policy exceeds the 128 KiB limit.");
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (run === generation) start("inference", bytes);
  } catch (error) { if (run === generation) fail(error.message); }
  event.target.value = "";
};
function context(id) {
  const canvas = $(id), bounds = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(bounds.width * dpr); canvas.height = Math.round(bounds.height * dpr);
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); return [ctx, bounds.width, bounds.height];
}
function drawScene(state) {
  const [ctx, width, height] = context("scene"), scale = width / 6, floor = height * .74;
  const cart = width / 2 + state[0] * scale, poleLength = Math.min(scale, height * .5);
  ctx.strokeStyle = "#34453a"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, floor + 22); ctx.lineTo(width, floor + 22); ctx.stroke();
  for (const limit of [-2.4, 2.4]) { const x = width / 2 + limit * scale; ctx.setLineDash([4, 6]); ctx.beginPath(); ctx.moveTo(x, 24); ctx.lineTo(x, floor + 22); ctx.stroke(); }
  ctx.setLineDash([]); ctx.fillStyle = "#76937d"; ctx.fillRect(cart - 30, floor - 12, 60, 24);
  ctx.fillStyle = "#a5b2a8";
  for (const offset of [-20, 20]) { ctx.beginPath(); ctx.arc(cart + offset, floor + 16, 6, 0, Math.PI * 2); ctx.fill(); }
  ctx.strokeStyle = "#b6ee63"; ctx.lineWidth = 9; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(cart, floor - 6); ctx.lineTo(cart + Math.sin(state[2]) * poleLength, floor - 6 - Math.cos(state[2]) * poleLength); ctx.stroke();
  ctx.fillStyle = "#ecf1e8"; ctx.beginPath(); ctx.arc(cart, floor - 6, 5, 0, Math.PI * 2); ctx.fill();
}
function chart(id, points, maximum, target, moving) {
  const [ctx, width, height] = context(id), left = 48, right = width - 12, top = 22, bottom = height - 34;
  const first = points[0]?.[0] ?? 0, last = Math.max(first + 1, points.at(-1)?.[0] ?? 1);
  const x = (value) => left + (value - first) / (last - first) * (right - left);
  const y = (value) => bottom - value / maximum * (bottom - top);
  ctx.font = "11px system-ui"; ctx.fillStyle = "#a5b2a8"; ctx.textAlign = "right";
  for (let i = 0; i <= 2; i++) {
    const value = maximum * i / 2; ctx.fillText(maximum < 1 ? value.toExponential(1) : value.toFixed(0), left - 8, y(value) + 4);
    ctx.strokeStyle = "#34453a"; ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(right, y(value)); ctx.stroke();
  }
  if (target != null) { ctx.strokeStyle = "#a5b2a8"; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(left,y(target));ctx.lineTo(right,y(target));ctx.stroke();ctx.setLineDash([]); }
  const line = (series, color) => { ctx.strokeStyle = color;ctx.lineWidth = 1.5;ctx.beginPath();series.forEach((point,index)=>{if(index)ctx.lineTo(x(point[0]),y(point[1]));else ctx.moveTo(x(point[0]),y(point[1]));});ctx.stroke(); };
  line(points, moving ? "#76937d" : "#b6ee63");
  if (moving) line(points.map((point,index)=>{const batch=points.slice(Math.max(0,index-19),index+1);return [point[0],batch.reduce((sum,p)=>sum+p[1],0)/batch.length];}),"#b6ee63");
  ctx.fillStyle="#a5b2a8";ctx.textAlign="left";ctx.fillText(first.toLocaleString(),left,bottom+18);ctx.textAlign="right";ctx.fillText(last.toLocaleString(),right,bottom+18);ctx.textAlign="center";ctx.fillText("Transitions",(left+right)/2,height-3);
}
function drawCharts() { chart("returns-chart", returns, 500, 475, true); chart("rate-chart", rates, .0004, null, false); }
window.addEventListener("resize",()=>{drawScene(latest?.state ?? [0,0,0,0]);drawCharts();});
drawScene([0,0,0,0]);drawCharts();requestAnimationFrame(frame);
const initialGeneration = generation;
$("bundled").disabled = true;
try {
  const [response, metadataResponse] = await Promise.all([
    fetch(new URL("models/cartpole.mpk", document.baseURI)),
    fetch(new URL("models/cartpole.json", document.baseURI)),
  ]);
  if (!response.ok || !metadataResponse.ok) throw new Error("Bundled policy or qualification report failed to load.");
  const metadata = await metadataResponse.json();
  const bytes = new Uint8Array(await response.arrayBuffer());
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  const hash = Array.from(digest, value => value.toString(16).padStart(2, "0")).join("");
  if (hash !== metadata.sha256) throw new Error("Bundled policy does not match its qualification report.");
  const report = metadata.qualification.find(run => run.seed === metadata.selected_training_seed);
  if (!Number.isFinite(report?.test_mean)) throw new Error("Bundled policy has no qualification score.");
  bundledBytes = bytes;
  $("model-evidence").textContent = `Bundled model · mean ${report.test_mean} over ${report.test_episodes} held-out episodes.`;
  $("bundled").disabled = false;
  if (generation === initialGeneration) start("inference", bundledBytes);
} catch (error) {
  $("model-evidence").textContent = "Bundled model unavailable.";
  if (generation === initialGeneration) fail(error.message);
}
