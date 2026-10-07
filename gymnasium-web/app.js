const $ = (id) => document.getElementById(id);
const taskName = new URL(location.href).searchParams.get("env") === "mountain-car" ? "mountain-car" : "cartpole";
const task = taskName === "mountain-car"
  ? { title: "MountainCar-v0", hz: 30, minimum: -200, maximum: 0, target: -110, rateMaximum: .0014, replay: 2000 }
  : { title: "CartPole-v1", hz: 50, minimum: 0, maximum: 500, target: 475, rateMaximum: .0004, replay: 1000 };
$("environment").value = taskName;
$("environment").onchange = () => { location.search = new URLSearchParams({ env: $("environment").value }).toString(); };
document.querySelector("h1").textContent = task.title;
$("score-target").textContent = `Target mean ≥${task.target}`;
$("scene").setAttribute("aria-label", taskName === "mountain-car" ? "MountainCar simulation: a car climbs the right hill" : "CartPole simulation: a cart balances an upright pole");
let worker, generation = 0;
/** @type {"loading" | "running" | "advancing" | "pausing" | "paused" | "failed"} */
let phase = "loading";
let currentMode = "inference", initialBytes, initialSource, bundledBytes, latest, lastTime = 0, debt = 0;
let returns = [], rates = [], speedWindow = { time: performance.now(), transitions: 0 };

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
  worker?.terminate();
  currentMode = mode; initialBytes = bytes; initialSource = source; latest = undefined;
  setPhase("loading"); debt = 0; returns = []; rates = [];
  $("error").hidden = true;
  $("mode").textContent = mode === "training" ? "Training · DQN" : "Inference";
  $("policy-source").textContent = mode === "training" ? `Fresh model · seed ${runSeed}` : source;
  $("status").textContent = "Starting Rust worker";
  $("pause").textContent = "Pause";
  ["transitions", "episodes", "return", "updates"].forEach((id) => $(id).textContent = "0");
  ["rate", "epsilon", "loss"].forEach((id) => $(id).textContent = "None");
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
        $("status").textContent = mode === "training" ? "Collecting replay transitions" : "Running frozen policy";
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
  $("status").textContent = phase === "paused" ? "Paused" : currentMode === "inference" ? "Running frozen policy"
    : snapshot.optimizer_steps ? "Training in this browser" : `Collecting replay transitions · ${snapshot.transitions}/${task.replay.toLocaleString()}`;
  const now = performance.now();
  if (phase === "running" && now - speedWindow.time >= 1000) {
    const achieved = (snapshot.transitions - speedWindow.transitions) * (1000 / task.hz) / (now - speedWindow.time);
    $("achieved").textContent = `${achieved.toFixed(1)}× actual`;
    speedWindow = { time: now, transitions: snapshot.transitions };
  }
  drawScene(snapshot.state); drawCharts();
}
function frame(now) {
  const elapsed = Math.min((now - lastTime) / 1000, .1); lastTime = now;
  if (phase === "running" || phase === "advancing") {
    // One in-flight batch bounds control latency. Speed never changes physics or updates per transition.
    debt = Math.min(debt + elapsed * task.hz * Number($("speed").value), 64);
    if (phase === "running" && debt >= 1) { const steps = Math.floor(debt); debt -= steps; advance(steps); }
  } else { debt = 0; }
  requestAnimationFrame(frame);
}
$("pause").onclick = () => {
  const transitions = { running: "paused", advancing: "pausing", paused: "running", pausing: "advancing" };
  const next = transitions[phase];
  if (!next) return;
  setPhase(next); debt = 0;
  $("status").textContent = phase === "pausing" ? "Pausing after the current batch" : phase === "paused" ? "Paused" : "Running";
  speedWindow = { time: performance.now(), transitions: latest?.transitions ?? 0 };
  if (phase === "paused" || phase === "pausing") $("achieved").textContent = "0.0× actual";
};
$("step").onclick = () => { if (phase === "running" || phase === "paused") { setPhase("paused"); advance(1); } };
$("restart").onclick = () => start(currentMode, initialBytes, initialSource);
$("train").onclick = () => start("training");
$("bundled").onclick = () => { if (bundledBytes) start("inference", bundledBytes); };
$("export").onclick = () => post({ command: "export" });
$("import").onchange = async (event) => {
  const file = event.target.files[0]; if (!file) return;
  const run = generation;
  try {
    if (file.size > 131072) throw new Error("Policy exceeds the 128 KiB limit.");
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (run === generation) start("inference", bytes, `Uploaded policy: ${file.name}`);
  } catch (error) { if (run === generation) fail(error.message); }
  event.target.value = "";
};
function context(id) {
  const canvas = $(id), bounds = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(bounds.width * dpr); canvas.height = Math.round(bounds.height * dpr);
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); return [ctx, bounds.width, bounds.height];
}
function drawScene(state) {
  if (taskName === "mountain-car") { drawMountainCar(state); return; }
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
function drawMountainCar(state) {
  const [ctx, width, height] = context("scene");
  const x = position => 24 + (position + 1.2) / 1.8 * (width - 48);
  const y = position => height - 24 - (Math.sin(3 * position) * .45 + .55) * (height - 60);
  ctx.strokeStyle = "#76937d"; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i <= 120; i++) { const position = -1.2 + 1.8 * i / 120; if (i) ctx.lineTo(x(position), y(position)); else ctx.moveTo(x(position), y(position)); }
  ctx.stroke();
  ctx.strokeStyle = "#b6ee63"; ctx.beginPath(); ctx.moveTo(x(.5), y(.5)); ctx.lineTo(x(.5), y(.5)-30); ctx.stroke();
  ctx.fillStyle = "#b6ee63"; ctx.beginPath(); ctx.moveTo(x(.5), y(.5)-30); ctx.lineTo(x(.5)+18,y(.5)-23); ctx.lineTo(x(.5),y(.5)-16); ctx.fill();
  ctx.save(); ctx.translate(x(state[0]), y(state[0])-9);
  ctx.rotate(Math.atan2(-1.35 * Math.cos(3 * state[0]) * (height-60), (width-48)/1.8));
  ctx.fillStyle = "#b6ee63"; ctx.fillRect(-15,-10,30,14); ctx.fillStyle = "#a5b2a8";
  for (const offset of [-10,10]) { ctx.beginPath(); ctx.arc(offset,5,5,0,Math.PI*2); ctx.fill(); }
  ctx.restore();
}
function chart(id, points, maximum, target, moving, minimum = 0) {
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
  if (moving) line(points.map((point,index)=>{const batch=points.slice(Math.max(0,index-19),index+1);return [point[0],batch.reduce((sum,p)=>sum+p[1],0)/batch.length];}),"#b6ee63");
  ctx.fillStyle="#a5b2a8";ctx.textAlign="left";ctx.fillText(first.toLocaleString(),left,bottom+18);ctx.textAlign="right";ctx.fillText(last.toLocaleString(),right,bottom+18);ctx.textAlign="center";ctx.fillText("Transitions",(left+right)/2,height-3);
}
function drawCharts() { chart("returns-chart", returns, task.maximum, task.target, true, task.minimum); chart("rate-chart", rates, task.rateMaximum, null, false); }
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
  $("model-evidence").textContent = `Bundled model · mean ${report.test_mean} over ${report.test_episodes} held-out episodes.`;
  $("bundled").disabled = false;
  if (generation === initialGeneration) start("inference", bundledBytes);
} catch (error) {
  $("model-evidence").textContent = "Bundled model unavailable.";
  if (generation === initialGeneration) fail(error.message);
}
