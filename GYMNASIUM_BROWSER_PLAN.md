# Gymnasium browser training and inference

Date: 2026-10-07. Status: implementation plan, not a completed browser feature.

Every delivered environment must train locally in the browser, show agents
learning, and separately run inference with a bundled qualified model.
Complete one environment through all gates before adding the next.

## Scope and current changes

The first catalog contains the five Classic Control, four Toy Text, three
Box2D, and eleven MuJoCo base tasks, plus the variants listed below. Atari,
PettingZoo, and the custom ecosystem remain later work.

Avian is removed from default Gymnasium builds, headless example builds, and
MuJoCo example builds. Avian3D and the shared Avian development dependencies
are removed. Avian2D remains an optional ecosystem dependency. Ecosystem commands
now require `--features ecosystem-inference`; the existing web ecosystem package
already selects that feature. Native ecosystem physics retains parallel support.

The former Avian example directories are now `examples/box2d-approximation` and
`examples/mujoco-approximation`. Their handwritten dynamics, CLI target names,
and existing work are preserved. Their policies and historical scores do not
qualify the replacement ports. Conditional dependency acknowledgements in example
targets apply only when a user explicitly enables the ecosystem feature too.

## Original environment contracts

Keep the existing Gymnasium oracle commit
[`7a1191388aa4aa973d3a5e4b039899cd99cc991f`](https://github.com/Farama-Foundation/Gymnasium/tree/7a1191388aa4aa973d3a5e4b039899cd99cc991f).
The local catalog identifies its version as 1.3.0. Upgrade only with a reviewed
semantic diff and regenerated fixtures. Pin the physics engine separately.

Classic Control and Toy Text translate upstream task logic directly into Rust.
[Box2D tasks](https://gymnasium.farama.org/environments/box2d/) retain Box2D;
[MuJoCo tasks](https://gymnasium.farama.org/environments/mujoco/) retain MuJoCo
and the original MJCF models. Rust may call the original engines through
bindings. Bevy owns presentation and scheduling, not replacement physics.

For each task, inventory constructor options, action and observation types,
precision, reset distributions, dynamics, rewards, info fields, termination,
truncation, and rendering geometry. Record each unsupported option explicitly.
Test injected state/action trajectories against a dependency-pinned Python oracle.
Use exhaustive transitions where feasible and statistical tests for randomness.
Do not claim NumPy seed-stream identity from matching integer seeds.

## Browser experience

Each environment page exposes distinct `Train` and `Inference` modes.
The canvas, mode control, primary action, speed, and relevant metrics are visible
without opening settings. Preserve the existing site's visual conventions.
Use a stacked canvas and charts on narrow screens and an adjacent arrangement
on desktop. Labels and focus states must remain usable at 200% zoom.

Training starts from a fresh initialization when the user selects `Start training`.
Loading a checkpoint for continued training is an explicit alternative.
The live view shows a contributing rollout lane and its policy update number.
A separate evaluation preview, if present, is labelled as evaluation.
Never replay a prerecorded success or bundled policy as evidence of live training.

The training view provides start, pause, resume, stop, reset, seed, configuration,
checkpoint download, and resumable-run export/import. Reset names the training
state it discards. It does not overwrite the bundled model.
Changing the environment stops the prior run and releases its resources.

Show raw episode return and a labelled rolling mean against environment
transitions. Plot fixed-seed validation return separately, with the score gate
drawn on that chart. Allow wall-clock time as an alternative horizontal axis.
Show environment transitions, optimizer updates, episodes, elapsed active time,
success rate, and measured transitions per second. Preserve full exported metrics
while bounding the points rendered by the chart.

Show optimizer learning rate as a separate chart against optimizer updates.
PPO exposes actor and critic rates separately; tabular methods expose their
step size. A fixed learning rate appears as a flat line. Score improvement is
learning progress, not optimizer learning rate. Do not invent a universal
minimum optimizer learning rate: a larger value need not learn faster or better.
Configuration and schedule changes are visible events and start a distinct
qualification recipe.

Inference automatically loads the environment's bundled best qualified model.
It provides pause, reset, single-step, seed, speed, model upload/download,
and restore-bundled controls. It performs no optimizer updates. Show the selected
model, its qualification score, evaluation sample size, and provenance.
An incompatible upload leaves the last working model active.

## Speed and time semantics

Offer 1x, 2x, 4x, 8x, and 16x in both modes. Keep each environment's original
fixed timestep. Increasing speed requests more complete simulation steps per
wall-clock second; it never multiplies gravity, timestep, reward, discount,
learning rate, or the ratio of optimizer updates to collected transitions.

Define 1x using the environment's physics timestep, or its pinned render period
for discrete tasks without physical time. Display the chosen time basis.
For parallel training, count simulated seconds per lane separately from total
transitions across all lanes. Do not report additional lanes as faster playback.

Measured speed equals simulated seconds per lane divided by active wall-clock
seconds. The ratio is dimensionless. Compute simulated seconds as completed
steps times seconds per step. Exclude paused wall time and display requested
speed separately from achieved speed. Hardware may not sustain 16x; show the
measured shortfall rather than dropping training transitions or changing physics.

Render the latest completed state at display cadence. Bound queued snapshots
and metric batches. Pause freezes simulation and optimizer progress after the
current atomic update completes. Single-step in inference advances exactly one
environment step. Backgrounding the page pauses by default and exposes that
state. Resume does not accumulate missed wall-clock work.

## Runtime boundaries and feasibility gates

The current training modules and autodiff backend are mostly native-gated.
Browser inference support does not establish browser training support.
[Burn's WASM guide](https://burn.dev/books/burn/advanced/web-assembly.html)
documents Flex and WebGPU execution. It does not qualify this project's DQN,
PPO, optimizer, or checkpoint-resume paths on the browser.

First prove one forward/backward/update/save/reload cycle with the locked Burn
0.21 dependencies in a browser worker. Compare outputs and an optimizer update
against a native fixture with measured tolerances. Start with a portable CPU
backend; evaluate WebGPU acceleration separately. No server-side training is
an acceptable substitute for browser training.

Run simulation, rollout collection, and optimization in a dedicated worker.
Keep DOM controls and presentation responsive on the main thread. The
[worker API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers)
provides message passing and isolates worker execution from DOM access.
Transfer owned snapshots at bounded intervals rather than sharing mutable model
state. Batch inference across available lanes; avoid copying an entire replay
buffer or model to render each frame.

Before implementation, define a versioned worker message schema with validated
run identity, sequence, command acknowledgement, payload limits, and errors.
Use closed modes `Train` and `Inference`. Use closed lifecycle states `Loading`,
`Ready`, `Running`, `Pausing`, `Paused`, `Completed`, `Stopped`, and `Failed`.
Document permitted commands and resource ownership for every state. Derive button
availability from state.

Reject stale messages from stopped or replaced runs.
Serialize checkpoint activation at a safe boundary and reset recurrent memory.

Split portable environment, agent, optimizer, and checkpoint bytes from native
filesystem, process, clock, and thread adapters. Preserve existing public APIs.
An inference export contains policy parameters and observation normalization;
a resumable training export additionally needs optimizer, RNG, replay/rollout,
scheduler, counters, and recurrent state as applicable. Specify the format and
compatibility rules before implementing import. Test continuation equivalence.

Prove a Box2D browser build against the pinned Python engine before replacing
LunarLander. Match engine version, solver settings, joints, collisions, and sleep
semantics. A newer Box2D major version is not an interchangeable replacement.

Prove a MuJoCo browser build with the same model and engine version used natively
before expanding its ports. The current native binding pins MuJoCo 3.9.0;
the [upstream programming guide](https://mujoco.readthedocs.io/en/stable/programming/index.html)
describes the C API and native builds. A supported browser binding and build
strategy remain unverified. Resolve that gap in a prototype, including model
loading, numeric agreement, memory growth, licensing, and deployment headers.
Do not label these engine integrations implementation-ready yet.

## Score and learning gates

The first nine defaults inherit their existing
[catalog gates](tests/fixtures/gymnasium/catalog.json). The registry thresholds
for physics tasks come from the
[pinned registry](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/__init__.py).
Values explicitly marked proposed below are local project requirements.

Run three independently trained seeds. Each must pass its task gate on an
untouched test suite after checkpoint selection on separate validation seeds.
Use 100 test episodes per seed unless a task specifies a different count.
Report each seed and aggregate distributions; a single successful seed is
insufficient. Freeze evaluation policy, confidence method, and test seeds before
training. Report unshaped environment return for qualification.

- CartPole-v1: mean return at least 475 and at least 90% of episodes reach 500 steps.
- MountainCar-v0: mean at least -110 and at least 95% reach position 0.5.
- MountainCarContinuous-v0: mean at least 90 and at least 95% reach position 0.45.
- Pendulum-v1: mean at least -200 and upright dwell rate at least 0.70.
  Dwell uses steps 50 through 199 inclusive, absolute angle at most 15 degrees,
  and absolute angular velocity at most 1 radian per second.
- Acrobot-v1: mean at least -100 and target reach rate at least 95%.
- CliffWalking-v1: exact return -13, 100% goal completion, and no cliff entries;
  retain the 200-step project safety cap.
- FrozenLake-v1: success lower 95% Wilson bound at least 0.70 over 10,000 episodes.
- Taxi-v4: 100% deliveries and no illegal pickup/dropoff over all 300 starts,
  with at most 200 steps each. Its current mean-return gate needs the correction
  described below before qualification can pass.
- Blackjack-v1: mean-return lower 95% confidence bound at least -0.08 and
  improvement over random at least 0.25 over 100,000 games. Define the confidence
  estimator and paired baseline sampling in the executable gate.
- LunarLander-v3: mean at least 200; proposed additional safe-landing rate at least 95%.
- BipedalWalker-v3: mean at least 300.
- CarRacing-v3: mean at least 900. Preserve the original pixel observation contract;
  a compact state-vector policy is a separate task.
- InvertedPendulum-v5: mean at least 950.
- InvertedDoublePendulum-v5: mean at least 9100.
- Reacher-v5: mean at least -3.75.
- Pusher-v5: mean at least 0, as registered. Audit attainability against the pinned
  reward formula before accepting this as an executable gate.
- Swimmer-v5: mean at least 360.
- Hopper-v5: mean at least 3800.
- HalfCheetah-v5: mean at least 4800.
- Walker2d-v5: proposed project mean at least 4000; upstream supplies no registry threshold.
- Ant-v5: mean at least 6000.
- Humanoid-v5: proposed project mean at least 6000; upstream supplies no registry threshold.
- HumanoidStandup-v5: proposed project mean at least 100,000; upstream supplies
  no registry threshold.

Registered variants receive separate models and gate results. FrozenLake8x8-v1
requires a lower 95% Wilson bound of 0.85 over 10,000 episodes.
CliffWalkingSlippery-v1 requires 0.90 by the same method and sample count.
LunarLanderContinuous-v3 retains the 200 return threshold; BipedalWalkerHardcore-v3
retains 300. Inventory constructor variants before each port, including
Blackjack reward modes and Taxi rain/fickleness. Never reuse a default's score
claim for a different configuration.

The Taxi fixture enumerates 300 equiprobable starts. Reverse shortest-path
evaluation of its deterministic dry transitions gives a maximum mean of 7.93:
terminal reward 20 minus one per preceding move. Therefore the literal
combination of that fixture, exhaustive evaluation, and mean at least 8 is
impossible. Regenerate the Python oracle first. If the oracle confirms these
transitions, replace the exhaustive mean gate with per-start optimal return and
retain 8 only as separately labelled registry metadata. This plan does not
silently change the existing fixture or claim it passes.

Treat optimizer learning rate as a recipe parameter. Freeze its initial value,
schedule, batch size, optimizer, exploration, network, and update cadence before
qualification. Reject nonfinite parameters or gradients and invalid schedules.
Use transition budgets to measure learning speed, independently of playback.

Proposed initial budgets per seed are 200,000 transitions for CartPole;
1 million each for MountainCar and Acrobot; 2 million each for continuous
MountainCar and Pendulum; 100,000 training episodes for Blackjack; 10,000 for
CliffWalking and Taxi; and 100,000 for FrozenLake and the slippery/8x8 variants.
Allow 5 million transitions for LunarLander and the first three MuJoCo tasks;
10 million for BipedalWalker, CarRacing, Pusher, Swimmer, Hopper, HalfCheetah,
Walker2d, and Ant; and 30 million for Humanoid, HumanoidStandup, and Hardcore.
These are proposed acceptance budgets, not measured convergence claims.

Evaluate at 0%, 10%, 25%, 50%, 75%, and 100% of budget. Require a positive
validation improvement over initialization by 25%, and the full score gate by
100%. Compare independent seed means and report uncertainty. Preserve failed
runs. A budget or recipe change requires a new version and a complete rerun;
do not extend only the failed seed until it passes. Record time-to-gate and
transitions-to-gate for both native and browser runs on named hardware.

## Bundled best models

Choose the best checkpoint by the predeclared validation metric among candidates
that meet the qualification protocol. Run the final test suite once after that
selection. Do not choose a checkpoint or seed using final test scores.
Describe it as the best qualified model from recorded runs, not a global optimum.

For every default and advertised variant, bundle the model, normalization,
environment and engine versions, observation/action schema, training recipe,
seed, source commit, artifact hash, and validation/test evidence. Fetch only the
selected environment's artifacts. Check hashes and compatibility before activation.
Do not ship a random or unqualified fallback under the best-model label.

Require a native-trained artifact to load in the browser and a browser-trained
artifact to load natively. Check inference numerically and behaviorally.
Keep model export distinct from full training-resume export. Downloaded exports
must remain useful without the user's browser storage or a private run directory.

## Implementation sequence and exit gates

1. Repair CI and establish a preserved source baseline. Run dependency exclusion,
   fixture, Rust, lint, and Nix checks without automatic crate publication.
2. Make CartPole the full vertical slice. Extract portable dynamics and DQN,
   prove worker training, add both modes and charts, qualify three seeds, bundle
   the selected model, and verify the deployed page. Do not proceed on inference alone.
3. Repeat all gates for MountainCar, continuous MountainCar, Pendulum, and Acrobot.
4. Repeat for CliffWalking, FrozenLake, Taxi, and Blackjack, including their variants.
5. Pass the Box2D browser feasibility gate. Replace and qualify LunarLander,
   BipedalWalker, and CarRacing individually, including registered variants.
6. Pass the MuJoCo browser feasibility gate. Qualify InvertedPendulum,
   InvertedDoublePendulum, Reacher, Pusher, Swimmer, Hopper, HalfCheetah, Walker2d,
   Ant, HumanoidStandup, and Humanoid individually.
7. Publish the complete tested catalog and its machine-readable qualification
   reports. Resume custom Avian or more complex examples afterward.

## Verification and release gates

Dependency checks must pass for default, headless, and MuJoCo configurations:

```sh
python3 tests/test_gymnasium_dependencies.py
python3 tests/fixtures/gymnasium/test_contracts.py
```

Run the exact repository Rust gates in the development environment:

```sh
nix develop --command cargo fmt --all -- --check
nix develop --command cargo test
nix develop --command cargo clippy --all-targets --all-features -- -D warnings
nix develop --command cargo test -p bevy-gym-web --locked
nix flake check
nix build .#web-dist
nix run .#web-check
```

Run the applicable personal-lint workflow from the installed skill without
altering project lint policy. Add branch coverage for changed production code
and close every reachable changed branch. CI must run tests, not just compilation.
Add a separate browser-training build and qualification job before claiming
that the existing ecosystem browser suite covers the new catalog.

Browser acceptance must prove actual parameter changes and improved held-out
performance from a fresh model, with training requests blocked from the network.
It must also prove inference performs zero updates. Test 1x/2x/4x/8x/16x,
pause/resume, stop/reset, route changes, hidden-tab behavior, export/import,
checkpoint corruption, unavailable GPU, worker crash, and recovery.

Stall an old run, start a new run, resolve the new run first, then resolve the
old run. Assert that the new run remains displayed. During a stalled training
update, prove that the UI still accepts controls. After pause acknowledgement,
assert that step and update counts remain unchanged. Fixed-seed runs at each
speed must agree after the same transition/update counts within declared backend
tolerances. Inference single-step must advance once with no optimizer update.

Inspect desktop and mobile at each environment's first delivery. Check keyboard
controls, chart labels and units, stable canvas layout, overflow, and 200% zoom.
Run Chromium, Firefox, and WebKit where supported. An unsupported device or
browser shows an explicit capability error; it cannot count as a passing target.

Set provisional interaction targets of 100 ms p95 for control acknowledgement
and 33.3 ms p95 main-thread frame time during training on named reference hardware.
Record cold startup, compressed transfer, peak memory, and 30-minute resource
stability. No unbounded replay, chart, snapshot, or worker queue is acceptable.
Display achieved speed when hardware cannot meet requested speed.

Full learning qualification is an artifact-producing job, separate from fast
PR smoke tests. A browser optimizer smoke test cannot substitute for the task's
score gate. Version and retain the browser, device, backend, recipe, curves,
metrics, model hashes, and failed runs with each qualification report.

Deploy the exact tested static artifact and verify its final URL in a browser.
Resolve private-source hosting, browser engine assets, cross-origin requirements,
and access policy before deployment. Training and inference remain client-side.

## Current verification boundary

CartPole is deployed with browser training, inference, controls, curves, and a
qualified bundle. Three native training seeds passed. Chromium trained offline
and scored 500 across 200 held-out episodes. Chromium, Firefox, and WebKit
passed the functional browser checks. Desktop and mobile views were inspected.

MountainCar has shared float64 dynamics, boundary tests, browser task selection,
and a bundled qualified model. Native training seeds 42, 43, and 44 all passed
within 310,000 transitions. Chromium trained offline for 390,000 transitions
and scored -102.41 with 200/200 goals. The deployed visual review remains
pending. Other environments are not yet browser-qualified.
Continuous MountainCar now has a shared core and native adapter. A pinned NumPy
2.4.4 oracle checks 201 transitions bit for bit, including clipped force,
raw-action penalties, reverse motion at the goal, and both precision states.
Its new production files have 100% line coverage. Browser training, model
qualification, and deployment for this task remain unfinished.
The Box2D/MuJoCo engine replacements remain planned.

The broader controls and performance checks above remain acceptance work,
including physical tab-switch verification, explicit worker crashes, speed-independent replay,
200% zoom, latency budgets, and 30-minute resource stability.
