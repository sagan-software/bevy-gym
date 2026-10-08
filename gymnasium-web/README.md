# Browser Gymnasium

Run CartPole, MountainCar, and continuous MountainCar at <https://sagan-software.github.io/bevy-gym/>.
The browser worker runs Bevy Gym's Rust environments and Burn learners locally.
Training starts from a fresh model. Inference loads a frozen policy.

```sh
NO_COLOR=true nix develop --command trunk serve --config gymnasium-web/Trunk.toml
```

Open <http://127.0.0.1:8081>. The page shows episode returns, their rolling mean,
optimizer learning rate, TD loss, and the actual achieved simulation speed.
At 1×, CartPole advances 50 transitions per second with its original 20 ms
physical timestep. MountainCar advances 30 transitions per second, matching
Gymnasium playback. Speed changes pacing, with one optimizer opportunity per
transition. Hardware limits can reduce achieved speed.

Pause takes effect after the current worker batch, at most 64 transitions.
Downloaded policies contain inference parameters, not resumable training state.
Uploads are validated in a temporary worker before replacing the active session.
Invalid, oversized, or timed-out uploads leave the current session intact.
Starting another session cancels pending validation; late results are ignored.

A hidden page pauses after its current worker batch. Returning to the page
keeps it paused until Resume is selected. A session started while hidden
waits without advancing. Visibility-event tests hold a worker response to
verify that simulation and optimizer progress stop together.

## Verified CartPole model

The bundled model was trained through the browser session API with seed 42.
It reached a validation mean of 500 after 60,000 transitions and scored 500 in
all 200 held-out episodes. Training seeds 42, 43, and 44 passed the release gates
within 60,000, 90,000, and 60,000 transitions respectively. Seed 43's held-out
mean was 497.59, with 199 of 200 episodes reaching the time limit.

[Model provenance and qualification results](models/cartpole.json) record the
architecture, model hash, validation seeds, held-out seeds, and each run's curve.
The model uses the default `DqnConfig`, including Adam learning rate 0.0003.
Learning rate uses optimizer updates on its horizontal axis. Episode return
uses environment transitions.

## Verified MountainCar model

The default MountainCar-v0 port uses float64 internal state, the original
clipping rules and reward, and an external 200-transition cap. It requires
both position at least 0.5 and velocity at least zero to terminate.
Custom reset bounds and nonzero goal-velocity options are not exposed yet.
Diagnostic state injection rejects nonfinite coordinates and positions whose
terrain angle `3 * position` would overflow.

Training uses DQN with learning rate 0.001, normalized observations, and terrain
potential scale 25. Potential shaping changes only the optimizer reward;
charts and score gates use the original environment reward. The default native
example uses the same scale. The three qualification runs reached their selected
checkpoints after 310,000, 290,000, and 180,000 transitions for seeds 42, 43, and 44.

Their held-out means were -104.015, -104.08, and -104.005, with 200/200 goals each.
The bundled model comes from offline Chromium training with seed 42. Its
validation mean was -100.83 after 390,000 transitions, exceeding the native
models. It scored -102.41 with 200/200 goals on separate held-out seeds.
[Model provenance](models/mountain-car.json) records all four runs and the hash.

The gate requires mean at least -110, at least 190/200 goals, and improvement
of at least 80 from initialization, within one million transitions per seed.
Validation uses 100 seeds; the 200 held-out seeds are disjoint.

## Verified continuous MountainCar model

MountainCarContinuous-v0 uses the original force clipping, raw-action reward
penalty, goal at position 0.45, and external 999-transition limit. PPO collects
64 transitions from each of eight environments before updating its actor and
critic. The view shows the first contributing environment.

Actor and critic learning rates are 0.003 and 0.001. Both appear on the
learning-rate chart. Training uses terrain potential scale 25; episode returns
and qualification scores retain the original reward. At 1×, each environment
advances 30 transitions per second. Total transitions include all eight lanes;
the displayed actual speed accounts for that lane count.

Fresh training seeds 42, 43, and 44 passed after 10,240 transitions each.
Their 200-episode held-out means were 90.82, 96.06, and 98.44, with 200 goals
each. The bundle uses seed 44, selected by validation mean before test scoring.
The gate requires mean at least 90 and at least 190 goals in 200 episodes.
[Model provenance](models/mountain-car-continuous.json) retains the recipe,
selection evidence, source hashes, model hash, and disjoint seed ranges.

Chromium also trained a fresh policy with network access disabled after worker
initialization. It scored 95.43 with 200/200 goals. Loading that browser-trained
policy natively reproduces its mean within 0.00001 on the same held-out seeds.
The bundled native-trained policy is evaluated in all three browser engines.
Desktop checks also require the return chart to fit a 1280×800 viewport.

## Continuous-action model records

The shared recurrent PPO learner compiles with `browser-training` on WebAssembly.
`RecurrentPpoPolicy::to_bytes` and `load_bytes` use the locked Burn 0.21
named MessagePack record. They preserve actor, critic, and recurrent inference
parameters; optimizer and rollout state are excluded.

Byte loading rejects missing biases, incompatible actor and critic dimensions,
and nonfinite parameters before activating the model. The focused native test
proves that both actor and critic change after an update and survive a byte
round trip. The checkpoint validator has 100% line coverage. Its tensor-to-float
conversion error cannot be constructed with the fixed float32 backend and is
not exercised.

## Visual fidelity

CartPole and both MountainCar tasks preserve Gymnasium's 600 by 400 scene proportions,
procedural shapes, and colors. The dashboard scales each scene without stretching.
[Visual comparisons](../docs/visual-comparisons/README.md) include official GIF
contact sheets, browser recordings, fixed-state fixtures, and comparison gates.

## Checks

`nix run .#browser-runtime-check` verifies that all three browser engines can
create a page before CI compiles the site. It records engine versions and
sets a 20-second test timeout per engine.

```sh
nix develop --command cargo test -p bevy-gym-browser
nix develop --command cargo test -p bevy-gym-browser --test learning -- --ignored --nocapture
nix develop --command cargo test -p bevy-gym-browser --test mountain_car_learning -- --ignored --nocapture
nix develop --command cargo test -p bevy-gym-browser --test continuous_learning continuous_seed_ -- --ignored --nocapture
nix develop --command cargo clippy -p bevy-gym-browser --all-targets -- -D warnings
nix develop --command cargo clippy -p bevy-gym-browser --target wasm32-unknown-unknown -- -D warnings
```

The CartPole learning gate starts three independent models, selects each checkpoint on
20 validation seeds, and evaluates it on 200 different seeds. Each seed must
reach mean return 475, at least 90% full-length episodes, and improvement of at
least 400 over its initial validation mean within 200,000 transitions.
The command saves reports and policies under `runs/browser-cartpole/`.

Continuous MountainCar qualification saves reports and selected policies under
`runs/browser-mountain-car-continuous/`. Retain Chromium's qualification JSON and
MPK under its `browser-seed-42/` subdirectory before checking browser-to-native
compatibility:

```sh
nix develop --command cargo test -p bevy-gym-browser --test continuous_learning browser_trained_continuous_policy_loads_natively -- --ignored --nocapture
```

CartPole uses Gymnasium's default Euler equations and float64 state.
Reset values have Gymnasium's uniform distribution, but use SplitMix64 rather
than NumPy PCG64. Equal numeric seeds do not produce the same Python sequence.
The explicit state-injection tests compare against the pinned Python oracle.

Other Gymnasium environments remain in the
[browser implementation plan](../GYMNASIUM_BROWSER_PLAN.md).
The [Pendulum port](PENDULUM.md) records its implemented session and renderer
contracts and pending qualification gates.

## Deployment assets

Each build places the application, worker, WebAssembly, models, and images in
one SHA-256-addressed directory. The application resolves asset URLs relative
to its own module. This prevents a cached worker from loading with a newer page
that supports additional environments. Trunk's hashed CSS remains at the root.

The post-build hook versions unpublished staging output before Trunk publishes
it. Unit tests check deterministic naming and invalidation for each asset type.
A browser regression blocks every old unversioned application, worker, and
model URL and requires continuous MountainCar inference to start successfully.

The deployment also runs `nix run .#gymnasium-check`. This launches the built
site under `/bevy-gym/`, exercises its controls and policy round trip, and trains
a fresh model in a real browser worker before checking held-out scores.
Chromium performs full qualification with the network disconnected after worker
initialization. Firefox and WebKit check training updates, inference, controls,
and rejection of stale loading results. Full qualification cases for Firefox
and WebKit are explicitly skipped.

The UI suite checks all three engines, including all speed settings for each
environment, single-step inference, the WebKit pause regression, and upload failures.
MountainCar passed offline training, scoring -102.41 across
200 held-out episodes after 390,000 transitions. Qualification jobs retain
both metric reports and the selected policy record as downloadable artifacts.

Native session coverage records 48 of 49 facade lines and 96 of 97 discrete-session
lines. The uncovered facade span propagates a Pendulum initialization error
that the fixed valid recipe cannot produce. Unhit regions propagate errors from
configuration creation, policy calls,
environment stepping, and optimizer updates. Tests do not inject failures into
an active optimizer. WebAssembly worker behavior is checked
by browser tests; native LLVM cannot execute its browser entry point.

The continuous-session facade, rollout, training, task profile, and observation
modules have 100% line coverage. Unhit regions propagate configuration,
inference, and optimizer errors that tests do not inject into an active learner.
Rollout advantage calculation has 100% region coverage.
The episode summary reports 139 of 145 lines, while its annotated production
lines all have hits. LLVM also reports one function with mismatched counters
after a clean coverage run. That reporting discrepancy remains unresolved;
these measurements do not establish complete branch coverage.

## Worker contract

The worker name selects `cartpole`, `mountain-car`, or `mountain-car-continuous`
for its lifetime.
An empty name retains the original CartPole default; other names fail before
initialization. The existing command shapes remain unchanged.
MountainCar snapshots place position and per-step velocity in the first two
state coordinates and zero the remaining two.
Continuous snapshots also report the critic rate and loss, plus the number of
contributing environments. Inference has one environment and no optimizer state.

The worker announces protocol `1` before accepting commands. The page checks
that version before sending work. A worker instance identifies its run;
restarting terminates that instance and rejects its queued responses. FIFO
message delivery orders commands. Only one advance request is outstanding;
policy export may queue behind it.

The page has six lifecycle states: loading, running, advancing, pausing, paused,
and failed. An advance moves running to advancing. Its response returns to
running unless a pause moved it to pausing, in which case it becomes paused.
A worker error terminates the worker and enters failed. A new session returns
to loading. Invalid seed input leaves the previous session intact.

Commands are JSON objects tagged by `command`: `start_training` takes a `u32`
seed, `start_inference` takes a seed and policy byte array, `advance` takes an
integer step count from 1 through 256, and `export` takes no extra fields.
Unknown fields and commands are rejected. The worker limits JSON messages to
1 MiB; the page limits policy uploads to 128 KiB. Responses are tagged by
`event`: `ready`, `started`, `snapshot`, `policy`, or `error`. An error includes
a diagnostic message. A new worker is required after a page-handled error.
