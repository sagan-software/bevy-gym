# Browser Gymnasium

Run CartPole training and inference at <https://sagan-software.github.io/bevy-gym/>.
The browser worker runs Bevy Gym's Rust environment and Burn DQN learner locally.
Training starts from a fresh model. Inference loads a frozen policy.

```sh
nix develop --command trunk serve --config gymnasium-web/Trunk.toml
```

Open <http://127.0.0.1:8081>. The page shows episode returns, their rolling mean,
optimizer learning rate, TD loss, and the actual achieved simulation speed.
The 1× through 16× controls keep the original 20 ms physical timestep and one
optimizer opportunity per transition. Hardware limits can reduce achieved speed.
Pause takes effect after the current worker batch, at most 64 transitions.
Downloaded policies contain inference parameters, not resumable training state.

## Verified CartPole model

The bundled model was trained through the browser session API with seed 42.
It reached a validation mean of 500 after 60,000 transitions and scored 500 in
all 200 held-out episodes. Training seeds 42, 43, and 44 passed the release gates
within 60,000, 90,000, and 60,000 transitions respectively. Seed 43's held-out
mean was 497.59, with 199 of 200 episodes reaching the time limit.

[Model provenance and qualification results](models/cartpole.json) record the
architecture, model hash, validation seeds, held-out seeds, and each run's curve.
The model uses the default `DqnConfig`, including Adam learning rate 0.0003.
Learning rate and episode return are separate charts.

## Checks

```sh
nix develop --command cargo test -p bevy-gym-browser
nix develop --command cargo test -p bevy-gym-browser --test learning -- --ignored --nocapture
nix develop --command cargo clippy -p bevy-gym-browser --all-targets -- -D warnings
nix develop --command cargo clippy -p bevy-gym-browser --target wasm32-unknown-unknown -- -D warnings
```

The learning gate starts three independent models, selects each checkpoint on
20 validation seeds, and evaluates it on 200 different seeds. Each seed must
reach mean return 475, at least 90% full-length episodes, and improvement of at
least 400 over its initial validation mean within 200,000 transitions.
The command saves reports and policies under `runs/browser-cartpole/`.

The shared dynamics use Gymnasium's default Euler equations and float64 state.
Reset values have Gymnasium's uniform distribution, but use SplitMix64 rather
than NumPy PCG64. Equal numeric seeds do not produce the same Python sequence.
The explicit state-injection tests compare against the pinned Python oracle.

Other Gymnasium environments remain in the
[browser implementation plan](../GYMNASIUM_BROWSER_PLAN.md).

The deployment also runs `nix run .#gymnasium-check`. This launches the built
site under `/bevy-gym/`, exercises its controls and policy round trip, and trains
a fresh model in a real browser worker before checking held-out scores.
Chromium performs full qualification with the network disconnected after worker
initialization. Firefox and WebKit check training updates, inference, controls,
and rejection of stale loading results. All seven checks passed on this host;
the two full qualification cases for Firefox and WebKit are explicitly skipped.

Native coverage records 100% of the CartPole dynamics, action validation, state
validation, batch-budget, and error-display lines. Session coverage is 81 of 88
lines. Its unhit paths propagate configuration, observation, optimizer, and
out-of-range action errors that the fixed valid session configuration does not
produce. Native worker command handling covers 76 of 80 lines; the no-op native entry
point and defensive error propagation remain unhit. WebAssembly worker
behavior is checked by browser tests rather than native LLVM instrumentation.

## Worker contract

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
