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

Native coverage records 100% of the CartPole dynamics, action validation, state
validation, batch-budget, and error-display lines. Session coverage is 78 of 85
lines. Its unhit paths propagate configuration, observation, optimizer, and
out-of-range action errors that the fixed valid session configuration does not
produce. The native no-op worker entry point is unhit. WebAssembly worker
behavior is checked by browser tests rather than native LLVM instrumentation.
