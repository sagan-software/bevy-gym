# Pendulum browser port

The browser session API supports `Task::Pendulum` with recurrent PPO training
and frozen inference. The renderer preserves the original 500 by 500 scene
and torque-arrow image. The page controls are implemented and tested locally.
Model qualification remains pending, so Pendulum is not yet in the deployed menu.

The shared environment follows [Gymnasium Pendulum at revision
7a119138](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/classic_control/pendulum.py).
The catalog version is 1.3.0; dynamics fixtures pin NumPy 2.4.4. Custom gravity
and reset bounds are not exposed. The uniform reset distribution uses SplitMix64;
numeric seeds do not reproduce NumPy's reset stream.

## Session contract

PPO observes angle cosine, angle sine, and angular velocity divided by
8 radians per second. The action is torque in [-2, 2] newton metres.
Physical state retains float64 precision; observations and actions use float32.
Each transition advances 0.05 seconds. The external time limit truncates after
200 transitions and retains the terminal observation for value bootstrapping.

Training scales rewards by 0.1 for optimization. Snapshots and episode totals
retain the original reward. Eight lanes each contribute 64 transitions before
an update. Actor and critic learning rates are 0.003 and 0.001.
The actor has 32 recurrent units; the critic has hidden widths 64 and 32.
The recipe uses gamma 0.99, GAE lambda 0.95, four epochs, four sequences per
minibatch, initial log standard deviation -0.5, and zero entropy coefficient.

Inference uses deterministic mean actions and owns no optimizer. Recurrent
memory resets after each episode. Snapshot state is angle in radians,
angular velocity in radians per second, last torque in newton metres, and zero.
Last torque is zero before the first action. Batch size does not alter inference
transitions, episode resets, returns, or exported policy parameters.

The worker name is `pendulum`. Command and snapshot formats retain protocol 1.
Wrong observation dimensions and corrupt policy records fail before activation.
Action length is checked before finiteness; invalid actions leave state and
reward unchanged.

## Verification

```sh
nix develop --command cargo test -p bevy-gym-browser --test pendulum_session
nix develop --command cargo test -p bevy-gym-browser --test pendulum_learning
nix run .#gymnasium-check -- --grep 'rendering matches pendulum'
nix run .#gymnasium-check -- --grep 'Pendulum|delayed Pendulum image'
```

Four integration tests cover both network updates, native policy compatibility,
original rewards, the 200-transition limit, frozen parameters, episode resets,
batch-size agreement, and incompatible policies. Unit tests exercise malformed
action lengths, each nonfinite torque, observation normalization, and reward scaling.

Four Pygame reference images cover upright and downward states, zero torque,
positive torque, and negative torque. All twelve comparisons pass across
Chromium, Firefox, and WebKit with the existing pixel tolerance. Integer sprite
placement preserves Pygame's horizontal and vertical flip rules.

Fifteen control and evaluation checks pass across the three browser engines.
They cover both optimizer updates, export and frozen replay, single stepping,
desktop and mobile layouts, image failure and retry, and stale image completions.
The stale-completion checks resolve the newer request before either completing
or rejecting the older request. A loaded CPU caused one Firefox update timeout;
the complete fifteen-check suite passed after background jobs were paused.

The worker's `evaluate_pendulum` command takes a `u32` seed and a policy byte
array. It returns `event: "evaluation"` with a `score` object containing numeric
`reward` and integer `upright_steps` from 0 through 150. It evaluates exactly
200 deterministic actions in an independent environment and changes no active
session. Other worker tasks reject the command before loading its record.
Corrupt records and nonfinite actions produce the existing `error` response.

The overflow regression uses finite float32 parameters whose recurrent products
produce a nonfinite torque. It verifies `Pendulum torque must be finite` and
proves that the active session still advances from its first transition.
The fixture generator uses the committed continuous MountainCar record envelope
and Python msgpack 1.1.2. Regenerate it from the repository root:

```sh
nix shell --impure --expr '
  let project = builtins.getFlake (toString ./.);
      pkgs = import project.inputs.nixpkgs { system = builtins.currentSystem; };
  in pkgs.python3.withPackages (p: [ p.msgpack ])
' --command python gymnasium-web/tests/fixtures/make_overflow_policy.py
```

Native LLVM coverage records 60 of 61 lines in the score evaluator. The remaining
line propagates a policy-forward error. The fixed observation width, bounded
physical state, validated record, and finite action check prevent its input and
memory validation failures in this evaluator. Backend conversion failures are
not injected. The missing-torque fallback is also unhit because record validation
requires one action coordinate. This does not establish complete branch coverage.

Strict workspace Clippy and the browser package's personal lint checks pass.
The root package's personal lint command reports 33 existing diagnostics outside
this change, including function length and complexity. Its changed-line filter
reports no diagnostics; the complete root personal lint gate remains unresolved.

The [original torque image](assets/clockwise.png) is copied without modification.
The [Gymnasium MIT license](assets/GYMNASIUM-LICENSE) applies to that image
and the ported renderer.

## Qualification still required

Each of three independent training seeds must achieve mean raw return at least
-200 and upright dwell rate at least 0.70. Dwell counts the states after
zero-based transitions 50 through 199 inclusive, before any reset. A state
qualifies when its normalized absolute angle is at most 15 degrees and its
absolute angular velocity is at most 1 radian per second.
Both evaluators compare angle cosine against `cos(pi / 12)` and use the source
float64 state before reset. Reports retain every episode's raw return and dwell
count; aggregate dwell divides their sum by 150 samples per episode.

Native profile `pendulum-native-ppo-v1` uses training seeds 42, 43, and 44.
Each run selects its highest mean on 20 seed-specific validation episodes,
breaking ties by the earliest checkpoint. Those seeds come from
`SeedConfig::from_root(seed).validation`.

A second, separately retained profile uses actor rate 0.0003 with the same
seeds, architecture, optimizer, rollout, reward scale, evaluation rule, and
two-million-transition budget. Seeds 42 and 43 passed common validation with
means -193.12975 and -195.49872 and dwell rates 0.98547 and 0.98133. Seed 44 is
still training. No held-out test score has been opened for this profile.
The browser's active qualification run still uses the original actor rate.

Final selection evaluates each candidate
on seeds 10,000 through 10,099. Every candidate must pass both gates before any
test seed is opened. The bundled candidate is the highest common-validation mean,
breaking ties by the lowest training seed. Tests use seeds 400,000 through
400,199 for each candidate. Checkpoints above two million transitions are
ineligible, including the native collector's final rounded-up rollout.

```sh
nix develop --command cargo test -p bevy-gym-browser --test pendulum_learning \
  qualify_three_native_seeds_before_bundling -- --ignored --nocapture
```

Browser profile `pendulum-browser-ppo-v1` trains seed 42 while offline. It checks
the same 100 validation seeds every 10,240 transitions and at the exact
two-million-transition limit. When a candidate first passes both gates, the run
freezes it before evaluating seeds 600,000 through 600,199. The report retains the baseline,
validation history, individual test episodes, policy hash, browser version,
CPU model, and elapsed time. Qualification has an eight-hour process timeout;
the transition budget and score gates remain unchanged.

```sh
env -u LD_LIBRARY_PATH -u LD_PRELOAD BEVY_GYM_QUALIFY_PENDULUM=1 \
  nix run .#gymnasium-check -- --project chromium \
  --grep 'Pendulum browser PPO passes offline'
```

The full browser learning run is an explicit release gate and is skipped by the
default browser suite. Copy its `pendulum-learning.json` and `pendulum-policy.mpk`
into `runs/browser-pendulum/browser-v1/` before checking native compatibility:

```sh
nix develop --command cargo test -p bevy-gym-browser --test pendulum_learning \
  browser_trained_pendulum_policy_replays_held_out_episodes_natively \
  -- --ignored --nocapture
```

Compatibility requires each native return to differ by less than 0.001 from its
browser return, exact per-episode dwell counts, and both qualification gates.

The first native run ended with SIGTERM at 71,680 recorded transitions.
Its artifacts remain under `runs/pendulum-ppo/pendulum-oracle-v1-seed42/`.
Fresh durable runs preserve the same recipe and two-million-transition budget.
An interrupted inference checkpoint cannot resume optimizer state.
No Pendulum model is bundled or advertised as qualified yet.
