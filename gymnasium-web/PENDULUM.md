# Pendulum browser port

The browser session API supports `Task::Pendulum` with recurrent PPO training
and frozen inference. The renderer preserves the original 500 by 500 scene
and torque-arrow image. Model qualification and the page controls remain pending.

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
nix run .#gymnasium-check -- --grep 'rendering matches pendulum'
```

Four integration tests cover both network updates, native policy compatibility,
original rewards, the 200-transition limit, frozen parameters, episode resets,
batch-size agreement, and incompatible policies. Unit tests exercise malformed
action lengths, each nonfinite torque, observation normalization, and reward scaling.

Four Pygame reference images cover upright and downward states, zero torque,
positive torque, and negative torque. All twelve comparisons pass across
Chromium, Firefox, and WebKit with the existing pixel tolerance. Integer sprite
placement preserves Pygame's horizontal and vertical flip rules.

The [original torque image](assets/clockwise.png) is copied without modification.
The [Gymnasium MIT license](assets/GYMNASIUM-LICENSE) applies to that image
and the ported renderer.

## Qualification still required

Each of three independent training seeds must achieve mean raw return at least
-200 and upright dwell rate at least 0.70. Dwell counts the states after
zero-based transitions 50 through 199 inclusive, before any reset. A state
qualifies when its normalized absolute angle is at most 15 degrees and its
absolute angular velocity is at most 1 radian per second.

The first native run ended with SIGTERM at 71,680 recorded transitions.
Its artifacts remain under `runs/pendulum-ppo/pendulum-oracle-v1-seed42/`.
Fresh durable runs preserve the same recipe and two-million-transition budget.
An interrupted inference checkpoint cannot resume optimizer state.
No Pendulum model is bundled or advertised as qualified yet.
