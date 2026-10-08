# Acrobot

The native example and browser worker share `bevy_gym::environments::Acrobot`.
The bundled DQN reaches the goal in all 200 held-out episodes, with mean return
`-82.555` in native inference, Chromium, Firefox, and WebKit.
[Model provenance](models/acrobot.json) retains the recipe, selection results,
episode scores, learning curves, failed first profile, and model hash.

## Environment contract

The port follows [Gymnasium revision 7a119138](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/classic_control/acrobot.py)
with NumPy 2.4.4. It uses the default book equations and fourth-order Runge–Kutta
integration with a 0.2-second timestep. The source derives from RLPy; its
[BSD-3-Clause notice](../LICENSES/ACROBOT-BSD-3-Clause.txt) ships with the site.

- Actions 0, 1, and 2 apply torques -1, 0, and +1. Other indices are rejected.
- Reset rounds each uniform `[-0.1, 0.1)` coordinate to float32. Later state
  evolution uses float64. Observations contain the two angle sine/cosine pairs
  and two angular velocities, projected to float32.
- Angle wrapping retains both endpoints of `[-pi, pi]`. Angular velocities are
  clipped to `[-4*pi, 4*pi]` and `[-9*pi, 9*pi]` radians per second.
- The goal requires tip height strictly greater than one metre above the pivot.
  That transition earns zero; every other transition earns -1.
- The external `TimeLimit` truncates after 500 transitions. Natural termination
  takes precedence when the goal is reached on transition 500.
- Reset uses SplitMix64. Equal numeric seeds differ from NumPy PCG64 sequences.
  Custom reset bounds, link parameters, timestep, torque noise, and NIPS dynamics
  are not exposed by this default port.

The Python oracle checks 2,133 transitions and 33 reset seeds. Focused tests
cover action and diagnostic-state validation, velocity clipping, strict goal
height, both wrap endpoints, and termination at the time limit. All three new
core modules have 100% measured line and region coverage.

## Learning and frozen inference

DQN uses hidden layers `[128, 128]`, Adam rate 0.0003, discount 0.99, batch size
128, a 200,000-transition replay buffer, and a 5,000-transition warmup. The target
network updates every 1,000 optimizer updates. Exploration decreases from 1 to
0.05 over 200,000 transitions. The browser trains one environment; native
qualification uses 16 environments.

Angular velocities are divided by float32 representations of `4*pi` and `9*pi`.
Potential shaping adds `gamma * next_potential - current_potential` to the
optimizer reward, using tip height minus two metres and scale 10. Only natural
termination sets the next potential to zero. Charts and scores retain the
original environment reward.

The first native profile failed common validation for seeds 42 and 43; no
held-out seeds were opened. The second profile uses 100 validation episodes
and a stricter training stop of -90, while qualification remains -100 with at
least 95% goal completion. Its one-million-transition budget is unchanged.

- Seed 42 selected transition 110,000: validation -85.71, held-out -84.535.
- Seed 43 selected transition 130,000: validation -85.19, held-out -82.555.
- Seed 44 selected transition 130,000: validation -86.31, held-out -86.58.

Each native seed reached all 200 held-out goals. Common validation seeds
`10000..10100` selected seed 43 before test seeds `700000..700200` were opened.
Chromium trained a separate model offline for 50,000 transitions. It scored
-86.585 with 200/200 goals on `800000..800200`. The host was a T490 with an Intel
Core i7-8565U; Chromium was 147.0.7727.15. Concurrent training jobs were active.

Native replay of the browser-trained policy scored -86.65 with 200/200 goals.
Two episode returns differed: seed 800021 was -214 in Chromium and -218 natively;
seed 800173 was -227 and -236. The other 198 returns agreed. The model was not
selected using these replay differences. Short transition probes for both
policies agree across all three engines within `1e-9` absolute state error;
the largest observed error was `1.7763568394002505e-15`.

Inference retains no optimizer or exploration state. Repeated canonical exports
remain identical. Native file records contain different metadata from byte
records, so compatibility compares canonical exports after loading.

## Playback and rendering

Browser 1× advances five transitions per second, matching the physical timestep.
The native reference player uses Gymnasium's 15 frames per second. Playback
speed does not change physics, update cadence, or checkpoint selection.

The canvas preserves the original 500×500 geometry, cyan links, yellow joints,
and target line. Four pinned Pygame images cover downward, upright,
bent, and goal states. All twelve browser comparisons pass. Acrobot foreground
selection excludes faint edge pixels using a channel threshold of 80; matching
retains the shared two-pixel distance, 80-unit channel tolerance, and 0.5% limit.
These checks do not establish byte-identical rasterization.

Browser tests cover real parameter updates, frozen export/import, pause,
single-step inference, worker crash recovery, desktop/mobile layout, all five
speeds, and the physical timestep. Deployed motion review remains separate.
The [reviewed recording and captures](../docs/visual-comparisons/README.md#acrobot)
show local inference, fresh training, and mobile layout.

## Reproduce the gates

```sh
nix develop --command cargo test --test acrobot_contract
nix develop --command cargo test --example acrobot
nix develop --command cargo test -p bevy-gym-browser --test acrobot_session
nix run .#gymnasium-check -- --grep 'Acrobot|acrobot'
```

The ignored native qualification test requires completed runs under
`runs/acrobot-dqn/acrobot-oracle-v2-seed{42,43,44}-durable`. Browser qualification
requires `BEVY_GYM_QUALIFY_ACROBOT=1`; transition compatibility requires retained
probe files and `BEVY_GYM_ACROBOT_COMPATIBILITY=1`. Ordinary browser tests always
exercise the bundled model and functional gates.
