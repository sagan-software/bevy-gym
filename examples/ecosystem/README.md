# Ecosystem curriculum examples

This directory contains four incremental recurrent Burn PPO examples built on
one Avian2D ecosystem simulation:

1. `ecosystem-survival`: one bunny learns to eat, drink, and stay alive.
2. `ecosystem-competition`: several bunnies share one learned policy and
   compete for finite resources.
3. `ecosystem-predator-prey`: bunnies eat spawned food while foxes hunt
   bunnies; both drink from the well.
4. `ecosystem-obstacles`: procedural trees, rocks, and damaging thorn bushes
   extend the predator-prey curriculum.

The implementation order, fixed evaluation gates, video format, and current
evidence are tracked in [PLAN.md](PLAN.md). Primary-source design research is
kept in [RESEARCH.md](RESEARCH.md).

All policies act from egocentric Avian2D raycasts, physiology, and per-agent
LSTM state. They do not receive map coordinates for food, water, other agents,
predators, or obstacles. Local observation, centralized training-state, action,
and recurrent shapes remain stable across stages so checkpoint transfer is
real weight reuse.

Running an example without arguments starts live training and opens its visual
demo. The Bevy Inspector egui panel shows mean training and evaluation reward,
fixed-seed survival time, reward change per iteration, PPO diagnostics, and the
durable run directory. It also controls world playback, training, perception,
reward weights, physiology, movement, and episode duration.

The perception overlay starts enabled for agent 0. Each agent has two forward
eye origins with dense frontal rays and sparse peripheral rays. Conjugate gaze
turns both eyes within the configured head-relative limit. Each line is an active
post-physics semantic sector stored in that agent's next observation. A dim line
marks an empty sector. A colored line stops at the perceived hit: food is
yellow, water is blue, agents are white or orange, obstacles are gray, thorns
are magenta, and boundaries are brown. Use `Show perception rays` and `Ray
source` in the HUD to hide the overlay or inspect another agent. If the selected
agent dies, the filter follows the first living agent.

The collapsed `Experiment tuning` panel provides these live controls:

- active perception rays in even pairs from 2 through 36;
- food or prey and completed-drink reward weights;
- maximum and starting HP, satiation, and hydration points;
- need-loss, starvation-damage, and dehydration-damage intervals in seconds;
- translation-only need cost and movement speed;
- eye-gaze range;
- episode duration in seconds.

Ray counts remove sparse peripheral pairs before dense frontal pairs. Each ray
keeps its narrow sensing arc, so lower counts create real blind gaps. Inactive
observation slots remain zero and the actor input stays 340 values wide. The
policy emits forward, body-turn, and gaze actions. Checkpoints created for the
old 339-input, two-action profile are incompatible with this profile.

Training accepts a changed profile only between complete PPO iterations. This
keeps one rollout batch on one reward and dynamics definition. Use `Apply +
restart visible world` to apply the same profile to the rendered evaluation
world. Every applied profile is written into the run metrics. Purple graph
markers identify training changes. A settings change resets the displayed
reward curve and fixed-seed checkpoint-selection baseline. The run config stores
the latest profile, and `eval` or `watch` loads it with the checkpoint. Rewards
from different profiles are not directly comparable.

The default survival profile has 5 maximum HP and starts with 5 HP. Satiation
and hydration each have 5 maximum points and start at 3. Both needs lose one
point every five simulated seconds. Starvation deals 1 HP every three seconds,
and dehydration deals 1 HP every two seconds while the matching need is zero.
Food adds one satiation point. A completed one-second drink adds one hydration
point. Translation advances the need clock 25% faster; body rotation and gaze
do not add movement cost.

The episode ends after 20 simulated seconds by default. Its terminal survival
reward is `survived_seconds * remaining_hp / maximum_hp`. Food and drink bonuses
are added when those events occur. Their pre-consumption need multiplier is
125% at or below 10%, 100% through 50%, 90% through 75%, 75% below 90%, and zero
from 90% upward. The signed locomotion output maps to forward throttle: `-1`
stops, `0` uses half throttle, and `1` uses full throttle. Fewer rays remove
peripheral pairs before frontal pairs. Shorter episodes produce updates sooner
but can remove delayed consequences. These controls are intended for visual
experiments.

```sh
cargo run --example ecosystem-survival
cargo run --example ecosystem-competition
cargo run --example ecosystem-predator-prey
cargo run --example ecosystem-obstacles
```

The survival demo uses the verified seed-157 profile: six PPO updates, 16
rollout episodes per update, 16 fixed-seed evaluation episodes after every
update, a 20-second horizon, and 8x visual playback. The other stages retain
their shorter seed-42 demo budgets. Override these settings after the mode name
when needed:

```sh
cargo run --example ecosystem-survival -- demo \
  --iterations 12 --rollout-episodes 4 --eval-episodes 6 \
  --episode-seconds 20 --seed 42 --speed 8
```

Use WASD, arrow keys, left drag, or middle drag to pan. Use the mouse wheel or
`+` and `-` to zoom. Pointer input over the HUD does not move the camera. Press
space to pause the world, `R` to reset it, and `F1` to open the Bevy world
inspector. The HUD controls remain available while Burn trains on its worker
thread.

Headless training requires an explicit command. Disabling default features
also excludes the rendering and Inspector-egui dependencies:

```sh
cargo run --no-default-features --release --example ecosystem-survival -- train
cargo run --no-default-features --release --example ecosystem-survival -- \
  train --iterations 32 --rollout-episodes 4 --episode-seconds 20 --seed 42
```

The current seed-157 headless run improved fixed-seed mean reward from 0.000 to
44.150 within six updates. On 100 disjoint seed-907 episodes, the random
checkpoint scored 0.400 and survived 18.708 seconds on average. The selected
checkpoint scored 44.240, survived the complete 20 seconds in every episode,
and collected both food and water in every episode. See
[EVIDENCE.md](EVIDENCE.md) for the commands, per-update results, confidence
intervals, resource events, and rendered screenshots.

To inspect an existing checkpoint without training, use `watch`:

```sh
cargo run --release --example ecosystem-survival -- \
  watch --checkpoint runs/ecosystem-survival-ppo/<run-id> --seed 101 --speed 4
```

Current commands and learning results are recorded in
[EVIDENCE.md](EVIDENCE.md). Qualifying videos remain pending until their held-out
gates pass.

The `video` command reads every immutable `step-*.mpk` checkpoint from one run
and produces a deterministic H.264/yuv420p MP4 plus a typed JSON manifest:

```sh
cargo run --release --example ecosystem-survival -- \
  video --checkpoint runs/ecosystem-survival-ppo/<run-id> \
  --output examples/ecosystem/media/survival.mp4
```

Default timing is exact: 10 seconds of the selected best checkpoint, 5 seconds
for each chronological checkpoint, then 20 seconds of the selected best
checkpoint. The command refuses to overwrite an existing MP4 or manifest.
