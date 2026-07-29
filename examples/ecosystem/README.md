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
demo. The Bevy Inspector egui panel shows the current policy, exact episodic
reward, training and evaluation survival time, PPO losses, entropy, iteration
count, gain from the random-policy baseline, and durable run directory. It also
controls world playback speed, world pause, world reset, training pause, and
training stop.

The perception overlay starts enabled for agent 0. Each line is an active
post-physics semantic sector stored in that agent's next observation. A dim line
marks an empty sector. A colored line stops at the perceived hit: food is
yellow, water is blue, agents are white or orange, obstacles are gray, thorns
are magenta, and boundaries are brown. Use `Show perception rays` and `Ray
source` in the HUD to hide the overlay or inspect another agent. If the selected
agent dies, the filter follows the first living agent.

The collapsed `Experiment tuning` panel provides these live controls:

- active perception rays from 1 through 36;
- alive, food or prey, and absorbed-water reward weights;
- starting health;
- hunger and thirst speed;
- health damage speed;
- episode step limit.

Ray counts select an evenly spaced subset of the 36 reserved directions. Each
ray keeps its narrow sensing arc, so lower counts create real blind gaps.
Inactive observation slots remain zero and the actor input stays 339 values
wide. Existing policies and recurrent memory therefore remain compatible.

Training accepts a changed profile only between complete PPO iterations. This
keeps one rollout batch on one reward and dynamics definition. Use `Apply +
restart visible world` to apply the same profile to the rendered evaluation
world. Every applied profile is written into the run metrics. Purple graph
markers identify training changes. The learning graph uses survival time, so
its points remain comparable when reward weights change. Episodic reward does
not remain comparable across reward profiles.

Alive reward provides dense feedback and usually improves early credit, but a
large value can reward passive survival. Food and water bonuses make resource
discovery more important, but sparse bonuses increase variance. Lower starting
health, faster need drain, and faster damage create urgency while reducing time
for exploration. Fewer rays reduce spatial detail and usually slow resource
discovery, but they make learned scanning behavior easier to see. Shorter
episodes produce PPO updates sooner but can cut off delayed consequences. These
controls are intended for short visual experiments, not production model
selection.

```sh
cargo run --example ecosystem-survival
cargo run --example ecosystem-competition
cargo run --example ecosystem-predator-prey
cargo run --example ecosystem-obstacles
```

The survival demo uses the verified seed-157 profile: six PPO updates, 16
rollout episodes per update, 16 fixed-seed evaluation episodes after every
update, a 1,200-step horizon, and 8x visual playback. The other stages retain
their shorter seed-42 demo budgets. Override these settings after the mode name
when needed:

```sh
cargo run --example ecosystem-survival -- demo \
  --iterations 12 --rollout-episodes 4 --eval-episodes 6 \
  --max-steps 1200 --seed 42 --speed 8
```

Use WASD or arrow keys to pan. Use the mouse wheel or `+` and `-` to zoom.
Press space to pause the world, `R` to reset it, and `F1` to open the Bevy
world inspector. The HUD controls remain available while Burn trains on its
worker thread.

Headless training requires an explicit command. Disabling default features
also excludes the rendering and Inspector-egui dependencies:

```sh
cargo run --no-default-features --release --example ecosystem-survival -- train
cargo run --no-default-features --release --example ecosystem-survival -- \
  train --iterations 32 --rollout-episodes 4 --max-steps 1200 --seed 42
```

The first command is the verified no-argument headless profile. Its recorded
seed-157 run improved fixed-seed mean survival from 56.181 to 78.768 seconds in
six updates. Fresh-process evaluation on 100 disjoint episodes improved from
54.367 to 79.427 seconds. See [EVIDENCE.md](EVIDENCE.md) for confidence
intervals and resource-use measurements.

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
