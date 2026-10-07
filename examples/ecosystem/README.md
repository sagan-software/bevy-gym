# Ecosystem curriculum examples

This directory contains eight incremental recurrent Burn PPO examples built on
one Avian2D ecosystem simulation:

1. `ecosystem-forage`: one bunny learns to perceive, approach, and eat food.
2. `ecosystem-sprint`: food disappears after five seconds, and earlier contact
   earns more reward.
3. `ecosystem-gorge`: ephemeral food alternates between opposite banks; leaving
   the visible bridge causes death.
4. `ecosystem-survival`: the transferred movement policy learns to eat, drink,
   and stay alive.
5. `ecosystem-shelter`: the bunny retains homeostasis while seeking shelter.
6. `ecosystem-competition`: several bunnies share one learned policy and
   compete for finite resources.
7. `ecosystem-predator-prey`: bunnies eat spawned food while foxes hunt
   bunnies; both drink from the well.
8. `ecosystem-obstacles`: procedural trees, rocks, and damaging thorn bushes
   extend the predator-prey curriculum.

The implementation order, fixed evaluation gates, video format, and current
evidence are tracked in [PLAN.md](PLAN.md). Primary-source design research is
kept in [RESEARCH.md](RESEARCH.md).

All policies act from egocentric Avian2D raycasts, physiology, and per-agent
LSTM state. They do not receive map coordinates for food, water, other agents,
predators, or obstacles. Local observation, centralized training-state, action,
and recurrent shapes remain stable across stages so checkpoint transfer is
real weight reuse.

The transfer order is `forage -> sprint -> gorge -> survival -> shelter ->
competition -> predator-prey -> obstacles`. `ecosystem-curriculum` trains that
order and evaluates every earlier lesson after each promotion.

The sprint lesson gives every food item a five-second fixed-step lifetime. Its
bonus decreases with the remaining lifetime, so reaching food sooner is better
than approaching it slowly. Sprint promotion requires at least 80% food success
with mean contact within four seconds. The gorge lesson retains that deadline
and alternates food between banks. Gorge promotion requires at least 80%
opposite-bank success with mean contact within five seconds. The gorge floor is
lethal outside a brown bridge with solid rails. Perception rays classify both
the gorge and bridge.

Running an example without arguments starts live training and opens its visual
demo. The Bevy Inspector egui panel shows mean training and evaluation reward,
fixed-seed survival time, reward change per iteration, PPO diagnostics, and the
durable run directory. It also controls world playback, training, perception,
reward weights, physiology, movement, and episode duration.

The perception overlay starts enabled for agent 0. Each agent has one ray origin
centered between two rendered eyes, with dense frontal rays and sparse peripheral
rays. Conjugate gaze turns the ray fan within the configured head-relative limit. Each line is an active
post-physics semantic sector stored in that agent's next observation. A dim line
marks an empty sector. A colored line stops at the perceived hit: food is
yellow, water is blue, agents are white or orange, obstacles are gray, thorns
are magenta, and boundaries are brown. Use `Show perception rays` and `Ray
source` in the HUD to hide the overlay or inspect another agent. If the selected
agent dies, the filter follows the first living agent.

Use `Show hitbox/hurtbox shapes` to draw cyan hurtboxes and orange mouth
hitboxes. An active mouth hitbox turns bright red-orange. The overlay includes
food, thorn, and well interaction sensors and works in the 3x3 training replay.

The collapsed `Experiment tuning` panel provides these live controls:

- active perception rays in even pairs from 2 through 24;
- food or prey and completed-drink reward weights;
- maximum and starting HP, satiation, and hydration points;
- need-loss, starvation-damage, and dehydration-damage intervals in seconds;
- translation-only need cost and movement speed;
- eye-gaze range;
- episode duration in seconds.

Ray counts remove sparse peripheral pairs before dense frontal pairs. Each ray
keeps its narrow sensing arc, so lower counts create real blind gaps. Each ray
stores normalized hit distance followed by ten one-hot semantic channels: food,
well, bunny, fox, solid obstacle, thorn, boundary, shelter, gorge, and bridge.
Inactive observation slots remain zero and the actor input stays 275 values wide.
The actor receives no separate food or well summary, species channel, or stage
channel. The policy emits forward, body-turn, gaze, and attack actions. Attack opens a
forward mouth hitbox. Food, water, and prey interactions require the open mouth
to overlap the target. Checkpoint profile 30 deliberately rejects earlier actor
schemas.

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
point every 60 simulated seconds. Starvation and dehydration each deal 1 HP
every 10 seconds while the matching need is zero. In survival, food restores
satiation to 5 and a completed one-second drink restores hydration to 5.
Translation advances the need clock 25% faster; body rotation and gaze do not
add movement cost.

The survival map regenerates five solid obstacles for each episode. The well is
shallow water: agents can cross it at half speed. Solid collisions use Avian
contact normals and pre-solver velocity to apply speed-scaled damage. Agents
turn at up to 16 radians per second, and ground traction removes lateral sliding.

The survival episode ends after 120 simulated seconds by default. Each live
step rewards survival while penalizing normalized physiological deficit and
rewarding resource recovery. A surviving agent also receives a bounded
terminal bonus of `5 * (1 - normalized_drive)`, so maximum terminal stats earn
the full bonus. The signed locomotion output maps to forward throttle: `-1`
stops, `0` uses half throttle, and `1` uses full throttle. The signed attack
output opens the mouth when it is positive. Shorter episodes produce updates
sooner but can remove delayed consequences. These controls are intended for
visual experiments.

```sh
cargo run --features ecosystem-inference --example ecosystem-forage
cargo run --features ecosystem-inference --example ecosystem-sprint
cargo run --features ecosystem-inference --example ecosystem-gorge
cargo run --features ecosystem-inference --example ecosystem-survival
cargo run --features ecosystem-inference --example ecosystem-shelter
cargo run --features ecosystem-inference --example ecosystem-competition
cargo run --features ecosystem-inference --example ecosystem-predator-prey
cargo run --features ecosystem-inference --example ecosystem-obstacles
```

The survival demo uses the seed-157 profile: 16 PPO updates, nine parallel
rollout environments per update, 16 fixed-seed evaluation episodes after every
fourth update and the final update, a 120-second horizon, and 1x visual
playback. Each of the nine environments supplies recurrent sequences and
returns to the same PPO update. The numbered PPO counter advances only after
all nine episodes finish and their pooled optimizer update completes. The HUD
reports collection, evaluation, teacher training, memory training, and PPO
optimization as separate worker activities.
All nine use the survival environment contract; independent seeds vary their
maps without substituting forage lanes. Before offline survival initialization
finishes, the first window shows a labelled nine-environment initialization
preview. That preview loops until the first contributing optimizer batch is
ready, and the HUD identifies it as preview data.
After that update, the renderer replays those exact trajectories in a labelled
3x3 grid. If training produces traces faster than playback, the renderer keeps
the newest pending trace without blocking training. The renderer interpolates
between fixed physics states at 1x. The return graph adds one point for every
training episode and keeps fixed-seed evaluation means as a separate series.
After a terminal state, the grid pauses for one wall-clock second. Each agent
shows its episode return and a fading signed reward delta above its physiology
bars. The other stages retain sequential collection and their shorter seed-42
demo budgets. Override these settings after the mode name when needed:

```sh
cargo run --features ecosystem-inference --example ecosystem-survival -- demo \
  --iterations 12 --rollout-episodes 9 --eval-episodes 6 \
  --episode-seconds 120 --seed 42 --speed 1
```

Use WASD, arrow keys, left drag, or middle drag to pan. Use the mouse wheel or
`+` and `-` to zoom. Pointer input over the HUD does not move the camera. Press
space to pause the world, `R` to reset it, and `F1` to open the Bevy world
inspector. The HUD controls remain available while Burn trains on its worker
thread.

Headless training requires an explicit command. Disabling default features
also excludes the rendering and Inspector-egui dependencies:

```sh
cargo run --features ecosystem-inference --no-default-features --release --example ecosystem-survival -- train
cargo run --features ecosystem-inference --no-default-features --release --example ecosystem-survival -- \
  train --iterations 16 --rollout-episodes 9 --episode-seconds 120 --seed 157
```

The accepted default-profile run is
`runs/ecosystem-survival-ppo/final-homeostasis-profile-20260801/best.mpk`.
Its 16 updates collected exactly 172,800 steps from 144 complete training
episodes. All 16 validation episodes survived 120 seconds; mean terminal health
and satiation were 100%, mean hydration was 72.5%, and 43.75% ended with every
stat at maximum. On 100 disjoint 120-second episodes, all agents survived; mean
terminal health was 100%, satiation was 95.6%, hydration was 72.8%, and 31%
ended with every stat at maximum.

A 100-episode disjoint 300-second stress test reached the full horizon in 85%
of episodes, with 15 dehydration deaths. An eight-update 300-second continuation
performed worse on its disjoint test at 80% survival, so it is not the selected
checkpoint. The evidence supports reliable 120-second survival and improved
five-minute survival. It does not establish indefinite survival.

To inspect an existing checkpoint without training, use `watch`:

```sh
cargo run --features ecosystem-inference --release --example ecosystem-survival -- \
  watch --checkpoint runs/ecosystem-survival-ppo/<run-id> --seed 101 --speed 4
```

Current commands and learning results are recorded in
[EVIDENCE.md](EVIDENCE.md). Qualifying videos remain pending until their held-out
gates pass.

The `video` command reads every immutable `step-*.mpk` checkpoint from one run
and produces a deterministic H.264/yuv420p MP4 plus a typed JSON manifest:

```sh
cargo run --features ecosystem-inference --release --example ecosystem-survival -- \
  video --checkpoint runs/ecosystem-survival-ppo/<run-id> \
  --output examples/ecosystem/media/survival.mp4
```

Default timing is exact: 10 seconds of the selected best checkpoint, 5 seconds
for each chronological checkpoint, then 20 seconds of the selected best
checkpoint. The command refuses to overwrite an existing MP4 or manifest.
