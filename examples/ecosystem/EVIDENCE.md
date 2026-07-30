# Ecosystem implementation and learning evidence

This log separates implementation proof, exploratory probes, and qualifying
evaluation. Raw run directories remain ignored under `runs/`; the commands and
results needed to interpret them are recorded here.

## Current rounded-geometry and integer-physiology proof

This 2026-07-29 section supersedes the historical mechanics and learning runs
below. Earlier checkpoints use different physiology, reward, geometry, or
tensor contracts and are not evidence for the current example.

The short-ray defect came from the well's drinking sensor. It was a semantic
square collider, so a ray cast while the agent was inside it returned the
sensor's nearby exit boundary. The drinking sensor is now a nonsemantic circle.
Perception ray queries also accept only entities with a semantic collider, so
interaction triggers cannot clip or occlude perception. Agents use oval
colliders and visuals. Food, wells, trees, rocks, and thorns use circular
colliders and matching circular visuals. The square map retains rectangular
boundary walls.

The current survival profile uses 5 maximum HP, 5 maximum satiation, and 5
maximum hydration. An episode starts at 5 HP and 3 points in each need. Both
needs lose one point every five simulated seconds. Starvation deals 1 HP every
three seconds, and dehydration deals 1 HP every two seconds while the matching
need is zero. Food adds one satiation point. Each completed one-second drink
adds one hydration point. Translation advances the need clock by an additional
25%; rotation and gaze do not add movement cost.

The default horizon is 20 simulated seconds. The final transition adds
`survived_seconds * remaining_hp / maximum_hp` to resource bonuses. Food and
drink bonuses use the need before consumption. The multiplier is 125% at or
below 10%, 100% through 50%, 90% through 75%, 75% below 90%, and zero from 90%
upward. Checkpoint selection uses mean episode reward rather than survival time,
which avoids arbitrary selection when all policies reach the 20-second horizon.

Fresh headless training used this command:

```sh
cargo run --no-default-features --release --example ecosystem-survival -- \
  train --run-id integer-survival-final-v6-20260729
```

The fixed-seed evaluation curve was:

```text
iteration 0: evaluation reward 0.000, survival 18.450 s
iteration 1: training reward 7.806, evaluation reward 0.500, survival 18.544 s
iteration 2: training reward 10.950, evaluation reward 4.125, survival 19.381 s
iteration 3: training reward 18.312, evaluation reward 40.619, survival 20.000 s
iteration 4: training reward 26.569, evaluation reward 43.450, survival 20.000 s
iteration 5: training reward 23.100, evaluation reward 44.150, survival 20.000 s
iteration 6: training reward 37.644, evaluation reward 44.150, survival 20.000 s
```

The six updates completed in 18,634 joint steps. The initial food and well are
separate, visible from the spawn, outside contact range, and ordered along one
learnable route. This prevents the water-only local optimum found in two
discarded exploratory runs.

Fresh processes evaluated the random and selected checkpoints on 100 disjoint
seed-907 episodes:

```sh
cargo run --no-default-features --release --example ecosystem-survival -- eval \
  --checkpoint runs/ecosystem-survival-ppo/integer-survival-final-v6-20260729/checkpoints/step-000000.mpk \
  --episodes 100 --seed 907
cargo run --no-default-features --release --example ecosystem-survival -- eval \
  --checkpoint runs/ecosystem-survival-ppo/integer-survival-final-v6-20260729/best.mpk \
  --episodes 100 --seed 907
```

```text
random:  reward 0.400, survival 18.708 s, 95% CI [18.587, 18.825]
         ate and drank 0.0%, food events 0, water 0.000, deprivation deaths 95
learned: reward 44.240, survival 20.000 s, 95% CI [20.000, 20.000]
         ate and drank 100.0%, food events 108, water 120.000, deaths 0
```

Rendered checkpoint playback used the same selected checkpoint and seed:

```sh
cargo run --release --example ecosystem-survival -- watch \
  --checkpoint runs/ecosystem-survival-ppo/integer-survival-final-v6-20260729/best.mpk \
  --seed 907 --speed 4
```

The inspected [final perception rays](../../ai/bmad-output/implementation-artifacts/screenshots/ecosystem-integer-survival-final-v6-rays.png)
show both visible resources, the oval agent, the circular well, and rays that
continue to the wall instead of stopping at the drinking sensor. The exact
rendered training command was also run:

```sh
cargo run --release --example ecosystem-survival
```

The [completed training HUD](../../ai/bmad-output/implementation-artifacts/screenshots/ecosystem-integer-survival-final-v6-complete.png)
shows iteration 6, 18,634 steps, a 44.15-point fixed-seed reward gain, and
20.00-second survival. The [final rendered learning graph](../../ai/bmad-output/implementation-artifacts/screenshots/ecosystem-integer-survival-final-v6-learning-graph.png)
shows mean reward, reward change per iteration, and fixed-seed survival under
the same profile-3 mechanics. Scrolling this HUD left the camera scale
unchanged.

The post-training review also verifies complete-drink allocation under scarce
water, fair recipient rotation, full deprivation intervals after a need reaches
zero, zero bootstrap after terminal reward, profile-3 checkpoint validation,
bounded CLI horizons, full final-step age accounting, and explicit attribution
for combined deprivation and thorn damage. Every saved policy now has an
immutable tuning sidecar. Watch, evaluation, resume, and video reject mismatched
bunny and fox profiles. Evaluation, watch, and video require the exact stage;
training resume permits only the current stage or its direct predecessor. Video
also rejects profile changes between segments and records the effective tuning
in its manifest. Failed checkpoint replacement invalidates the complete policy
set before any policy overwrite.

The final rendered ecosystem example suite passed 79 tests. The headless
ecosystem example suite passed 64 tests. `cargo test` passed 54
library tests and one documentation test, with one documentation test ignored.
Strict all-target, all-feature Clippy passed with warnings denied. The
personal-lint workflow found no new candidate-local diagnostic. Its Cargo stage
remains blocked by 15 pre-existing CartPole lifetime diagnostics and three
pre-existing ecosystem 100-line diagnostics. Its Dylint stage remains blocked
because the isolated environment lacks `wayland-client.pc`. The latest log is
stored outside the repository under `~/.cache/rust-personal-lints/logs/`.

## Historical implementation proof

All four stages construct deterministic Avian2D worlds and complete random
joint-action smoke rollouts. The shared simulation currently includes dynamic
agent colliders, solid boundaries and obstacles, sensor resources and thorns,
finite/refilling well water, periodic food, physiology and death, predation,
fixed local rays, and a padded centralized critic state.

The reusable Burn trainer includes a local-observation LSTM actor, a separate
centralized critic, per-agent host memory, bounded continuous actions, intact
sequence chunks, masked GAE, clipped PPO loss, minibatched actor/critic Adam
updates, species parameter sharing, deterministic policy evaluation, and a
single-record actor/critic checkpoint round trip.

Focused gates last passed on 2026-07-29:

```text
cargo test --example ecosystem-survival --all-features
48 passed; 0 failed

cargo test --no-default-features --example ecosystem-survival
38 passed; 0 failed

cargo test
54 library tests and 1 documentation test passed

cargo clippy --all-targets --all-features -- -D warnings
passed
```

The exact `cargo run --example ecosystem-survival` command opened the visual
world before baseline evaluation and kept rendering during background PPO
updates. Manual inspection confirmed the reward pulse, current return, gain
from baseline, graph, world controls, training controls, and `F1` world
inspector. The inspected run was interrupted after three updates and is not
learning evidence.

The recurrent updater now weights actor and critic optimization plus actor
loss, critic loss, entropy, and approximate-KL diagnostics by valid recurrent
timesteps. A mixed 1-step/32-step regression proves the 1:32 weighting instead
of equal sequence weighting.

An earlier all-target attempt exposed a process-global Burn seed race between
algorithm-local initialization locks. Model initialization now uses one shared
backend lock; the exact all-target command passes without warnings.

The exact personal-lint run still fails on pre-existing CartPole lifetime
annotations and four pre-existing ecosystem functions above its private
100-line threshold. Its Dylint toolchain also lacks the Wayland pkg-config
entry. The current log is outside the repository at
`~/.cache/rust-personal-lints/logs/run-1918738-1785314414995`. The configured
strict Cargo Clippy gate remains clean.

## Historical default survival learning proof

The current actor has separate observation-dependent means and one learned
state-independent log standard deviation per action dimension. That actor
refactor changed the effective ecosystem initialization from `-0.5` to the
generic `-1.0` default. Action standard deviation fell from about 0.61 to 0.37,
and the visual survival example stopped discovering enough food and water.
Earlier ecosystem checkpoints predate the separate `mean_head` field and do
not load into the current actor.

The fix makes initial log standard deviation an explicit validated PPO setting.
The ecosystem profile starts and caps it at `-0.5`; the generic recurrent PPO
default remains `-1.0`. Survival also uses the declared 1,200-step episode
horizon instead of the former 600-step headless default. Reward remains pure
survival time: `1.0` per simulated second alive, with zero food and water
bonuses.

Before the fix, the former no-argument headless profile completed 62,466 steps
without beating its 50.025-second initial evaluation. Its final evaluation was
48.612 seconds. The fixed no-argument command was then run from a clean process:

```sh
cargo run --no-default-features --release --example ecosystem-survival -- \
  train --run-id final-default-headless-20260729a
```

The fixed run completed 59,971 environment steps and improved at every
evaluation point:

```text
initial  56.181 s
update 1 57.337 s
update 2 61.431 s
update 3 62.662 s
update 4 62.975 s
update 5 72.487 s
update 6 78.768 s, 95% CI [69.350, 89.462]
```

This is a 40.2% fixed-seed validation gain in six updates. Eat-and-drink
success rose from 12.5% to 68.75%. Food events rose from 11 to 22, and absorbed
water rose from 7.938 to 25.736 units.

Fresh-process evaluation used 100 disjoint episodes with seed 10001:

```text
step zero: 54.367 s, 95% CI [52.469, 56.740], ate+drank 15%, 54 food events,
           30.742 water units
best:      79.427 s, 95% CI [75.163, 83.593], ate+drank 56%, 153 food events,
           157.384 water units
```

The held-out gain is 46.1%, and the learned lower confidence bound exceeds the
step-zero mean. This proves current-architecture lifetime and resource-use
learning. The 56% held-out mechanism rate remains below the 70% curriculum
qualification threshold, so this is not a three-seed curriculum qualification.

CartPole was run as an independent integration control. Its unshaped DQN run
improved from 8.917 to the 500-step maximum after 70,000 steps. This isolates
the ecosystem failure to its recurrent exploration profile rather than the
shared collection, optimizer, checkpoint, or evaluation path.

The exact rendered command also completed all six updates and reproduced the
56.181-to-78.768 fixed-seed curve. The inspected window showed the live policy,
36 perception rays, learning graph, reward and dynamics controls, and PPO
progress. The current checkpoint is:

```text
runs/ecosystem-survival-ppo/final-default-headless-20260729a/best.mpk
```

All older survival results below remain useful historical evidence, but their
checkpoints are incompatible with the current actor record shape.

## Sparse-food survival correction

The rendered seed-157 policy exposed a reward exploit: with 12 initial food
items and capacity 20, the solo bunny could spin locally and encounter food too
often. Survival now starts with two food items and has capacity two. Consuming
one leaves one item available until the fixed 20-step replenishment interval;
if both are gone, the simulation immediately restores one at a new procedural
location so the map never has zero food. The regression test
`survival_food_supply_stays_between_one_and_two_with_delayed_replenishment`
covers the `2 -> 1 -> delayed 2` transition and the one-item floor.

The controlled from-scratch rerun reused seed 157 and the previous 12-iteration,
eight-world, `log_std_max=-0.5` configuration:

```text
runs/ecosystem-survival-ppo/sparse-food-157
48,920 environment steps
validation: 51.025 s at step zero -> 47.100 s at iteration 12
selected checkpoint: step zero (no learned improvement)
```

On 50 fixed held-out episodes (`seed=10001`), the selected step-zero checkpoint
reached 53.932 seconds with CI `[51.438, 56.738]` and ate-and-drank success in
12% of episodes. The iteration-12 checkpoint reached 52.176 seconds with CI
`[50.056, 54.926]` and 8% ate-and-drank success. The new food mechanics remove
the plentiful-food regime, but the old training budget does not learn the new
search problem.

The failure was a delayed-credit mismatch rather than an unstable optimizer.
With `gamma=0.999`, GAE lambda `0.98`, and 0.1-second steps, the trace weight
`gamma * lambda = 0.97902` has an effective horizon of only 47.66 steps, or
4.77 simulated seconds. Eating and drinking affect death tens of seconds later,
so sparse resource discoveries received almost no action credit. The training
CLI now validates `--gamma` and `--gae-lambda`, and the ecosystem default uses
GAE lambda `1.0`, extending the effective trace horizon to 1,000 steps, or 100
simulated seconds.

Three from-scratch replications used 16 rollout worlds per update,
`gamma=0.999`, GAE lambda `1.0`, and `log_std_max=-0.5`:

```text
seed 157: sparse-full-credit-157, 16 iterations, 190,581 steps
seed 163: sparse-full-credit-163, 12 iterations, 129,091 steps
seed 269: sparse-full-credit-269, 12 iterations, 105,938 steps
```

Fresh-process evaluation on the same 100 held-out episodes (`seed=10001`)
measured:

| Training seed | Step-zero mean | Learned mean | Learned 95% CI | Gain | Ate and drank |
| --- | ---: | ---: | --- | ---: | ---: |
| 157 | 53.823 s | 109.875 s | [106.179, 113.340] | 104.1% | 89% |
| 163 | 56.484 s | 86.275 s | [82.364, 90.238] | 52.7% | 62% |
| 269 | 48.671 s | 61.825 s | [59.636, 64.022] | 27.0% | 24% |

All three learned confidence intervals are wholly above their corresponding
step-zero intervals. Seed 157 produced 538 food events versus 49 at step zero;
seed 163 produced 141 versus 24. This is a substantial sparse-food learning
improvement reproduced across three independent initialization and rollout
seeds. Seed 269's food-and-water mechanism rate remains below the 70% formal
curriculum gate, so retain that lineage as robustness evidence for lifetime
learning rather than a selected predecessor.
The historical selected curriculum source was
`runs/ecosystem-survival-ppo/sparse-full-credit-157/best.mpk`.
Its actor record is incompatible with current code.

Two verified mechanics limitations remain follow-up work rather than blockers
for this lifetime result: drinking withdraws the full per-step request even
when the agent cannot absorb it, and sparse replacements rotate across all 24
padded critic food slots instead of the configured two live slots. The constant
`1.837876` entropy also shows that `log_std_max=-0.5` saturates the variance
head; split policy and entropy diagnostics before interpreting combined actor
loss as policy movement.

A short deterministic preview of the seed-157 run showed translation from the
upper map to the well and onward across the map during the closing best-policy
segment, rather than stationary tight-circle spinning. Its local diagnostic
artifacts are `behavior-preview.mp4`, `behavior-preview-video-manifest.json`,
and `behavior-contact-sheet.png` in that ignored run directory.

All survival evidence below this section was collected before the sparse-food
correction. Preserve it as historical optimizer evidence, but do not cite it as
qualification of the current survival environment or use its checkpoint as a
current-stage curriculum source.

## Historical survival learning probes

The original half-extent 25 map produced identical 46.7-second deaths across
eight 60-second trials. Exploration did not discover both delayed resource
loops. The declared introductory map was reduced to half-extent 10, historically
with 12 initial food items and a 20-step food interval. Later stages expand to
14, 20, and 25, preserving one fixed tensor contract.

A first checkpointed probe with the standard `gamma=0.99` improved validation
mean lifetime from 72.867 to 89.017 seconds, then regressed. Pooling four
independent episodes per update removed single-episode update bias but did not
fix the delayed-credit mismatch.

The accepted probe used `gamma=0.999`, GAE lambda `0.98`, actor learning rate
`0.0001`, critic learning rate `0.0003`, three epochs, eight sequence chunks per
optimizer minibatch, and four ecosystem episodes per iteration:

```sh
cargo run --quiet --example ecosystem-survival -- train \
  --iterations 6 --rollout-episodes 4 --max-steps 1200 \
  --eval-episodes 8 --eval-interval 2 --seed 103 \
  --run-id long-horizon-probe-103
```

Validation mean lifetime rose from 65.050 seconds at step 0 to 96.100 seconds
at step 17,245. This is a 47.7% single-seed validation gain. The selected
checkpoint was reloaded through the `eval` command on a disjoint seed stream:

```text
20-episode initial checkpoint
mean lifetime: 66.570 s
bootstrap 95% interval: [61.485, 72.889] s
ate and drank: 20.0%

20-episode selected checkpoint
mean lifetime: 86.919 s
bootstrap 95% interval: [77.844, 95.959] s
ate and drank: 65.0%
```

The selected checkpoint improves disjoint mean lifetime by 30.6%, and its
bootstrap lower bound exceeds the initial mean. This is credible learning
evidence, but it is not final qualification: the declared gate requires at
least 100 test episodes, three training seeds, and 70% of episodes containing
both eating and drinking. The current resource-mechanism result is five
percentage points short.

A continuation probe loaded that selected checkpoint and reached 102.861
seconds on its validation stream. On the same 20-episode disjoint stream used
above, the continuation best produced:

```text
mean lifetime: 93.894 s
bootstrap 95% interval: [83.619, 104.339] s
ate and drank: 70.0%
```

This clears the numerical single-run gates against the original 66.570-second
step-zero result. It still needs 100 test episodes and three independent
training seeds before qualification.

The same random and continuation-selected checkpoints were then evaluated in
fresh processes on one identical 100-episode disjoint stream (`seed=10001`):

```text
random step-000000: mean 67.050 s, 95% CI [63.768, 70.696], ate+drank 26.0%
continuation best:  mean 97.110 s, 95% CI [92.492, 101.456], ate+drank 75.0%
```

The selected policy improves mean lifetime by 44.8%, its lower confidence
bound exceeds the random mean, and the resource mechanism exceeds 70%. This
qualifies that one policy on the full held-out episode count. Release-level
survival qualification still requires two more independent training seeds.

Two follow-up robustness configurations exposed update variance rather than
being folded into a success average:

```text
seed 113, four worlds/update: 47.837 -> 54.300 validation (+13.5%), failed
seed 127, eight worlds/update: 80.162 -> 89.699 validation (+11.9%), failed
seed 131, eight worlds/update, entropy coefficient 0.0005:
  73.124 -> 73.124 validation (+0.0%), failed
```

The pooled runs often reached 100-second stochastic training lifetimes while
their deterministic actor means regressed. Lowering the entropy bonus did not
repair that divergence. Inspection found that recurrent chunks were optimized
in trajectory order on every PPO epoch, giving terminal chunks systematic
last-update influence. The trainer now deterministically shuffles intact chunks
with the dedicated rollout RNG between epochs; time order inside each LSTM
chunk is unchanged.

The first corrected run used eight pooled worlds, the original entropy
coefficient, and seed 137. Validation improved from 75.987 to 101.587 seconds
after 84,933 joint steps. Fresh-process evaluation on the same 100 disjoint
environment seeds produced:

```text
random step-000000: mean 74.886 s, 95% CI [71.401, 79.014], ate+drank 37.0%
shuffled best:      mean 100.992 s, 95% CI [97.732, 104.423], ate+drank 92.0%
```

That is a 34.9% held-out gain, with the lower confidence bound above the random
mean and the resource mechanism comfortably above 70%. It is the first fully
qualified seed using the corrected updater. Two additional corrected training
seeds are still required before calling the stage robust.

Corrected seed 139 did not replicate the result within the same 12-iteration,
eight-world budget. Its validation best rose from 74.049 to only 78.274 seconds
(+5.7%), with a 50% eat-and-drink rate. Stochastic training lifetime reached
96.5 seconds while actor-mean validation remained weak. This run is retained as
a failed robustness trial. Deterministic chunk shuffling fixes update-order
bias, but variance-to-mean transfer still needs a controlled remedy before the
remaining survival seeds and later curriculum stages can be qualified.

A bounded-exploration trial then capped the actor log standard deviation at
`-1.0` (standard deviation at most 0.368) for corrected seed 149. Validation
improved from 52.862 to 59.062 seconds after 53,613 joint steps (+11.7%). The
actor entropy stayed at 0.837877, the exact value implied by the cap, and
stochastic training lifetime remained close to actor-mean validation. This
removed the large stochastic-to-deterministic gap but suppressed exploration
too strongly to meet the 20% survival gate. The run is retained as another
failed robustness trial.

The intermediate `log_std_max=-0.5` setting retained more exploration while
constraining variance. Corrected seed 151 selected a 106.111-second validation
checkpoint from a 59.975-second baseline after 80,267 joint steps. Fresh-process
evaluation on 100 disjoint environment seeds produced:

```text
random step-000000: mean 57.282 s, 95% CI [55.507, 59.159], ate+drank 5.0%
bounded best:       mean 108.118 s, 95% CI [104.215, 111.679], ate+drank 94.0%
```

That is an 88.7% held-out gain, and the lower confidence bound exceeds the
random mean. Seed 151 is the second fully qualified corrected survival seed.
One more independent seed is required for the predeclared robustness claim.

Corrected seed 157 independently selected a 117.624-second validation
checkpoint from an 80.749-second baseline after 85,212 joint steps.
Fresh-process evaluation on the same 100 disjoint environment seeds produced:

```text
random step-000000: mean 70.278 s, 95% CI [66.757, 74.168], ate+drank 32.0%
bounded best:       mean 117.138 s, 95% CI [115.596, 118.496], ate+drank 99.0%
```

That is a 66.7% held-out gain. Together, corrected seeds 137, 151, and 157 all
pass the survival lifetime, confidence, and resource-mechanism gates. The
survival stage therefore meets the predeclared three-seed robustness standard.

## Later-stage implementation probes

The first fully corrected competition lineage retrained directly from the
qualified survival seed-157 checkpoint for twelve four-ecosystem updates with
seed 263 and `log_std_max=-0.5`. Fixed held-out evaluation over 50 corrected
ecosystems (`seed=10001`) produced:

```text
random:     51.641 s, 95% CI [50.734, 52.640], ate+drank 2.5%
transfer:   84.094 s, 95% CI [80.765, 87.578], ate+drank 69.0%
curriculum: 80.765 s, 95% CI [77.738, 83.849], ate+drank 64.0%
```

The selected policy is 56.4% above random, all four identities consume
resources, its maximum identity advantage is 4.5%, and evaluation records
38,900 Avian contacts. It regresses 4.0% from transfer and retains only 68.9%
of the qualified solo lifetime, however, so
`runs/ecosystem-competition-ppo/corrected-curriculum-263` is a failed
retention/curriculum-learning lineage rather than qualifying evidence.

A separate seed-269 lineage at `log_std_max=-1.0` never improved over its
step-zero validation policy. Fresh held-out evaluation of its selected
checkpoint reproduces the 84.094-second transfer result (71.8% solo retention),
so it contains no learned competition improvement. A seed-271 low-variance
continuation from the seed-263 checkpoint selected 78.693 seconds held out,
2.6% below its input and 67.2% of solo lifetime. These runs remain ignored
failed diagnostics:

```text
runs/ecosystem-competition-ppo/corrected-low-variance-269
runs/ecosystem-competition-ppo/corrected-continuation-271
```

The corrected optimizer and environment expose that the previous optimization
recipe does not reliably preserve solo performance. Predator-prey must not use
these checkpoints as qualified curriculum inputs.

The competition probe directly loaded the continuation checkpoint. Its
eight-ecosystem disjoint comparison improved mean lifetime from 68.309 to
72.671 seconds and recorded 4,130 Avian agent-contact observations. The 6.4%
gain is below the declared 15% gate, and eating-plus-drinking completion fell
from 43.8% to 34.4%. Competition is mechanically valid but unqualified.

The first explicit spawn-rotation run loaded the qualified seed-157 survival
policy and trained for 54,215 joint steps. On 50 disjoint ecosystems, the
random competition policy, transferred policy, and selected curriculum policy
measured:

```text
random:     51.932 s, 95% CI [51.028, 52.916], ate+drank 3.0%
transfer:   77.988 s, 95% CI [75.106, 81.346], ate+drank 58.5%
curriculum: 79.202 s, 95% CI [76.486, 81.842], ate+drank 75.5%
```

The curriculum policy is 52.5% above random. All four stable bunny identities
consumed resources, the max-to-min identity lifetime spread was 9.0%, and
39,910 Avian contact observations proved active blocking and pushing. The run
passes the random and mechanism gates but retains only 67.6% of the 117.138-
second solo predecessor result. It therefore remains unqualified under the
literal 80% retention gate.

Two `log_std_max=-1.0` continuations improved deterministic transfer. The first
reached 89.891 seconds held out; the second reached 92.228 seconds, still 1.482
seconds below the retention threshold. A four-update continuation with
per-update validation selected a policy at its first update. On the same 50
disjoint ecosystems it produced:

```text
curriculum: 99.627 s, 95% CI [96.290, 102.579], ate+drank 92.0%
```

This is 91.3% above the 51.932-second random baseline and retains 85.1% of the
117.138-second solo result. All four identities consumed resources, the largest
identity mean advantage above the population mean was 13.6%, and evaluation
recorded 44,540 Avian contacts. This lineage passes every competition gate.
Two independent training lineages are still required before calling the stage
robust.

The predator-prey smoke probe loaded the competition bunny checkpoint and
created a separate shared fox learner. Four disjoint ecosystems recorded 55
feeding events, 6 predations, 12.360 water units, and 2,768 agent contacts.
No fox both hunted and drank in that small set, so no learning claim is made.

The obstacle smoke probe loaded both predator-prey policies. Four disjoint
ecosystems recorded 318.4 thorn hit points of damage, 972 agent contacts, 34
feeding events, 2 predations, and 24.080 water units. This proves the hazard
path is active, not that agents have learned avoidance.

## Interactive checkpoint rendering

The render-enabled survival target passed its 12 unit tests and strict scoped
Clippy gate. The real continuation checkpoint launched a 1280x720 Vulkan window
and remained active for the bounded runtime probe:

```sh
timeout 60s cargo run --example ecosystem-survival --features render -- \
  watch --checkpoint runs/ecosystem-survival-ppo/continuation-probe-105 \
  --seed 101 --speed 8
```

The viewer uses deterministic mean actions and retains one LSTM memory per
living agent. The visual projection is read-only; it does not add global world
coordinates to actor observations. The HUD labels playback `EVAL` and shows
the stage, checkpoint, step time, playback speed, agent/resource counts,
average physiology, and the configured actor and critic learning rates.

No competition, predator-prey, obstacle, or video stage is yet claimed as
qualified. Interactive rendering is implemented but does not itself establish
learning.

## Interactive perception and tuning probe

On 2026-07-29, the survival demo ran with one rollout episode, one or two PPO
iterations, and a 200-step horizon. The rendered window was inspected at its
native desktop resolution. Agent 0 showed all 36 post-physics perception
sectors. Empty sectors reached sight range, while food, well, and boundary hits
stopped at their stored distance and used distinct colors.

The Inspector-egui HUD showed the enabled ray checkbox, agent selector, reward
weights, starting health, need drain, damage, timeout, and explicit visible
world restart control. The focused all-feature example suite passed 42 tests.
The headless suite passed 34 tests. Simulation tests independently verify ray
projection, profile validation, reset health and horizon changes, and exact
reward-event composition. This probe verifies presentation and control wiring;
it does not establish a learning-rate claim for any tuning profile.

The final repository `cargo test` passed 53 library tests and one documentation
test. `cargo clippy --all-targets --all-features -- -D warnings` passed. The
personal-lint gate retained only its documented CartPole lifetime diagnostics,
four pre-existing ecosystem 100-line diagnostics, and missing Wayland
pkg-config dependency. The new ray and tuning surfaces add no personal-lint
diagnostic.

On 2026-07-29, the visual HUD was inspected again after adding the perception
count control. The slider displayed the full validated range from 1 through 36
inside the experiment section that opens at startup. The default world rendered
all 36 sectors. A focused reduced-profile test
proved that four rays select forward, left, rear, and right tensor slots, retain
the 339-value observation width, and leave a 30-degree target outside every
fixed-width ray. The all-feature example suite now passes 44 tests, and the
headless suite passes 35 tests. Training and evaluation metrics record
`experiment/perception_ray_count` beside each applied profile.

## Checkpoint video pipeline

The survival run produced a temporary six-segment smoke video using shortened
one-second durations. This validated the same default sequencing path without
representing the output as qualifying media:

```sh
cargo run --example ecosystem-survival --features render -- video \
  --checkpoint runs/ecosystem-survival-ppo/continuation-probe-105 \
  --output /tmp/ecosystem-survival-video-smoke.mp4 --fps 1 \
  --intro-seconds 1 --checkpoint-seconds 1 --outro-seconds 1 --seed 201
```

`ffprobe` confirmed six frames, H.264, 1280x720, and `yuv420p`. The generated
manifest recorded exact frame ranges and SHA-256 hashes for opening best,
`step-000000`, `step-008045`, `step-014823`, `step-021754`, and closing best.
A contact sheet was inspected at original resolution. Every frame showed the
top-down environment and HUD with the correct checkpoint label. The video
window uses a 1.0 scale-factor override so desktop HiDPI settings cannot change
the output dimensions.

## Historical energy-aware binocular survival proof

On 2026-07-29, the final 340-input, three-action checkpoint profile was trained
from a fresh process with the no-render command below. Checkpoint profile 2 is
incompatible with earlier 339-input, two-action ecosystem checkpoints.

```sh
cargo run --no-default-features --release --example ecosystem-survival -- \
  train --run-id need-weighted-lr3-20260729
```

The fixed validation-seed curve was:

```text
iteration 0: 38.087 s
iteration 1: 40.431 s
iteration 2: 38.462 s
iteration 3: 40.281 s
iteration 4: 41.719 s, selected best
iteration 5: 40.450 s
iteration 6: 40.781 s
```

The selected checkpoint gained 9.5% within four updates and the run completed
six updates in 38,521 joint
steps. A prior random-layout candidate regressed from 27.950 to 25.669 seconds.
A rotated route with resources farther away stayed flat at 27.656 to 27.644
seconds. Placing the first food and well inside the initial binocular search
route produced the fast learning curve. Food and absorbed-water rewards now
use the reserve before consumption: 125% at or below 10%, 100% through 50%,
90% through 75%, 75% below 90%, and zero from 90% upward. The base defaults are
8 per food event and 10 per absorbed water unit. The final action mapping
converts the signed network output to throttle: `-1` stops, `0` uses half
throttle, and `1` uses full throttle. Physical velocity, body heading, and the eye-cone centerline
agree, so the policy can move while turning and cannot move backward while
looking forward. Agent, food, well, tree, rock, and thorn colliders are
rectangles or squares with the same dimensions as their rendered visuals.

Fresh processes then evaluated the random step-zero checkpoint and selected
checkpoint on 100 disjoint seed-907 episodes:

```sh
cargo run --no-default-features --release --example ecosystem-survival -- eval \
  --checkpoint runs/ecosystem-survival-ppo/need-weighted-lr3-20260729/checkpoints/step-000000.mpk \
  --episodes 100 --seed 907
cargo run --no-default-features --release --example ecosystem-survival -- eval \
  --checkpoint runs/ecosystem-survival-ppo/need-weighted-lr3-20260729/best.mpk \
  --episodes 100 --seed 907
```

```text
random:  39.124 s, 95% CI [38.566, 39.752], ate+drank 100.0%
learned: 42.445 s, 95% CI [41.530, 43.474], ate+drank 100.0%
```

The disjoint gain is 8.5%. Food events increased from 104 to 118. The learned
evaluation consumed 136.481 water units and recorded 4,619.696 overconsumption
damage, down from 7,196.906 for the random policy. Training reduced
overconsumption deaths from 27 to 18 while increasing mean survival. The other
learned deaths were 18 from starvation and 64 from combined deprivation. These
diagnostics prove that agents can traverse the square food sensors, consume
both resources, and improve under the need-weighted reward.

The exact rendered command completed the same six-update visual demo with
binocular rays enabled, two eye origins, the learning graph, optimizer rates,
and live controls:

```sh
cargo run --example ecosystem-survival
```

The inspected captures are [live training](../../ai/bmad-output/implementation-artifacts/screenshots/ecosystem-training-live.png)
and [completed training](../../ai/bmad-output/implementation-artifacts/screenshots/ecosystem-training-complete.png).
The focused tests cover translation-only drain, free body turn and gaze,
forward binocular limits, paired live ray counts, exact fullness boundaries,
overconsumption damage and attribution, profile persistence, HUD pointer
capture, world drag input filtering, forward-only throttle, and traversable food
pickup sensors and every need-reward zone boundary. The final headless example
suite passed 50 tests. The final all-feature example suite passed 61 tests.
`cargo test` passed 54 library tests
and one documentation test, with one documentation test ignored. Strict
all-target, all-feature Clippy passed with warnings denied.

The personal-lint workflow reported no candidate-local diagnostic. Its first
stage remains blocked by 15 pre-existing CartPole lifetime diagnostics and
three pre-existing ecosystem 100-line diagnostics. Its Dylint stage remains
blocked because the isolated tool environment lacks `wayland-client.pc`.
