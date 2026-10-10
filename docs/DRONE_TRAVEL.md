# Drone travel lesson

Status: reusable environment, PPO training, and frozen inference implemented.
The standalone scene displays an unqualified RL trial. No travel checkpoint is qualified.
Historical tracking weights are imitation-trained and do not qualify this lesson.

## Interrupted run recovery

The seed-19 endurance trial stopped with `No space left on device` after 1,520
complete PPO updates. Its saved update-1,500 selection survives all five roots,
but fails heading, settling and return gates. The stopped artifacts remain in
`runs/drone-travel/endurance-seed19-20261009`.

The recovery run uses the same immutable binary, seed, qualified recovery checkpoint
and 2,400-update budget per stage. It writes to the separate directory
`runs/drone-travel/endurance-seed19-recovery-20261010` under
`bevy-gym-travel-endurance-seed19-recovery-20261010.service`.
This replays training from initialization with fresh optimizers; it does not restore
optimizer state from update 1,520. The original recipe retains the learner and
optimizer across stages only after their fixed selection gates pass.

The [recovery evidence](./progress/drone-travel-storage-recovery.json) records source,
binary and prerequisite hashes, the launch command and compared optimizer prefix.
Inspect the service and current selection records before reporting progress.
The first fifteen optimizer records and update-20 selection report match the original
exactly. The update-20 checkpoint bytes differ, so byte reproducibility is not established.
Held-out and independent-run evidence remain required.

The recovery run completed all 2,400 updates and exhausted its budget.
All five final selection episodes survived 1,000 actions. Seed 0 failed with
0.279 radians of heading error and no final settled streak; the limit is pi/12.
Independent held-out endurance evaluation also failed despite all 32 episodes surviving.
Its worst heading error was 0.659 radians, and its shortest final settled streak was zero.

The [completion record](progress/drone-travel-recovery-final.json) and
[complete held-out suite](progress/drone-travel-recovery-final-held-out.json) preserve
checkpoint and executable hashes, terminal state and all 128 episodes. Near, far and
fast results measure untrained transfer; those stages were never reached in training.
The seed-23 batched trial continues independently. No travel checkpoint is qualified.

## Batched training trial

The [seed-23 trial](./progress/drone-travel-batched-trial.json) uses the batched PPO
optimizer at `e8b7941`, with a 24,000-update limit per stage. It transfers the same
qualified recovery actor and inserts the same zero-weight heading input. The critic,
Adam optimizers, counters and seed streams start fresh. Rewards, physical rules,
observations, actions and promotion gates remain unchanged.

The ten-update smoke run completed 5,120 RL transitions and failed selection.
The first ten optimizer records in the larger trial match that smoke run exactly.
This does not establish checkpoint-byte reproducibility or qualification. The older
seed-19 recovery run exhausted its budget; its artifacts remain preserved.

Before reporting progress, inspect
`bevy-gym-travel-batched-seed23-20261010.service` and
`runs/drone-travel/batched-seed23-20261010`. Held-out evaluation and independent
run evidence remain required before promotion is reported.

## Seed-23 endurance selection and independent evaluation

The batched seed-23 run passed its five-root endurance selection at update 1,860,
after 952,320 transitions. All five episodes survived, with at least 244 final
settled actions. The mean return was 854.3. The trainer then continued into near travel.

Independent evaluation through the same immutable executable failed endurance.
All 32 held-out episodes survived, but maximum heading error reached 0.542 radians
against pi/12 permitted. The shortest final settled streak was zero. The
[selection record](progress/drone-travel-endurance-seed23.json) and
[complete 128-episode suite](progress/drone-travel-endurance-seed23-held-out.json)
preserve checkpoint identity and every case.

Endurance remains unqualified. The immutable batched trial advances after selection
alone. The current trainer requires separate-process validation before each transition.
The older process retains its historical behavior; its transitions do not establish qualification.

## Independent promotion validation

After a five-root selection passes, the trainer launches the same executable in a
separate process. The process loads the saved policy without a learner or exploration
sampler. It evaluates endurance through the proposed stage, in curriculum order.
The parent checks the checkpoint SHA-256, recomputes every physical gate and checks
that the exit status agrees. It saves the child stdout as `<stage>-<update>.promotion.json`.

Each stage uses 32 ordered validation roots, from `u64::MAX - 1024` through
`u64::MAX - 1055`. Training excludes these high-half roots. They differ from the
five selection roots and the final held-out roots `u64::MAX - 1` through
`u64::MAX - 32`. Repeated validation can influence stage transitions, so final
held-out evaluation remains a separate release requirement.

If validation fails its gates with exit status 1, training continues in the same
stage. If loading, process execution, report parsing, checkpoint identity or status
agreement fails, training stops before any transition. Budget exhaustion cannot
promote a stage. Successful transitions retain the existing learner and optimizers.

To inspect a frozen checkpoint through near travel:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --lesson travel \
  --evaluate-promotion path/to/checkpoint.mpk --travel-stage travel-near
```

The command emits one compact JSON report and exits 0 only when every prerequisite
passes. It requires an explicit lesson and stage. It creates no training output.

The private `travel-promotion-v1` report has exactly three required members:
`schema`, a string equal to `travel-promotion-v1`; `checkpoint_sha256`, a string
matching the saved bytes' 64 lowercase hexadecimal SHA-256 digits; and `stages`,
an array containing the exact curriculum prefix. Stage tags are `travel-endurance`,
`travel-near`, `travel-far` and `travel-fast`. Each stage object requires `stage`
and `episodes`; the episode array contains exactly the 32 ordered roots above.

Each episode requires `seed` (`u64`), `steps` (`u16`), `reward` (`f64`),
`survived` (boolean), `first_arrival` (null or a nonzero `u16`),
`settled_actions` (`u16`), `final_distance` (`f32`, metres),
`final_heading_error` (`f32`, radians) and `final_speed` (`f32`, metres per second).
Missing or unknown object members fail parsing. Missing arrival differs from null;
null parses but fails qualification. Physical measurements must be finite and
nonnegative before the existing survival, arrival, settling and return gates apply.
JSON decoding retains these numeric types; consumers must preserve full `u64` seed
precision. Object member order is irrelevant; stage and episode array order is required.

The [promotion evidence](progress/drone-travel-promotion.json) records tests,
coverage gaps and the immutable bounded PPO probe. The probe uses seed 23 and
1,900 updates per stage under
`bevy-gym-travel-promotion-seed23-20261010.service`, with output in
`runs/drone-travel/promotion-seed23-20261010`. Inspect its current state and reports
before claiming that a real training transition exercised this gate. No travel
checkpoint is qualified, and browser 3v3 remains unfinished.

## Run the travel scene

```sh
nix develop --command cargo run --features robots --example drone-travel-scene
```

For browser playback:

```sh
nix develop --command scripts/build_drone_skills.sh
python3 -m http.server 8000 --directory robot-web/skills-dist
```

Open [the travel trial](http://localhost:8000/travel/). Run or Space starts and pauses
playback; Step or N applies one policy action; Reset or R restarts seed 42 and clears
recurrent memory. The scene runs the same `TravelTask` near-stage environment and 13-feature
encoder as training.

The actor chooses every motor fraction. The camera and goal
marker read observations only. The marker's radius is 0.5 m; its arrow shows the
requested heading. Rendering does not move the physical body or select actions.

The embedded `docs/progress/drone-travel-trial.mpk` is original seed-11 near-travel
update 120, after 61,440 travel transitions from the qualified recovery actor.
Its SHA-256 is `0f5a36039389e68c66281029d77eb2d7397bba8483e4a1f366092f512397d5f7`.
It failed selection, and its full run later exhausted 600 updates without promotion.
The scene labels it `Unqualified RL` and derives the displayed identity from loaded
bytes.

The seed-42 episode survives 1,000 actions but finishes approximately 0.97 m
from the goal, with 150.7 degrees heading error and 1.60 m/s speed. Completion is
not qualification. Load and inference errors prevent further actions; no fallback
controller exists. Training remains a separate command below.

[Scene evidence](progress/drone-travel-scene.json) records native/WASM action parity,
coverage gaps, desktop/mobile recordings, inspected contact sheets, and licensed
asset provenance. Mobile checks use a 390-pixel browser viewport, not a physical
mobile device. Native window interaction remains unverified.

## Environment

`DroneTravel` uses the same private physics world as `DroneHover::disturbed()`.
A `DroneDestination` fixes the position and heading for the environment's lifetime.
The policy must choose all four motor fractions. The destination supplies task
information; it never generates a route, modifies an action, or moves the body.

Construct a destination with
`DroneDestination::try_from((position, heading))`, using `Vec3` and `Vec2`, respectively.
Position coordinates
are metres in the existing right-handed world: +X right, +Y up, and -Z forward.
X and Z must be strictly inside `(-10, 10)`; Y must be strictly inside `(0, 10)`.
All coordinates must be finite. Invalid position returns
`InvalidDroneDestination::Position` before heading validation.

Heading components correspond to world X and Z. Every finite nonzero heading is
accepted, including subnormal and maximum finite magnitudes. Normalization
preserves direction; input magnitude is discarded. Non-finite or zero headings
return `InvalidDroneDestination::Heading`. No position is clamped. Validated fields
are private and immutable.

The observation contains a read-only `DroneObservation` and the destination.
Position is measured in metres, linear velocity in m/s, and angular velocity in
rad/s. Orientation is a unit quaternion from body coordinates to world coordinates.
The neural-network encoding below fixes scaling and heading representation.
The environment does not expose a mutable physics world.

Actions retain the hover contract: four finite motor fractions in `[0, 1]`, ordered
front-left, front-right, rear-right, rear-left. Each command lasts 20 ms, with four
5 ms physics substeps. Contact or flight-region exit terminates the episode.
Subsequent actions return the unchanged terminal observation and zero reward.

## Reset and reward

Construction matches disturbed reset seed zero. Explicit reset seeds restart the
existing disturbed reset stream; omitted seeds continue it. Reset retains the
destination. The [recovery guide](DRONE_RECOVERY.md) specifies the position, tilt,
heading, and velocity distribution. Arrival does not end the episode automatically.
Use `TimeLimit` to impose an action horizon.

Continuing reward is:

```text
upright   = clamp((1 + body_up · world_up) / 2, 0, 1)
alignment = clamp((1 + body_forward · desired_heading) / 2, 0, 1)
reward    = upright × alignment / (1 + distance_squared / (1 m²))
```

The dot products, factors, and reward are dimensionless. `distance_squared` is
measured in square metres. Body forward is -Z. Desired heading is the normalized
world vector `(heading.x, 0, heading.y)`. Termination earns zero. This is a local
reward definition, not a reproduction of another environment.

[Flightmare's quadrotor environment][flightmare], inspected on 2026-10-09,
separates physical stepping, reset randomization, and state-based rewards. Its
coordinate system and reward differ from this lesson. Travel reuses this project's
existing solver and reset contract.

## Verification and remaining work

Run the public environment tests:

```sh
nix develop --command cargo test --no-default-features --features robots --test drone_travel
```

Tests compare every physical snapshot against disturbed hover under identical
seeds and action fixtures. They cover coordinate boundaries, heading normalization,
validation order, terminal behavior, and seeded resets. Internal tests distinguish
position, heading, and uprightness reward factors. Fixtures test physics only;
they are not autonomous example controllers or learned qualification evidence.

[Validation evidence](progress/drone-travel-environment.json) records native/WASM
results and coverage. All 101 measured lines, including tests, and 16 branch outcomes
were hit. The unreachable normalization-error region is recorded separately.
Strict native/WASM checks pass; changed-line personal lint is clean. Strict personal
lint still fails on the unchanged repository backlog.

The shared recurrent PPO collector now accepts the environment's typed observation.
A [collection regression](progress/drone-travel-collector.json) checks 512 deterministic
one-step travel episodes and cleared per-episode memory. Its body-only encoder is
a test fixture, not the travel actor contract. Existing hover/recovery collection
and qualification remain green. Collector coverage records 226/226 lines, including
tests, and 6/6 branch outcomes across both instantiations. Two existing critic-error
propagation regions remain unhit; the evidence records their exact locations.

## Training and frozen inference

Run preparatory endurance followed by three travel stages, with a finite budget per stage:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --lesson travel --seed 11 --updates 600 \
  --output runs/drone-travel/seed11
```

The source is the qualified PPO recovery checkpoint
`docs/progress/drone-recovery-transfer.mpk`, SHA-256
`7c2b9a6f2a0288faa27676d1514848710f4d5c24cd77505e75cf4990576a12b1`.
The trainer verifies its bytes against the embedded reference before transfer.
It copies the recurrent actor and adds one zero-weight heading input. The critic,
optimizers, and sampling streams start fresh. A `transfer.json` record precedes
training. Tests prove unchanged actions and recurrent memory at transfer.

The native `transferred_heading_input_learns_from_ppo` test starts with identical
actions and memory for opposite heading inputs. After one PPO update, both depend
on the heading input. Its samples and advantages are isolated optimizer fixtures.
This establishes trainability of the inserted input, not travel qualification.
The [evidence record](./progress/drone-travel-heading-learning.json) preserves the
test, browser checks and coverage boundaries.

Each update collects eight lanes of 64 actions through `TravelTask` and the shared
`DroneTravel` implementation. Policy weights remain fixed during collection;
PPO updates follow the 512 transitions. Each lane owns its recurrent memory.
Stage changes reset lanes and memory while retaining the trained agent and optimizer.

Evaluate a saved candidate without training:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --lesson travel --seed 42 \
  --evaluate-checkpoint runs/drone-travel/seed11/travel-endurance-20.mpk \
  --travel-stage travel-endurance
```

The command requires an existing 13-input checkpoint and creates no output files.
Missing, incompatible, or invalid inference fails visibly; there is no fallback.
`--travel-stage` accepts `travel-endurance`, `travel-near`, `travel-far`, or
`travel-fast`, defaults to
`travel-near`, and requires `--evaluate-checkpoint`. Evaluation requires
`--lesson travel` and conflicts with `--updates` and `--initialize-from`.
A single episode explicitly reports that it does not establish qualification.

### Held-out evaluation

Evaluate a frozen selected checkpoint in a new process:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --lesson travel \
  --evaluate-held-out runs/drone-travel/seed11/travel-endurance-20.mpk \
  > runs/drone-travel/seed11/held-out.json
```

The fixed `travel-v1` suite evaluates endurance, near, far, and fast in that order.
Each stage uses 32 roots, descending from `u64::MAX - 1` through `u64::MAX - 32`.
These roots are excluded from training and selection. Each episode starts with fresh
recurrent memory. Every stage applies its existing full-horizon gates; failure in
one stage does not skip later stages. No optimizer or exploration sampler is created.

The evaluator reads the checkpoint once and hashes the exact bytes moved into
validated loading. It uses [sha2 0.10.9](https://docs.rs/sha2/0.10.9/sha2/), already
present in the lockfile and now a direct development dependency for this example.
The command writes JSON to stdout and creates no files itself. The shell command
above saves stdout. A failed behavioral gate still emits the complete report and
returns exit status 1; all gates passing returns 0. Load or inference errors fail
visibly. Missing and incompatible checkpoints produce no report.

`--evaluate-held-out` requires `--lesson travel`. It conflicts with explicit
`--seed`, `--updates`, `--initialize-from`, `--evaluate-checkpoint`, and `--travel-stage`.
The default seed is unused. This mode always runs the full fixed suite.

The report contains:

- `mode`: the string `held-out-evaluation`.
- `suite`: the string `travel-v1`; changes to these distributions, roots, or gates
  require a new suite identity.
- `checkpoint`: the supplied path; `checkpoint_sha256`: 64 lowercase hexadecimal
  characters identifying the loaded bytes.
- `evaluation`: `passed` only when every stage passes; otherwise `failed`.
- `qualification`: `not established by evaluation alone`.
- `stages`: four ordered objects containing `stage`, `evaluation`, and the 32
  ordered `episodes` in the measurement format below.

The report establishes behavioral results for exact bytes. It cannot establish how
those bytes were trained. Qualification also requires the recorded RL training
source, source checkpoint, run manifest, and checkpoint chain. Use these final roots
for independent evaluation after selection; do not feed their episodes into training
or tune promotion thresholds against them. No current travel candidate is qualified.

### Optimizer records

Each travel training invocation creates or truncates `optimization.jsonl` in its
output directory. Use a fresh directory to preserve earlier runs. Each completed
PPO update writes one UTF-8 JSON object followed by LF, then flushes the writer
before selection or the next rollout. This includes updates between checkpoint
selections. Flush makes data available to readers; it does not promise power-loss
durability. A write failure can leave a partial final line.

Every object contains these required members; none may be absent or null:

- `lesson`: one of `travel-endurance`, `travel-near`, `travel-far`, `travel-fast`.
- `update`: positive integer index within the stage, starting at one.
- `optimizer_steps`: cumulative integer actor/critic minibatch update count.
- `optimizer_updates`: positive integer minibatch update count for this call.
- `valid_samples`: positive integer count of distinct rollout transitions, currently 512.
- `actor_loss`: finite number, the clipped surrogate loss including the entropy term.
- `critic_loss`: finite number, half the mean squared value residual.
- `entropy`: finite number, pre-tanh diagonal Gaussian differential entropy in nats.
- `approximate_kl`: finite number, mean old-minus-current action log probability in nats.
- `actor_learning_rate` and `critic_learning_rate`: finite numeric optimizer step sizes.

Losses and step sizes are dimensionless under this lesson's dimensionless reward
and feature encoding. Metrics average valid timesteps across optimizer passes.
Actor loss, entropy and the sampled KL estimate may be negative; these values are
preserved. Object member order is unspecified. Rows follow stage and update order;
the update index restarts when a stage changes. No row is an evaluation or promotion.

NaN and either infinity stop the trainer with `non-finite PPO metric: FIELD` before
writing that record. Creation, write and flush errors also stop training visibly.
The completed optimizer update is not rolled back, and no subsequent rollout or
selection runs after the error. The existing checkpoint cadence and qualification
gates are unchanged. [Validation](progress/drone-travel-progress.json) records
error-path tests, native/WASM results and measured coverage.

A bounded [critic-warmup comparison](progress/drone-travel-optimization-probe.json)
used zero or 20 critic-only updates before 20 PPO updates, with training seed 11.
Both final policies failed all five survival evaluations. Mean returns were 208.7
without warmup and 244.7 with warmup, below the unchanged source's 279.2.
Warmup is not enabled in the trainer. This probe used selection seeds, never the
held-out suite, and does not establish improvement or qualification.

### Observation encoding

The actor and critic each receive 13 dimensionless features. In order, these are
body-frame destination displacement XYZ divided by 2 m, body-frame world-up XYZ,
body-frame linear velocity XYZ divided by 2 m/s, body-frame angular velocity XYZ
divided by 2 rad/s, and signed heading error divided by pi radians.
Heading error is `atan2((forward × desired).y, forward · desired)`; positive values
rotate around world +Y. The last feature lies in `[-1, 1]`. An exactly vertical
forward vector has undefined horizontal heading and explicitly encodes positive zero.
The up vector and recurrent memory still describe tilt. Outputs retain the four
motor fractions and ordering specified above; no controller converts goals to actions.

### Sampling and promotion

`TravelTask` samples one immutable destination per episode. Explicit seeds restart
both streams at episode zero; omitted seeds advance the wrapping episode index.
`SeedConfig::environment_episode(0, episode)` controls the disturbed body reset.
Stream 1 draws radius, azimuth, height, and heading in that order. Azimuth and heading
use the sampler's float range from minus pi to pi. Position is
`(radius × cos(azimuth), height, radius × sin(azimuth))`; heading is
`(cos(heading), sin(heading))` in world X/Z for near, far, and fast travel.
Endurance instead fixes position `(0, 2, 0)` metres and derives heading once from
the seeded initial body's forward vector projected onto world X/Z. The destination
then remains immutable; it does not follow the body or choose motor actions.
This reset uses one additional temporary physics world, with constant time and space
relative to episode length. The original travel sampling streams are unchanged.

The fixed stage profiles are:

- `travel-endurance`: original hover position and initial heading, first arrival by
  action 500, mean return at least 800.
- `travel-near`: radius 0.5–1 m, height 1.5–2.5 m, first arrival by action 500,
  mean return at least 600.
- `travel-far`: radius 2–4 m, height 1–4 m, first arrival by action 500,
  mean return at least 500.
- `travel-fast`: radius 4–6 m, height 1–5 m, first arrival by action 250,
  mean return at least 500.

Every episode lasts at most 1,000 actions, or 20 seconds. Arrival requires distance
at most 0.5 m and absolute heading error at most pi/12 radians. Every evaluated
episode must survive the full horizon and satisfy both bounds plus speed at most
0.5 m/s for its final 100 consecutive actions. Arrival deadlines are inclusive:
500 actions is 10 seconds; 250 is 5 seconds; 100 settling actions is 2 seconds.

Every twentieth update and the final budgeted update save weights, reload frozen
inference, and evaluate ordered seeds `0, 1, 2, 42, 18446744073709551615`.
All episode gates and the mean-return gate must pass before the next stage.
Exhaustion returns an error and preserves the failed candidate. These selection
results do not replace independent held-out qualification. The command above supplies
its behavioral evaluation; a passing trained candidate remains pending.

Each stage record contains `lesson`, `update`, `passed`, `optimizer_steps`, and
ordered `episodes`. Episode fields are `seed`, `steps`, summed unscaled `reward`,
`survived`, `first_arrival`, `settled_actions`, `final_distance` in metres,
`final_heading_error` in radians, and `final_speed` in m/s. `first_arrival` is a
one-based action index or JSON null when absent. Non-finite floating diagnostics
serialize as JSON null. Inference adds `mode`, `checkpoint`, `stage`, and an honest
`qualification` message; it does not update the checkpoint.

[Training implementation evidence](progress/drone-travel-policy.json) records native,
WASM, CLI, lint, and measured coverage results. No travel policy has qualified.
The seed-11 trial and failed selection records remain under
`runs/drone-travel/seed11-20261009`; its immutable binary and source manifest preserve
the exact launch version. See [current status](EXAMPLE_STATUS.md) before restarting.

Next: investigate the failed trial, qualify a frozen candidate on independent
held-out goals, and add the separate scene with recorded browser playback. The
default curriculum still runs hover and recovery only. Existing hover/recovery
videos do not establish travel competence.

[flightmare]: https://github.com/uzh-rpg/flightmare/blob/master/flightlib/src/envs/quadrotor_env/quadrotor_env.cpp

## Endurance prerequisite experiment

Frozen diagnostics found a transfer gap: the qualified recovery actor terminated
between actions 691 and 729 on all five original-target hover episodes extended to
20 seconds. Its existing 500-action qualification remains valid. The same actor
terminated between actions 592 and 671 on the five near-travel episodes. Update 120
survived all five extended hover episodes but missed the position gate; update 400
terminated on all five. These observations do not isolate every cause of PPO failure.

[Diagnostic and validation evidence](progress/drone-travel-endurance.json) preserves
the measured episode endings and source identity. The added endurance prerequisite
trains the longer horizon before goal changes. It retains the existing reward,
actuators, PPO settings, and every near/far/fast gate. Its success is unproven.
The original trial remains preserved. A separate seed-11 experiment runs under
`bevy-gym-travel-endurance-seed11-20261009.service`, with artifacts in
`runs/drone-travel/endurance-seed11-20261009`. Neither trial establishes travel competence.
