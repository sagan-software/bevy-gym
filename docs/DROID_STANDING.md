# Articulated droid standing

Status: PPO training and frozen inference run through the physical environment.
No standing policy is qualified. The standalone mannequin scene runs frozen RL inference.

Inspect the evaluated seed-17 candidate at PPO update 7,920:

```sh
nix develop --command cargo run --features robots --example droid-standing-scene
```

Build and serve the browser lessons:

```sh
nix develop --command scripts/build_drone_skills.sh
python3 -m http.server 8000 --directory robot-web/skills-dist
```

Open <http://localhost:8000/standing/>. Run starts playback; Step requests one learned
action; Reset restores seed 42 and clears recurrent memory. Space, N and R provide
the same controls. V cycles one, four and sixteen actions per presentation tick.

Each action still advances at most 20 ms of physical time. These rates do not guarantee
a wall-clock frame rate. Playback starts paused and waits for the complete model rig.

Train a new candidate in a new directory:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example droid-standing -- train --seed 7 --updates 600 \
  --output runs/droid-standing/seed7
```

An existing output directory is rejected. Budget exhaustion exits unsuccessfully and
preserves the candidate. Selection runs every 20 updates and at the final update.
Each selection saves `standing-N.mpk`, its `.json` provenance, and
`standing-N.evaluation.json`. The optimizer journal flushes after every update.

Continue a saved RL actor and critic in a new directory:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example droid-standing -- warm-start \
  --checkpoint runs/droid-standing/seed7-recovery-20261010/standing-600.mpk \
  --seed 13 --updates 2400 \
  --output runs/droid-standing/seed13-warm-start
```

Warm start preserves actor and critic parameters. It creates fresh Adam optimizers,
optimizer counters, minibatch ordering, environments, sampling streams and episode
memory. It does not resume optimizer state. The seed controls the new training
streams; it does not claim to initialize the imported parameters. The update budget
and subsequent checkpoint counters describe only the new run.

Metadata, SHA-256, architecture and finite parameters are validated before creating
output. The run retains the exact imported bytes as `initial.mpk` and their validated
seven-member sidecar as `initial.json`. `warm-start.json` emits exactly three required,
non-null members, once each, in this order:

- `schema`: the string `droid-standing-warm-start-v1`.
- `seed`: the unsigned 64-bit seed for the new run.
- `source`: the complete seven-member `droid-standing-v1` source record.

This is an emitted audit artifact with no parser. Preserve it with subsequent
checkpoints to retain the training lineage. Existing output directories are rejected.
Qualification gates remain unchanged.

Evaluate a saved candidate without training or modifying files:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example droid-standing -- evaluate \
  --checkpoint runs/droid-standing/seed7/standing-600.mpk
```

Replace `600` with the saved update number. Omission of `--seed` evaluates all
32 held-out roots. `--seed 42` inspects one episode and cannot qualify a checkpoint.
Failed evaluation prints its complete report and exits unsuccessfully. Missing,
corrupt or incompatible inference stops before an episode, without a fallback.

The native CLI and browser tests reuse `DroidStanding`, the same encoder, model and
evaluator. File-based CLI operations are native-only; its WASM entry point returns
an explicit error. The frozen browser scene uses no optimizer. The reset-only
`droid-standing-contract` example remains available for inspecting physical anchors.

## Frozen episode trace

Inspect every physical boundary without training or modifying the checkpoint:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example droid-standing -- trace \
  --checkpoint runs/droid-standing/seed7-recovery-20261010/standing-600.mpk \
  --seed 42
```

The native command writes JSONL to standard output. The default seed is 42.
It uses the same frozen session as the viewer and retains only the current frame.
Each action comes from the recurrent RL actor. No optimizer runs.

Exit 0 means the trace completed, including a failed standing episode.
It does not qualify the checkpoint.

Invalid checkpoints fail before any record with exit 1. Invalid arguments exit 2.
Inference, nonfinite physical state and output errors stop tracing with exit 1.
An output error can leave completed records and an incomplete final record.
The checkpoint and its sidecar remain unchanged.

The local `droid-standing-trace-v1` emitted profile follows
[RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) JSON and the
[JSON Lines convention](https://jsonlines.org/).
Output is UTF-8 without a byte-order mark. Each compact object occupies one line,
followed by LF, including the last object. There are no empty lines.
This command provides no trace parser or alternate input grammar.

The first object has these required members, once each, in this order:

- `kind`: `"origin"`.
- `schema`: `"droid-standing-trace-v1"`.
- `seed`: the unsigned 64-bit episode seed.
- `source`: the validated checkpoint record, with `schema`, `algorithm`, `seed`,
  `update`, `transitions`, `optimizer_steps` and `sha256`, in that order.
- `horizon`: the integer 1000, measured in policy actions.
- `policy_interval_ms`: the integer 20, the nominal milliseconds per action.

`source` retains the existing `droid-standing-v1` provenance contract.
Its algorithm is `"PPO"`; its SHA-256 is 64 lowercase hexadecimal characters.
The seed is an unsigned 64-bit integer; update and optimizer counts are positive.
`transitions` equals `update` times 512 environment transitions per update.
Decode 64-bit integers exactly. JavaScript `Number` cannot represent every valid seed,
including the captured seed `18446744073709551615`.

Subsequent objects have these required members, once each, in this order:

- `kind`: `"frame"`.
- `step`: the completed action count, starting at 0 and increasing by one through
  at most 1000. Frame 0 records the reset state.
- `status`: `"continuing"`, `"terminated"` or `"truncated"`.
- `torques`: null at reset; otherwise 26 finite dimensionless fractions in inclusive
  [-1, 1], ordered by `DroidActuator::ALL` as documented below.
- `observation`: 204 finite actor features from the current physical boundary,
  ordered by the encoder documented below.
- `segments`: thirteen physical objects, ordered pelvis, torso, head,
  left-upper-arm, left-forearm, right-upper-arm, right-forearm, left-thigh,
  left-calf, left-foot, right-thigh, right-calf, right-foot.

Each segment has these required members, once each, in this order:

- `position`: three finite world coordinates, XYZ in metres.
- `orientation`: four finite quaternion components, XYZW, rotating segment axes
  into world axes.
- `linear_velocity`: three finite world components, XYZ in metres per second.
- `angular_velocity`: three finite world components, XYZ in radians per second.
- `floor_contact`: a boolean indicating active contact with the floor.

No other member is null. The final frame has a terminal or truncated status.
Frame N records the torques applied to frame N-1 and their resulting physical state.
Its observation supplies the next action when the episode continues; recurrent
memory persists between actions. A terminal 5 ms substep can end an action early,
so the nominal interval does not establish exact elapsed time.

The serializer emits finite `f32` values without application rounding.
Decode them as `f32` to preserve their bits, including signed zero and subnormals.
NaN and infinities fail before frame serialization rather than becoming JSON null.

The [capture manifest](progress/standing-trace/capture.json) preserves ten complete
failed selection episodes and compressed JSONL hashes. The source checkpoint at
update 600 terminates after 78 to 96 actions. The warm-start checkpoint at update
200 terminates after 88 to 131 actions. Each set contains three terminal calf-floor
contacts and two terminal pelvis-height violations. These observations do not
establish a single cause of failure or standing qualification.

## Physical model

`DroidStanding` implements `Env` with `DroidAction`, `DroidObservation` and unit info.
The private Rapier 0.36 world contains thirteen dynamic segments, twelve joints and
a fixed floor. Total mass is 70 kg.

Rapier derives each segment's inertia from its
box collider and mass. Every action lasts 20 ms, with four 5 ms physics substeps.
Gravity is 9.81 m/s² downward. Bodies cannot sleep and use continuous collision detection.

The [joint guide](https://rapier.rs/docs/user_guides/rust/joints/) controls Rapier's
joint semantics.
The [force guide](https://rapier.rs/docs/user_guides/rust/rigid_body_forces_and_impulses/)
controls persistent torque application. These APIs provide mechanics; the local
robot dimensions, masses, torque bounds and limits are application choices.
This simplified robot is not a validated human biomechanics model.

Each collider has friction 0.8. Adjacent linked segments cannot collide with each
other; nonadjacent self-contact remains enabled. The floor's upper face is y=0,
with half-width 10 m. Contacts and joint constraints never choose policy actions.
There are no posture motors, damping controllers, animation clips or root movement.

The [collision profile](../src/robots/droid/geometry.rs) records every box centre,
half extent, mass, anchor and angular limit. Coordinates are metres in a common
bind frame: +X right, +Y up, -Z forward. All segment axes align in the bind pose.
A unit quaternion rotates each segment's local axes into world coordinates.

Masses are pelvis 12 kg, torso 24 kg, head 5 kg, each upper arm 2 kg, each forearm
including hand 1.5 kg, each thigh 7 kg, each calf 3.5 kg and each foot 0.5 kg.

## Action contract

`DroidAction::try_from([f32; 26])` rejects the entire action when any fraction is
non-finite or outside inclusive [-1, 1]. It returns `InvalidDroidAction` before
physics can receive the action. Valid fractions, including signed zero, are preserved.
`DroidActuator::ALL` fixes this order:

1. Spine X, Y, Z.
2. Neck X, Y, Z.
3. Left shoulder X, Y, Z; left elbow Y.
4. Right shoulder X, Y, Z; right elbow Y.
5. Left hip X, Y, Z; left knee X; left ankle X, Z.
6. Right hip X, Y, Z; right knee X; right ankle X, Z.

At each substep, the environment clears previous user torques. For each actuator,
torque equals its dimensionless fraction times its Nm bound times its current
parent-local unit axis rotated into world coordinates. The child receives that
world torque and the parent receives its opposite. Zero requests no actuator torque;
passive constraints and contacts still act. Axis directions follow the right-hand rule.

Maximum absolute torque is 80 Nm per spine axis, 8 Nm per neck axis, 30 Nm per
shoulder axis, 20 Nm per elbow, 120 Nm per hip axis, 100 Nm per knee and 60 Nm per
ankle axis. Limits below use Rapier angular solver coordinates in radians, not
an independently defined Euler-angle decomposition. Translation is locked at every joint.

- Spine X/Y/Z: [-0.5, 0.5]. Neck X/Y/Z: [-0.4, 0.4].
- Shoulders X/Y/Z: [-1.8, 1.8].
- Left elbow Y: [-2.4, 0.1]. Right elbow Y: [-0.1, 2.4]. Other elbow axes are locked.
- Hips X: [-1.5, 1.5], Y: [-0.7, 0.7], Z: [-0.6, 0.6].
- Knees X: [-2.4, 0.12]. Other knee axes are locked.
- Ankles X: [-0.65, 0.65], Z: [-0.4, 0.4]. Ankle Y is locked.

## Observation and lifecycle

`DroidObservation::body` returns a read-only `DroidBodyState`. Each state contains
world position in metres, a segment-to-world unit quaternion, world linear velocity
in m/s, world angular velocity in rad/s and an active-floor-contact flag.
Joint-relative rotations and velocities can be derived from linked segment states.
The actor encoding below derives its features from this physical observation API.

`DroidBody::ALL` orders pelvis, torso, head, left upper arm, left forearm, right upper
arm, right forearm, left thigh, left calf, left foot, right thigh, right calf, right foot.
Callers cannot access the physics world or replace its body transforms through this API.

Reset samples one common yaw in ±π radians and pitch/roll in ±0.01 radians, then
lifts the bind pose by 0.03 m. All velocities start at zero. The common transform
preserves coincident joint anchors. `reset(Some(seed))` restarts the reset stream;
`reset(None)` continues it. Reset rebuilds all bodies, contacts and solver state.

A substep terminates when the pelvis falls below 0.5 m, a non-foot segment contacts
the floor, or any segment leaves x/z inclusive [-5, 5] m or y inclusive [0, 3] m.
Non-finite position, orientation or velocity also terminates. The first terminal
substep freezes the observation; later actions return that observation and zero reward.
The environment has no implicit horizon. A trainer must impose its declared time limit.

Continuing reward is `upright * height / (1 + drift + speed)`:

- `upright` is `(1 + torso_up.y) / 2`, clamped to [0, 1].
- `height` is pelvis height in metres divided by 0.99 m, clamped to [0, 1].
- `drift` is squared horizontal distance from the origin in m² divided by 1 m².
- `speed` is squared pelvis speed in m²/s² divided by 1 m²/s².

Termination earns zero. Reward shaping is not a qualification criterion. Standing
promotion thresholds, held-out seeds and perturbation cases must be frozen before
training. The retained checkpoint records and evaluations below do not establish qualification.

## Mannequin provenance

Bind anchors come from the neutral-armour mannequin with dark joints in
`assets/robots/survival/mannequin.glb`. Quaternius released Universal Animation Library
Standard v3.0 under CC0. The [manifest](../assets/robots/survival/manifest.json) preserves
source URLs, licence, original archive hashes and recolouring details.
The current model SHA-256 is
`a0b15a89d3b21bf20ac48f8ac32fa16f20a9f32b170dfacd01a5375d7190d97f`.

The profile rotates the model's +Z-forward bind coordinates by π about Y. Spine uses
`spine_01`; neck uses `neck_01`; shoulders use `upperarm_l/r`; elbows use `lowerarm_l/r`;
hips use `thigh_l/r`; knees use `calf_l/r`; ankles use `foot_l/r`. Collision boxes
approximate the armoured segments.

The viewer projects all thirteen mapped bones from
one physical observation. It caches global bind transforms after scene instantiation
and transform propagation, then removes each segment centre before applying its
physical position and rotation. Unmapped parent bones retain their bind offsets
relative to their nearest mapped ancestor. Rendering never writes physical state.

Missing or duplicate mapped bones prevent playback. A lost mapped transform stops
playback with a visible mapping error. Checkpoint and inference failures also remain
visible and stop actions. The GLTF loader may load authored clips; the viewer starts
no authored clip and installs no animation graph.

## Validation and remaining work

The [evidence record](progress/droid-standing-physics.json) preserves source hashes,
red-test output, native/browser results, coverage, lint gaps and the concurrent travel
run snapshot. Four public droid tests and seven internal physics tests pass. The
browser suite passes 180 tests, including all four public droid contract tests.

Native coverage hits all 439 instrumented droid-module lines, including tests, and
all 30 production branch outcomes. Constant profile data and enum declarations have
no instrumented line counts. Failing assertion branches in tests remain unexecuted.
Strict repository Clippy passes; strict personal Clippy retains unchanged repository
findings. Changed-line personal Rust lint is clean. Two unchanged Nix source-filter
findings remain at `flake.nix:119` and `flake.nix:200`.

The [training evidence](progress/droid-standing-training.json) records the separate
PPO, checkpoint and CLI checks. Training infrastructure does not establish balance.
The [scene evidence](progress/droid-standing-scene.json) records native and browser
checks, screenshots, playback video, source hashes and measured coverage gaps. The
[recording](progress/standing-scene/playback.mp4) shows single-step, reset, speed
selection and the failed candidate falling. Native window interaction remains
unverified. No existing pursuit controller was extended.

Cross-platform policy parity compares all 26 torque outputs across 62 identical
204-value inputs, carrying recurrent memory, with absolute tolerance 0.00001.
Each platform separately matches direct policy inference and physical stepping.
Exact native/browser physical trajectory replay is unverified: seed 42 ended after
62 native actions and 54 browser actions. Initial input differences reached
0.0000002384185791015625 and grew through the episode.

Rapier 0.36 enables `enhanced-determinism`, but its
[determinism contract](https://rapier.rs/docs/user_guides/rust/determinism/) also
requires identical initial values and deterministic calculations outside the solver.
The divergence has not been isolated. No exact physical replay claim is made.

## Actor encoding and PPO

The actor receives 204 dimensionless values. Let `R` be the inverse pelvis rotation,
`p` its world position in metres, `v` its world velocity in m/s, and `w` its world
angular velocity in rad/s. All triples use XYZ order.

The first twelve values are `R * ((0, 0.99, 0) m - p) / 1 m`,
`R * world_up`, `R * v / (2 m/s)`, and `R * w / (4 rad/s)`.
Each remaining segment follows `DroidBody::ALL`, excluding pelvis, and contributes:

1. `R * (segment_position - p) / 1 m`.
2. Its up and forward (-Z) unit vectors rotated into the pelvis frame.
3. `R * (segment_velocity - v) / (2 m/s)`.
4. `R * (segment_angular_velocity - w) / (4 rad/s)`.
5. Its active floor-contact flag, encoded as zero or one.

The velocity features are rotated differences of world velocities, not derivatives
in a rotating frame. Two orientation vectors preserve equivalent quaternion signs.
Encoding borrows the observation and fills one fixed array in O(13) time and space.
No reward, episode outcome or privileged critic information enters the actor.

The actor has a 64-unit LSTM and 26 bounded outputs. The separate 128/64-unit critic
receives the same physical features for this single-agent lesson. Eight independent
lanes retain separate recurrent memory and sampling streams. Each contributes
64 transitions per update; the policy stays fixed during all 512 transitions.
These lanes are independent skill episodes, not a shared competition world.

PPO uses actor learning rate 0.0003, critic learning rate 0.001, discount 0.995,
GAE lambda 0.95, initial log standard deviation -2, entropy coefficient 0.001,
four epochs and minibatches of eight sequences. Training starts from random weights.
No imitation targets, posture controllers or constant-action substitutes are used.
The decoder checks width 26 before validating every finite torque fraction in [-1, 1].

## Frozen qualification gates

Selection roots are `[0, 1, 2, 42, 18446744073709551615]` in that order.
Held-out roots descend from `18446744073709551614` through `18446744073709551583`.
Training roots exclude these partitions. Evaluation starts fresh recurrent memory
for each episode and requests the frozen actor's mean action without optimizer calls.

Every episode must survive 1,000 actions (20 seconds), earn at least 800 total reward,
and finish with at least 100 consecutive stable actions. Each stable action requires:

- Pelvis height in inclusive [0.85, 1.15] m.
- Torso-up projection in inclusive [0.9659258, 1], approximately within 15 degrees of vertical.
- Horizontal pelvis distance from the origin in inclusive [0, 0.5] m.
- Pelvis speed in inclusive [0, 0.25] m/s.
- Both feet in active floor contact.

The evaluator rejects missing, reordered or extra cases and non-finite measurements.
Selection alone cannot qualify a checkpoint. All 32 held-out cases and independent
training provenance must pass. These thresholds were fixed before the retained seed-7
trial; a failed run does not change them.

## Checkpoint record

The required UTF-8 JSON sidecar is an object with exactly seven required members:
`schema` is the string `droid-standing-v1`; `algorithm` is the string `PPO`;
`seed` is an unsigned 64-bit integer; `update` is a positive unsigned 32-bit integer;
`transitions` is an unsigned 64-bit integer equal to `update * 512`;
`optimizer_steps` is a positive unsigned 64-bit integer; and `sha256` is exactly
64 lowercase ASCII hexadecimal characters identifying the weight bytes.

Unknown, duplicate, missing or null members are rejected. Objects cannot replace the
closed string values. Member order is accepted freely; emission follows the order
above. Parsing retains numeric values, validates the derived transition count, and
stores the hash as 32 bytes. Identity validation precedes architecture validation.

The sidecar establishes byte identity and internal consistency. It cannot independently
prove training history or competence. Preserve the launch command, source revision and
patch, source hashes, binary hash, optimizer journal and independent evaluations.
The journal rejects non-finite PPO metrics before JSON serialization or another rollout.

## Retained seed-7 trial

The original seed-7 run stopped with `No space left on device` after 217 complete
optimizer records. Its last saved checkpoint, update 200, failed all five selection
cases and all 32 held-out cases. The travel seed-19 run also stopped on storage
exhaustion after 1,520 complete updates. Its update-1,500 candidate survived all five
selection horizons but failed qualification. The
[interruption record](progress/robot-storage-interruption.json) preserves both results.

The recovery run restarts from random weights with the same seed, immutable binary
and 600-update recipe. It does not resume optimizer state or overwrite the stopped
run. Artifacts are in `runs/droid-standing/seed7-recovery-20261010`; the user unit is
`bevy-gym-standing-seed7-recovery-20261010.service`. The source manifest, binary and
launch record remain under `/home/sagan/.cache/bevy-gym-quality-validation`.

All 217 complete optimizer records match the original prefix exactly. Update-20
selection measurements also match, while checkpoint bytes differ. No checkpoint
byte reproducibility claim is made.

The run completed all 600 updates and exhausted
its budget without passing. Final selection lasted 78–96 actions, with no stable
standing sequence. Independent held-out evaluation failed all 32 cases. The
[recovery record](progress/droid-standing-recovery.json) preserves the launch, prefix
audit, final selection, checkpoint hash and optimizer-journal hash. The
[held-out report](progress/droid-standing-recovery-evaluation.json) preserves every case.

The seed-13 trial imported update 600 and started fresh optimization.
Its budget was 2,400 new updates. The unit is
`bevy-gym-standing-warm-start-seed13-20261010.service`; output is
`runs/droid-standing/seed13-warm-start-20261010`. The launch bundle is
`/home/sagan/.cache/bevy-gym-quality-validation/standing-warm-start-seed13-20261010`.
The [warm-start record](progress/droid-standing-warm-start.json) preserves source
identity, validation and coverage gaps. Inspect live state before reporting results.

A [native training profile](./progress/droid-standing-training-profile.json) records
five-update trials using the same saved RL actor and seed 17. Warmed PPO optimization
medians were 5.32 seconds for the baseline, 5.05 seconds with root optimization level
3, 5.06 seconds with native Burn defaults, and 5.04 seconds with native SIMD. Collection
took about 0.21 seconds per 512 transitions. None met the predeclared 2.66-second
optimizer target.

Each configuration ran once alongside both live training runs; these
measurements do not establish a reliable gain. The native feature and profile patches
were reverted.

Recurrent PPO now evaluates each optimizer minibatch in one LSTM call. Shorter
sequences receive trailing zero observation rows. The optimizer selects valid output
rows before computing probability densities and losses. Each sequence retains its
own initial memory, and the loss averages only valid timesteps. Batched tensor memory
scales with lane count times maximum sequence length. When sequence lengths differ,
padding adds work.

The [batching record](./progress/recurrent-ppo-batching.json) preserves the original
failing resource test, scalar comparisons, native/browser gates and coverage. The
five-update padded prototype reduced median warmed optimization from 5.32 to 0.47
seconds. The final executable completed ten updates in 7.00 seconds. Observed warmed
update intervals had a 0.70-second median, including collection, optimization and
journal flush. Polling and scheduling make those intervals approximate. This smoke
run failed standing selection and does not qualify a policy.

The seed-13 trial completed all 2,400 updates and exhausted its budget. Its final
selection lasted 85 to 100 actions, with no survivors. Independent evaluation through
the original executable failed all 32 held-out cases, lasting 83 to 128 actions.
The [completion record](./progress/droid-standing-warm-start-final.json) and
[complete held-out report](./progress/droid-standing-warm-start-final-evaluation.json)
preserve the failure.

The [next trial](./progress/droid-standing-batched-trial.json) uses the verified batched
executable, seed 17 and a 24,000-update limit. It imports seed-13 update 660, which had
the highest mean reward across the 120 fixed selection reports. That parent remains
unqualified. Adam state, counters, samplers and episode memories start fresh. The
physical rules, rewards and qualification gates remain fixed. Before reporting
progress, inspect unit `bevy-gym-standing-batched-seed17-20261010.service` and output
`runs/droid-standing/seed17-batched-20261010`.

## Seed-17 held-out checkpoint evaluation

Update 7,920 had the highest selection mean return among 402 completed reports.
That selection rule chose the checkpoint before held-out evaluation. Frozen
inference through the immutable training executable failed all 32 held-out cases.
Episodes lasted 208–445 actions, with a mean of 328, against 1,000 required.
Every final stable streak was zero. Improved selection return does not qualify standing.

The [complete evidence](progress/standing-seed17-update7920/evaluation.json) records
both selection and held-out results, checkpoint and executable hashes, and the
exact command. Checkpoint bytes remained unchanged. The
[episode CSV](progress/standing-seed17-update7920/episodes.csv) and
[plot](progress/standing-seed17-update7920/held-out.png) retain every case.
The seed-17 run has now exhausted its 24,000-update budget with exit 1.
No standing policy is qualified.

## Evaluated candidate in the browser

The standalone scene now bundles the exact update-7,920 policy and its sidecar.
The older update-100 artifacts remain preserved. The scene labels the current
candidate `Unqualified RL` and shows its update and checkpoint digest.

The [browser evidence](progress/standing-seed17-update7920/browser-evidence.json)
records source and artifact hashes, checks and verification gaps. The
[desktop recording](progress/standing-seed17-update7920/browser-desktop-one.mp4) shows
1× playback; the [mobile recording](progress/standing-seed17-update7920/browser-mobile.mp4)
shows 16× playback. Both end after 418 actions. Native seed-42 inference ends after
366 actions. Full physical trajectory parity remains unestablished.

The new native/WASM test replays 366 identical 204-feature actor inputs with fresh
initial memory and compares all 26 torque fractions within absolute tolerance 0.00001.
It retains recurrent state between inputs. All 230 browser tests pass, including this
replay and the historical 62-frame replay. Native default tests, 24 session tests,
26 scene tests, formatting and strict repository Clippy pass.

The camera follows observed body bounds and leaves physical and policy state unchanged.
A mobile frustum test checks every terminal body centre; invalid points preserve the
camera. Desktop and mobile render checks keep the fallen mesh in frame.

Coverage hits five of six new camera branch outcomes. `frame_body`'s `None` outcome
at `examples/robots/standing_scene/view.rs:275` remains unhit because the session
exposes authoritative finite body observations. Direct helper tests reject NaN and
both infinities. Native LLVM tests do not execute bundled loading at
`examples/robots/standing_scene/mod.rs:68` or system registration at lines 120–126.
Browser playback exercises those paths.

Native window interaction and foreground
preview-pane visibility remain unverified. Changed-line personal Rust lint is clean;
its strict Clippy child still fails on existing diagnostics outside changed lines.

## Completed seed-17 run

Update 22,940 has the highest selection mean return across all 1,200 reports.
The ranking used selection results only, before this candidate’s held-out evaluation.
Five of 32 held-out episodes survived 1,000 actions; all final stable streaks were zero.
Mean duration was 665.53125 actions. Frozen evaluation exited 1.
The [record](progress/standing-seed17-update22940/summary.json) retains the unchanged
checkpoint hash, full ranking, selection results, held-out results and commands.

Selection seed 42 survived with return 826.67025, but its final torso-up projection
was 0.73860, below the required 0.96593. The compressed traces retain every
policy-selected torque and physical boundary for selection seeds 0 and 42.
The current reward can exceed its threshold while posture fails. This does not
establish a physics or optimizer defect. The next experiment will test a training-only
reward profile emphasizing posture and foot support while preserving qualification gates.

## Training-only posture reward

The optional `--reward-profile posture-v1` applies posture and foot-support weights
to the original training reward. `original` remains the default. Frozen `evaluate`
and `trace` reject the training option and retain the original qualification gates.
The actor keeps the same 204 features and 26 direct-torque fractions.

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example droid-standing -- warm-start \
  --checkpoint docs/progress/standing-seed17-update22940/checkpoint.mpk \
  --reward-profile posture-v1 --seed 23 --updates 24000 \
  --output runs/droid-standing/posture-seed23
```

The imported actor and critic remain unqualified. Adam, counters, samplers and
episode memory start fresh. This command does not resume optimizer state.

Let `u` be the torso-up projection on world Y, clamped to [-1, 1]. The training
reward is the original reward times `exp(-(1 - u) / s)` times the contact weight.
The dimensionless scale `s` is exactly `0.03407418727874756`, derived from the
original f32 upright threshold. The posture weight is 1 when upright and
`exp(-1)` at that threshold. Inverted poses receive less weight than horizontal poses.

The dimensionless contact weight is 1 for both feet, 0.5 for exactly one foot,
and 0.25 for neither foot. Contacts come from the same post-action physics snapshot.
Terminal episodes retain zero reward before weighting. Reset streams, physical rules,
torques, observations and qualification gates remain unchanged.

Before collecting transitions, this mode writes `training-reward.json`. It emits
seven required members as compact UTF-8 JSON without a final newline. The schema
string is `droid-standing-posture-reward-v1`; `seed` is an unsigned integer.
`posture_up_projection_scale` is the number `0.03407418727874756`.
`contact_both`, `contact_one` and `contact_none` are the numbers 1.0, 0.5 and 0.25.
`evaluation` is the string `original-standing-v1`.

No member is absent or null; the writer emits no extensions. There is no profile
input parser. The CLI test asserts the complete emitted text and member order.
The original mode emits no profile artifact. A failed profile write stops training
before rollout collection. The unchanged checkpoint sidecar requires the run record
and reward artifact to identify this training recipe.

The [validation record](progress/droid-standing-posture-reward.json) retains
source hashes, commands, coverage gaps and the 20-update seed-23 smoke trial.
That trial used 10,240 transitions and failed each of the five selection episodes.
It has no held-out evaluation or qualification claim. All 245 browser tests,
31 native learning tests and 14 CLI tests pass. The reward wrapper has full
measured production line, region, function and branch coverage.

## Training-only pelvis-height reward

The optional `--reward-profile posture-height-v1` multiplies the `posture-v1`
training reward by `exp(-((h - t) / s)^2)`. `h` is the authoritative post-action
pelvis Y position above the zero-height floor, in metres. The target `t` is
1.0 metre, the midpoint of the original f32 height bounds. The scale `s` is
0.1499999761581421 metres, half their span after conversion to f64.
The normalized error and Gaussian weight are dimensionless. The weight is 1
at the target and `exp(-1)` at both original height bounds.

[Isaac Lab v3.0.0-EA](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab/isaaclab/envs/mdp/rewards.py#L102)
uses a squared root-Z height-error penalty on flat terrain. This local experiment
uses a positive Gaussian weight and pelvis Y. It does not adopt the upstream reward.

`original` remains the default. Both earlier reward profiles retain their behavior
and artifact shapes. The PPO actor still receives 204 own-body features and chooses
26 direct-torque fractions. Physics, reset streams, terminal rewards and frozen
qualification gates remain unchanged. `evaluate` and `trace` reject reward overrides.

```sh
nix develop --command cargo run --features robots --example droid-standing -- \
  warm-start \
  --checkpoint docs/progress/standing-posture-seed23-update23540/checkpoint.mpk \
  --reward-profile posture-height-v1 --seed 37 --updates 24000 \
  --output runs/droid-standing/posture-height-seed37-reproduction
```

This imports the recorded PPO actor and critic. Adam, counters, samplers,
environments and recurrent memory start fresh. It does not resume optimizer state.

Before collection, `training-reward.json` emits one object with nine required
members as compact UTF-8 JSON without a final newline. Its exact seed-37 text is:

```json
{"contact_both":1.0,"contact_none":0.25,"contact_one":0.5,"evaluation":"original-standing-v1","pelvis_height_scale_m":0.1499999761581421,"pelvis_height_target_m":1.0,"posture_up_projection_scale":0.03407418727874756,"schema":"droid-standing-posture-height-reward-v1","seed":37}
```

`schema` and `evaluation` are the fixed strings shown above. `seed` is the run's
unsigned 64-bit integer. The other six members are JSON numbers with the exact
values shown above. Contact weights and posture scale retain their earlier meanings;
the two height members use metres. Every member appears once in the displayed order.

No member is absent or null, and the writer emits no extensions. There is no input
parser or normalization. A failed profile write stops before collecting transitions.

The [smoke record](progress/standing-posture-height-smoke-seed37/summary.json)
retains the source and executable hashes, parent identity, checkpoint, profile,
optimizer journal, selection outcomes and coverage. The 20-update trial consumed
10,240 transitions and performed 140 optimizer updates. All five selection episodes
failed and none survived the horizon. Its checkpoint is unqualified; held-out cases
were not evaluated.

The [24,000-update trial](progress/droid-standing-posture-height-seed37.json)
uses the same immutable executable, parent and seed. Its first 20 optimizer records
match the smoke trial exactly. It stopped at update 12,860 after its first passing
selection report. Frozen held-out qualification failed; the checkpoint was not promoted.

Validation passes 132 default native tests, 480 robot native tests, 318 WASM tests
and 15 CLI tests. Strict Clippy and personal Rust/Python lint pass. All 43 changed
production coverage regions were hit. The reward wrapper has 73 of 73 production
regions and six of six branch outcomes hit.

The coverage record lists unchanged error propagation from runtime construction,
models, action decoding, optimizers, serialization and file operations. It also
lists unhit qualification success returns. Nix lint still fails on the ten recorded
baseline findings outside this change.

## Seed-31 selection and held-out result

The posture trial stopped at update 21,180 after its first passing fixed selection
report. All five selection cases passed; the prior 1,058 reports failed.
This selection rule chose the checkpoint before held-out evaluation.

Frozen inference failed the unchanged 32-case held-out gate. Twenty-nine episodes
survived 1,000 actions and twenty-seven finished with at least 100 stable actions.
Standing remains unqualified; the checkpoint was not promoted into the curriculum.
The [record](progress/standing-seed31-update21180/summary.json) retains checkpoint
identity, warm-start lineage, optimizer journal, selection history and every held-out case.

## Seed-37 selection and held-out result

The height trial passed all five selection cases at update 12,860 after 642 failed
reports. Frozen inference failed the unchanged 32-case held-out gate. Twenty-four
episodes survived 1,000 actions and twenty-three finished with at least 100 stable
actions. The [record](progress/standing-seed37-update12860/summary.json) retains
checkpoint identity, optimizer journal, selection history and every held-out case.

The [seed-43 continuation](progress/droid-standing-posture-height-seed43.json)
imports this unqualified checkpoint with fresh Adam. Its parent was chosen by mean
fixed selection return among the first-passing seed-31 and seed-37 candidates.
Held-out results were excluded from that ranking. It passed selection at update
100. Frozen held-out qualification failed: 26 of 32 episodes survived and 25
finished with at least 100 stable actions. The
[result](progress/standing-seed43-update100/summary.json) retains the checkpoint
and evaluation. No lesson has been promoted.

## Recorded optimizer and selection experiments

Choose a new output directory for each run. This command continues the recorded
seed-43 RL weights; it does not qualify or promote a lesson:

```sh
cargo run --no-default-features --features robots --example droid-standing -- \
  warm-start --checkpoint docs/progress/standing-seed43-update100/checkpoint.mpk \
  --seed 73 --updates 1001 --reward-profile posture-height-v1 \
  --optimizer-profile low-noise-v1 --selection-policy best-passing-budget \
  --output runs/droid-standing/low-noise-next
```

`--optimizer-profile original` preserves the historical recipe and artifact set.
`conservative-v1` uses actor learning rate 0.00003, critic learning rate 0.001,
zero entropy bonus and one epoch. `low-noise-v1` also starts Gaussian exploration
at log standard deviation -3.5, approximately 0.0302 before `tanh`. Random
initialization uses that scale. Warm starts replace only the exploration scales;
learned mean parameters and critic parameters remain unchanged. Both warm-start
optimizers start fresh.

 Observation width, torque bounds, physics and frozen
standing gates remain unchanged.

A selected recipe writes `training-optimizer.json` before collecting transitions.
The conservative record has exactly `schema`, `actor_learning_rate`,
`critic_learning_rate`, `entropy_coefficient` and `epochs`. Its schema is
`droid-standing-conservative-optimizer-v1`. The low-noise record uses schema
`droid-standing-low-noise-optimizer-v1` and adds `reset_initial_log_std: -3.5`.
The original recipe writes no optimizer-profile artifact.

`--selection-policy first-passing` preserves the original stop at the first passing
fixed selection report. Its zero exit status establishes selection only.
`best-passing-budget` collects the complete budget and retains the passing checkpoint
with the highest mean unscaled selection return. Equal returns retain the earliest
observed update. Held-out cases never rank checkpoints.

`selection-budget.json`
records the budget, rule and selected checkpoint, or null when none passed.
Budget completion returns a nonzero exit status even when a candidate passed.
Neither stopping policy promotes a lesson or establishes held-out qualification.

The [archived experiments](progress/standing-control-trials-20261011/) retain source
patches compressed with gzip, launch records, optimizer journals, selection histories
and checkpoint identity:

- Seed 59 collected 101 updates and selected update 20. Held-out evaluation failed:
  24 of 32 episodes survived and finished with at least 100 stable actions.
- Seed 61 collected 101 updates and selected update 20. Held-out evaluation failed:
  21 episodes survived and 20 finished with at least 100 stable actions.
- Seed 67 collected 101 updates. No checkpoint passed selection; none was evaluated
  on held-out cases.
- Seed 71 collected 1,001 updates and selected update 60. Held-out evaluation failed:
  28 episodes survived and 27 finished with at least 100 stable actions.

The [motion comparison](progress/standing-control-trials-20261011/motion-comparison.json)
measures frozen inference on seed 42. The published policy's peak segment angular
speed was 154.30 rad/s; seed 59's selected policy reached 157.29 rad/s.
Lower exploration noise did not fix frozen-policy flailing. Standing, walking and
competitive combat remain unqualified. The portable source patch passed
`git apply --check` against its recorded base commit.

## Quiet reward and motion selection

`--reward-profile posture-quiet-v1` multiplies the posture-height training reward
by `1 / (1 + M / S²)`. `M` is the largest squared angular speed among all thirteen
segments, in rad²/s². `S` is 4 rad/s. The weight is dimensionless. Components convert
to f64 before squaring. Terminal rewards, body dynamics and torque bounds retain
the original standing contract.

The reward record uses schema `droid-standing-posture-quiet-reward-v1`. It retains
the posture-height members and adds `angular_speed_scale_rad_s: 4.0`,
`angular_speed_measure: "maximum squared segment angular speed"` and
`angular_speed_weight: "inverse quadratic"`. The record is written before collection.

Quiet selection requires the original standing gates and an additional motion gate.
After 100 settling actions, every segment must remain at or below 8 rad/s at all
900 remaining post-action boundaries. The five ordered selection roots remain
`0`, `1`, `2`, `42` and `18446744073709551615`. Selection ranks passing checkpoints
by mean original standing return; held-out cases never rank checkpoints.

Quiet selection reports add `control_episodes` and `control_passed`. Each control
score contains `seed`, `steps`, `checked_actions`, `peak_rad_s` and `survived`.
`peak_rad_s` is null before any post-settling boundary was checked. Otherwise, it
is the largest measured segment speed in rad/s. Missing measurements, incomplete
episodes and physical termination fail the gate. Other reward profiles retain
their previous selection reports.

Run the separate frozen checker without changing weights:

```sh
cargo run --no-default-features --features robots --example droid-standing -- \
  evaluate-control --checkpoint path/to/checkpoint.mpk
```

Omitting `--seed` checks all 32 fixed held-out roots. Adding `--seed 42` inspects one
episode. The report includes both standing and control scores, separate pass flags,
and their combined result. Invalid checkpoint identity stops before evaluation.
Neither a passing episode nor budget completion promotes a lesson. The motion gate
excludes the first two seconds; visual review must include that initial motion.

The [seed-83 record](progress/standing-quiet-trials-20261011/seed83/summary.json)
retains 1,001 actual PPO updates and 512,512 transitions. Three checkpoints passed
original standing selection; all three failed motion selection. No checkpoint was
selected for held-out evaluation. This immutable trial predates fused arithmetic
and integrated motion selection; its recorded motion rule was applied separately
to every standing candidate. Fresh runs with seeds 89 and 97 each completed
5,001 PPO updates and 2,560,512 transitions. Neither produced an original standing
selection pass.

Their [records](progress/standing-quiet-trials-20261011/) retain
failed checkpoints for diagnosis; neither run reached held-out evaluation.

The [verification record](progress/standing-quiet-trials-20261011/verification.json)
identifies native, browser, CLI and coverage evidence. These checks establish
mechanics and rejection behavior. No quiet standing policy is qualified.

## Small initial action head

`--optimizer-profile gentle-v1` requires fresh initialization. It multiplies the
fresh action-mean head weights and bias by 0.01 before collecting transitions.
It retains recurrent parameters, the critic, all 26 full torque bounds and the
original action decoder. This changes initial policy parameters, not actuator limits.
Imported checkpoints reject this profile before optimization or output creation.

The profile uses actor learning rate 0.00003, critic learning rate 0.001, zero entropy
coefficient, one epoch and initial log standard deviation -5.0. Its optimizer record
uses schema `droid-standing-gentle-optimizer-v1` and adds
`reset_initial_log_std: -5.0` and `mean_head_scale: 0.01` to the five conservative
members. Older profiles retain their previous record shapes.

[Stable Baselines3's pinned initialization](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L557)
uses an action-head orthogonal gain of 0.01. This local profile multiplies existing
Burn parameters by 0.01; it does not reproduce that orthogonal initialization.

The operation detaches scaled tensors before tracking gradients and resets actor
Adam. Tests verify recurrent and critic preservation, retained Gaussian exploration,
actual PPO updates and exact byte reload.

The [seed-101 pilot](progress/standing-gentle-trials-20261011/seed101/summary.json)
completed 21 actual PPO updates and 10,752 transitions. No checkpoint passed.
Frozen update-21 inference on root 42 terminated after 25 actions. Its first maximum
torque fraction was 0.00335, and its peak segment speed reached 24.00 rad/s.

Three independent runs with roots 107, 113 and 131 have recorded budgets of 10,001
updates each. Their launch records remain diagnostic until fixed selection and
independent held-out evaluation pass. No gentle standing policy is qualified.

The [verification record](progress/standing-gentle-trials-20261011/verification.json)
records 25 CLI tests, native and browser checks, and per-object coverage. The fixed
valid fresh configuration cannot exercise its construction-error propagation path.

## Exploration bound recovery

The PPO learner projects stored log standard deviations into the configured bounds
after every actor Adam step. Forward-only clamping previously let parameters move
outside those bounds, where gradients could not restore exploration. In a public
regression, reversing variance preferences for 100 updates left both bounds unchanged.
The corrected learner recovers inward from both bounds.

Projection retains the parameter ID and actor Adam state. It detaches the projected
tensor before tracking the next gradient. Private tests verify exact lower, interior
and upper values, gradient tracking, and unchanged mean, memory and critic values.
Frozen inference retains its existing clamp and checkpoint format.

The [verification record](progress/ppo-variance-boundary-20261011/verification.json)
retains the failing public test, passing public tests, native and browser gates,
personal lints, and fresh coverage of all 16 projection regions. The pinned
[Burn 0.21 parameter implementation](https://github.com/tracel-ai/burn/blob/v0.21.0/crates/burn-core/src/module/param/base.rs)
retains identity through `Param::map`; its bytes match the installed dependency.
This optimizer correction requires new physical evaluation. It does not qualify
an existing standing, walking or competitive policy.

## Full episode and early motion failures

The [fresh seed-151 run](progress/standing-gentle-trials-20261011/fresh-exploratory-head-seed151/summary.json)
and [critic warm-up seed-191 run](progress/standing-gentle-trials-20261011/critic-warmup-seed191/summary.json)
failed selection. The [smaller-step seed-193 continuation](progress/standing-gentle-trials-20261011/critic-warmup-small-step-seed193/summary.json)
completed 5,001 actor updates after 100 critic-only updates. It also failed every
selection checkpoint. Long survival alone does not establish controlled movement.

The [seed-179 continuation](progress/standing-gentle-trials-20261011/exploratory-continuation-seed179/summary.json)
completed 10,001 updates. Only update 8,720 passed all five original standing and
post-settling limb-control selection cases. The predeclared selection rule selected
that candidate without held-out ranking. Frozen evaluation reproduced selection,
but only 28 of 32 held-out episodes survived. Its first-action arm speed was about
157 rad/s. The candidate remains rejected.

All selection records, selected bytes,
frozen evaluator source and full selection motion traces are retained.

The [passive comparison](progress/standing-gentle-trials-20261011/motion-diagnosis/summary.json)
uses zero torque only as an isolated mechanics fixture. Its first-action segment
speeds stayed below 0.0002 rad/s across the five selection roots, then the body fell
within 24–26 actions. This does not establish a qualified controller or a solver defect.

The [seed-211 stronger-motion pilot](progress/standing-gentle-trials-20261011/safe-episode-seed211/summary.json)
terminates training when any segment exceeds 8 rad/s, including the first action.
Its training reward subtracts 0.1 times mean squared torque fraction from the original
posture-height reward; a speed-limit failure returns -1. Original bodies, torque
bounds, integration and qualification gates remain unchanged. All 501 updates failed
selection. Short episodes split a 512-transition rollout into many recurrent sequences;
eight-sequence minibatches and four epochs produced up to 256 optimizer steps per
update. These research records do not qualify standing, walking or combat.

## Per-channel exploration initialization

The learner can reset exploration with one log standard deviation per torque
channel. It validates width, all finite values and inclusive configured bounds
before changing parameters or optimizer state. Success retains learned means,
recurrent and critic parameters, critic Adam, counters and minibatch state. Actor
Adam starts fresh. Collect a new rollout after resetting exploration.

The [verification record](progress/per-action-exploration-20261011/verification.json)
retains the missing-API regression, public and private tests, native and browser
gates, personal lints, and fresh per-object coverage. All 43 recorded method regions
and both outcomes of its three recorded branches were hit. This API initializes
trainable exploration parameters; it supplies no mean action and qualifies no policy.

The full-batch [seed-211 pilot](progress/standing-gentle-trials-20261011/safe-full-batch-seed211/summary.json)
completed 1,001 updates with one optimizer step per 512 transitions. Its
[seed-223 continuation](progress/standing-gentle-trials-20261011/safe-full-batch-continuation-seed223/summary.json)
completed 10,001 updates. A [larger-step seed-223 trial](progress/standing-gentle-trials-20261011/safe-full-batch-larger-step-seed223/summary.json)
changed only the actor learning rate from 0.00003 to 0.0003 and completed
1,001 updates. Every selection checkpoint failed. None reached held-out evaluation.

## Calibrated exploration trials

The [actuator response diagnostic](progress/standing-gentle-trials-20261011/actuator-response/summary.json)
uses 260 isolated signed torque fixtures over five reset roots. First-action
angular-speed response spans about 279-fold across channels. The initializer
scales Gaussian exploration by measured response, with standard deviation capped
at 0.2 and floored at exp(-9). These parameters remain trainable. The diagnostic
is neither an autonomous example nor a guarantee for simultaneous or later actions.

The [calibrated seed-211 pilot](progress/standing-gentle-trials-20261011/calibrated-exploration-seed211/summary.json)
completed 1,001 updates and failed all 51 selection checkpoints. Frozen update-540
[motion traces](progress/standing-gentle-trials-20261011/calibrated-motion/summary.json)
show first-action speeds near 4.17 rad/s but later speeds of 33–46 rad/s before
falling. This remains failed standing.

The [seed-227 continuation](progress/standing-gentle-trials-20261011/calibrated-continuation-seed227/launch.json)
has a 10,001-update budget and retains seed-211 update-960 parameters with fresh
Adam states. The [seed-229 additive-cost trial](progress/standing-gentle-trials-20261011/additive-motion-seed229/launch.json)
retains seed-179 update-8,720 actor and critic parameters, then resets exploration
to the calibrated channel scales. Its 1,001-update budget uses posture-height
reward minus 0.1 times maximum segment angular speed squared divided by (8 rad/s)².

The additive cost preserves physical dynamics and natural termination. Later
requests freeze the terminal frame with zero reward. Original selection gates,
held-out partitions and full first-two-second inspection remain required. Launch
records report pending outcomes; they establish no qualified policy.
