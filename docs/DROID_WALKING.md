# Droid walking

`DroidWalking` provides the native and browser-compatible physical task for commanded
walking. No walking policy, PPO runner, browser walking scene or qualified walking
lesson is supplied yet. The continuous curriculum and competitive 3v3 remain unfinished.

## Quick start

Inspect the seeded body and command without advancing physics or choosing actions:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example droid-walking-contract
```

Run the external mechanics checks:

```sh
nix develop --command cargo test --no-default-features --features robots \
  --test droid_walking
```

The reset-only guide is not an autonomous agent demonstration. Deterministic actuator
fixtures exist only in isolated tests. A future example must use validated RL policy
inference for every torque request. If inference fails, the example must stop visibly.

## Physical task

The task retains the [standing body](DROID_STANDING.md): thirteen physical segments,
70 kg total mass and twenty-six bounded torque channels. Joint geometry, limits,
friction, gravity and the original reset pose distribution remain unchanged. Its
standalone floor has 30 m half-width; the original standing floor retains 10 m.
Each action advances at most four 5 ms substeps, or 20 ms.

`DroidVelocityCommand::try_from(Vec2)` accepts finite X/Z velocity with represented
speed at most 1 m/s. `Vec2.x` means world X velocity; `Vec2.y` means world Z velocity.
Validation preserves f32 components and signed zeros without normalization.
For example, represented `(0.6, 0.8)` exceeds the exact speed limit and is rejected.
The default command is `(0.5, 0)` m/s.

The command stays fixed for the environment's lifetime, including resets. It is
lesson setup and never selects an actuator request. Observations contain only the
original own-body snapshot and that command. Public access exposes neither mutable
physics nor unchecked command construction.

## Lifecycle and reward

Validation checks every segment's finite position, orientation, linear velocity and
angular velocity before any torque application. It then checks pelvis height below
0.5 m, all segment bounds, and non-foot floor contacts, in that order. Allowed centres
are X/Z in inclusive [-29, 29] m and Y in inclusive [0, 3] m. Only feet may contact the
floor. `DroidBody::ALL` order selects the first segment within each failure category.

`DroidWalkingStatus` is `Running`, `Fallen(reason)` or `Failed(segment)`. The task
checks physics after each substep and stops immediately on either terminal outcome.
Both outcomes return `EpisodeStatus::Terminated` and zero reward; subsequent requests
retain the stopped state until reset. A failed snapshot can contain nonfinite values.
Partially advanced failed physics is not rolled back.

When the task is running, reward multiplies foot support, torso uprightness, pelvis height
fraction and horizontal velocity tracking. If either foot contacts
the floor, support is 1. Otherwise, support is 0.

Uprightness is torso up-Y clamped to [0, 1]; height fraction is
pelvis height divided by its 0.99 m bind height, also clamped to [0, 1]. Tracking is
`exp(-(error_x² + error_z²) / 0.25)`, with squared velocity error and scale in m²/s².
Both velocity components convert to f64 before subtraction and squaring.

Use `TimeLimit` with 1,000 actions for a nominal 20-second episode. Reward measures
this physical task; it does not establish learned walking. Before training, freeze
selection and held-out partitions plus travel, balance, energy and foot-sliding
gates. Promotion must retain parent identity and require independent evaluation.
Failed training or an exhausted budget cannot promote a lesson.

## Actuator work and elapsed time

`DroidWalking::mechanical_work_joules()` returns the accumulated absolute mechanical
actuator work estimate. `DroidWalking::elapsed()` returns advanced solver time as
`Duration`. Reset clears both measurements. A fall retains their final values;
requests after termination change neither value. Failed physics makes work unavailable
with `None`, including when failure follows a completed substep. Elapsed time includes
that completed 5 ms substep.

For each actuator, world torque in Nm uses the same parent orientation and bounded
fraction as the original operator. Relative child-minus-parent angular velocity is
in rad/s. Their dot product gives actuator power in watts. Each substep estimates
work in joules as the midpoint of absolute endpoint powers times 0.005 seconds.
Each actuator contributes separately, so opposing actuator powers cannot cancel.
Angular components widen to f64 before subtraction and power arithmetic.

[Rapier's force contract](https://rapier.rs/docs/user_guides/rust/rigid_body_forces_and_impulses/)
retains applied torque through a solver step. The measurement uses that held torque
at both boundaries. This trapezoidal estimate excludes gravity, contact work and
electrical losses; it does not establish exact continuous work or battery consumption.
Reading it exposes no mutable solver or additional actor input. Existing observation,
action, reward and `Info = DroidWalkingStatus` contracts retain their shapes.

The [work verification record](progress/droid-walking-work.json) records native and
browser tests, strict Clippy, personal lints, source identity and fresh per-object
coverage. Public tests cover reset, nonzero actuation, passive falling and terminal
freezing. Private tests cover both torque signs, opposing powers, finite angular
extremes and invalid physics before and after a substep. All 58 work regions and
160 environment regions executed, including both sides of every recorded branch.
Foot-sliding measurement and walking qualification gates remain unfinished.

## Verification

External tests retain the reset stream, validate exact command boundaries, prove
commands cannot change actuation, compare all twenty-six signed actuator requests
with the original standing operator, and freeze terminal physics. Private tests
cover diagnostic order, every nonfinite measured quantity, coordinate/contact
boundaries, stopping before actuation, and worked reward values.

The [foundation record](progress/droid-walking-foundation.json) retains source hashes,
commands, native/browser outcomes and fresh production coverage. The task is custom
Rapier mechanics over the project's mannequin; it is not a Gymnasium Humanoid port.
The [curriculum](ROBOT_SKILL_CURRICULUM.md) records the remaining browser and training work.

## Foot-contact sliding

`DroidWalking::foot_contact_sliding_distance_metres()` returns the cumulative
sliding estimate in metres as `Some([left, right])`. Reset clears both values.
A fall retains the final values, and terminal requests freeze them. Failed
physics returns `None`.

For each completed 5 ms substep, the measurement resolves each foot's solver
contact points through the current body poses. Point velocity includes linear
velocity and rotation about the body's centre of mass. Horizontal X/Z speed
in m/s widens to f64 before arithmetic. Each foot adds its maximum contact-point
speed times 0.005 seconds. When a foot has no solver contacts, it adds zero.
The floor is stationary.

The measure includes predictive solver constraints and uses discrete substep
boundaries. It does not identify loaded contact or integrate exact continuous
slip. It never enters actor observations, rewards or actuator requests.
[Rapier 0.36's contact resolver](https://docs.rs/rapier3d/0.36.0/rapier3d/geometry/struct.ContactManifoldData.html#method.solver_contact_world_points)
controls the conversion from stored contact anchors to world points.

The [sliding verification record](progress/droid-walking-slip-20261011/verification.json)
records eight public walking tests, native and browser gates, strict Clippy and
personal lints. Fresh per-object coverage records 30/30 point-velocity regions,
48/48 contact-query regions and 177/177 walking-environment regions. All recorded
branch sides execute. Fixtures cover rotation, vertical velocity, finite angular
extremes, both collider orders, reset, accumulation, terminal freezing and failed
physics. These measurements supply future walking evaluation; they do not qualify
a walking policy.

## Walking actor inputs

`DroidWalkingObservation::actor_features()` returns 207 values from one immutable
snapshot. The first 204 describe the body's physical state. The last three describe
the requested velocity in the pelvis frame. A tilted pelvis requires three command
axes even though the world command has only X/Z components.

The root groups are height-only displacement toward 0.99 m, world up, linear velocity
divided by 2 m/s and angular velocity divided by 4 rad/s, each ordered XYZ.
The height target has no horizontal return-to-origin displacement.

Twelve other segments follow `DroidBody::ALL` excluding pelvis. Each contributes relative position
in metres, local up, local forward, relative linear velocity divided by 2 m/s,
relative angular velocity divided by 4 rad/s and contact as 0 or 1. All vectors use
the pelvis frame. The final command is world `(X, 0, Z)` velocity divided by 1 m/s,
then rotated into that frame. Rotation preserves equivalent quaternion signs.

The encoder excludes reward, actuator work, foot sliding, other robots and critic
state. Nonfinite measurements remain nonfinite for policy rejection. The reset-only
guide now prints the feature count and command suffix without selecting an action.
The existing standing encoder and its 204-input checkpoint contract remain unchanged;
a standing checkpoint cannot silently supply a compatible walking policy.

The [verification record](progress/walking-encoding-20261011/summary.json) retains
public regression failures, ten public tests, four private encoder tests, native and
browser gates, personal lints and fresh coverage. Tests check every physical group,
command isolation, translation, quaternion signs and nonfinite measurements.
A walking trainer, qualified policy and browser lesson remain unfinished.
