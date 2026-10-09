# Drone motor damage

Status: actuator failure implemented; learned recovery and visuals pending, 2026-10-08.

## Quick start

```sh
nix develop --command cargo run --no-default-features --features robots --example drone-damage
```

The guide hovers for two seconds, fails the front-left motor, and keeps commanding
half power. It reports the resulting crash. This is a force-failure baseline.
It retains body mass and collision geometry and does not train a policy.

`DroneHover::fail_motor(DroneMotor::FrontLeft)` disables a named actuator.
`DroneObservation::motor_state` returns `Working` or `Failed`.
Repeated failure succeeds during flight. After termination, failure returns
`DroneEpisodeEnded` without changing the terminal observation. Reset restores
all four actuators.

The next lesson must recover after a motor fails during flight. It must preserve
healthy-flight behavior, show the failed motor, and evaluate adaptation separately
from ordinary hover. Later pursuit lessons combine this with a moving target and
perception. A successful hover policy alone does not complete this requirement.

## Sources and physical limits

[Flightmare's supplement, section 5.2](https://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf)
uses three direct motor-thrust outputs after losing one propeller. Its state
includes position, orientation, linear velocity, and body angular velocity.
Its reward excludes yaw angle and yaw rate. This supports testing position and
tilt recovery while permitting rotation. Its experiment does not establish that
our different geometry, inertia, force limits, or reward will recover.

The existing model has four parallel thrust axes and alternating reaction moments.
For one missing corner, zero static roll and pitch torque require its opposite
corner's thrust to be zero. The remaining diagonal pair has the same reaction
sign. Positive lift therefore produces yaw torque. Require neither fixed heading
nor zero yaw rate in a damaged-hover qualification.

The current force limit also matters. Each motor supplies at most `m * g / 2`
newtons, where `m = 1 kg` and `g = 9.81 m/s²`. The remaining balanced diagonal pair
has only enough total force to equal the upright body's weight. It has no static
upward acceleration margin. Dynamic recovery with three motors needs measured
behavior; do not assume it follows from the intact model's thrust margin.

## Invariant scratchpad

- Motor identities are the four named corners in the existing action order:
  front left, front right, rear right, rear left. Forward is body -Z.
- Each motor is working or failed. Repeated failure is idempotent.
- Failure removes that motor's force and reaction moment before the next substep.
  The other motors retain their commands and force limits.
- An action remains four validated fractions in `[0, 1]`. Failure never bypasses
  action validation or lets a caller access the physics world.
- Damage cannot change the body pose, velocity, reward history, or random stream
  at the instant it is applied. Physics determines subsequent motion.
- Reset restores the configured episode's initial condition and motor state.
  A terminal episode remains absorbing; damage cannot resume it or alter its
  already-returned terminal observation.
- Read-only motor state accompanies the physical observation. Derive visual motor
  activity from that state and the accepted command.
- Keep healthy-model serialization, observations used by existing encoders, reset
  streams, and qualified native/WASM rollouts unchanged.
- Qualify one motor failure first, covering every corner. Multiple failed motors
  must still have defined physics, even when the lesson cannot recover from them.

The public API uses the closed `DroneMotor` identifier and the
`DroneEpisodeEnded` error. Health storage and the physics world remain private.
It exposes no repair operation during flight.

## Training and visual milestones

1. Add a short guide that fails one named motor during a seeded flight.
2. Verify the force boundary and lifecycle before training a damaged-flight policy.
3. Add motor-state features to a separate damaged-flight encoder and checkpoint recipe.
4. Train intact flight, fixed-corner failure, then randomized corner and failure time.
5. Evaluate unseen seeds, every motor, and failures after a stable approach.
6. Show failed actuators and learned recovery in the native and browser viewers.
7. Add projectile hits, detached parts, and the changed collision and inertia model.
8. Combine damage with pursuit, lost-target search, and adversarial target behavior.

The first force-failure milestone retains the collision proxy and body mass.
It must be labeled actuator failure. It does not complete visible physical
separation of a destroyed thruster. Detached-part mass, inertia, and collision
changes require their own documented model and regression evidence.

Use position error, tilt, survival time, post-failure altitude loss, and motor
commands as evaluation metrics. Record angular velocity without penalizing yaw
merely because it differs from intact hover. Freeze held-out thresholds before
selecting a trained checkpoint. Compare with the existing intact policy and a
constant-thrust baseline on exactly the same damage schedule.

## Acceptance checks

- [x] Existing intact native and browser qualification remains green.
- [x] Every motor identity maps to the documented corner and reaction sign.
- [x] Each failed motor contributes exactly zero force and reaction moment.
- [x] All sixteen working/failed combinations have defined actuator behavior.
- [x] The all-failed case follows the existing power-off dynamics.
- [x] Repeated failure, reset, terminal failure, and cross-environment isolation pass.
- [x] Compile-fail tests reject raw identifiers and writable physics/health access.
- [ ] Native/WASM damage rollouts agree within the recorded physics tolerance.
- [ ] The learned policy passes every frozen one-motor-failure qualification case.
- [ ] Browser training, checkpoint inference, and visible damage are demonstrated.
- [ ] Detached parts, target pursuit, perception, and attacks have separate evidence.

The native and WASM integration tests compare all sixteen actuator combinations
against explicit zero commands. Two private tests check immediate state preservation
and removal of previously accumulated forces. Existing frozen healthy rollouts and
all fourteen learning tests still pass in the browser. A cross-target numeric
fixture for damaged motion remains pending.

Validation passed root tests, strict all-target/all-feature Clippy, the native
robot suite, and `nix run .#drone-browser-check`. Six documentation tests passed;
one existing plugin example remains ignored. The runnable guide ended 36 actions
after failure. Personal-lint discovery found no diagnostics on changed lines.
The full personal strict gate still reports the existing library/test backlog.

[Coverage evidence](progress/drone-damage-coverage.json) records 320/320 lines
and 16/16 branches in `hover.rs`, including internal tests; observation and error
methods also have complete measured line coverage. The guide was run separately
without coverage instrumentation. No damaged-flight policy is qualified yet.
