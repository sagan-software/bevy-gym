# Shared robot world

Run the reset-only guide:

```sh
nix develop --command cargo run --features robots --example robot-world-contract
```

The guide reads the initial positions of three drones and three droids.
It does not select actions or advance physics. Competitive rules, qualified
team policies and browser match playback remain unfinished.

## Bodies and clock

`RobotWorld` owns one private Rapier 0.36.0 solver with `enhanced-determinism`.
All six robots share collision detection and integration. The world contains
42 dynamic bodies, one fixed floor, 43 colliders and 36 passive joints.
Their total dynamic mass is 213 kg: three 1 kg drones and three 70 kg droids.

The drone body, rotor geometry and force operator come from the
[flight environment](ROBOT_ENVIRONMENT.md). The thirteen-segment droid body,
joint limits and twenty-six direct torque actuators come from
[standing](DROID_STANDING.md). The standalone lessons retain their original
floors, reset distributions, rewards and termination rules.

Gravity is `(0, -9.81, 0)` m/s². The shared floor covers X/Z from -30 to 30 m,
has its top at Y = 0 m and uses friction coefficient 0.8.
Drone centres start at `(-3, 2, lane)` m. Each droid's original rig receives
an identity rotation and translation `(3, 0.03, lane)` m.
The lanes are -2, 0 and 2 m in `RobotSlot::ALL` order.
These placements and the shared floor are application policy.

One actuator interval lasts 20 ms and contains four common 5 ms solver steps.
During each substep, all six actuator requests enter the solver before integration.
Rotor forces and joint torque axes use the current physical orientation.
Elapsed time advances only after all four substeps succeed.

## Snapshot and action boundary

`RobotSnapshot` captures all six physical states at one decision boundary.
`RobotSlot` has the closed variants `First`, `Second` and `Third`.
`RobotId` derives its team from `Drone(slot)` or `Droid(slot)`.
Neither type exposes solver handles.

`RobotActions::new` binds three validated `DroneAction` values and three
validated `DroidAction` values to that snapshot. Each array follows
`RobotSlot::ALL`. The caller must infer all six requests from trained RL
policies before calling `RobotWorld::advance`. This API validates actuator
shape, bounds and frame ownership; it does not establish policy provenance.

The full snapshot is authoritative environment state. Actor observations still
require sensor isolation under [the competition contract](DRONES_VS_DROIDS.md).
The foundation does not supply actor encoders, centralized critics, weapons,
health, rewards, episode termination or policy selection.

`RobotFrame` retains an opaque in-process world/reset scope and a positive
boundary ordinal. Moving a world preserves its frame. Reset creates a fresh
scope. Frames cannot be publicly constructed or serialized.
Elapsed time derives from completed intervals rather than a separate clock.

## Failure and reset

`advance` rejects a wrong-world, stale or previous-reset frame with `WrongFrame`.
`ClockExhausted` rejects ordinal overflow. Both fail before actuation.
`NonFinite(RobotId)` identifies a robot with nonfinite physical state.
Validation checks position, orientation and both velocities for every body.

An advancement failure returns `RobotWorldError::Rejected(cause)` and stops
the world. Later snapshot and advancement requests return `Stopped(cause)`.
The error retains its original cause through `Error::source`.
A ready-world snapshot can reject nonfinite state without mutating the world.

A solver failure after a substep can leave partially advanced bodies.
The world stops and retains the last successfully published frame and elapsed
time; it does not roll physics back. `reset` rebuilds every body and constraint,
clears the stopped state and returns a fresh snapshot.

## Verification

The [evidence record](progress/robot-world-foundation.json) retains source hashes,
commands, test counts, coverage and frozen RL comparisons.
Six public boundary tests pass natively and in Chrome/WASM.
Three compile-fail checks reject raw solver access, forged frames and incomplete
action arrays. Private tests verify shared collision, mass and finite-state checks.

The new world files have full measured line coverage. Two defensive production
regions remain unhit: post-substep validation error propagation at
`simulation.rs:178`, and the stopped alternative inside private validation at
`simulation.rs:198`. The first has no reproduced solver failure from supported
finite actions. The second is bypassed by the stopped-world entry guard.

Two complete frozen droid RL traces match the previous binaries and published
traces byte for byte. All 64 drone promotion cases also match the previous
evaluator byte for byte. Those candidates remain unqualified.
These checks establish tested body-model preservation and WASM execution;
visible browser 3v3 inference remains unfinished.

## Frozen six-controller inference

Run the shared inference checks:

```sh
nix develop --command cargo test --features robots --test droid_skill_session shared_world
nix run .#drone-browser-check
```

The private example session loads one frozen PPO actor per team and retains
separate recurrent memory for each robot. All six actors read the same physical
snapshot before any request advances the world. Current memories remain unchanged
while requests and next memories are staged. Successful common advancement commits
all six next memories together. Inference or actuator validation failure stops
every controller before physics; a world failure retains the last published frame.

Each drone uses the retained seed-7 hover update 280. Each droid uses seed-17
standing update 22,940. The [hover transfer evidence](progress/drone-checkpoint-transfer.json)
establishes standalone hover qualification. The
[standing evidence](progress/standing-seed17-update22940/summary.json) records failed
standing qualification. Neither checkpoint qualifies a shared competitive task.

Loading requires the exact recorded hover bytes and a digest-bound standing record.
Missing, changed or incompatible weights cannot activate playback or a fallback.

Drone actors receive twelve own-body hover features. Droid actors receive the
original 204 standing features, with target displacement relative to their lane.
Targets are passive reset setup: drone centres at `(-3, 2, lane)` m and droid
standing targets at `(3, 0.99, lane)` m. Other robots' state, reward and critic
features cannot enter these actor encoders. These inputs do not implement combat
sensors, tactics, firing or learned alert states.

The diagnostic clip stops at 20 seconds, after 1,000 common actuator frames.
Reset rebuilds all bodies and clears every ready controller's memory and last action.
A failed controller remains failed after reset. Tests compare the complete clip
with six direct recurrent-policy replays and verify hidden-state isolation.
Native/WASM parity replays 128 recorded frames with identical inputs and separate
memories, using a fixed action-fraction tolerance of 0.00001.

The [inference evidence](progress/robot-world-inference.json) records hashes,
validation and coverage. Every recorded production branch outcome is covered.
Three defensive production
paths remain unhit: loading the known-valid embedded hover reference, rejecting
the constant initial world snapshot and rejecting constant world reconstruction
during reset. The session is backend inference evidence; its browser scene,
competitive rules, shared-task training and qualification remain unfinished.
