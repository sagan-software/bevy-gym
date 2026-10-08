# First drone environment contract

Status: headless physics and typed Bevy integration implemented. Updated 2026-10-08.
Rendered flight, browser rollouts, and learning remain pending.
This defines the first P1 checkpoint in [the roadmap](EXAMPLE_ROADMAP.md).
It is an original game simulation informed by the references, not an ARC Raiders
source port or a claim about Embark's unpublished controller.

## Sources and engine

Use optional `rapier3d` 0.36.0, with `enhanced-determinism` and without its
parallel solver. The [published Rust package](https://crates.io/crates/rapier3d/0.36.0)
requires Rust 1.86. Its `PhysicsWorld`, rigid-body force, and contact-query source
were inspected. The [Rapier guide](https://rapier.rs/docs/user_guides/rust/getting_started/)
documents WASM support and the determinism feature. Bevy 0.18 renders an observation;
the environment owns a private Rapier world, so it needs no Bevy physics-plugin
version coupling. Native/WASM comparison remains an implementation gate.

[Flightmare's pinned dynamics](https://github.com/uzh-rpg/flightmare/blob/d4218aedac18cbe9364a0a0df10ab992c4b65e4f/flightlib/src/dynamics/quadrotor_dynamics.cpp)
uses body-frame motor positions, inertia, thrust mapping, and alternating reaction
torques. [The pinned PyBullet drone implementation](https://github.com/learnsyslab/gym-pybullet-drones/blob/7ebad1ecabd28a7000add2d05f888aa2e837c2cc/gym_pybullet_drones/envs/BaseAviary.py)
applies each motor force in its link frame and sums alternating reaction torques.
These control the comparison cases, not our game's mass or reward choices.

## Invariants and public boundary

- The environment is either flying or ended. An ended episode is absorbing until
  reset: no physics step, unchanged observation, zero reward, terminated status.
- `DroneAction` contains exactly four finite thrust fractions in inclusive `[0, 1]`.
  `TryFrom<[f32; 4]>` rejects invalid values before they can reach the simulation.
  No public field, mutable accessor, deserializer, or raw-world escape bypasses it.
- Motor order is front-left, front-right, rear-right, rear-left, viewed from above.
  The coordinate system is right-handed: +X right, +Y up, and -Z forward.
- `DroneObservation` reports position in metres, unit orientation quaternion,
  world linear velocity in metres/second, and world angular velocity in radians/second.
  Its fields are read-only through accessors. Physics remains authoritative.
- Reset with a supplied seed restarts the reset stream. Reset without a seed
  continues it. Rebuild solver state on reset so cached contacts cannot leak episodes.
- A fixed 20 ms policy interval contains four 5 ms physics steps. Use `Duration`
  for these intervals. Force and point coordinates rotate into world space on
  every substep, including when one action spans changing attitude.
- A ground contact or departure from the bounded flight region ends the task.
  Evaluate contact and region limits after each substep. Keep external time limits
  in the existing `TimeLimit` wrapper.

## Physical model

The first lesson uses a rigid body and four ideal thrust actuators. It includes
gravity, rotational inertia, contact impulses, and reaction torque. Rotor lag,
aerodynamic drag, wind, and damage belong to later separately tested lessons.

For rotor `i`, force is `u_i * F_max * R * Y`: `u_i` is a dimensionless command,
`F_max` is maximum force in newtons, `R` rotates the body frame into the world,
and `Y` is the body-up unit vector. Apply that force at `p + R * r_i`, where
`p` and `r_i` are position and motor offset in metres. Rapier computes the
resulting moment in newton-metres. Add alternating yaw moments
`sign_i * kappa * u_i * F_max * R * Y`, where `kappa` is in metres.

Choose `F_max = m * g / 2` per rotor, with mass `m` in kilograms and gravity
magnitude `g = 9.81 m/s²`. Four commands of 0.5 therefore balance an upright body.
This is a force command, not rotor RPM. A power-off action must accelerate downward;
equal commands must cancel moments; an asymmetric command must change attitude.
A tilted body must accelerate along its tilted thrust axis.

The body mass is 1 kilogram. The collision proxy has half-extents
`(0.287, 0.104, 0.297)` metres. Motor centres are 0.2505 metres to either side,
0.0875 metres above, and 0.2606 metres forward or behind. The reaction arm
`kappa` is 0.016 metres, with signs `+1, -1, +1, -1` in action order.
[Asset attribution and alignment](../assets/robots/README.md) record the source
model transform. The proxy omits individual rotor blades. A renderer must never
set the authoritative body pose.

Reward is `upright / (1 + distance_squared / (1 m²))`. Distance is measured from
`(0, 2, 0)` metres. `upright` is `(1 + body_up.dot(world_up)) / 2`, clamped to
`[0, 1]`. Ground contact or leaving `x,z ∈ [-10,10]` and `y ∈ [0,10]` metres
terminates the episode and earns zero. Reset starts upright with independent
uniform offsets of ±0.2 metres on X/Z and ±0.1 metres on Y.

The [28-line hover guide](../examples/robots/hover.rs) holds four commands of 0.5
for ten simulated seconds. It is a diagnostic baseline. No trained policy is
qualified for this task yet.

## Acceptance checklist

- [x] A runnable guide and external integration test fail because the API is absent.
- [x] Action endpoints 0 and 1 pass; adjacent out-of-range values, NaN, and both
      infinities fail independently in each of four motor positions.
- [x] Compile-fail evidence rejects unchecked actions and raw-world access.
- [x] Equal seeds reproduce reset and rollout; unseeded resets advance the stream.
- [x] Power-off acceleration matches gravity; symmetric hover preserves position
      and attitude; asymmetric thrust produces the predicted torque directions.
- [x] Thrust follows body orientation after rotation and across all four substeps.
- [x] Ground contact, region exit, repeated terminal steps, and reset after each
      termination have direct behavioral evidence.
- [x] A private invariant has an internal test after external tests pass.
- [x] Native and WASM builds use the same environment implementation.
- [x] Two Bevy environments run through typed action and transition messages.
- [ ] Native/WASM rollout comparison passes in an actual browser.
- [x] The model has provenance, a retained license, and a documented actuator map.
- [ ] Bevy desktop and browser render the real state; visual checks cover motion
      and repeated resets. Baseline and learned-policy labels remain distinct.
- [x] Required root Rust gates and changed-line personal lints pass.
- [x] New drone source lines and branches have 100% measured coverage.
- [ ] Full-package personal lints pass; baseline findings remain in the audit.
- [x] Final prose-only documentation, the guide, and documentation tests pass.
