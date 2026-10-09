# Drone recovery lesson

Status: native and browser frozen-policy inference implemented, 2026-10-09.
Standalone checkpoint transfer remains pending.

Use `DroneHover::disturbed()` for a hover task that needs feedback control.
`DroneHover::default()` and its committed traces remain unchanged. The new constructor
uses the same validated actions, observations, dynamics, reward, and termination rules.

[Flightmare's pinned reset implementation][flightmare-reset]
randomizes position, linear velocity, and normalized orientation before a flight.
This supports varying initial conditions during training. Our lesson uses the bounded
distribution below; it does not reproduce Flightmare's reset distribution or reward.

## Invariants and API choice

- Episode states remain flying and ended. Reset profiles are privately either calm
  or disturbed. The profile persists across every reset.
- Seeds accept the existing full `u64` range. Only `DroneAction` accepts externally
  supplied motor values; its validation remains unchanged.
- Every observation is derived from the private solver. No new state setter,
  unchecked action path, or engine accessor is public.
- Add one named constructor. The initial episode uses seed zero; explicit seeded
  resets restart the stream, and omitted seeds continue it.
- Keep the calm position offsets. Disturbed resets also sample independent yaw in
  ±π radians, pitch and roll in ±π/12 radians, world linear velocity in ±0.5 m/s
  per axis, and world angular velocity in ±0.5 rad/s per axis.
- Compose orientation as `R_y(yaw) * R_x(pitch) * R_z(roll)`, with the rightmost
  rotation applied first. Use the deterministic physics rotation implementation.
- Draw position, yaw, pitch, roll, linear velocity, and angular velocity in that
  order. Physics steps do not consume the reset stream.
- Generate only finite, bounded states. The initial body remains above the floor.
  Rebuild solver state before each episode, including after a crash.

The reset adds constant work and storage for one body. It creates no public mutable
configuration. The named constructor avoids a second environment type with identical
actions and observations. Later curriculum stages need their own reviewed contract.

## Acceptance checklist

- [x] A runnable guide and external tests fail because the new constructor is absent.
- [x] The constructor and every reset produce the documented finite state distribution.
- [x] Seeded and continuing resets reproduce trajectories after termination.
- [x] Constant half-thrust fails the five held-out seeds within 500 actions.
- [x] Existing calm traces remain byte-for-byte unchanged.
- [x] Disturbed traces and all public tests match in the browser.
- [x] The viewer can select calm or disturbed starts and shows the selected lesson.
- [x] Show a real browser recording of both starts and inspect the narrow layout.
- [x] Cover new physics and session branches; record remaining UI coverage gaps.
- [x] Run required Rust gates and changed-line personal lints.
- [x] Finish with prose-only documentation, the guide, and documentation tests.

The [learning lesson](DRONE_LEARNING.md) now supplies a qualified checkpoint and
held-out comparisons against constant half-thrust. Learned inference and training
controls are available in the existing combined viewer.

## Run and inspect

The [recovery command](../examples/robots/recovery.rs) now loads the qualified
RL curriculum checkpoint and runs a disturbed episode:

```sh
nix develop --command cargo run --no-default-features --features robots --example drone-recovery
```

Use `--checkpoint PATH --seed 42` to select compatible weights and a reset seed.
Missing, corrupt, or incompatible weights fail before any environment action.
The command reports checkpoint path, inference mode, and the complete episode score.
It shares the [hover lesson's](DRONE_HOVER.md) observation/action contract, reward,
episode horizon, recurrent memory handling, and checkpoint provenance.
Only the reset distribution changes, as specified above.

Run recovery independently through the shared curriculum trainer:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --lesson recovery --seed 7 --updates 600 \
  --output runs/drone-recovery/new-run
```

This mode starts from random weights. When `--lesson` is omitted, the
[curriculum trainer](DRONE_CURRICULUM.md) transfers hover weights and optimizer
state into recovery. The older [direct trainer](DRONE_LEARNING.md) also starts
recovery from random weights. All paths use `DroneHover::disturbed()`.
Promotion requires five selection survivors, mean return at least 400, and mean
final distance at most 0.5 metres. The frozen checkpoint survived all 32 held-out
recovery episodes. Navigation, combat, and damaged flight remain unqualified.

The earlier constant-thrust guide terminated seed 42 during action 274, after at
most 5.48 simulated seconds. Its initial linear velocity was approximately
`(-0.282, 0.301, -0.160)` m/s. All five baseline seeds, 0, 1, 2, 42, and `u64::MAX`,
terminated before 500 actions. Those results are historical physics evidence.

### Separate frozen-policy scene

```sh
nix develop --command cargo run --features robots --example drone-recovery-scene
```

The [browser build guide](../robot-web/skills/README.md) serves this lesson at
`/recovery/`. The scene embeds the recorded RL curriculum checkpoint and exposes
only Run, Step, and Reset. Reset restores seed 42 and clears recurrent memory.
Every motor command comes from inference; failures stop playback visibly.
The [recording](progress/skill-scenes/recovery-desktop.mp4) completed 500 actions
and ended 0.26 metres from the target. This one episode illustrates playback;
the 32-seed evaluation establishes qualification. See the
[validation record](progress/drone-skill-scenes.json) for gates and coverage gaps.

### Historical viewer evidence

Open the viewer, then select Calm start or Disturbed start:

```sh
nix develop --command cargo run --features robots --example drone-flight
```

Selecting a start creates a paused seed-42 episode. Reset retains that choice and
restores the hover command. The [optimized browser recording](progress/drone-recovery.mp4)
shows both starts, termination, reset, and a paused single step. The
[390-pixel frame](progress/drone-recovery-narrow.png) shows the wrapping controls.
Native window interaction and mobile device input remain unverified.

The [coverage record](progress/drone-recovery-coverage.json) records all gates and
their limits. Physics and session code have 100% measured native line and branch
coverage. New UI construction lines lack native coverage hits; browser rendering
and interaction supply separate behavioral evidence. Full-package personal lints
retain 51 errors and two warnings in unchanged files.

[flightmare-reset]: https://github.com/uzh-rpg/flightmare/blob/d4218aedac18cbe9364a0a0df10ab992c4b65e4f/flightlib/src/envs/quadrotor_env/quadrotor_env.cpp
