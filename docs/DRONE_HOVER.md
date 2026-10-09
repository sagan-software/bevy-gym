# Drone hover lesson

The native `drone-hover` command runs frozen RL inference. The shared trainer
supports standalone hover training. The separate scene runs the same frozen policy.

## Run

Run from the repository root:

```sh
nix develop --command cargo run --no-default-features --features robots --example drone-hover
```

Select another compatible checkpoint and reset seed explicitly:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-hover -- --checkpoint docs/progress/drone-curriculum.mpk --seed 42
```

The command prints the checkpoint path, inference mode, and complete episode score.
Missing, corrupt, or incompatible weights return an error before any action.
A custom checkpoint is not qualified merely because it loads or survives one episode.
The native CLI reads files; it is not a browser entry point.

## Scene

```sh
nix develop --command cargo run --features robots --example drone-hover-scene
```

The [browser build guide](../robot-web/skills/README.md) serves this lesson at
`/hover/`. Run, Step, and Reset control playback. The policy supplies every motor
command; the scene has no manual, imitation, or fallback controller.
Reset restores seed 42 and clears recurrent memory. Policy errors stop playback
and remain visible. The scene embeds the recorded checkpoint; custom checkpoint
selection remains available through the native inference command above.

## Environment and policy

The lesson uses `DroneHover::default()` and the same episode implementation as
[curriculum evaluation](DRONE_CURRICULUM.md). The solver owns the body pose.
The actor chooses all four motor fractions every 20 ms. Each fraction must be finite
and within `[0, 1]`, ordered front-left, front-right, rear-right, rear-left.
Inference retains recurrent memory within each episode and resets it between episodes.

The twelve inputs contain target displacement, world up, linear velocity, and angular
velocity in body-frame XYZ order. Displacement is divided by 2 metres, velocity by
2 m/s, and angular velocity by 2 rad/s. World up has unit length.
The fixed target is `(0, 2, 0)` metres. There is no route or targeting controller.

Reward is uprightness divided by `1 + distance_squared / (1 m²)`.
The [environment contract](ROBOT_ENVIRONMENT.md) defines uprightness, reset bounds,
contact termination, and flight-region termination. The episode ends after at most
500 actions. Scores report unscaled return, final target distance, and survival.

## Training and qualification

Train hover independently with the shared curriculum runner:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --lesson hover --seed 7 --updates 600 \
  --output runs/drone-hover/new-run
```

Omitting `--lesson` trains hover and then transfers its actor, critic, and optimizer
into disturbed recovery:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --seed 7 --updates 600 \
  --output runs/drone-curriculum/new-run
```

Use a new output directory. Standalone training starts from random weights.
Promotion requires all five selection episodes to survive, mean return of at least
400, and mean final distance of at most 0.5 metres. Budget exhaustion fails the lesson.

The default checkpoint is `docs/progress/drone-curriculum.mpk`, SHA-256
`8b94182a1368f5449a52db5c0859160fc5819d6772c0547f31d240df7a0a465a`.
The [existing evidence](progress/drone-curriculum.json) records PPO training without
imitation, seed 7, 280 hover updates, 20 recovery updates, and 153,600 transitions.
Its held-out recovery result was 32/32 survivors, mean return 443.4709, and mean
final distance 0.1580 metres. This does not qualify navigation, damage, or combat.

A separate calm-start replay passed all 32 held-out seeds in native and browser
tests. Native mean return was 464.4299, with mean final distance 0.06235 metres.
The [scene evidence](progress/drone-skill-scenes.json) includes the test gates,
coverage limits, and browser recordings.

The separate [recovery command](DRONE_RECOVERY.md) uses the same checkpoint with
disturbed starts. Historical constant-thrust evidence remains in the earlier
records; neither native lesson command uses that controller now.
