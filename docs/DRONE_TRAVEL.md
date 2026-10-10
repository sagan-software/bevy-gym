# Drone travel lesson

Status: reusable environment implemented; RL training, qualification, and scene
remain unfinished. Historical tracking weights are imitation-trained and do not
qualify this lesson.

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
The later neural-network encoding must state its scaling and heading representation.
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

The next implementation must sample progressively harder goals, transfer a qualified
RL actor, and collect rollouts through this same environment. Position and heading
must reach independent held-out gates. Approach speed and settling time need
explicit stage criteria before training. Keep policy inference failure visible.
Training and inference commands, qualified travel checkpoints, a standalone scene,
and recorded browser playback remain pending. Existing hover and recovery browser
recordings do not establish travel competence.

[flightmare]: https://github.com/uzh-rpg/flightmare/blob/master/flightlib/src/envs/quadrotor_env/quadrotor_env.cpp
