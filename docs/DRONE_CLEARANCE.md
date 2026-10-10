# Drone clearance

Status: the core clearance environment and physical range observations are implemented.
Geometry curriculum, training, frozen inference and qualification remain unfinished.
No clearance policy is bundled.

## Quick start

Inspect the fixed destination and six sensor readings without issuing an agent action:

```sh
nix develop --command cargo run --no-default-features --features robots --example drone-clearance
```

`DroneClearance::new(destination, boxes)` accepts validated `DroneDestination` and
`DroneObstacle` values. Its `Env` implementation reuses disturbed travel physics,
motor commands and rewards. Reset retains the destination and boxes.
`observation().travel()` reads body and task information; `observation().ranges()`
reads that same physical snapshot's six local rays. Snapshot fields stay private.

The example prints observations only. It has no agent, training mode or checkpoint.
The clearance actor encoding, opening/braking/cover curriculum and qualification
cases remain unfinished. Travel qualification remains a prerequisite for RL transfer.

For a range-only snapshot:

```sh
nix develop --command cargo run --no-default-features --features robots --example drone-ranges
```

`DroneHover::ranges()` reads one physical snapshot. Use
`DroneRanges::distance(DroneRangeDirection::Down)` to read a direction.
A hit returns `Some(DroneRangeDistance)`; `metres()` reads its distance.
A missed ray returns `None`.

Use `DroneHover::disturbed_with_obstacles(boxes)` to combine validated collision boxes
with the recovery lesson's disturbed initial motion. Empty input matches
`DroneHover::disturbed()` exactly. Reset retains the boxes and restores working motors.
Any solid contact ends flight; the caller must supply feasible lesson geometry.

The [geometry evidence](progress/drone-disturbed-obstacles.json) records seed-zero
construction, seeded and unseeded reset streams, platform contact and frozen terminal reads.
That milestone passed 236 browser tests and 128 focused native robot tests.
Default native tests, formatting, strict Clippy and changed-line personal Rust lint passed.
Its coverage hit both used constructor instantiations; two unused generic
placeholder instances had zero counts. It added no production branch.

## Range contract

The local sensor profile uses six body-frame rays in this order:

- `Forward`: local -Z.
- `Back`: local +Z.
- `Left`: local -X.
- `Right`: local +X.
- `Up`: local +Y.
- `Down`: local -Y.

Every ray starts at the physical body centre and rotates with body orientation.
The closest solid hit within ten metres is returned. The floor and fixed obstacles
are included; the drone collider and sensor volumes are excluded. A hit at exactly
ten metres remains `Some`; an origin inside another solid reports zero.
Distances start at the centre and do not represent swept-body clearance or surface gaps.

`DroneRangeDistance::try_from(f32)` rejects NaN and either infinity with
`InvalidDroneRangeDistance::NonFinite`. It then rejects finite values outside
[0, 10] metres with `OutOfRange`. Accepted bits, including negative zero, are preserved.
The tuple field and reading array are private. Directions use a closed enum;
callers cannot supply an unchecked numeric direction or access the physics world.

Reads use current collider poses, including before the first solver step.
They never advance physics, consume reset randomness or issue an actuator command.
Six rays over N colliders take O(N) time and O(1) auxiliary space.
The [Parry 0.31.1 ray API][parry-ray]
controls geometric intersections; the ten-metre limit and direction profile are local policy.
The [Rapier 0.36.0 world API][rapier-world]
owns the colliders. This implementation scans them directly without refreshing
broad-phase state during a read.

## Verification and remaining work

The [environment evidence](progress/drone-clearance-environment.json) records the
missing-API failure, staged implementation, native/WASM checks and coverage.
All 238 browser tests, default native tests and 131 focused robot tests pass.
Formatting, strict Clippy and changed-line personal Rust lint pass.
Every new function definition and region executes, including the runnable example
and private task test. Two unused linked `Env::reset` and `Env::step` copies remain
unhit; each method executes in public integration tests.

Compile-fail documentation prevents access to owned physics or construction of
mixed snapshots. Empty geometry preserves travel observations, reward bits, status
and reset streams. Platform tests prove retained geometry and destination, zero
terminal reward and frozen body/range snapshots. No avoidance skill is inferred from tests.

The [evidence record](progress/drone-ranges.json) preserves source hashes, the missing-API
failure, exact checks, boundary cases and coverage. That milestone passed 234 browser tests,
including the range contract, and 126 focused robot tests. Default native tests,
formatting, strict repository Clippy and changed-line personal Rust lint passed.

Its coverage executed every instrumented line, function and branch in the three range
modules, including their internal test, and the runnable example. The public hover
wrapper is executed. Enum declarations and constant data have no executable line counts.
The private solver-invariant panic at `src/robots/ranges/observation.rs:70` is unhit;
public callers cannot inject malformed solver output. No allocation measurement is claimed.

The clearance lesson still needs its geometry curriculum, actor encoding, promotion
gates, RL training, qualified checkpoint and browser scene. Its core environment
reuses travel rewards, physical sensors and validated motor actions. Policies must
choose every avoidance and braking action; a planner cannot provide routes or controls.
Travel qualification remains a prerequisite. The complete 3v3 environment remains unfinished.

[parry-ray]: https://docs.rs/parry3d/0.31.1/parry3d/query/trait.RayCast.html#method.cast_ray

[rapier-world]: https://docs.rs/rapier3d/0.36.0/rapier3d/pipeline/struct.PhysicsWorld.html
