# Initial quality audit

Baseline: `232e801`, inspected 2026-10-08. This is a scoped initial pass, not a
completed review of every production branch. Remaining areas are listed in
[EXAMPLE_STATUS.md](EXAMPLE_STATUS.md).

## Confirmed example problems

`unused-crate-dependencies = "warn"` in `Cargo.toml` applies to individual example
targets. The examples contain 432 `use crate_name as _;` lines. CartPole already
allows `unused_crate_dependencies` but still imports `tokio`, `clap`, optional
`avian2d`, and optional `mujoco_rs` only as `_`. These imports explain no CartPole
concept and do not make the tutorial easier to use. Audit side-effect registration
before removing any import. Allow the dependency lint explicitly for examples
instead of manufacturing dependency use.

Classic Control and Toy Text examples total 7,236 lines. CartPole alone has 1,719.
Their training, filesystem, process, video, and visualization workflows obscure
the smallest environment/policy loop. Preserve useful complete applications, but
move orchestration behind reusable implementation and expose short guided examples.
Do not replace the tutorial with a launcher that hides every relevant API.

`examples/mujoco-approximation/ant.rs` opens with a long copied environment reference
and observation table. This mixes API reference and tutorial purposes. Link to the
versioned contract and explain only the actions and observations used by the lesson.

## Confirmed delivery gaps

`gymnasium-web/index.html` exposes CartPole, MountainCar, continuous MountainCar,
and Acrobot through a selector. It is not the requested thumbnail-based Bevy-style
gallery. Pendulum has renderer work and pending qualification described elsewhere.
Current documents do not establish all 23 tasks as browser-ready.

The manifest declares only three native MuJoCo examples behind `mujoco`; eleven
separate approximation examples exist. The native feature depends on a native
MuJoCo runtime. These facts do not establish faithful browser MuJoCo support.

Existing six-frame reference sheets cover all 23 Gymnasium tasks. They establish
reference availability, not visual equivalence or trained-policy performance.

## Public API risks requiring discriminating tests

These are review targets, not proven defects without their stated counterexamples.

- `BevyGymPlugin::with_tick_rate` accepts an arbitrary `f64`, then passes it to
  `Time::<Fixed>::from_hz`. Test zero, negative, NaN, infinity, and valid endpoints.
  Prefer an additive validated rate API while preserving the existing signature.
- Plugin configuration and reset schedules are global resources. Test two distinct
  environment plugins with different seed schedules and reset settings before
  claiming isolation. Their message types are generic; their configuration is not.
- `ActionResponse` correlates by entity, without an observation/episode token.
  Stall response A, reset the environment, issue B, resolve B, then resolve A.
  Determine whether stale actions can step a new episode before proposing a fix.
- `EnvComponent.env` and observation/statistic fields are publicly mutable.
  Document intended advanced access and distinguish it from a future safe beginner
  integration path. Making these fields private would be an API migration.
- `step_system` clones observations/actions and allocates a map and outcome vector
  each tick. Measure realistic observation sizes and active environment counts
  before optimization. Do not infer a bottleneck solely from the implementation.

## Validation boundary

Baseline formatting passed. Full behavioral, compiler, lint, coverage, browser,
and deployment validation remains pending. No runtime bug is claimed solely from
API appearance, line count, or a historical handoff document.
