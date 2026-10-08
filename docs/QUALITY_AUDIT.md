# Initial quality audit

Baseline: `232e801`, inspected 2026-10-08. This is a scoped initial pass, not a
completed review of every production branch. Remaining areas are listed in
[EXAMPLE_STATUS.md](EXAMPLE_STATUS.md).

## Confirmed example problems

The dependency-only import problem is fixed. `unused-crate-dependencies` now uses
`allow` because all examples share package dependencies while each teaches a
subset. Removed bare dependency imports from all 45 existing example source files
that contained them, preserving named extension-trait imports. The direct search,
root tests, and all-target/all-feature strict Clippy pass. This cleanup does not
qualify the examples' dynamics, rendering, or learned policies.

Classic Control and Toy Text examples total 7,236 lines. CartPole alone has 1,719.
Their training, filesystem, process, video, and visualization workflows obscure
the smallest environment/policy loop. Preserve useful complete applications, but
move orchestration behind reusable implementation and expose short guided examples.
Do not replace the tutorial with a launcher that hides every relevant API.

`examples/mujoco-approximation/ant.rs` opens with a long copied environment reference
and observation table. This mixes API reference and tutorial purposes. Link to the
versioned contract and explain only the actions and observations used by the lesson.

## Confirmed delivery gaps

The new thumbnail index links to the manual drone viewer and five Classic Control
tasks. Pendulum now appears in the existing environment selector. The public
deployment remains gated by the browser workflow; the previous run timed out.
Current documents do not establish all 23 tasks as browser-ready.

The manifest declares only three native MuJoCo examples behind `mujoco`; eleven
separate approximation examples exist. The native feature depends on a native
MuJoCo runtime. These facts do not establish faithful browser MuJoCo support.

Existing six-frame reference sheets cover all 23 Gymnasium tasks. They establish
reference availability, not visual equivalence or trained-policy performance.

## Reset defects fixed

Automatic reset previously matched the public `EpisodeEndEvent` by pool-local
`EnvId`. Two distinct environment types could both have ID zero, so ending the
first also reset the second. `tests/reset_isolation.rs` reproduced this by expecting
zero completed episodes in the continuing environment and observing one.

Reset seed schedules were also a single resource. Registering a second plugin
replaced the first plugin's schedule. A regression expected seed 101 after the
first plugin's terminal step and observed 201.

Completion now travels through a private message parameterized by environment
type and addressed to the exact entity. Each spawned entity retains its own seed
schedule. Public message types and signatures are unchanged. Five regression
tests cover both defects, a despawn before automatic reset, a continuing step,
and a consumer clearing public transition reports before reset. Clearing those
reports cannot prevent the private lifecycle transition.

The stepper now separates parallel environment steps from serial message emission.
It derives the follow-up from the transition and preserves public message order.
No performance improvement is claimed. The reset and step source files have full
measured line and branch coverage in the focused suite.

## Public API risks requiring discriminating tests

These are review targets, not proven defects without their stated counterexamples.

- `BevyGymPlugin::with_tick_rate` accepts an arbitrary `f64`, then passes it to
  `Time::<Fixed>::from_hz`. Test zero, negative, NaN, infinity, and valid endpoints.
  Prefer an additive validated rate API while preserving the existing signature.
- Reset schedule isolation is fixed. `GymConfig` and `Time<Fixed>` remain global.
  Define the intended behavior when plugins request different tick rates before
  choosing a compatible configuration API.
- `ActionResponse` correlates by entity, without an observation/episode token.
  Stall response A, reset the environment, issue B, resolve B, then resolve A.
  Determine whether stale actions can step a new episode before proposing a fix.
- `EnvComponent.env` and observation/statistic fields are publicly mutable.
  Document intended advanced access and distinguish it from a future safe beginner
  integration path. Making these fields private would be an API migration.
- `step_system` clones observations/actions and allocates a map and outcome vector
  each tick. Measure realistic observation sizes and active environment counts
  before optimization. Do not infer a bottleneck solely from the implementation.

## Personal-lint backlog

Repository strict Clippy passes. The stricter personal suite still reports 50
errors and two warnings in unchanged code. The
[diagnostic inventory](progress/personal-lint-backlog.json) records each location.
Most findings concern function length; others concern complexity, wildcard enum
matches, midpoint expressions, and a stale lint expectation. Fix these in scoped
follow-up changes with behavioral tests. Do not hide them with broader allowances.
The user-requested unused-dependency allowance is the only lint-policy exception.

The same personal rules report no diagnostics on this checkpoint's changed lines.
This scoped pass does not make the full-package gate green.

## Validation boundary

The exact root `cargo test`, formatting, all-target/all-feature strict Clippy, and
robot WASM compile checks pass. The drone suite covers eight external contracts
and five internal invariants. Manual browser flight has visual evidence; no trained
drone controller is qualified.
[Coverage evidence](progress/drone-foundation-coverage.json) records source hashes,
command scope, full-file coverage, and uncovered changed lines. All 339 instrumented
added source lines were hit, including internal test code. No added instrumented
line is uncovered. The full `plugin.rs` line rate is 75.54%; its uncovered paths
are outside the changed lines. Module declarations and comments are not executable
coverage targets. Two compile-fail documentation tests separately protect unchecked
action construction and raw-world access.

The viewer adds twelve tests for playback, completion, reset, keyboard shortcuts,
clicks, model alignment, rotor pivots, and read-only projection. Its session has
full measured coverage. Graphics startup and asset-state projection retain gaps
in [the viewer record](progress/drone-viewer-coverage.json). The default coverage
export omitted example source; the record uses the instrumented example executable.
Browser interaction is separate evidence and does not count as native coverage.

The current deployment still offers three Classic Control tasks. Existing source
work for additional tasks does not establish deployed qualification. Every remaining
area in the status document still requires its own behavioral and visual evidence.
