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

The public `PpoTrainer` currently stores configuration without a training method.
`PpoAgent` constructs an actor and critic and exposes inference, but has no optimizer
update. These types document a trainer boundary; they do not provide a usable
training entry point. The recovery lesson uses the working `RecurrentPpoAgent`.
Review these overlapping names before designing a smaller public training API.

`ContinuousPpoExample` requires `Vec<f32>` actions and observations, which forces
typed environments through adapters. The recovery collector keeps `DroneAction`
and `DroneObservation` through simulation and converts them only at the network
boundary. Its private lesson code can inform a later compatible public API.

The synchronous training guide also exposed a lint that bans `std::fs` writes.
A function-local exception keeps checkpoint output synchronous. The guide uses
no asynchronous runtime solely to satisfy that lint.

The thumbnail index links to the drone viewer and five Classic Control tasks.
The local drone viewer offers learned inference and manual control. Pendulum now
appears in the existing environment selector. The public gallery and manual viewer
are deployed; newer checkpoints remain gated by the browser workflow.
Current documents do not establish all 23 tasks as browser-ready.

The manifest declares only three native MuJoCo examples behind `mujoco`; eleven
separate approximation examples exist. The native feature depends on a native
MuJoCo runtime. These facts do not establish faithful browser MuJoCo support.

Existing six-frame reference sheets cover all 23 Gymnasium tasks. They establish
reference availability, not visual equivalence or trained-policy performance.

## Local package gate, 2026-10-09

`cargo package --list --allow-dirty -p bevy-gym` fails with
`No such file or directory (os error 2)` in the quality worktree. A file-system
trace ends while Cargo opens the missing nested reference path
`ref/godot-rl-agents-examples/godot_rl_agents_plugin`. The trace is retained in
`ragdoll-package-files.trace` under the validation cache. Packaging remains unverified.

The shooter viewer now owns its unpublished vendored ragdoll dependencies in a
separate `publish = false` workspace package. The library manifest has no normal
path-only ragdoll dependencies. Cargo requires published normal path dependencies
to have registry versions; see the [Cargo publication rules](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#local-paths-in-published-crates).

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

Repository strict Clippy passes. The stricter personal suite still reports 51
errors and two warnings in unchanged code. The
[diagnostic inventory](progress/personal-lint-backlog.json) records each location.
Most findings concern function length; others concern complexity, wildcard enum
matches, midpoint expressions, and a stale lint expectation. Fix these in scoped
follow-up changes with behavioral tests. Do not hide them with broader allowances.

This audit changed the workspace lint level only for unused dependencies. The
training guide also retains the scoped synchronous-file exception described above.

The strict pass stops in library code, so its filtered output cannot qualify an
example it never checks. A separate discovery pass retains warnings without
promoting them to errors and reaches the inference viewer. It reports no viewer
diagnostics. The full-package strict gate remains unresolved.

## Validation boundary

The exact root `cargo test`, formatting, all-target/all-feature strict Clippy, and
robot WASM compile checks pass. The drone suite covers eight external contracts
and five internal invariants. The recovery policy now passes separate native and
browser qualification. [The learning guide](DRONE_LEARNING.md) records the limits
of those results; damage recovery, perception, and pursuit remain unqualified.

[Coverage evidence](progress/drone-foundation-coverage.json) records source hashes,
command scope, full-file coverage, and uncovered changed lines. All 339 instrumented
added source lines were hit, including internal test code. No added instrumented
line is uncovered. The full `plugin.rs` line rate is 75.54%; its uncovered paths
are outside the changed lines. Module declarations and comments are not executable
coverage targets. Two compile-fail documentation tests separately protect unchecked
action construction and raw-world access.

The viewer now has 25 tests for playback, completion, reset, keyboard shortcuts,
clicks, model alignment, rotor pivots, policy failures, and read-only projection.
Its session and controller have full measured line coverage. Graphics startup and
asset-state projection retain gaps in the
[inference record](progress/drone-inference-coverage.json). The default coverage
export omits example source when Cargo uses a separate build directory; the record
explicitly exports the instrumented example executable.
Browser interaction is separate evidence and does not count as native coverage.

Existing source work for additional tasks does not establish deployed qualification. Every remaining
area in the status document still requires its own behavioral and visual evidence.
