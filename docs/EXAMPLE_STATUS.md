# Examples execution status

Updated: 2026-10-08. Goal status: active.

## Resume here

Read [the roadmap](EXAMPLE_ROADMAP.md), [quality audit](QUALITY_AUDIT.md),
[reference research](EXAMPLE_RESEARCH.md), and [drone contract](ROBOT_ENVIRONMENT.md).
Continue P1 with learned recovery control, batched training, and a curriculum.
The disturbed-start task and native/browser comparison pass locally. The manual
viewer and first comparison checkpoint are pushed; monitor CI and deployment.
Manual Bevy flight runs in WASM.
Perception, damage adaptation, pursuit, and the jumping quadruped remain pending.
Do not start later port families before the custom robot milestones.

Working checkout: `/home/sagan/Code/github.com/sagan-software/bevy-gym-quality`.
Branch: `quality-roadmap-20261008`. Baseline: `232e801`.
Original checkout: `/home/sagan/Code/github.com/sagan-software/bevy-gym`.
It has 1,304 staged deletions, four additions, and three untracked example folders.
Preserve its index and working files. Do not use `git add -A` there.

GitHub remote is `github`; `origin` is a separate GitLab repository. Publish
authorized milestones to GitHub `main`. Configured identity is Bill Curry,
`bill@sagan.software`. Inspect author, committer, message, and trailers before
finalizing each commit. Never add AI attribution.

## Completed work

- Created the persistent goal, isolated worktree, complete roadmap, and initial audit.
  No matching tracked reviewer instructions were found outside the `ai` instructions.
  Root `AGENTS.md` is an ignored broken Nix-store symlink; use session instructions.
- Inventoried all 23 Gymnasium reference sheets. Located pinned Unity scenes and
  all eleven Godot README-video source folders and per-example notices.
- Downloaded and inspected all 17 Unity images and eleven Godot videos through
  individual 20-frame contact sheets. Verified the 28 media hashes.
- Downloaded the official ARC Raiders documentary and Leaper gameplay. Inspected
  aerial, damage, locomotion, training, and dense motion sheets. A continuous
  jump clip remains necessary for measured timing. Both video hashes are recorded.
- Verified AI Warehouse's public README says its example code is private.
- Confirmed installed `yt-dlp` 2026.08.19 and FFmpeg. `yt-dlp` is already committed
  in NixOS dotfiles at `modules/dev/default.nix:156`; T490 enables that module.
  No dotfiles edit or activation is needed.
- Retained Nate Gazzard's CC BY 3.0 quadrotor GLB, license, attribution, and motor map.
  Inspected its four named rotors and 4,564 triangles in a rendered view and video.
  Shared the [screenshot](progress/drone-model.png) and
  [turntable recording](progress/drone-model.mp4) with the user. These show asset
  inspection, not Bevy flight or learned control.
- Allowed `unused_crate_dependencies` and removed bare dependency-only imports
  from 45 existing example files. Extension-trait imports remain named.
- Reproduced and fixed cross-environment automatic resets and overwritten seed
  schedules. Five external regression tests protect isolation and reset lifecycle.
- Implemented the optional `robots` module with validated `DroneAction`, read-only
  `DroneObservation`, and a private Rapier 0.36 world in `DroneHover`.
  The 28-line guide demonstrates a seeded, time-limited constant hover command.
- Added native robot tests and a robot WASM compile check to CI.

## Verification

Passed after the source changes:

- `nix develop --command cargo fmt --all -- --check`.
- `nix develop --command cargo test`: 119 unit/integration tests and two documentation
  tests passed. One existing plugin documentation example remains ignored.
- `nix develop --command cargo test --features robots --lib --test drone_hover`:
  87 library tests and eight external drone tests passed.
- `nix develop --command cargo clippy --all-targets --all-features -- -D warnings`.
- `nix develop --command cargo check --no-default-features --features robots,browser --target wasm32-unknown-unknown`.
- Focused branch coverage: 87 library tests, eight drone tests, and five reset tests.
  All new drone source lines and branches were hit. All 339 instrumented added
  source lines were hit, including internal test code. The changed reset and step
  source files have full line and branch coverage. Source hashes and exact scope
  are in [the coverage record](progress/drone-foundation-coverage.json).
- The personal lint suite reports no diagnostics on changed lines against `9cc5672`.
  Its command uses `--rust --package bevy-gym --all-targets --all-features --no-deps
  --extra-rustc-arg=-Aunused_crate_dependencies --changed-range 9cc5672` through the
  installed `scripts/nix-safe run lint -- --repo` entry point.

The full-package personal gate remains unresolved: 50 errors and two warnings
outside changed lines. [The audit](QUALITY_AUDIT.md) and
[diagnostic inventory](progress/personal-lint-backlog.json) retain them. The scoped
pass does not override that failure. Full Nix flake checks have not run in this audit.

The headless guide passed: seed 42 retained position
`(0.096625954, 1.931982, -0.088559546)` metres across ten simulated seconds.
`cargo test --features robots --doc` passed four tests, including both new
compile-fail guards; one existing plugin example remains ignored. Changed Markdown
and Cargo TOML checks pass. The remaining CI commands also passed: both Python
contract/dependency checks, robot tests without default features, both browser
package test suites, and browser-package Clippy for native and WASM targets.
Opt-in long training qualification tests remain ignored by their existing configuration.
The new viewer has run in the T3 browser with manual controls. A trained drone policy
and new deployment remain unqualified. Native/browser comparison now passes below.

Earlier missing-API tests failed with `E0432` before production code existed.
The initial reset regressions also failed before their fixes. These logs are retained
as `drone-red.log` and `reset-red.log`. Runtime coverage does not replace the
compile-fail checks for unchecked action construction and raw-world access.

## Browser and visual status

T3 preview is available. Inspected the official Bevy gallery and deployed Bevy Gym
page. The deployment offers three Classic Control environments. Started fresh
CartPole training and observed replay collection, then paused it. No optimizer
update was observed during that initial review. The preview reports `visible: false`;
the page reports
`visibilityState: visible`. Foreground training speed remains unverified.
The preview resize tool failed. Narrow layouts were instead inspected in
390-pixel same-origin frames; mobile device and touch behavior remain unverified.
Historical browser qualification is not current audit evidence.

The manual viewer checkpoint lives in `examples/robots/flight.rs`, its `flight/` modules,
`robot-web/`, `gallery/`, and `scripts/build_drone_viewer.sh`. Twelve viewer tests pass.
A click-event regression failed before switching buttons to Bevy pointer click observers.
Buttons use one hit target. Apparent browser click misses were traced to preview input:
the tool omits movement, and synthetic offsets needed correction for 1.75× display scaling.
Corrected pointer input verified reset, motor selection, single-step, run, and pause.
The runtime remains manual control; rotor spin illustrates thrust rather than measured RPM.

The [flight recording](progress/drone-flight.mp4) shows hover, power-off, ground contact,
climb, pause, reset, and asymmetric thrust. Its [contact sheet](progress/drone-flight-contact-sheet.png)
was inspected. Desktop and 390-pixel frame layouts were inspected. The preview resize
API remains unavailable, so this does not establish mobile-device or touch qualification.
The initial debug WASM is 103 MB. The tested optimized release is 34,801,765 bytes;
local gzip level 9 produces 10,685,070 bytes. This is not a measured network transfer.
`bevy-gym-flight-release-build-20261008` passed; its log is `flight-release-build.log`.
The abandoned shared-cache release unit was stopped to let tests use the build lock.

The local viewer is at `http://100.105.254.50:8767/`; the local gallery review is at
`http://100.105.254.50:8768/examples/`. Both serve ignored build/review directories.
The public routes will be `/bevy-gym/robots/hover/` and `/bevy-gym/examples/` after deployment.

A temporary asset viewer serves only `runs/quality-research/media` at
`http://100.105.254.50:8766/drone-viewer.html` through
`bevy-gym-drone-viewer-lan-20261008.service`. It is an inspection page, not a
published example. Use the flight recording above for the Bevy physics state.

## Viewer checks and remaining gaps

The final Rust edits passed the exact root test, formatting, and strict Clippy commands,
the twelve viewer tests, debug and optimized WASM builds, and personal lints scoped
against `decad5c`. The full-package personal backlog remains unresolved. Shell lint,
JavaScript syntax, workflow syntax, six gallery routes, and all thumbnail files pass.

[Viewer coverage](progress/drone-viewer-coverage.json) records the final source hashes.
The session has full line and branch coverage. Native window setup, graphics/asset
initialization, loading/failure text, and color projection retain instrumented gaps.
The CPU tests do not start a native window. Asset-loader failure fixtures remain
pending; the successful browser load does not cover those failure branches.
Browser video covers visible behavior; it does not supply native coverage hits.
The native window and touch input remain unverified.

## Native/browser physics comparison

The [comparison guide](DRONE_PARITY.md) and [evidence record](progress/drone-parity.json)
record 1,871 complete native/browser results with identical float bits and statuses.
The ten cases cover hover, falling, climbing, roll, pitch, yaw, changing motors,
absorbing termination, seeded reset, and continuing reset. All eight public drone
contract tests and four recovery tests also pass in the T3 browser. Screenshots
were shown to the user.

The fixture contains 24,323 observation floats and 1,841 rewards. It is generated
from native physics, so independent gravity and torque checks remain necessary.
Both compared targets used test-profile builds. Other configurations are unqualified.
Root Rust gates, WASM Clippy, Nix lint, and actionlint passed. Personal lints found
no changed-line diagnostics; the current full scan retains 51 distinct errors and
two warnings in unchanged files. The ignored fixture generator's ten lines remain
unhit in the coverage run; the comparison itself has full measured branch coverage.

`nix run .#drone-browser-check` is wired into CI. Its wrapper built and printed help
locally; the first headless CI execution is pending. Interactive local suites ran on
ports 8769, 8770, and 8771. Their `wasm-bindgen-test-runner` servers remain in user units.
The first comparison checkpoint is pushed as `9f98657`.

## Disturbed starts

`DroneHover::disturbed()` adds finite initial tilt and velocity while preserving
the calm constructor and all 1,420 prior trace results. The [recovery contract](DRONE_RECOVERY.md)
defines the bounds, rotation order, reset behavior, and acceptance checks.
The 25-line recovery guide demonstrates the constant-thrust failure case.
Seed 42 terminates during action 274, after at most 5.48 simulated seconds.

The viewer offers Calm start and Disturbed start; reset preserves the choice.
The optimized [recording](progress/drone-recovery.mp4) shows both profiles, drift,
termination, reset, and a single step. The desktop and 390-pixel frame were inspected.
All 16 viewer tests pass. Physics and session code have full measured native line
and branch coverage; [the record](progress/drone-recovery-coverage.json) retains
UI construction and asset-state coverage gaps. Root tests, strict Clippy, WASM Clippy,
Nix lint, and actionlint pass. Both guides and documentation tests pass after the
final prose edit. This checkpoint awaits commit and publication.

There is no learned drone controller yet. Constant half-thrust solves calm hover
but fails the five recovery seeds. Require held-out improvement before describing
a policy as learned recovery control. Training seeds must exclude evaluation seeds.

## Published checkpoints

- `2843952`: roadmap and initial audit. GitHub
  [CI run 37809258024](https://github.com/sagan-software/bevy-gym/actions/runs/37809258024) passed.
- `9cc5672`: inspected visual references. GitHub
  [CI run 37819131154](https://github.com/sagan-software/bevy-gym/actions/runs/37819131154) passed.
- `5b4a87c`: remove dependency-only example imports. Pushed to GitHub `main`.
- `f268233`: isolate automatic resets and seed schedules. Pushed to GitHub `main`.
- `decad5c`: typed drone hover, licensed asset, guide, and validation evidence.
  Pushed to GitHub `main`; remote revision verified.
  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37830890969) passed.
  [Browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37830890801)
  reached its 45-minute job limit during Gymnasium qualification. Deployment was skipped.
  The viewer checkpoint raises the job limit to 90 minutes without removing tests.
- `bfe088d`: manual drone viewer, licensed font, examples index, recordings, and
  coverage gaps. Pushed to GitHub `main`; remote revision verified.
  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37841619148) passed.
  [Browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37841619198)
  is running. The new public routes are not yet verified.
- `9f98657`: native/browser bit comparison and browser contract-test harness.
  Pushed to GitHub `main`; remote revision verified.
  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37847009522) is running.
  [Browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37847009460)
  is pending behind the viewer build. Pages already uses `cancel-in-progress: false`;
  new checkpoints do not cancel the active build.

## Local evidence and active validation

Another agent freed disk space. This run has not removed shared Cargo caches.
Builds, targets, logs, and reference media live under
`/home/sagan/.cache/bevy-gym-quality-validation/`. The `/dev/shm/bevy-gym-quality-*`
paths are symlinks to those directories. Recreate them after a reboot. Builds use
four jobs, or two for coverage/WASM, with development and test debug information off.

Detached units use `bevy-gym-*-20261008.service`. A passed unit has
`SubState=exited` and `ExecMainStatus=0`; status zero while running is not a pass.
Logs append across reruns, so read the final invocation.
Both `bevy-gym-quality-ci-remainder-20261008` and `bevy-gym-drone-guides-20261008`
passed. All viewer validation units passed. The release build and WASM execution
also passed.
Full coverage uses `RUSTC_BOOTSTRAP=1` only for nightly branch instrumentation.
The command and output scopes are retained in the coverage record.

Research downloads are under ignored `runs/quality-research/`; durable source URLs,
revisions, hashes, and conclusions are committed. Do not depend on ignored files
as the only record of a completed check. `HANDOFF.md` has ten pre-existing `MD060`
table diagnostics; this run's handoff notice does not change those tables.

## Pending audit areas

Public configuration validation and global tick-rate behavior; asynchronous response
correlation; all environment transitions; trainer numerical correctness; recurrence
and multi-agent accounting; checkpoint validation; worker cancellation and stale
results; rendering fidelity; asset integrity; tutorial paths; complete CI and gallery.

## Handoff update rule

Show screenshots and short videos at each visual checkpoint. Label source references,
asset inspection, simulations, and learned-policy recordings accurately. The user
explicitly requested regular visual updates on 2026-10-08.

After each checkpoint, record its commit, pushed revision, CI/deployment outcome,
tests and exact coverage gaps, visual artifacts, learning evidence, blockers, and
one concrete next action. Keep the goal active until the entire roadmap is done.
