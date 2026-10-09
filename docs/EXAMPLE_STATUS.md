# Examples execution status

Updated: 2026-10-08. Goal status: active.

## Resume here

Read [the roadmap](EXAMPLE_ROADMAP.md), [quality audit](QUALITY_AUDIT.md),
[reference research](EXAMPLE_RESEARCH.md), and [drone contract](ROBOT_ENVIRONMENT.md).

The [browser curriculum](DRONE_BROWSER_TRAINING.md) is qualified. Seed 7 passed
calm hover at update 300 and disturbed recovery after 20 more updates. Its saved
checkpoint survived 32/32 held-out episodes natively and passes the same frozen
qualification in WASM. Mean return is 426.9936; mean final distance is 0.3300 m.

The earlier native curriculum passed at 300 total updates. Both checkpoints and
their separate evidence remain under `docs/progress`.

Browser curriculum controls and worker transitions are published as `5c40d23`.

[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37874595348) passed;
[Pages](https://github.com/sagan-software/bevy-gym/actions/runs/37874595397) is running.

The final 25 worker tests, 20 controller tests, fifteen browser learning tests,
root tests, and strict native/WASM Clippy pass. The selected-worker personal
checks pass with no raw diagnostics. Three large test functions missed by earlier
changed-line filtering were split by behavior. Coverage and the optimized release
build pass. The full root personal-lint backlog remains.

The seed-7 browser run is complete. Do not retrain it to recover evidence.

`docs/progress/drone-browser-curriculum.json` contains the run and qualification;
`drone-browser-curriculum.mpk` contains its final 46,343-byte checkpoint.

The saved curriculum bundle remains at `http://100.105.254.50:8777/robots/hover/`.

Its server is `bevy-gym-curriculum-ui-server-20261009.service`, and its runtime is
`build-460aff23e4292a9214631a7180419359b53bd03d4bc1ae1898455e45b38d883b`.

The final gate units end with `final-gates2-20261009`, `final-wasm-20261009`,
`final-personal2-20261009`, and `final-coverage-20261009`, prefixed by
`bevy-gym-browser-curriculum-`.

Logs are under `/home/sagan/.cache/bevy-gym-quality-validation/logs`.

The final coverage, native/strict, and personal-lint units exited successfully.

The guide/documentation command also finished: eight curriculum tests and six
documentation tests passed; one existing plugin example remains ignored.

The [motor-failure API and guide](DRONE_DAMAGE.md) are pushed as `c3ae4ad`.
Native/WASM tests cover all sixteen actuator combinations, reset, and terminal
rejection. Compile-fail tests protect identifiers and private health state.
[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37871626023) passed;
[Pages](https://github.com/sagan-software/bevy-gym/actions/runs/37871625987) passed.
The native curriculum's earlier pending Pages build was superseded.
The damage model currently preserves mass and collision geometry.

The inspected Leaper footage and licensed animated target candidate are published
as `17c92c2`. [The research record](EXAMPLE_RESEARCH.md) links their visual evidence.
The target's valid GLB is under `runs/quality-research/media/quaternius-adventurer.glb`;
the earlier `.gltf` download is a quota-error HTML page and must not be used.

The visible actuator-failure checkpoint is published as `0821b4d`.
[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37877715282) passed;
[Pages](https://github.com/sagan-software/bevy-gym/actions/runs/37877715381) is pending.
All 35 viewer tests,
four public damage tests, root tests, strict native/WASM Clippy, and actual browser
robot tests pass. Personal discovery found no diagnostics in the changed robot
files. The full personal strict gate retains its existing library/test backlog.
Six documentation tests pass; one existing plugin example is ignored.

The browser demonstrates paused failure, a stopped rotor, its red marker, a
36-action fall under constant half-thrust, and reset. The final desktop and narrow
views pass without temporary CSS overrides. The accessible canvas label includes
F. Small-screen scene height is now at least 760 pixels, which clears the control
panel from the drone. Native window and mobile touch input remain unverified.
[The evidence record](progress/drone-visible-damage.json) retains artifact hashes,
validation counts, runtime versions, and the recording's verification boundary.

`DroneHover::observation` exposes a read-only snapshot for paused viewers. Its
external regression first failed with `E0624`, then passed. Viewer tests first
failed on the missing control/session/marker API, then passed. Clippy required a
`# Panics` section for the private-body invariant; the final gate passes.

Completed validation services use the prefix `bevy-gym-visible-damage-` and suffix
`-20261009`: `gates2`, `wasm`, `personal2`, `coverage2`, and `release2`.
Logs remain in the validation log directory. The first personal invocation failed
before analysis because it lacked Nix's Wayland pkg-config paths; `personal2`
ran inside `nix develop`. Personal caches remain separate from native build caches.

Final coverage hits every changed executable line except seven startup lines
exercised by the browser. The coverage record includes those exact line numbers,
source hashes, and the existing asset/error branch gaps. The headless damage
guide also ends 36 actions after failure.

The preview is `http://100.105.254.50:8778/robots/hover/?build=final`, served by
`bevy-gym-visible-damage-server-20261009` from `runs/quality-research/site-visible-damage`.
Its runtime is `build-85b4ba1609c7a202f04f63a69e36ee1003b6aa94968938e312796b26f5f35bd6`.
No recording is active. The preview is 1280×800 CSS pixels.
Do not resize during recording.

The scheduled-failure baseline is published as `307876c`.

The handoff commit `850ebf1` superseded the baseline CI run.

[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37879381806) passed;
[Pages](https://github.com/sagan-software/bevy-gym/actions/runs/37879345358) is pending.

The new
`drone-damage-baseline` guide compares 32 seeds, four motors, and two failure times.

Both controllers reached damage in all 256 cases and then crashed. Constant
half-thrust lasted 36–37 actions; the intact policy lasted 34–38 actions.

Results and source/checkpoint hashes are in `docs/progress/drone-damage-baseline.json`.

Seven assessment tests pass natively and in the actual browser. Root tests,
strict all-target/all-feature Clippy, and `nixfmt --check flake.nix` pass.

The focused personal Nix check also passes with no findings.

Personal Rust discovery has no diagnostics in the changed files; its full-project
strict backlog remains. Coverage hits every measured assessment and guide source
line, plus both outcomes of the assessment's two instrumented conditions.

Four test panic lines remain intentionally unhit.

Final logs use `damage-baseline-gates4`, `damage-baseline-wasm4`,
`damage-baseline-personal3`, and `damage-baseline-coverage-final2` under
`/home/sagan/.cache/bevy-gym-quality-validation`. The first browser run inherited
an incompatible `LD_LIBRARY_PATH`; the passing command removes that variable
before `nix run .#drone-browser-check`. Do not wrap this runner in `nix develop`,
which can restore the conflicting library path.

Damage-aware training implementation: `32be3f9`.

The guide is `drone-train-damage`; its separate policy has sixteen inputs.

The shared rollout collector now supports typed drone tasks and fixed-width encoders.

Existing healthy checkpoint qualification still passes. Sixteen damage-training tests
pass natively and in the browser. Root tests, viewer/curriculum tests, strict
native/WASM Clippy, and focused personal Rust/Nix checks pass. No changed-file
personal diagnostics remain; the full-project strict backlog remains.

Coverage hits the added helper code. Successful CLI promotion and final completion
remain unmeasured; no damage-aware checkpoint has passed its gates yet.

The final logs are `damage-training-gates2`, `damage-training-wasm2`,
`damage-training-personal2`, and `damage-training-coverage-final`, with supplemental
one/two-update and filesystem-failure CLI checks in the same validation directory.

The final guide/documentation log is `damage-training-guide-final3`: the one-update
run reported budget exhaustion, and six documentation tests passed with one
existing ignored plugin example.

Seed 7 finished under `bevy-gym-damage-curriculum-20261009.service` with exit 1.
Its log is `/home/sagan/.cache/bevy-gym-quality-validation/damage-curriculum-seed7.log`;
checkpoints are under `runs/drone-damage-curriculum-seed7`. The front-left lesson
exhausted its 600-update budget. The scheduled-failure lesson was not reached.
Do not restart the completed unit or weaken the frozen promotion thresholds.

The sixteen-input run passed intact hover at update 260. At front-left update 600,
all five healthy cases crashed before the scheduled failure. This records lost
intact-flight performance during fixed-failure training. Preserve the reports and
hover-260 checkpoint; assess healthy rehearsal before a new training recipe.
[The result](progress/drone-damage-curriculum-seed7.json) retains the final gates
and checkpoint hashes. No damaged-flight checkpoint is qualified.

The viewer now loads local twelve-input recovery and sixteen-input motor-failure
checkpoints. Native CLI validation happens before opening a window. Browser file
validation preserves the existing policy on rejection and ignores stale reads.
The recorded `progress/drone-checkpoint-file.mp4` shows hover-260 followed by a
front-left failure and crash. It does not demonstrate damage adaptation.
Screenshots record intact flight, the crash, and the narrow error state.

The recording used the release-compiled viewer before WASM size optimization,
with the unchanged prior training worker. Desktop and 390-pixel layouts were
inspected. The OS file picker, native window, and touch input remain unverified;
the browser test supplied a File to the visible input, then clicked Watch file.

Gates: 43 native viewer tests, 16 damage-training tests, root tests, strict native
and WASM Clippy, browser physics/training tests, and 25 JavaScript tests pass.
The file-reader module has 100% measured Node line, branch, and function coverage.
Rust coverage gaps are recorded in `progress/drone-checkpoint-file-coverage.json`.
Personal Rust discovery has no changed-file diagnostics; the strict full-project
backlog remains. The focused personal shell check passes.

The destruction checkpoint also splits overlong paragraphs in these handoff
and damage documents without changing their recorded evidence.

The checkpoint-file optimized bundle passed under
`bevy-gym-checkpoint-file-build-20261009.service`. Its runtime is
`build-a45fa58d3e7d57459a6ae39367f7d06c6e6f265ccfbebedb4a60eadbd7187748`.
The browser replayed hover-260 through all 500 intact actions with final height
1.87 metres. Published checkpoint `5069766` passed
[CI run 37883811556](https://github.com/sagan-software/bevy-gym/actions/runs/37883811556).
Its Pages run `37883811666` was pending at the last check.

Rotor bursts, soft smoke, rotor mesh removal, and colliding crash debris are now
implemented. All 49 viewer tests, root tests, format checking, and strict native
and WASM Clippy pass. Changed-file personal lint checks pass; the full-project
backlog remains. [Coverage](progress/drone-destruction-coverage.json) records the
native startup and defensive missing-pose gaps. The optimized browser bundle passes.

The [recording](progress/drone-destruction.mp4) shows rotor smoke, a step-36 crash,
colliding debris, and reset. Desktop and 390-pixel layouts were inspected.
[The visual record](progress/drone-destruction.json) retains hashes and limitations.

The final preview is `http://100.105.254.50:8780/robots/hover/?build=soft-smoke-final`.
Its runtime is `build-42ebdf885689457dd8e9749a99f645a2023906e8fc0e01954f76a9978c4b673c`.
The server unit is `bevy-gym-destruction-server-20261009`; no recording is active.
The release unit `bevy-gym-destruction-soft-smoke-build-20261009` exited successfully.
Validation logs use the `drone-effects-` prefix in the existing cache directory.

Next, implement the blockout arena and player-controlled humanoid robot from
[DRONE_PURSUIT_GAME.md](DRONE_PURSUIT_GAME.md). Pistol looting, rotor weak points,
telegraphed drone fire, cover, windows, pipes, and adversarial training remain
planned. The plan and reference contact-sheet evidence are saved.

Preserve the healthy checkpoint and its qualification. The damage contract records
the current model's static thrust limit; do not assume damage recovery follows
from the intact-flight force budget.

Perception, damage adaptation, pursuit, and the jumping quadruped
remain pending. Do not start later port families before the custom robot milestones.

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

## Browser training worker checkpoint

The worker now reuses the recovery lesson's model, encoder, collector, and evaluator.
Its private protocol supports fresh training, one 512-transition update, checkpoint
export, and independent evaluation. Runs stop at 260 updates. Failed updates discard
the run; malformed requests preserve it. The library API and tutorial examples
are unchanged. There are no lint-only imports or new third-party crate versions.

All twelve native protocol and invariant tests pass. Root tests, root strict
Clippy, worker native/WASM Clippy, selected-package personal lints, and shell/workflow
checks pass. [Coverage](progress/drone-worker-coverage.json) records the exact
initialization/serialization error gaps and unmeasured WASM adapter paths.
The root personal-lint backlog remains open.

The actual browser worker completed two updates and exported changed weights.
A 10 ms page timer fired 288 times during the second update. The first update's
policy failed all five selection episodes. Do not describe that checkpoint as
qualified. [The browser record](progress/drone-worker-browser.json) retains the
observed metrics, model size, browser version, and WASM hash.

The full seed-7 browser run is complete and its worker has been terminated.
[The retained model and record](progress/drone-browser-training.json) include all
selection milestones and final qualification. It survived 32/32 episodes with
mean return 469.4906 and mean final distance 0.2131 m. Eleven learning tests passed
in the actual browser, including both bundled and browser-trained qualification.

The new panel supports start, pause, resume, discard, learning-rate metrics,
checkpoint playback, and checkpoint download. Fifteen host-controller tests cover
cancellation, stale replies, validation, completion, and transport errors.
Twenty-eight viewer tests pass. Root tests, strict native/WASM Clippy, formatting,
workflow checks, and shell checks pass. The strict personal-lint backlog remains
unresolved; warning-enabled discovery found no new candidate diagnostics.
[Coverage](progress/drone-training-ui-coverage.json) records exact remaining gaps.

The final optimized release build passes. Desktop and 390×780 layouts were
inspected in T3. Watch checkpoint returned focus to the canvas and scrolled to
the scene. The saved final browser model reached action 500 in a separate
[recording](progress/drone-browser-trained-flight.mp4). Its panel remained idle
because playback loaded the saved model through the checkpoint API.

The local review server uses port 8773 and `runs/quality-research/site-training`.
The earlier 8768 full-training page is no longer active. Do not restart training
to recover evidence; model bytes and metrics are saved in `docs/progress`.
The interactive WASM test server on port 8774 completed eleven tests successfully.

## Foundation verification

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

The manual viewer is deployed and verified. The recovery policy passes native
and browser qualification below and is selectable in the local inference viewer.

Earlier missing-API tests failed with `E0432` before production code existed.
The initial reset regressions also failed before their fixes. These logs are retained
as `drone-red.log` and `reset-red.log`. Runtime coverage does not replace the
compile-fail checks for unchecked action construction and raw-world access.

## Browser and visual status

T3 preview is available. Inspected the official Bevy gallery and deployed Bevy Gym
page. The deployment offers three Classic Control environments. Started fresh
CartPole training and observed replay collection, then paused it.

No optimizer
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

That checkpoint used manual control. Rotor spin illustrates thrust rather than measured RPM.

The [flight recording](progress/drone-flight.mp4) shows hover, power-off, ground contact,
climb, pause, reset, and asymmetric thrust.

Its [contact sheet](progress/drone-flight-contact-sheet.png) was inspected.

Desktop and 390-pixel frame layouts were inspected. The preview resize
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

`nix run .#drone-browser-check` passes in CI. Its wrapper built and printed help
locally. Interactive local suites ran on
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
final prose edit. Checkpoint `88f2c5f` is pushed and passes CI.

Constant half-thrust solves calm hover but fails the five recovery seeds.
The following training checkpoint supplies a learned controller with separate
qualification seeds. Training seeds exclude both evaluation partitions.

## Qualified recovery policy

The native training guide and shared lesson implementation are under
`examples/robots/train.rs` and `examples/robots/learning/`. The lesson uses the
existing recurrent PPO optimizer and keeps the library API unchanged.
[The contract](DRONE_LEARNING.md) records seed partitions, observation units,
rollout boundaries, and acceptance checks. Ten native tests and ten browser tests
pass, including a real optimizer update and the bundled-policy qualification.

`bevy-gym-drone-learning-run-20261008.service` ran the first seed-7 experiment.

It wrote checkpoint and evaluation pairs beneath `runs/drone-recovery/seed7-v1/`.

That ignored directory also retains the exact source snapshot used for the run.

The run allowed 2,000 updates of 512 transitions each. Its log is
`/home/sagan/.cache/bevy-gym-quality-validation/logs/drone-learning-run.log`.

The run was stopped after the selected 133,120-transition checkpoint passed
qualification. Do not overwrite or delete this run. Only that selected checkpoint
is qualified. The guide now defaults to the selected 260-update budget.

The selected checkpoint survived all five selection episodes, with mean return
468.670. On 32 distinct final seeds, all episodes survived ten seconds, mean return
was 443.324, and mean final distance was 0.341 m. Constant half-thrust survived none
and scored 47.621. The same frozen qualification gates pass in the browser.
The policy is bundled at `assets/robots/recovery.mpk`; its SHA-256 starts `bb0bef6e`.
[The learning guide](DRONE_LEARNING.md) links the curve, raw results, and full hash.

The final root test, strict Clippy, WASM Clippy, Nix lint, and workflow checks pass.

Personal lints report no changed-line diagnostics and retain the full-package
backlog of 51 errors and two warnings. [Coverage](progress/drone-learning-coverage.json)
records 350/350 lines and 14/14 branches across the guide and helper files, including
internal test code. Both JSON and model write failures were exercised.

The file-based ignored qualification helper ran separately without coverage;
the bundled qualification ran under coverage. Native window and touch checks remain
unavailable. Browser training controls were added in the later UI checkpoint.

The updated `drone-browser-check` wrapper and its learning suite pass CI.
Interactive browser execution is verified at port 8772 through
`bevy-gym-drone-learning-browser-20261008.service`.

## Learned inference viewer

The viewer shares the training lesson's encoding and model recipe. Its controller
is manual, learned, or failed. A failure pauses playback before physics advances.
Reset preserves learned weights and clears memory; manual reset restores hover.
No library API changed. [The inference guide](DRONE_INFERENCE.md) records the contract.

All 25 viewer tests and ten learning tests pass. The exact root test, formatting,
all-target/all-feature strict Clippy, WASM viewer Clippy, and optimized build pass.
[Coverage](progress/drone-inference-coverage.json) records 292/292 instrumented
changed lines, including tests. Session and policy code have full line coverage;
all fourteen session branches are hit. Unchanged graphics initialization and UI
asset-state/color projection retain the recorded coverage gaps.

The strict personal lint pass remains blocked by 51 existing library errors.
The filtered output alone does not prove that example Clippy ran. An additional
`--no-deny-warnings` discovery pass reaches the viewer without hiding diagnostics;
it reports no viewer diagnostics. Dylint retains two existing warnings.
A separate strict example-only attempt stops on three existing wildcard enum
matches in `dqn.rs`, `ppo.rs`, and `recurrent_ppo/mod.rs`.

The [19-second recording](progress/drone-inference.mp4),
[contact sheet](progress/drone-inference-contact-sheet.png),
[desktop screenshot](progress/drone-inference.png), and
[narrow screenshot](progress/drone-inference-narrow.png) were inspected and shared.
Seed 42 reaches action 500 under the learned policy and ground contact during
action 274 under constant half-thrust. Learned selection works through the button
and keyboard shortcut. Reset and single-step are included in the recording.

Desktop and 390-pixel frame layouts pass the visual check. The gallery's scrollbar
leaves 375 content pixels; neither surface overflows horizontally. Mobile touch
and native window interaction remain unverified. T3 recording keeps frames
advancing; idle preview wall-clock speed is unverified. Synthetic pointer movement
needs `button: -1` and device-scale-adjusted coordinates before the tool's click.

Local inference is served at `http://100.105.254.50:8768/robots/hover/`.

The optimized WASM is 36,444,646 bytes. Its checksum, source hashes, exact checks,
and remaining gaps are in the coverage record. Validation units use the prefix
`bevy-gym-inference-` and retain logs under the existing validation cache.

The hover, recovery, and one-update training guides pass after the final prose
edit. Four documentation tests pass; one existing plugin example remains ignored.

Changed Markdown passes its configured check.

## Deployment qualification cache

The [cache contract](BROWSER_QUALIFICATION_CACHE.md) allows Pages to reuse an
identical, previously verified Gymnasium build. The workflow includes source and
runner image identity, excludes prefix fallbacks, saves only after success, and
retains the qualifying commit/run/key. Drone output is always rebuilt separately.

Manual dispatch always reruns qualification. Actionlint and the current input-path
audit pass. Cache miss/save, exact-hit reuse, and manual-dispatch execution still
need CI evidence. No speed improvement has been measured yet.

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
  passed. Opened the public gallery and drone route in T3, then verified keyboard
  playback, climb, tilt, pause, and reset.

  The [live gallery screenshot](progress/examples-live.png),
  [viewer screenshot](progress/drone-live.png), and [recording](progress/drone-live.mp4)
  show the deployed build. This deployment predates the disturbed-start controls.

- `9f98657`: native/browser bit comparison and browser contract-test harness.

  Pushed to GitHub `main`; remote revision verified.

  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37847009522) passed,
  including the first headless browser contract and parity runs.

  [Browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37847009460)
  was replaced while pending by the next checkpoint. Pages uses `cancel-in-progress: false`;
  new checkpoints do not cancel the active build.

- `88f2c5f`: disturbed starts, manual comparison controls, guide, and browser tests.

  Pushed to GitHub `main`; remote revision verified.

  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37848277209) passed.

  [Browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37848277060)
  passed. Its newly deployed route has not received a separate visual check.

- `503cf81`: native recovery training, qualified checkpoint, browser learning tests,
  metrics, and training guide. Pushed to GitHub `main`; remote revision verified.

  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37853641788) passed,
  including the new headless browser learning suite.

  [Browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37853641770)
  passed. Its deployed route has not received a separate visual check.

- `12ff6a4`: learned-policy viewer, reset/failure guards, 25 viewer tests, and
  browser recording. Pushed to GitHub `main`; remote revision verified.

  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37857449446) passed.

  [Browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37857449451)
  was superseded while pending.

- `e2a450f`: cache qualified Gymnasium deployment output. Pushed to GitHub `main`.

  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37858799219) passed.

  Its pending deployment was superseded by the worker checkpoint.

- `4741945`: browser recovery training worker, protocol tests, and browser evidence.

  Pushed to GitHub `main`; remote revision verified.

  [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37861862828) passed.

  [Browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37861862768)
  is running Gymnasium qualification.

## Local evidence and active validation

Another agent freed disk space. This run has not removed shared Cargo caches.

Builds, targets, logs, and reference media live under
`/home/sagan/.cache/bevy-gym-quality-validation/`. The `/dev/shm/bevy-gym-quality-*`
paths are symlinks to those directories. Recreate them after a reboot. Builds use
four jobs, or two for coverage/WASM, with development and test debug information off.

For future personal-lint runs, use the separate `personal-target` and
`personal-build` directories under that cache root. The personal runner uses a
different Rust compiler; keep its Cargo artifacts separate from native validation.

Interactive shells also supply `CFLAGS`, while the service environment leaves it
unset. Blake3 records that variable in its build fingerprint. Use `env -u CFLAGS`
inside the Nix shell for future native commands so both invocation paths agree.

Detached units use `bevy-gym-*-20261008.service`. A passed unit has
`SubState=exited` and `ExecMainStatus=0`; status zero while running is not a pass.

Use a separate log path for each invocation; a reused path can retain stale output.

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
