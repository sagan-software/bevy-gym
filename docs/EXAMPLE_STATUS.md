# Examples execution status

Updated: 2026-10-10. Current work: RL-only drone/droid curriculum.

## Resume here

### Travel storage recovery, 2026-10-10

The seed-19 travel run stopped at update 1,520 with `No space left on device`.
Recovery replays its original immutable binary and seed in a new directory, with
the same qualified recovery checkpoint and 2,400-update budget per stage.
The service is `bevy-gym-travel-endurance-seed19-recovery-20261010.service`.
It creates fresh optimizers from the original recipe; it does not restore a saved
optimizer state. The stopped run remains preserved.

The [recovery evidence](./progress/drone-travel-storage-recovery.json) records the
launch and exact optimizer-prefix comparison. The
[travel guide](DRONE_TRAVEL.md#interrupted-run-recovery) identifies both directories.
Inspect live state before acting. Travel remains unqualified, and the seed-13
standing trial continues independently. The RL-only browser 3v3 goal remains active.

### Complete frozen standing traces, 2026-10-10

`droid-standing trace` records the reset and every applied RL action's resulting
physical state. It shares the viewer session, retains checkpoint provenance and
never runs an optimizer. The [guide](DROID_STANDING.md#frozen-episode-trace) defines
the emitted JSONL contract and errors. Ten complete failed selection episodes are
preserved in the [capture manifest](progress/standing-trace/capture.json).

Native tests, strict repository Clippy, the four scene-example tests and nine CLI
checks pass. The complete browser gate passes 220 tests, including policy-action
parity. The rebuilt standing viewer renders and steps at a 390 by 844 viewport.
The scene test facade also fixes the prior CI failure caused by missing collector
test imports; runtime inference retains its separate facade.

Seven of eight trace production branch outcomes are covered. The inference-error
handler lacks a direct injected failure; session failure tests cover its underlying
sticky error behavior. Candidate personal Rust lint is clean. Strict personal
Clippy retains unchanged repository errors. The
[evidence record](./progress/droid-standing-trace.json) names these boundaries.

The seed-13 warm-start trial remains bounded to 2,400 updates. Inspect its service
and output before acting. Standing qualification, later skills and browser 3v3
remain unfinished. The RL-only 3v3 goal remains active.

### Standing checkpoint continuation, 2026-10-10

`droid-standing warm-start` imports validated RL actor and critic parameters into
fresh Adam optimizers. It retains exact source bytes and provenance before training.
New counters and seed describe the new run; optimizer state is not resumed.
The [guide](DROID_STANDING.md) defines the command and audit artifact.

The seed-13 trial imports the failed seed-7 recovery checkpoint at update 600.
It runs at most 2,400 new updates under
`bevy-gym-standing-warm-start-seed13-20261010.service`, with a two-CPU quota and
2 GiB memory limit. Output is `runs/droid-standing/seed13-warm-start-20261010`.
The [evidence record](progress/droid-standing-warm-start.json) preserves the immutable
binary, source patch, input hash and checks. Inspect live state before acting.

Seven CLI checks pass. Three recurrent checkpoint/update tests pass in native and
browser builds; the complete browser gate passes 218 tests. Required worker native
and WASM Clippy commands pass. The frozen standing viewer uses a separate inference
facade, removing caller-dependent feature checks that broke worker CI. Its rebuilt
browser scene renders and steps a learned action.

Candidate personal Rust lint is clean; strict personal Clippy still reports unchanged
repository errors. The coverage record names the unreachable second configuration
error path. Standing qualification, later skills and browser 3v3 remain unfinished.
The RL-only 3v3 goal remains active.

### Standing viewer and storage recovery, 2026-10-10

The halted local thread is `01a122d8-148d-79f2-b593-21d8472e877f`.
Both standing seed 7 and travel seed 19 stopped with `No space left on device`.
The original dirty checkout remains untouched. The
[interruption record](progress/robot-storage-interruption.json) preserves the stopped
runs, checkpoint identities and failed evaluations.

`droid-standing-scene` now runs standalone frozen PPO inference in native and browser
builds. All 26 torques come from the learned actor. Thirteen mannequin bones project
from authoritative physical states; no authored animation runs. Run, pause, step,
reset and speed controls change presentation timing only. The candidate remains
explicitly unqualified. The [guide](DROID_STANDING.md) provides copyable commands.

[Scene evidence](progress/droid-standing-scene.json) records desktop and narrow
screenshots, inspected playback video, native/WASM gates and exact coverage gaps.
Identical-input policy-action parity passes across 62 recurrent frames. Exact
physical trajectory replay remains unverified: seed 42 ends after 62 native actions
and 54 browser actions. No trajectory or standing qualification is claimed.

A bounded recovery run restarts the same seed-7 recipe from random weights, using
its immutable training binary. It preserves the stopped output directory. All 217
complete optimizer records match the original prefix. The run completed all 600
updates without passing selection; independent evaluation failed all 32 held-out
cases. The [recovery record](progress/droid-standing-recovery.json) preserves the
launch, hashes and failed results. No training unit remains active for this task.

Next: diagnose the failed standing and travel policies before another bounded trial.
Standing and travel qualification, later skills, shared multi-agent training and browser
3v3 remain unfinished. The RL-only 3v3 goal remains active.

### Standing PPO workflow, 2026-10-09

`74a7832de870d2d6ef23051975ee9dd04cbcbd0e` is verified on GitHub main.
The original dirty checkout remains untouched. The standalone `droid-standing`
trainer and frozen evaluator now reuse the physical environment. The
[guide](DROID_STANDING.md) defines the 204-value actor input, 26 torque outputs,
fixed qualification gates, checkpoint profile and runnable commands.

The retained seed-7 trial is active under
`bevy-gym-standing-seed7-20261009.service`, with a 600-update budget. Artifacts are
in `runs/droid-standing/seed7-20261009`. The log is
`/home/sagan/.cache/bevy-gym-quality-validation/standing-seed7.log`; the immutable
binary, source patch and manifest are in the cache's `standing-seed7-20261009`
directory. At update 20, all five selection episodes failed after 27–35 actions.
No standing checkpoint is qualified. Inspect live state before taking action.

[Training evidence](progress/droid-standing-training.json) records 19 native standing
learning tests, five executable tests and 197 browser tests, including 17 standing tests.
The existing drone learning and damage training tests also pass. Strict native/WASM
Clippy and root tests pass. Candidate personal Rust and Python lint are clean;
strict personal Clippy still fails on unchanged repository diagnostics.

Measured native coverage hits 814 of 817 instrumented lines and 54 of 58 branch
outcomes across the changed collector and standing modules, including their tests.
It leaves the successful CLI exit, stable-streak increment
and successful trainer exit unhit. The update-20 cadence is observed in the retained
run but remains uninstrumented. Exact coordinates and source hashes are recorded.
No successful synthetic gate fixture is presented as trained balance.

Travel seed 19 remains active, with update 1412 observed and no promotion.
Standing selection at update 100 failed all five cases after 41–62 actions.
These are snapshots, not final run outcomes.
All earlier failed trials remain preserved.

Next: inspect both runs, implement the
physical mannequin frozen-policy scene, and evaluate selected checkpoints on held-out
roots. Later skills, shared multi-agent training and spectator 3v3 remain unfinished.
The curriculum goal remains active.

### Articulated droid mechanics, 2026-10-09

`6cbc50022d16244ec66391833a6d9eb9e21fc48e` is verified on GitHub main; its contracts
and strict CI jobs passed. The original dirty checkout remains untouched.

`DroidStanding` now implements physical standing mechanics with thirteen dynamic
segments, twelve limited joints and twenty-six validated torque outputs. Each output
applies equal and opposite torques to its linked bodies. Joint motors, animation and
root movement do not choose motion. The [guide](DROID_STANDING.md) documents the
model, action/observation contract, reward, reset rules and licensed mannequin anchors.

The reset-only `droid-standing-contract` example prints body centres. It does not
command an agent. Policy encoding, PPO training, inference, physical mannequin
rendering, qualification thresholds and checkpoints remain unfinished. This checkpoint
does not establish standing competence or complete the standalone standing lesson.

[Evidence](progress/droid-standing-physics.json) records four native public droid
tests, seven internal tests, 101 instrumented library tests and 180 browser tests.
Coverage hits all 439 instrumented droid-module lines, including tests, and all
30 production branch outcomes. Constant data has no instrumented counts; failing
test assertions remain unexecuted. No whole-package coverage claim is made.

Root tests, strict native/WASM Clippy, WASM compilation and formatting pass.
Changed-line personal Rust lint is clean. Strict personal Clippy retains unchanged
repository findings; personal Nix lint retains two unchanged source-filter findings
at `flake.nix:119` and `:200`. Ten unchanged Markdown paragraph-length findings remain.

The reset-only example runs, with all seven instrumented lines hit. Robot doctests
pass 12 cases with one existing ignore, including both droid privacy checks.
The standing scene and its visual evidence do not exist yet.

The seed-19 travel endurance unit remains active. The evidence snapshot records
update 684 and failed selection at update 680; inspect live state before acting.
Artifacts remain under `runs/drone-travel/endurance-seed19-20261009`, with per-update
metrics and the preserved source manifest. Prior failed trials remain unchanged.

Next: implement the standing actor encoding, PPO collection and checkpoint evaluation
through this environment, then the frozen-policy mannequin scene. Freeze qualification
criteria before training. Monitor seed 19 and independently evaluate any selected
travel candidate. Later skills, multi-agent training and spectator 3v3 remain unfinished;
the curriculum goal remains active.

### Per-update travel diagnostics and independent trial, 2026-10-09

`7e05631d1b70eddc614fe904d651ce0b10380eed` is verified on GitHub main. Its CI passed;
Browser preview remained pending at the last check. The travel trial scene remains
explicitly unqualified. The original dirty checkout was not edited.

Travel training now flushes `optimization.jsonl` after every completed PPO update.
The [guide](DRONE_TRAVEL.md#optimizer-records) defines each field, finite-value rules,
ordering and failure behavior. Invalid measurements and output failures stop training
before further rollouts or selection. These diagnostics do not alter agent actions,
rewards, optimizer settings, checkpoint cadence or promotion gates.

The [bounded comparison](progress/drone-travel-optimization-probe.json) tested zero
versus 20 critic-only updates before 20 PPO updates on seed 11. The five evaluation
records remained exactly unchanged during critic warmup. Both post-PPO policies
crashed in all five episodes; mean returns were 208.7 and 244.7, below the source's
279.2. Warmup also consumed minibatch-shuffle RNG, so this comparison does not isolate
critic fitting alone. Warmup remains disabled. Preserve both failed probe checkpoints
and the probe source under the paths recorded in its evidence.

`bevy-gym-travel-endurance-seed19-20261009.service` is active, with seed 19 and a
2,400-update maximum per stage. Its independent binary and source manifest are
preserved; it uses the original PPO settings and all unchanged selection/held-out
gates. Artifacts are under `runs/drone-travel/endurance-seed19-20261009` and the log is
`/home/sagan/.cache/bevy-gym-quality-validation/travel-endurance-seed19.log`.
Poll the live unit and progress file before acting; do not restart from an expired
observation. Seed-11 near-travel and endurance failures remain preserved.

[Validation evidence](progress/drone-travel-progress.json) records 31 native learning
tests, 17 curriculum tests, 14 CLI tests and 176 browser tests passing. Root tests,
strict native/WASM Clippy and WASM compilation pass. Changed-line personal Rust and
Python diagnostics are clean; strict personal Clippy and 45 unchanged Python
diagnostics remain repository-level gaps. Coverage hits all 150 logger lines,
including tests, both finite-validation branch outcomes and all seven changed
trainer lines. No whole-package coverage claim is made.

Next: monitor seed 19 through unchanged selection, diagnose its recorded optimizer
behavior, and independently evaluate any selected candidate before qualification.
Research the articulated standing environment while training runs. Travel
qualification, later drone lessons, physical droid lessons, multi-agent training
and the spectator 3v3 arena remain unfinished. The curriculum goal remains active.

### Travel trial scene and endurance failure, 2026-10-09

`fe83e779e7e9735efe4c6a533c7efb8d86daad11` was verified on GitHub main before this
checkpoint. `drone-travel-scene` now runs separately in native and browser builds,
using the exact training task and encoder. Run, pause, step and reset affect playback
only. The scene shows target position, heading, speed, checkpoint identity, and
`Unqualified RL` status. The [guide](DRONE_TRAVEL.md) provides copyable commands.

The embedded checkpoint is `docs/progress/drone-travel-trial.mpk`, original near-travel
seed 11 update 120, SHA-256
`0f5a36039389e68c66281029d77eb2d7397bba8483e4a1f366092f512397d5f7`.
It failed selection. Browser seed 42 survives the horizon but misses the position,
heading and speed gates. Desktop/mobile recordings and inspected contact sheets
are under `docs/progress/travel-scene/`; [evidence](progress/drone-travel-scene.json)
records source hashes and exact coverage gaps. No travel competence is claimed.

The endurance unit `bevy-gym-travel-endurance-seed11-20261009.service` has now
failed with exit status 1 after all 600 updates. The final evaluation has two
survivors out of five, no settled episodes, and no promotion. The
[failure record](progress/drone-travel-endurance-failed.json) preserves all 30
selection results, checkpoint hashes and source provenance. Artifacts remain in
`runs/drone-travel/endurance-seed11-20261009`; the log remains
`/home/sagan/.cache/bevy-gym-quality-validation/travel-endurance-seed11.log`.
No training job remains active for this task. Preserve both failed travel runs.

Native tests, strict native/WASM Clippy, WASM builds, and 173 browser tests pass.
Changed-line personal Rust lint is clean. Strict personal Clippy still fails on
unchanged repository diagnostics. Markdown syntax passes; personal Markdown retains
13 unchanged paragraph-length findings, with changed paragraphs clean.

Instrumented native coverage hits every changed
line in the session, presentation and travel-loading adapters. Renderer startup,
projection and target drawing have exact uncovered lines recorded in the evidence;
actual browser execution does not supply instrumented hit counts. Native window
interaction and browser inference fault injection remain unverified.

Next: diagnose the failed endurance optimization using recorded trajectories and
actor/critic measurements, then test a bounded training change without weakening
promotion or held-out criteria. Travel qualification, later drone lessons, physical
droid lessons, multi-agent training and the spectator 3v3 arena remain unfinished.
The active curriculum goal remains open.

### Held-out travel evaluation and original trial failure, 2026-10-09

`8a5a4cc971e6e463bfb6d4a5e507c3a59ab3ce2f` is verified on GitHub main. The original
`bevy-gym-travel-seed11-20261009.service` has now exhausted all 600 near-travel updates
and exited with status 1. Its terminal error is
`Lesson travel-near exhausted its update budget without passing.` None of the five
final episodes survived or settled. The [failure record](progress/drone-travel-seed11-failed.json)
preserves all selection summaries, checkpoint hashes, and final episodes. Preserve
all candidates, the source manifest,
and the log under their existing paths; do not relabel this run as qualified.

The separate endurance trial remains active. Its unit is
`bevy-gym-travel-endurance-seed11-20261009.service`, with artifacts under
`runs/drone-travel/endurance-seed11-20261009`. Inspect its live unit before acting.

`drone-curriculum --lesson travel --evaluate-held-out PATH` now evaluates all four
stages on 32 fixed held-out roots each. The report binds results to the SHA-256 of
the exact bytes loaded once. It preserves all stage results on behavioral failure,
returns a nonzero exit status, and never trains or modifies weights. Missing or
incompatible weights and conflicting modes fail before reporting or output creation.
A behavioral pass still requires independently verified RL provenance before qualification.

The [guide](DRONE_TRAVEL.md) specifies the fixed suite, output fields, command,
exit behavior and qualification boundary. The
[evidence](progress/drone-travel-held-out.json) records tests, coverage gaps, and the
original trial's final failure. Native learning tests (28, one ignored), 14 curriculum
unit tests, 11 CLI tests, root tests, strict native/WASM Clippy, WASM compilation,
browser tests and documentation checks pass. Changed-line personal Rust and Python
lint are clean; strict personal Clippy retains the unchanged repository backlog.
Coverage hit 99/99 evaluator lines and every production branch. The evidence records
one test-only branch and one inference-error propagation gap. No real trained policy
has passed this suite; synthetic success scores test aggregation only.
No travel or endurance checkpoint is qualified.
Next: inspect the endurance trial, diagnose any further failure from measured
trajectories, and evaluate a selected candidate independently before qualified transfer.
Browser travel playback, later drone lessons, physical droid lessons, multi-agent
training, and the spectator 3v3 arena remain unfinished.

### Travel endurance prerequisite experiment, 2026-10-09

`b643be7be5d3ad3208853f4ce61c70c235366f0c` is verified on GitHub main. It added
travel PPO training and separate frozen inference. Its remote CI was running and
Browser preview pending at publication; those states are not local test results.

Frozen diagnostics found that the qualified recovery actor terminates after
691–729 actions on all five original-target hover episodes extended to 1,000 actions.
Its existing 500-action qualification remains valid. Near-travel episodes terminate
at 592–671 actions before training. Later candidate 120 survives extended hover but
misses position gates; candidate 400 terminates on all five. The
[diagnostic evidence](progress/drone-travel-endurance.json) records exact endings.

The trainer now prepends `travel-endurance`: original hover position, immutable
initial heading, the same 1,000-action horizon and settling gates, and minimum mean
return 800. All near/far/fast profiles and thresholds remain unchanged. Reward,
actuators and PPO settings are unchanged. Tests fix initial geometry, repeatability,
and the original travel boundaries. This is an experimental prerequisite, not a
qualified controller or a proven remedy for every training failure.

At this earlier checkpoint, the original `bevy-gym-travel-seed11-20261009.service`
remained active and untouched. Its final failure is recorded above.
A separate `bevy-gym-travel-endurance-seed11-20261009.service` is active, with seed 11
and at most 600 updates per stage. Artifacts and the launch manifest are in
`runs/drone-travel/endurance-seed11-20261009`; its immutable binary is under
`/home/sagan/.cache/bevy-gym-quality-validation/travel-endurance-seed11-20261009/`.
The log is `/home/sagan/.cache/bevy-gym-quality-validation/travel-endurance-seed11.log`.
Inspect these live units before acting; do not restart either from an observation timeout.

Native learning tests (28, one ignored), curriculum unit tests (12), CLI tests (nine),
root tests, strict native/WASM Clippy, WASM compilation and the browser suite pass.
All 16 measured changed production lines were hit. The evidence records existing
trainer success/error gaps. Changed-line personal Rust and Python lint are clean;
strict personal Clippy still fails on the unchanged repository backlog.

Next: compare the completed trials, then qualify a frozen candidate on held-out
goals before adding qualified transfer or claiming competence. Travel browser scene
and video evidence, later drone lessons, physical droid lessons, multi-agent training,
and the spectator 3v3 arena remain unfinished.

### Travel PPO training and frozen inference, 2026-10-09

This checkpoint adds `drone-curriculum --lesson travel`. It transfers the
qualified seed-11 recovery actor, appends a zero-weight heading input, and starts a
fresh critic and optimizer. Position, heading, arrival deadline and final settling
gates are fixed before training. The three stages sample progressively farther goals;
the last stage halves the arrival deadline. All motor requests come from PPO policies.

`bevy-gym-travel-seed11-20261009.service` is running with seed 11 and at most 600
updates per stage. Artifacts are `runs/drone-travel/seed11-20261009`; `run.json` records
source and binary hashes, and `source.patch` preserves the exact launch sources.
The immutable launch binary is under
`/home/sagan/.cache/bevy-gym-quality-validation/travel-policy-seed11-20261009/`.
Its log is `/home/sagan/.cache/bevy-gym-quality-validation/travel-seed11.log`.
Do not restart the job solely because a later source edit changes the working binary.

Native learning tests pass (27, one ignored), as do 12 curriculum unit tests, nine
CLI tests, root tests, strict native/WASM Clippy, WASM compilation, and the actual
browser suite (25 learning tests). Frozen travel inference now runs through
`--evaluate-checkpoint`; missing/incompatible weights and conflicting modes fail
before output creation. The CLI regression proves no checkpoint mutation or output
files during inference. Changed-line personal Rust and Python lint are clean.
Strict personal Clippy still fails on the unchanged repository backlog.

The [travel guide](DRONE_TRAVEL.md) records commands, all three stage distributions,
the 13-input actor contract, rewards and selection gates. The
[implementation evidence](progress/drone-travel-policy.json) records measured coverage
and exact gaps. The training launch snapshot predates the later CLI inference and
bounded source-file validation edits; its preserved binary continues unchanged.

At update 380, selection failed: zero of five episodes survived, none arrived, and
all final settling counts were zero. No travel checkpoint is qualified. Promotion
criteria are unchanged. The next action is to inspect the completed trial, diagnose
loss of stable flight, and qualify a frozen candidate on independent held-out goals.
Travel scene/video evidence, combined progression through qualified travel, all
later drone lessons, physical droid locomotion and 3v3 competition remain unfinished.

The old damage service is no longer loaded. Its preserved log still ends with
`Lesson front-left exhausted its update budget without passing.` The absent unit's
default success fields are not evidence that the failed trial passed.

### Travel environment and typed collection, 2026-10-09

The environment checkpoint `081bace99e04ee5c873b1534bcb4e4a1c2d1d485` is on GitHub main.
`DroneTravel` now applies supplied motor actions through the existing disturbed-hover
physics. A validated immutable destination supplies position and heading. It changes
observations and reward without choosing actions. Public tests compare physical
snapshots with the original environment under identical seeds and commands.

The [guide](DRONE_TRAVEL.md) and [evidence](progress/drone-travel-environment.json)
record boundaries, units, reset semantics, reward, and unfinished work. Native tests,
strict native/WASM Clippy, root tests, the actual browser suite, and robot documentation
tests pass. Changed-line personal Rust and Nix lint pass. Strict personal Rust lint
still fails on the unchanged repository backlog. Coverage records 101/101 lines,
including tests, and 16/16 branch outcomes. The unreachable normalization-error
mapping region is the remaining gap; no full-package coverage claim is made.

The shared rollout collector now accepts typed travel observations. Its regression
checks 512 deterministic one-step travel episodes with cleared recurrent memory.
Native learning tests (20 passed, one ignored), curriculum unit tests (10), CLI tests
(6), strict native/WASM gates and the actual browser suite pass. Changed-line personal
lint is clean. [Collector evidence](progress/drone-travel-collector.json) records
226/226 lines including tests and 6/6 branch outcomes across instantiations. The two
existing critic-error propagation regions at rollout lines 130 and 133 remain unhit.
Travel uses a body-only test encoder here; no trained travel controller is implied.

At this earlier checkpoint, travel had no trained policy, training/inference command,
scene, or video. The newer section above supersedes its next action. Existing
imitation-only tracking is not a source checkpoint. Do not weaken gates on failure.

At that earlier checkpoint, no training job was active. Preserve both damage failures and the
completed seed-11 recovery transfer. Logs use
`/home/sagan/.cache/bevy-gym-quality-validation/travel-environment-` and
`/home/sagan/.cache/bevy-gym-quality-validation/travel-collector-`.
At the last remote check, `9a765cb` CI and Browser preview were still running;
`14f20dd` jobs were cancelled. Local validation is separate from remote CI.

### Qualified standalone checkpoint transfer, 2026-10-09

The browser scene checkpoint `14f20dd3e381b9120d5065bbf93d1a270fd46b6f` is pushed to
GitHub main. This checkpoint adds
`--lesson recovery --initialize-from qualified-hover` to `drone-curriculum`.
Only the recorded PPO hover checkpoint is accepted. Loaded actor and critic records
must match its embedded reference. Unsupported destinations fail before output
creation. Actor and critic weights transfer; the optimizer and sampling streams
start fresh. A `transfer.json` record precedes the first update. The original
in-memory curriculum still retains its optimizer between lessons.

The preserved hover source is now `docs/progress/drone-hover.mpk`, SHA-256
`5aa47c6941b1aa243c4eafcb0fafaf8c6fa9ee816944dc5a5ca0d3f95712403b`.
It passed 32/32 held-out calm episodes natively and in WASM. Native mean return was
459.7338 and mean final distance was 0.11701 metres.

A separate seed-11 trial passed recovery selection at update 20, using 88 optimizer
steps and 10,240 new transitions. Its service
`bevy-gym-recovery-transfer-seed11-20261009.service` exited successfully. Artifacts
remain in `runs/drone-recovery/transfer-seed11-20261009`. The frozen result is copied
to `docs/progress/drone-recovery-transfer.mpk`. Native held-out evaluation passed
32/32 recovery episodes with mean return 441.3443 and mean distance 0.26578 metres.
Hover replay also passed 32/32, with mean return 461.1305 and mean distance
0.16483 metres. All three new qualification tests also pass in actual WASM execution.
The [evidence](progress/drone-checkpoint-transfer.json) retains hashes and episodes.
Native learning tests (19 passed, one ignored), curriculum unit tests (10), CLI
tests (6), root tests, strict native/WASM Clippy, and the browser suite pass.
Changed-line personal Rust and Python lint pass. Strict personal Rust lint remains
blocked by the unchanged repository backlog. Transfer coverage is 71/71 lines
including tests and 2/2 branch outcomes. Main coverage is 53/56 lines and 11/14
branch outcomes; the evidence names error-propagation and existing loop gaps.
Existing browser videos do not show these new weights.

The original seed-11 damage run remains failed and preserved; it is a different run.
At that checkpoint, no training service was active. Logs use
`/home/sagan/.cache/bevy-gym-quality-validation/skill-transfer-`.

Next: implement drone travel through the shared RL environment and focused failing
tests, then train and qualify its checkpoint. Physical
droid lessons and the shared 3v3 arena remain unfinished.

### Standalone RL browser scenes, 2026-10-09

`drone-hover-scene` and `drone-recovery-scene` now run separate frozen-policy scenes.
They share the same `DroneHover` factories and encoding as curriculum training and
evaluation. Every motor command comes from the recorded PPO checkpoint. The renderer
reads physical poses; Run, Step, and Reset cannot select agent actions. Load or
inference failures stop playback without a fallback. Each scene identifies its
checkpoint. The gallery distinguishes these scenes from the historical workbench.

The [browser guide](../robot-web/skills/README.md), [hover guide](DRONE_HOVER.md),
[recovery guide](DRONE_RECOVERY.md), and [evidence](progress/drone-skill-scenes.json)
record the commands, contracts, qualification, and remaining limits. The unchanged
checkpoint is `docs/progress/drone-curriculum.mpk`, SHA-256
`8b94182a1368f5449a52db5c0859160fc5819d6772c0547f31d240df7a0a465a`.
New calm-start evaluation passed 32/32 held-out episodes natively and in WASM.
Native mean return was 464.4299 and mean final distance was 0.06235 metres.
The existing 32-seed disturbed-recovery qualification also passes again.

Native tests, strict native/WASM Clippy, the complete browser suite, and changed-line
personal Rust lint pass. Strict personal Clippy still fails on the unchanged
repository backlog; the changed-line pass reports no diagnostics. Shell and Nix
personal lint pass. Session coverage records
111/111 lines including its unit tests and 6/6 production branch outcomes. The
remaining unhit branch belongs to test setup; the fixed positive horizon cannot
exercise its error-propagation region. Renderer line coverage and browser fault
injection for displayed load/inference/model errors remain gaps. The evidence names
each boundary; no full-package coverage claim is made.

Actual T3 browser playback completed both 500-action episodes. Hover ended 0.06 metres
from the target; recovery ended 0.26 metres away. Playback controls and 1280-pixel and
390-pixel layouts were inspected. A mobile canvas sizing defect was fixed. Videos
and inspected contact sheets are under `docs/progress/skill-scenes/`. Review ran in
this thread because subagents were prohibited. Native window interaction and physical
mobile devices remain unverified. Local captures use development WASM bundles.

Seed 11 exhausted all 600 fixed front-left updates and exited with status 1.
`bevy-gym-rl-curriculum-seed11-20261009.service` is failed. The final error is
`Lesson front-left exhausted its update budget without passing.` Hover passed at
update 300. Training resets with the failed motor; evaluation requires intact flight
before a two-second failure and retained healthy flight. This distribution mismatch
is an investigation target, not a proven explanation for every failure. Preserve
both seed-7 and seed-11 artifacts. No training job is running for this task.

Logs use `/home/sagan/.cache/bevy-gym-quality-validation/skill-scenes-`.
The preview server on port 8782 serves `robot-web/skills-dist`.
The previous checkpoint `52c3a65eda215de134e91e4db44571d1ea7ffd5d` is on GitHub main
and its CI passed. Browser deployment remains queued behind the preceding build.
This scene checkpoint's remote CI and release deployment are unverified until pushed.

Next: implement qualified standalone checkpoint transfer. Later drone lessons,
physical droid lessons, competition, and the shared spectator 3v3 arena remain
unfinished. The active goal remains the complete RL-only curriculum and arena.

### Standalone training selection, 2026-10-09

The shared `drone-curriculum` trainer now accepts `--lesson hover` and
`--lesson recovery`. Omission retains the hover-to-recovery sequence and optimizer
transfer. Standalone modes start from random weights and use the same collector,
evaluator, and promotion criteria. An empty or invalid `--lesson` value fails before
output creation. A failed update budget never starts another lesson.

Three CLI tests cover both standalone lessons, default progression, and rejected
values. Native tests, strict Clippy, WASM checks, and changed-line personal lint pass.
[The evidence](progress/drone-skill-selection.json) records all ten added measured
Rust lines hit, existing coverage gaps, and corrected coverage-tool failures.
The [hover](DRONE_HOVER.md) and [recovery](DRONE_RECOVERY.md) guides have independent
training and inference commands. No new policy qualification is claimed.
Logs use `/home/sagan/.cache/bevy-gym-quality-validation/skill-selection-`.

The preceding inference checkpoint `e6baa3cc947075c6351b78d66d4f3fa7df7a67ba` is on
GitHub main. Its CI strict job passed; contracts and browser deployment were still
running or pending at the last check. Local evidence and remote CI remain separate.

Seed 11 passed hover at update 300 with five survivors, mean return 441.279, and
mean final distance 0.303 metres. It is training fixed front-left failure; update
200 still failed promotion. The original service, log, and checkpoint directory
remain unchanged. Preserve both this run and the earlier seed-7 failure.

Next: qualify calm-start replay of the saved curriculum policy, then implement
separate policy-only browser lesson scenes. Standalone checkpoint transfer, later
drone lessons, physical droid lessons, and the 3v3 arena remain unfinished.

### Frozen-policy skill commands, 2026-10-09

`drone-hover` and `drone-recovery` now infer every motor action from the qualified
RL curriculum checkpoint. Both use the same episode implementation as curriculum
evaluation. They retain recurrent memory within an episode and fail visibly for
missing, unreadable, corrupt, or incompatible weights. The CLI regression first
failed because the old constant-thrust commands ignored the checkpoint argument.

The [hover guide](DRONE_HOVER.md), [recovery guide](DRONE_RECOVERY.md), and
[validation record](progress/drone-skill-cli.json) describe the commands and limits.
The default checkpoint remains `docs/progress/drone-curriculum.mpk`, SHA-256
`8b94182a1368f5449a52db5c0859160fc5819d6772c0547f31d240df7a0a465a`.
Its existing held-out RL qualification passes again; no replacement training was run.
The seed-7 motor-failure failure remains preserved.

Native tests, strict native and WASM Clippy, the complete browser suite, Python
personal lint, and CLI failure/success checks pass. The worker's 25 protocol tests
also pass. Personal Rust discovery reports no candidate diagnostics. Strict personal
Clippy still reports the unchanged repository backlog. Coverage records all measured
entry-point lines, all 13 runner lines, all 40 shared episode lines through integration tests,
and both episode branch outcomes. The validation record identifies error-region gaps.
Logs use `/home/sagan/.cache/bevy-gym-quality-validation/skill-cli-`.

Seed 11 remains running under `bevy-gym-rl-curriculum-seed11-20261009.service`.
At the recorded update 280, hover promotion still failed despite five survivors.
Artifacts remain in `runs/rl-curriculum/seed11-20261009`; inspect the log for newer results.

The T3 viewer opens, but it still exposes historical constant-thrust and imitation
controls. That inspection is not new-scene qualification. Separate policy-only browser
scenes, standalone hover selection, and checkpoint transfer remain unfinished.
Next: add standalone lesson selection to the shared trainer, then isolate the two
browser lesson entry points. The droid lessons and shared 3v3 arena remain unfinished.
Remote CI for this checkpoint remains unverified.

### RL-only skill curriculum, 2026-10-09

The canonical `ai/AGENTS.md` now prohibits programmed example-agent decisions,
imitation-only substitutes, and silent controller fallbacks. Root discovery files
point to that source. Follow [the skill curriculum](ROBOT_SKILL_CURRICULUM.md) for
separate drone/droid lessons, prerequisite transfer, independent promotion, the
curriculum walkthrough, and the final trainable 3v3 scene.

The user superseded the earlier broad goal on 2026-10-09. Its unfinished audit and
reference ports remain deferred history, not the current execution objective.
Current objective: implement and qualify RL-only drone/droid skill scenes, curriculum
training and progression, and spectator-first competitive 3v3 inference.
A new service goal is active in continuation thread
`01a122d8-148d-79f2-b593-21d8472e877f`. The earlier unfinished goal was not marked complete.

The existing RL damage trainer is running seed 11 with a 600-update limit per lesson.
Unit: `bevy-gym-rl-curriculum-seed11-20261009.service`.
Log: `/home/sagan/.cache/bevy-gym-quality-validation/rl-curriculum-seed11.log`.
Artifacts: `runs/rl-curriculum/seed11-20261009`.
This is a fresh test of the existing recipe, not completed new-skill implementation.
The previous seed-7 damage run failed; do not replace that record with an inferred pass.

### Policy-only 3v3 priority, 2026-10-09

The latest request replaces the planned two-versus-two player game with a
spectator-first three-versus-three match. Follow [Drones vs. Droids](DRONES_VS_DROIDS.md).
All navigation, aim, firing requests, alert choices, and living locomotion must come
from reinforcement-trained policies. Keep physics and weapon limits as environment rules.

The audit confirms the current demonstration does not meet this requirement:
its flight network is imitation-trained, its navigation and firing are programmed,
and its living droid uses capsule movement and animation clips. No 3v3 training
or learned articulated droid locomotion has been completed. Do not label the existing
runtime as policy-only or treat its visual qualification as RL evidence.

Next: implement the six-agent shared-world action boundary and batched two-team
collector, with separate physical locomotion training before integrated qualification.
The new plan records research references, invariants, tests, and publish checkpoints.

### Ragdoll checkpoint, 2026-10-09

The current source adds physical mannequin death through the unpublished
`bevy-gym-pursuit-viewer` package. The viewer owns its camera and vendored ragdoll
dependencies; the published library has no normal path-only ragdoll dependencies.
The death camera lowers and widens its view. Reset restores animation and body count.

The initial browser build exposed a delayed Rapier sleeping-island panic after
settling and reset. The viewer now disables the adapter's optional forced-sleep timer
while retaining Rapier's automatic sleep. Extended native tests pass at 50 Hz and
60 Hz before and after that change; they do not reproduce the browser failure.
The browser recording and console provide the discriminating failure/success evidence.

Qualified runtime: `6bb43be4aa4b3c01f3ad18e1680cf19975c5adb00274318d860f96366cf32d57`.
URL: `http://100.105.254.50:8781/robots/pursuit/?build=6bb43be4`.
The optimized [recording](progress/shooter-ragdoll-final.mp4),
[contact sheet](progress/shooter-ragdoll-final.jpg), and
[evidence record](progress/shooter-ragdoll.json) show settled death, two resets,
and continued movement without console errors. The inspected 390x844 layout retains
the existing Fire/instruction overlap and incomplete touch aiming. Desktop controls
are qualified; mobile play remains unfinished.

Root tests, 115 viewer tests, 61 flight-viewer tests, strict native/WASM Clippy,
four asset tests, shell lint, and personal lint pass. Both raw personal-lint passes
contain no viewer diagnostics. The [coverage record](progress/shooter-ragdoll-coverage.json)
identifies native startup and test-assertion gaps. The ragdoll and animation files
have full measured line and branch coverage, including their tests.

Run `cargo test -p bevy-gym-pursuit-viewer --example drone-pursuit` through Nix.
Validation logs are under `/home/sagan/.cache/bevy-gym-quality-validation/`:
`ragdoll-sleep-gates.log`, `ragdoll-sleep-personal.log`, `ragdoll-sleep-coverage.log`,
`ragdoll-sleep-release.log`, and `ragdoll-sleep-browser.log`.
[Vendor provenance](../vendor/README.md) records the revision and compatibility edit.
The separate package-list failure is documented in [QUALITY_AUDIT.md](QUALITY_AUDIT.md).

Checkpoint `9c67238` passed the final local checks and is pushed to GitHub main.
CI and browser preview were still running or pending at the last check.
The policy-only 3v3 plan above supersedes the two-versus-two milestone. Authored strafing,
state lights, raytraced audio, mobile controls, and the broader examples roadmap
remain unfinished. Do not mark the overall goal complete.

### Weapon grip checkpoint, 2026-10-09

Projectile checkpoint `7cabda0` is confirmed on GitHub main. The current checkpoint
changes share one weapon frame between the pistol, both wrists, muzzle feedback,
and projectile launch. A two-bone arm solver preserves bone lengths; the muzzle
retracts before cover. Recoil derives from the pistol cooldown. Pelvis direction
changes now settle with a 100 ms time constant.

Native root tests, 113 scene tests, strict native/WASM Clippy, and asset checks pass.
The standard browser suite passes for this production code. Personal lint reports no candidate
diagnostics in either raw pass; unrelated
repository diagnostics remain. The [coverage record](progress/shooter-grip-coverage.json)
lists exact gaps. The [strafe recording](progress/shooter-grip-strafe.mp4) and
[close-up contact sheet](progress/shooter-grip-close.jpg) show the final source before
WASM optimization. The optimized release completed successfully.

Runtime: `9c2080fa21925205d022c8e2f3d75bb5a36e3ae61e9a0329bc2761f67f75a4c9`.
URL: `http://100.105.254.50:8781/robots/pursuit/?build=9c2080fa`.
The optimized [recording](progress/shooter-grip-final.mp4),
[contact sheet](progress/shooter-grip-final.jpg), and
[desktop capture](progress/shooter-grip-desktop.png) were inspected. The
[narrow capture](progress/shooter-grip-narrow.png) retains the existing Fire/instruction
HUD overlap; touch aiming remains incomplete. Desktop keyboard/mouse play is the
qualified path.

Next: run the guides/documentation checks, commit, and push this checkpoint.
Then integrate physical death. Compatibility probes compile the unmodified ragdoll
core and Rapier backend at upstream `05ca5a920c88cec9cfaa661c483aa4e26924e08f`
against Bevy 0.18.1 and bevy_rapier3d 0.34.0. Those probes have not established runtime
correctness. Their minimal manifests omit upstream lint configuration and emit
`unexpected_cfgs` warnings for `dylint_lib`.

The authoritative worktree remains `bevy-gym-quality`. Validation logs are under
`/home/sagan/.cache/bevy-gym-quality-validation/grip-*`; compatibility probes remain
under ignored `runs/quality-research/shooter/*compat-probe` directories.
Ragdolls, authored strafing, two-versus-two agents, state lights, and raytraced audio
remain unfinished. Do not mark the shooter or the broader roadmap complete.

### Projectile checkpoint, 2026-10-09

The authoritative worktree remains `bevy-gym-quality`. Camera checkpoint `09f7372`
and impact checkpoint `d0a53df` are pushed to GitHub main. The current changes add
swept player projectiles, muzzle flash/smoke, camera recoil, and player discharge
audio. Native and browser gates pass. The optimized build was inspected after a
rotor hit; the [recording](progress/shooter-projectile-final.mp4) and
[contact sheet](progress/shooter-projectile-final.jpg) retain that evidence.

The runtime is `538f1f48ffff24f85185adfb167955d5164087560218045c0cfce25df8e97659`
at `http://100.105.254.50:8781/robots/pursuit/?build=538f1f48`.
The [upgrade plan](DRONE_SHOOTER_UPGRADE.md#projectile-checkpoint) records behavior,
limits, test scope, and remaining work.

Next: align the physical projectile origin, visible muzzle, and both animated hands.
The current chest-origin tracer starts below the visible barrel. Directional
locomotion still changes abruptly. Ragdolls,
two drones/two droids, and state-aware lights/raytraced audio remain pending.

### Shooter camera and impact checkpoints, 2026-10-09

The current work follows [the shooter upgrade plan](DRONE_SHOOTER_UPGRADE.md).
The shoulder camera uses PanOrbit 0.34 with camera-relative WASD, right-button ADS,
a centre reticle, collision clearance, and an equipped starting pistol.
Weapon projection reads the current animated hand after animation evaluation.
Direction changes remain abrupt; the forward clip is still adapted for side travel.

The camera checkpoint `09f7372` is pushed to GitHub main.
The authoritative worktree is `bevy-gym-quality` on `quality-roadmap-20261008`.
The impact checkpoint adds validated physical impulses and persistent first-hit rotor
smoke. Native root tests, 99 scene tests, five public impulse tests, 20 navigation
tests, strict native/WASM Clippy, four asset tests, and the browser suite pass.

The browser suite includes all five impulse tests. Personal lint reports no candidate
Rust diagnostics; unrelated raw repository diagnostics remain. Nix lint reports
pre-existing unfiltered source roots at flake.nix:119 and :200.

The [hit recording](progress/shooter-rotor-hit.mp4) and
[contact sheet](progress/shooter-rotor-hit.jpg) show a first rotor hit, rotation,
and attached smoke. They use final Rust source before WASM optimization.
[Coverage](progress/shooter-impact-coverage.json) records the exact remaining gaps.
The impulse value has 37/37 covered lines and 8/8 branches, including test code.

Player shots used hitscan at the impact checkpoint; the current changes replace it.
The separate `bevy-gym-impacts` worktree is now an outdated scratch copy. Do not copy
its files over this worktree. The original `bevy-gym` checkout remains deliberately dirty.

Native root tests, 97 scene tests, 20 navigation tests, strict native/WASM Clippy,
asset contracts, and the standard browser suite pass. Personal lint's changed-line
filter is clean; its raw strict run still reports unrelated repository diagnostics.
[Coverage](progress/shooter-camera-coverage.json) records full line and branch coverage
for the camera, locomotion helper, and weapon projection, plus explicit gaps elsewhere.
The earlier mannequin checkpoint coverage also completed; its
[record](progress/drone-survival-coverage.json) replaces the pending status below.

The [ADS recording](progress/shooter-ads-strafe.mp4) and
[contact sheet](progress/shooter-ads-strafe.jpg) were inspected.
Camera orbit and ADS were verified in T3 tab `tab_1`; right-button drag works when
the embedded browser refuses pointer lock. Normal captured-mouse operation remains
unverified in that browser. Desktop and narrow layouts were inspected; this remains
a keyboard-and-mouse shooter, without a complete touch aiming interface.

Next after the current projectile checkpoint: improve directional animation, resolve bevy-ragdoll's
Bevy
0.19 boundary, then add two drones/two droids and state-aware lights/audio.
Do not report the complete shooter or the broader examples roadmap as finished.

The optimized impact build is `9a1e3e72b9565671c529a38e1cad7deaf965d1c79572f05516d12450e6515343`
at `http://100.105.254.50:8781/robots/pursuit/?build=9a1e3e72`.
The following camera evidence predates the impact checkpoint.
The [final recording](progress/shooter-camera-final.mp4),
[contact sheet](progress/shooter-camera-final.jpg), and
[desktop capture](progress/shooter-camera-desktop.png) use that build.

### Previous mannequin checkpoint

### Survival model checkpoint

Work continues in `/home/sagan/Code/github.com/sagan-software/bevy-gym-quality`.
The user selected the mannequin with neutral armor and dark joints.
The [survival upgrade plan](DRONE_SURVIVAL_UPGRADE.md) records the asset sources and remaining work.
Licensed CC0 models replace the primitive character, pistol, and cover blocks.
The scene now has authored animation, asset-loading protection, and a survival clock.

Native validation passes 86 scene tests, 20 navigation tests, root tests, formatting,
strict Clippy, WASM Clippy, and four asset contracts. The standard browser suite passes.
Personal lint reports no diagnostics on changed lines; unrelated repository diagnostics remain.
The optimized scene was inspected at 1280-by-800 and 390-by-844 CSS pixels.
The [desktop capture](progress/drone-survival-mannequin-desktop.png),
[narrow capture](progress/drone-survival-mannequin-narrow.png), and
[same-source preview recording](progress/drone-survival-mannequin-preview.mp4) show the models.
The recording predates WASM optimization; the two final captures use the optimized build.

The [qualification record](progress/drone-survival-model.json) identifies sources and runtime
hashes.
The browser uses build `7c2d064f979174ef2b8dd10331ae51a86c7c60e69986764d519c1a7734ca92d0`.
Its local URL is `http://100.105.254.50:8781/robots/pursuit/?build=survival-model-final`.
T3 tab `tab_1` is at the desktop viewport, with recording stopped.
The tool reports preview visibility as false; screenshots confirm browser rendering.

Branch coverage completed in `survival-branch-coverage.log` under
`/home/sagan/.cache/bevy-gym-quality-validation`. The corrected run used
`RUSTC_BOOTSTRAP=1`; the earlier stopped run is not coverage evidence.
See `progress/drone-survival-coverage.json` for exact file summaries and gaps.
Directional locomotion and hand attachment work continues in the camera checkpoint above.

The current game has one drone; defeating it ends the run. Buildings and the pipe remain blockout
meshes.
The navigator still uses programmed search over learned motor control.
Preserve these changes and the original checkout's unrelated dirty files.

### Playable navigation checkpoint

The playable pursuit scene now combines programmed search with learned motor control.
The `pursuit-search` guide demonstrates filtered sight and the motor loop without rendering.
Search receives current or remembered sightings, never the hidden character position.
It uses the authored static map to find viewpoints and routes around cover.
Learned search, hearing-directed navigation, damage adaptation, and adversarial humanoid training
remain unfinished.

The final 265 native trials all survived. Minimum lateral visibility was 2,092/3,000 samples.
Every house and pipe trial ended with ten uninterrupted seconds of sight.
All 53 pipe cases still miss the earlier 90% held-visibility threshold.
The [final record](progress/drone-navigation-final.json) preserves tested sources and logs.
Its trial archive is byte-identical to the earlier integration archive, verified by SHA-256.

The scene now accepts brief sightings that occur between navigation decisions.
Repeated remembered contacts cannot extend the investigation deadline.
After expiry, the sensor must clear or provide a visible contact before another investigation
starts.
A closed-door fixture verifies planned clear segments through a window view.
That fixture does not establish physical window-flight performance.

A failing regression reproduced the pipe camera obstruction. The camera now lowers under ceilings.
The [desktop capture](progress/drone-search-pipe-fixed-desktop.png),
[narrow capture](progress/drone-search-pipe-fixed-narrow.png), and
[recording](progress/drone-search-pipe-fixed.mp4) confirm the corrected view.
The desktop capture shows the drone outside the pipe after disabling the robot.
The original obstruction capture remains archived for comparison.

The local browser runs build
`ee5f5b24329264c454889da7f67c742db61a2fc7075f8899cbbae8e4f31136ba`
at `http://100.105.254.50:8781/robots/pursuit/?build=navigation-final`.
T3 preview inspection covered 390-by-844 and 1280-by-800 CSS-pixel views.
The tab is `tab_1`, at the desktop viewport, with recording stopped.
The latest open call confirmed a visible preview. The site server remains active.

Final native validation passed formatting, root tests, 74 scene tests, 20 navigation tests,
and strict all-target/all-feature Clippy. The exact WASM Clippy command also passed.
The standard browser suite passed, including 18 applicable navigation tests.
Two imported arena tests run only natively. Personal strict and discovery checks
reported no changed-line findings.

Nix lint still reports the two existing
`statix/unfiltered_source_root` findings at `flake.nix:119` and `flake.nix:200`.
Those lines are unchanged by this checkpoint.

Coverage executed every line in the new navigation modules and runnable guide.
[The coverage record](progress/drone-navigation-coverage.json) identifies four untested
branch outcomes, flight-error paths, and a test-only panic arm. Full branch coverage is not claimed.
The default cargo-llvm-cov report excludes examples. The recorded analysis instead
exports the instrumented scene, navigation test, and guide objects directly with LLVM.

All navigation build, native, browser, personal-lint, coverage, and trial jobs have finished.
The temporary research probe has been archived and removed. No training run is active.

The playable navigation checkpoint `1a15744` is pushed to GitHub `main`.
[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37961802709) and
[browser deployment](https://github.com/sagan-software/bevy-gym/actions/runs/37961802713)
were in progress at handoff. Inspect those runs before claiming remote validation.

The guide passed two episodes with 2,843/3,000 visible samples each.
Documentation tests passed seven cases with one existing ignored case.
Markdown validation passed. Continue the active roadmap with learned search and
cover behavior qualification.
Do not repeat the completed trial sweep unless controller behavior changes.

### Research preceding integration

The following records describe research before the playable navigation checkpoint.

Read [the roadmap](EXAMPLE_ROADMAP.md), [quality audit](QUALITY_AUDIT.md),
[reference research](EXAMPLE_RESEARCH.md), and [drone contract](ROBOT_ENVIRONMENT.md).

The [permanent-cover experiments](DRONE_SEARCH.md) distinguish search from
waiting for a target to reappear. Cover PPO 260 never saw the robot after it
stopped inside the pipe. A teacher using filtered sight and known arena waypoints
passed ten trials. Later distilled candidates learned both cover approaches, but
none passed all cover and moving-target checks. Do not install them in gameplay.

The recurrent run completed all 300 updates. No evaluated checkpoint saw the
robot during a held pipe sample. The [research archive](progress/drone-hide-candidate.json)
now contains its complete history and log. That run
has stopped. Its research sources are archived.

[Sequence imitation](RECURRENT_IMITATION.md) now trains through observation
history using the existing sample and memory types. Its 45-line guide learns
opposite actions from identical blank observations after different cues.
Six integration tests pass natively and in Chrome/WASM.

Root tests, formatting,
strict native/WASM Clippy, and changed-line personal strict/discovery checks pass.
All added instrumented library and guide lines and both guide conditional outcomes execute in
[coverage](progress/recurrent-sequence-coverage.json). The personal-lint backlog
outside changed lines remains. The standard browser check now includes the six
sequence tests. Nix lint still reports the two unchanged unfiltered-source findings
at `flake.nix:119` and `flake.nix:200`. The [validation
record](progress/recurrent-sequence-validation.json)
records commands, hashes, tooling retries, and boundaries.

The first [ordered sequence run](DRONE_SEARCH.md#ordered-sequence-imitation)
completed 300 updates. It learned each route at different checkpoints, but none
passed both routes across all five selection seeds. Its exact sources, results,
traces, logs, and selected weights are in the [sequence
archive](progress/drone-sequence-candidate.json).
The [path comparison](progress/drone-sequence-paths.png) shows actual solver paths.
No new navigator is qualified for gameplay.

The complete-episode comparison finished all 300 updates without a qualified
checkpoint. Its full history and terminal log are in the
[sequence archive](progress/drone-sequence-candidate.json). Update 180 survived
all ten cases but never saw the held pipe target; update 220 learned the pipe
view while crashing in every house case. Do not restart this unchanged recipe.

The [heading diagnostic](DRONE_SEARCH.md#heading-target-diagnostic) measured a
scalar yaw-label discontinuity in all ten teacher trials. A five-component
vector-heading teacher passes all ten native trials. Its 600-update learner run
finished successfully but produced no qualified checkpoint. The
[diagnostic archive](progress/drone-heading-diagnostic.json) retains exact sources,
measured counterexamples, calibration, evaluations, logs, and two reproduced traces.

The selected diagnostic model is `docs/progress/drone-vector-300.mpk`.
Temporary tests were archived and removed. In the seed-42 house trace, the drone
never reaches the teacher's 0.25-metre waypoint radius; its nearest sample is
0.389 metres away. Check safe arrival regions before repeating distillation.
Neither new candidate is qualified for gameplay.

The [arrival-region teacher](DRONE_SEARCH.md#intermediate-arrival-regions)
passed 74 native trials across both routes and 37 reset seeds. Every trial
survived 60 seconds and kept sight during the final ten seconds.
This verifies the sampled teacher routes, not learned navigation or every region point.

The arrival-region learner completed 600 updates. Candidate 560 passed 32/32
stationary, 30/32 house, and 32/32 pipe trials on fresh seeds. Lateral and
approaching targets each passed 0/32; three lateral trials crashed.
The [archive](progress/drone-arrival-region-candidate.json) preserves the exact
sources, complete histories, fresh-seed gates/results, and selected diagnostic model.
The [path comparison](progress/drone-region-qualification.png) shows the same first
fresh seed for all five routes. No candidate is qualified for gameplay.

All research services in this section have stopped. Temporary tests are archived
and removed. Production Rust and the playable scene remain unchanged.
Do not repeat the completed two-route distillation recipe unchanged.

The [navigation baseline](DRONE_NAVIGATION_BASELINE.md) now plans from filtered
sight and static geometry over the unchanged learned motor pilot. It receives no
target route or hidden coordinates. Six measured variants led to 265/265 surviving
trials; every hidden-target trial ended with ten uninterrupted seconds of sight.
The weakest moving-target sight fraction was 65.6%. All 53 pipe cases still fail
the earlier 90% held-visibility threshold, with sampled gaps up to 5.8 seconds.

The [research record](progress/drone-planner-baseline.json) and compressed raw
trials preserve sources, failures, logs, and restore instructions. All planner
services have stopped. The temporary runner is archived and removed.
No production Rust or browser controller changed. CI for `79abc4c` passed both
`strict` and `contracts` jobs.

Next, turn the observation-only planner into documented example code and focused
regression tests. Correct the eye-origin mismatch, define remembered-contact
initialization, and add a blocked-door/window-only case. Preserve the motor pilot's
zero-memory contract. Run coverage, strict native/personal/WASM checks, then
integrate the programmed search baseline into the playable scene with browser
screenshots and recordings. It must remain labeled separately from learned flight.
The full roadmap, including learned search and adversarial training, remains active.

The [navigation research](DRONE_NAVIGATION.md) is archived. Imitation 150 and
open PPO 140 each passed 192 native open-route cases. Open PPO crashed in all ten
house/pipe cases. Cover PPO 260 survived all 25 selection cases and ten repeated
cover audits, but its weakest house case had sight for only 40.5% of the episode.
Do not install these models in gameplay. Deliberate hidden-target search remains
unqualified, and no new navigator has browser verification.

The [research record](progress/drone-navigation-candidate.json) preserves exact
sources, restore paths, model hashes, raw results, logs, and failed checkpoints.
The three selected models and native path plots are committed beside it.
The earlier open-route and temporary-cover services finished. The earlier navigation tests
were archived and removed from `tests/`. Prototype warnings are recorded;
they are not clean production gates. Production Rust and the playable scene did
not change in this checkpoint.

The permanent-cover follow-up above now tests targets that remain hidden.
Hearing and vertical-clearance observations remain to be added.
Do not repeat the unchanged moving-goal or open-route experiments just to resume.

The [waypoint flight guide](DRONE_TRACKING.md) now runs in the rendered browser
viewer. Select Fly east, then Run. The same frozen imitation pilot flies eight
metres east and faces east. The destination marker derives from the typed goal;
the camera frames the route and closes in as the drone arrives. Reset retains
the pilot, repairs damage, and restores seed 42. No library API changed.

The [flight recording](progress/drone-tracking-flight.mp4),
[damage recording](progress/drone-tracking-damage.mp4), and
[narrow recording](progress/drone-tracking-narrow.mp4) show the release build.
Desktop and narrow flights reach step 500 with displayed distance 0.01 m.
Pointer controls work in both layouts. A rapid move-and-click regression exposed
Bevy's previous-frame click target; the controls now activate on primary press.
The [visual record](progress/drone-tracking-viewer.json) stores source/runtime
hashes, screenshots, observations, and verification limits. Actual mobile touch
and native window interaction remain unverified.

All 61 viewer tests, nine waypoint tests, root tests, the exact native robot CI
command, strict native/WASM Clippy, and full browser robot checks pass.
Changed-line personal Rust strict/discovery checks pass without candidate-local
diagnostics. The full personal-lint backlog remains. Nix was unchanged; its
existing unfiltered source-root warnings remain.

The [coverage record](progress/drone-tracking-viewer-coverage.json) covers every added
instrumented line except native window startup at `flight.rs:66` and system
registration at `flight.rs:90`. Both outcomes of new production branches execute.
Validation logs use `tracking-viewer-pointer-` in the shared validation cache.

Moving-goal and open-route navigation now have native research evidence above.
Failed-motor pursuit, learned hidden-target search, and browser navigation remain
unqualified. The playable combat scene still uses the old hover pilot. The flight viewer's
training panel still trains recovery only.

Do not retrain or requalify the unchanged waypoint model just to resume.
Preserve its zero recurrent memory and three-metre displacement limit.
The bundled weights retain SHA-256
`4e9f539f54b261bdaf67ab3636700a76b8c7fbe28d31d6579202b500c1d577e1`.
The nine tests include the 128 held-out heading cases and 80 bounded waypoint
cases. The [research record](progress/drone-heading-candidate.json) embeds the
training sources, restore paths, failures, and selection results.

All heading experiment services are stopped. No recording is active.
The prior waypoint viewer checkpoint `2b9fe8a` has
[passing CI](https://github.com/sagan-software/bevy-gym/actions/runs/37932039075).
Its [Browser preview run](https://github.com/sagan-software/bevy-gym/actions/runs/37932038999)
was pending when this research checkpoint was prepared. Check subsequent CI and
Pages runs separately; local browser verification does not prove deployment.
Preserve the original dirty checkout and work only in `bevy-gym-quality`.

## Earlier playable combat checkpoint

The playable combat checkpoint adds telegraphed drone return fire. A full 800 ms warning
precedes three moving rounds, spaced 120 ms apart. Solid cover intercepts shots;
window and pipe openings permit them. Three impacts disable the robot. Reset
restores health and removes shots, sound events, and playing sounds. The weapon
uses current sight only.

The healthy hover policy remains unchanged.

All 57 pursuit-viewer tests, 50 hover-viewer tests, eleven native weapon cases,
and eight Chrome/WASM weapon cases pass. Root tests, strict native/WASM Clippy,
the exact native robot CI command, and changed-line personal Rust gates pass.
[Coverage](progress/pursuit-return-fire-coverage.json) includes the runnable guide
and both outcomes of every added production branch. Native startup lines 346–351
and the existing invalid-solver-pose branch at line 244 remain unhit.
The full personal Rust backlog and existing Nix source-root warnings remain.
The Nix warning locations are now `flake.nix:119` and `flake.nix:200`.

The original WAV assets pass decoding and reproducibility tests. The browser audio
activation helper has 100% line, branch, and function coverage. All 29 Node tests
pass. Python and shell lint pass. A diagnostic Web Audio probe measured nonzero
output during warning and firing. Actual speakers and volume remain unverified.

The release build passed under `bevy-gym-pursuit-return-fire-audio-build-20261009`.
Its runtime is `build-e648e66df4ffc55ddee3fbe33bec19bdcb7a737dfec6d4ead38fed828a480149`.
The [recording](progress/pursuit-return-fire.mp4) shows warning, robot death,
and reset. Desktop and 390-pixel views fit. The
[visual record](progress/pursuit-return-fire.json) retains source hashes,
media hashes, audio measurements, and the diagnostic probe source.

The audio startup regression reproduced captured-constructor import ordering.
The entry module now installs activation before dynamically importing the WASM
bindings. Normal initialization and import failure have direct tests and V8
coverage. A real suspended context resumes on keyboard input; weapon output is
nonzero and reset returns it to silence. The MP4 contains video only.

Logs use the `pursuit-return-fire-` validation-cache prefix. The `ui-gates`,
`ui-ci-native`, `ui-browser`, and `ui-personal` logs cover final Rust sources.
`final-node` covers all 29 JavaScript cases. The release log also contains an
earlier, fixed `if-let` warning; the final release and strict gates pass.

The worktree is `bevy-gym-quality`; preserve the original dirty checkout.
No recording is active. The published combat CI passed; its Pages deployment
[37917936588](https://github.com/sagan-software/bevy-gym/actions/runs/37917936588)
is running. The research checkpoint `f5a2e06` also passed
[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37920078799).

The preceding hearing checkpoint is `5c91099`; its CI passed at
[run 37909451819](https://github.com/sagan-software/bevy-gym/actions/runs/37909451819).
Its [Pages build](https://github.com/sagan-software/bevy-gym/actions/runs/37909451860)
passed. The sight checkpoint's
Pages build, [run 37904786112](https://github.com/sagan-software/bevy-gym/actions/runs/37904786112),
passed. Local browser evidence and remote deployment status remain separate gates.

Next, train pursuit without passing hidden character coordinates into the actor.
Search must choose where to look; current camera direction follows physical body
orientation. Hearing retains no exact source position. Qualify approach and
reacquisition before training armed opponents or claiming learned combat.

The fixed follow camera can lose a nearby drone below the view; improve enemy
tracking alongside pursuit. Native window execution and actual mobile touch
input remain unverified. The larger roadmap and required port order remain active.

## Historical checkpoint notes

The remaining notes retain status as recorded at each checkpoint. Follow the
resume section above for current work.

The sensor checkpoint `7f5fd6a` adds a tested geometric sight model and the
`pursuit-sight` guide. The sensor checks exposed head, chest, and hip points
through the arena geometry. Its 90-degree cone reaches 20 metres; last-seen
memory expires at three seconds. Hidden movement cannot update the remembered
position. A one-millimetre ray extension closes wall-face rounding gaps.

Seven sensor tests pass in Chrome/WASM; nine native cases include the shared
arena tests. The guide runs and prints visible, remembered, and unknown states.
Root tests, strict Clippy, the expanded native CI command, and changed-line
personal Rust gates pass. [Coverage](progress/pursuit-sight-coverage.json) records
all instrumented sensor and guide lines hit, with both outcomes of every
instrumented sensor branch. The full personal Rust backlog remains.

Nix lint still reports pre-existing unfiltered-source warnings at
`flake.nix:119` and `flake.nix:199`; this checkpoint changes the browser test list.
Logs use the `pursuit-sight-` validation-cache prefix.

Checkpoint `7f5fd6a` supplied the sensor and guide before viewer integration.
The sensor contract and remaining work are in [the game plan](DRONE_PURSUIT_GAME.md).

The live-flight checkpoint `73e1194` connects the playable arena to live flight. The bundled
healthy hover policy drives `DroneHover` with arena obstacles. Rotor hits disable
motor forces; collision or flight-region termination emits one body destruction
event. Debris inherits flight velocity. Body damage stops further flight actions.
Reset restores the scene and clears recurrent policy memory.

All 34 pursuit-viewer tests, 50 hover-viewer tests, and 15 native shot tests pass.
Thirteen shot cases pass in Chrome/WASM, alongside the existing browser suites.
Root tests, strict native/WASM Clippy, and changed-line personal Rust gates pass.
The full-project personal lint backlog remains. [Coverage](progress/pursuit-flight-coverage.json)
records the unhit invalid-solver-pose guard and defensive missing-fragment-body branch.

The optimized build passed under `bevy-gym-pursuit-flight-build-20261009`.
Its runtime is `build-ca0fba74ba40dec27f75c61d39510e6de974d9c7d9560ded88943e7d6136235e`.
The preview is `http://100.105.254.50:8781/robots/pursuit/?build=live-flight`, served
by `bevy-gym-pursuit-server-20261009`. [The recording](progress/pursuit-flight.mp4)
shows hovering, rotor damage, loss of control, and crash debris. Desktop and narrow
views were inspected; narrow reset restores the drone and pistol without overflow.
[The visual record](progress/pursuit-flight.json) retains source and media hashes.

Logs use the `pursuit-flight-` validation-cache prefix. No recording is active.
Native window execution and mobile touch input remain unverified. The controller
is healthy hover only; pursuit and learned damage recovery are unfinished.

Preserve the existing player controls, damage, reset, and qualified hover behavior
while connecting the sight model and adding game observations and rewards.

## Earlier checkpoints

The flight-collision checkpoint `1a976a1` adds `DroneObstacle` and
`DroneHover::with_obstacles`. The arena can supply its immutable boxes without
exposing the solver or duplicating motor dynamics. The new thirty-line
`drone-obstacles` guide drops the drone onto a platform. Viewer integration is
next; the playable scene still uses its stationary target.

Seven new external tests pass natively and in Chrome/WASM. Existing hover, damage,
and native bitwise-parity tests pass. Root tests, strict all-target/all-feature
Clippy, and the changed-line personal Rust gate pass. The full-project personal
lint backlog remains. Nix lint reports two existing `unfiltered_source_root`
warnings at `flake.nix:119` and `flake.nix:199`; neither is on the changed command.

[Coverage](progress/drone-obstacles-coverage.json) records all changed measured
lines hit and both outcomes of the reported production branches. The guide entry
is executed separately. Logs use the `drone-obstacles-` validation-cache prefix.
The external test first failed because the requested API did not exist.

Next, construct the viewer's flight world from `Arena::blocks()`, omitting the
floor already owned by `DroneHover`. Use the bundled healthy hover policy as the
initial controller and label it accordingly. It does not demonstrate pursuit or
damage recovery. Project the authoritative flight pose into shot hitboxes without
resetting health.

Apply each rotor destruction to `fail_motor` before the next
flight action. Route a terminal collision through the existing one-time body
death event, and pass impact velocity into debris. Reset must rebuild flight,
clear policy memory, restore health, and remove presentation effects together.
Then add occlusion-aware sensing, telegraphed return fire, and pursuit training.

The destruction checkpoint `2f80f89` connects combat to rotor flashes, smoke, body explosions, and
eight colliding debris proxies. It shares the hover viewer's renderer and uses the
arena's immutable geometry for debris collisions. The target remains stationary;
flight coupling, sensing, return fire, and learned pursuit are unfinished.

All 26 pursuit-viewer tests, 50 hover-viewer tests, root tests, strict native/WASM
Clippy, and Chrome/WASM checks pass. Personal Rust discovery found no diagnostics
in the changed combat and destruction files; the full-project backlog remains.
Coverage hits every changed measured line except native plugin installation.
The defensive missing-body branches are recorded with their exact locations.

The optimized build passed under
`bevy-gym-pursuit-destruction-soft-flash-build-20261009`. Desktop and narrow browser
checks pass, including aimed rotor shots at 390 pixels. The final preview is
`http://100.105.254.50:8781/robots/pursuit/?build=soft-destruction`, served by
`bevy-gym-pursuit-server-20261009`. Its runtime is
`build-6f31f621d9cd24ccd25ceea0cbbd38a27056ded4c1f682bf35420b162d47fde2`.

[The recording](progress/pursuit-destruction.mp4) shows rotor smoke, body destruction,
colliding debris, and reset. [The visual record](progress/pursuit-destruction.json)
retains hashes and verification limits. A browser check found a flat polygon flash;
the shared renderer now uses a soft, camera-facing radial texture instead. Its
regression failed before that change and passes afterward.

Logs use the `pursuit-destruction-` prefix in the validation cache. No recording is
active. Native window execution and mobile touch input remain unverified. The hover
viewer was tested natively and compiled for WASM, but was not separately re-recorded.

Next, connect the drone to live flight and arena collision geometry. Rotor damage
must stop its motor forces, and fatal collisions must trigger the same destruction
path. Keep the qualified hover profile unchanged. Then add occlusion-aware sensing,
telegraphed return fire, and pursuit training.

The combat checkpoint `5164de6` connects visible pistol pickup, mouse aiming, ammunition,
body damage, and rotor weak points to the playable arena. The drone remains a
stationary target. Nineteen viewer tests, 49 flight-viewer tests, seven combat
cases, fourteen native shot cases, root tests, and strict native/WASM Clippy pass.
Chrome/WASM checks pass. Coverage hits all changed measured lines outside window
startup, with both outcomes of the measured production conditions covered.

The final optimized build and desktop/narrow browser review pass. The preview is
`http://100.105.254.50:8781/robots/pursuit/?build=combat-final`, served by
`bevy-gym-pursuit-server-20261009`. Its runtime is
`build-2b671bb45b2f66711aef3277fc1fae71a5e604e8b620bb66b0d075a46e678b31`.
The release unit `bevy-gym-pursuit-combat-layout-build-20261009` exited successfully.
No recording is active. Validation logs use the `pursuit-visible-combat-` prefix.

[The recording](progress/pursuit-visible-combat.mp4) shows pickup, body damage,
rotor damage and removal, body destruction, and reset. [The visual
record](progress/pursuit-visible-combat.json)
retains source and artifact hashes. The controls no longer obscure the drone in
narrow views. A failing regression also caught queued pickup being overwritten
between simulation ticks; the queue now retains it. A separate regression protects
pistol-to-hand alignment when aiming upward.

Personal shell checks pass. Personal Rust discovery has no diagnostics in the
changed combat files; the full-project backlog remains. Native window and mobile
touch input are unverified.

Next, connect combat damage to flight failure. Then
implement occlusion-aware sensing and telegraphed drone fire before pursuit training.
The seed-7 damage curriculum remains a failed run; preserve its healthy checkpoint
and frozen promotion gates. Do not restart it without the documented rehearsal change.

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

The destruction checkpoint is published as `13c76d1`.
[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37885879298) passed.
[Pages](https://github.com/sagan-software/bevy-gym/actions/runs/37885879336) is running.

The new `drone-pursuit` example provides an original articulated robot, a windowed
house, an open pipe, and cover. Player buttons and keyboard input produce the same
closed movement action. The 23-line `pursuit-walk` guide shows the headless loop.
No drone opponent, weapons, learned pursuit, or humanoid policy is connected yet.

Nine native arena tests and five viewer tests pass. Seven external arena cases
also pass in Chrome/WASM; the two internal tests are native-only. Root tests,
strict native/WASM Clippy, format checking, Nix formatting, and Actionlint pass.
Personal shell validation passes. Personal Rust discovery reports no diagnostics
in the changed arena files; the full-project strict backlog remains. The optimized
browser build and final desktop, interior, and narrow visual checks pass.

[Coverage](progress/pursuit-arena-coverage.json) hits every measured simulation and
presentation line and branch outside native window startup. The guide's process
entry and browser initialization error message remain uninstrumented. The first
browser check found a missed between-frame reset tap; its regression failed before
the input fix and now passes. Narrow layout and pipe traversal were inspected.

The personal Nix runner reports two `unfiltered_source_root` findings at lines
119 and 199 of `flake.nix`. Both unchanged roots already sit inside
`lib.fileset.toSource` with explicit file sets. These findings conflict with the
inspected source; no source filtering was removed or weakened.

The final optimized arena build passed under
`bevy-gym-pursuit-arena-contrast-build-20261009.service`. Its runtime is
`build-f5cf0f9b16eccbfcf358ea885afd30f45b774d275e3b1d151529e8e19c77fcce`.
The preview server is `bevy-gym-pursuit-server-20261009`, serving port 8781 from
`runs/quality-research/site-pursuit`. No recording is active.

[The recording](progress/pursuit-arena.mp4) shows pipe traversal, reset, and house
entry. [The evidence record](progress/pursuit-arena.json) retains source and media
hashes. Desktop, interior, and narrow screenshots are beside it. Native window
execution and mobile touch input remain unverified. The headless guide ran for
two simulated seconds, and `cargo test --features robots --doc` passed.

The playable arena checkpoint is published as `9e48258`.
[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37888862310) and
[Pages](https://github.com/sagan-software/bevy-gym/actions/runs/37888862433) are running.

The next combat checkpoint adds private body and rotor health, bounded pistol
ammunition, proximity pickup, and a simulation-time cooldown. All seven cases pass
natively and in Chrome/WASM. Strict Clippy, root tests, formatting, and Actionlint
pass. Personal Rust
discovery reports no diagnostics in the changed files; its full strict backlog remains.

[Coverage](progress/pursuit-combat-coverage.json) hits all 88 measured helper lines
and both outcomes of eleven instrumented conditions. The guide's process entry
and printing are checked by running it, not by coverage instrumentation.
The personal Nix check retains the same two unchanged source-filter findings
described above. Validation logs use the `pursuit-combat-` prefix in
`/home/sagan/.cache/bevy-gym-quality-validation`.

The combat-rules checkpoint is published as `09ceba4`.
[CI](https://github.com/sagan-software/bevy-gym/actions/runs/37889499029) passed;
[Pages](https://github.com/sagan-software/bevy-gym/actions/runs/37889499033) is pending.

The aimed-shot model now validates aim and target poses, resolves nearest body or
rotor hits, and blocks damage through static geometry. The headless combat guide
uses actual rays. Thirteen native cases pass, including two existing arena cases;
eleven external cases pass in Chrome/WASM. Root tests, strict Clippy, format checks,
Nix formatting, and Actionlint pass. Coverage hits all 113 measured helper lines
and both outcomes of thirteen instrumented conditions.

[The shot coverage record](progress/pursuit-shots-coverage.json) retains source
hashes and execution evidence. The personal Nix check still reports the two unchanged
source-filter false positives above. Logs use the `pursuit-shots-` prefix in the
validation cache. Personal Rust discovery reports no diagnostics in the changed
shot files or guide; the full-project strict backlog remains. The guide and
documentation tests run last.

The aimed-shot checkpoint is published as `41ca7e5`; its CI passed. The visible
combat checkpoint now connects these helpers to the arena. Next, connect drone
motor failure, occlusion-aware sensing, telegraphed fire,
pursuit training, and the humanoid policy. Keep the frozen recovery
checkpoints and promotion gates. The full sequence is in
[DRONE_PURSUIT_GAME.md](DRONE_PURSUIT_GAME.md).

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
- `nix develop --command cargo check --no-default-features --features robots,browser --target
wasm32-unknown-unknown`.
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
