# Drone curriculum

Status: native training qualified; browser playback verified, 2026-10-08.

Train calm hover before disturbed recovery. Keep the same actor, critic, and
optimizer across lessons. Reset environment lanes and recurrent memory when the
lesson changes. The existing direct-recovery guide and browser protocol remain
unchanged until this curriculum has its own qualification evidence.

## Contract

- The default sequence is calm hover, then disturbed recovery.
- `--lesson hover` and `--lesson recovery` each train only the selected lesson.
- Each lesson uses eight independent lanes and 64 actions per lane per update.
- Every twentieth update evaluates the five existing selection seeds independently.
- Advance only when all five episodes survive, mean return reaches 400, and mean
  final distance is at most 0.5 m. These thresholds are local lesson policy.
- A finite per-lesson update budget reports exhaustion without advancing.
- Evaluation never changes optimizer state or contributes training samples.
- Final qualification uses the existing 32 high-bit seeds and frozen gates.
- Preserve existing reset streams, direct-recovery results, and native/WASM parity.
- Use the existing public environment and optimizer APIs. Keep lesson helpers private.

Required checks are a failing integration test before implementation, selected
lesson construction, deterministic batches, independent evaluation, exact promotion
boundaries, budget exhaustion, and a real run with recorded lesson transitions.
Run repository Rust gates, personal lints, focused coverage, guides, and browser
checks before publishing the completed lesson. Add browser curriculum controls
only after the native training result passes final qualification.

## Research and damage limits

[Unity's curriculum documentation][unity-curriculum]
uses lesson completion criteria to advance environment difficulty. This lesson
uses independent evaluation scores as its completion criterion; it does not copy
Unity's configuration format or claim the same training algorithm.

[UZH's quadrotor learning supplement](https://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf)
drops yaw-angle and yaw-rate penalties for its motor-failure task. The
[UZH fault-tolerant controller](https://github.com/uzh-rpg/fault_tolerant_control)
is another reference for flight after a complete rotor failure.

The current drone has fixed, parallel thrust axes and alternating reaction moments.
With one corner motor absent, zero roll and pitch torque require its opposite
motor to produce zero thrust. The two remaining motors have the same reaction
sign. Positive lift therefore cannot also produce zero yaw torque in static hover.
This follows from the checked-in motor offsets and force model, not from an RL
training failure. Damaged flight must allow yaw rotation or use a changed actuator
model. Do not promise stationary four-axis hover after losing one rotor.

## Current evidence

The new example is `drone-curriculum`. Run it from the repository root:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --updates 600 --seed 7 \
  --output runs/drone-curriculum/my-run
```

For standalone training, add `--lesson hover` or `--lesson recovery`. These are the
control lesson names. Both use the same lesson factories, collector, evaluator, and
promotion criteria as the default sequence. Standalone training starts from random
weights unless qualified initialization is selected below. The default
sequence retains its optimizer across the hover-to-recovery transition.

Use a distinct output directory for each run. Checkpoints and scores include the
lesson and update in their filenames. The final update also receives an evaluation
when the budget is not divisible by twenty. The command exits unsuccessfully when
a lesson exhausts its budget. It never advances solely because time has elapsed.

Fourteen learning tests and eight curriculum tests pass natively. All fourteen
learning tests pass in the actual browser. Root tests, strict native/WASM Clippy,
and formatting pass. Personal-lint discovery reaches the new example without
candidate diagnostics; the existing full-package backlog remains unresolved.
The example allows the synchronous-file-I/O lint locally, with a reason.
The tutorial uses ordinary synchronous file writes.

[Coverage and CLI evidence](progress/drone-curriculum-coverage.json) record the
promotion boundaries, zero-budget rejection, and one/two-update exhaustion paths.
All 90 measured lesson-helper lines and eight branch outcomes were hit, including
its tests. Instrumented CLI coverage is 30/37 lines and 5/8 branch outcomes.
The full uninstrumented run exercised promotion, both lessons, and successful
completion. I/O failure propagation and instrumented success-path coverage remain
gaps; the coverage record names them.

The seed-7 run completed calm hover after 280 updates and disturbed recovery after
20 further updates. It retained one optimizer and used 153,600 training transitions.
The saved recovery model survived all 32 held-out episodes, with mean return
443.4709 and mean final distance 0.1580 m. Native and browser qualification passed.
This run used 300 updates; the earlier direct-recovery run used 260. These results
do not establish a curriculum efficiency advantage.

[The record](progress/drone-curriculum.json) retains every selection evaluation,
final qualification, model hash, and training location. The frozen
[checkpoint](progress/drone-curriculum.mpk) has a permanent native/WASM regression.
[The learning curve](progress/drone-curriculum-learning.png) shows both lessons.

[The flight recording](progress/drone-curriculum-flight.mp4),
[contact sheet](progress/drone-curriculum-flight-contact.png), and
[screenshot](progress/drone-curriculum-flight.png) show actual Bevy browser playback.
The native-trained checkpoint was loaded through the existing browser bridge.
The idle training panel did not produce this model. The displayed seed-42 episode
reached action 500. Browser curriculum training controls remain to be implemented.

The completed unit is `bevy-gym-curriculum-run-20261008.service`. Its log remains at
`/home/sagan/.cache/bevy-gym-quality-validation/logs/curriculum-run.log`.
Intermediate artifacts remain in `runs/drone-curriculum/seed7-initial`.

[unity-curriculum]: https://unity-technologies.github.io/ml-agents/Training-ML-Agents/#curriculum

## Qualified standalone transfer

Run recovery from the recorded PPO hover prerequisite:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example drone-curriculum -- --lesson recovery --initialize-from qualified-hover \
  --seed 11 --updates 600 --output runs/drone-recovery/qualified-transfer
```

`qualified-hover` is the only accepted source. It requires `--lesson recovery`.
Unsupported combinations fail before output creation. Missing or changed source
networks fail without a fallback. Loading compares the complete actor and critic
with the embedded qualified reference. Transfer starts a fresh optimizer and
sampling streams; the default two-lesson sequence retains its optimizer.

Before the first update, `transfer.json` records one object with nine required,
non-null members. `event` is the string `checkpoint-transfer`; `source` is
`qualified-hover`; `source_checkpoint` is `docs/progress/drone-hover.mpk`.
`qualified_record_sha256` is the lowercase SHA-256 of that committed reference
record, not of an arbitrary alternate serialization. `source_training` is
`PPO, seed 7, hover update 280`; `destination` is `recovery`; `optimizer` is `fresh`.
`optimizer_steps` is the integer zero; `seed` is the supplied unsigned 64-bit integer.
Member order is not promised. No optional members or null values are emitted.

The [transfer evidence](progress/drone-checkpoint-transfer.json) records the source,
selection evaluations, held-out episodes, hashes, validation, and coverage gaps.
The source passed 32/32 held-out hover episodes natively and in WASM. Seed 11
passed recovery selection after 20 updates and 10,240 new transitions. The saved
`docs/progress/drone-recovery-transfer.mpk` passed 32/32 recovery episodes and
32/32 hover replay episodes natively and in WASM. Native mean returns were
441.3443 and 461.1305, respectively. These transitions exclude prerequisite training.
This single trial does not establish a training efficiency advantage.

The transfer module records 71/71 measured lines, including tests, and both branch
outcomes. Main records 53/56 lines and 11/14 branch outcomes. Serialization,
known-valid embedded decoding, CLI filesystem errors, and existing loop paths
remain the exact gaps listed in the evidence. Strict native/WASM gates and the
browser suite pass. Changed-line personal lint is clean; strict personal lint
still fails on the unchanged repository backlog. Existing scene recordings use
the original curriculum checkpoint, not these new transferred weights.

## Travel extension

`--lesson travel` runs the separate near/far/fast PPO sequence described in the
[travel guide](DRONE_TRAVEL.md). It transfers the qualified recovery actor into
13-input observations with a zero-weight heading feature and a fresh critic and
optimizer. Frozen inference uses `--evaluate-checkpoint` and the same environment.
No travel candidate is qualified yet. The default hover/recovery sequence and its
qualification evidence are unchanged; combined progression through qualified travel
remains unfinished.
