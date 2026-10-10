# Robot skill curriculum

Status: implementation in progress; most lessons are not implemented or trained.
Updated: 2026-10-09. The user superseded the earlier broad examples goal.
This curriculum is the current execution objective. The earlier unfinished audit
and reference ports remain deferred. A new curriculum goal is active in the
continuation thread. The earlier blocked goal was not marked complete.

Deliver separate drone and droid skill examples, a curriculum walkthrough, and a
final competitive three-versus-three environment. Every autonomous action must come
from reinforcement-trained policies under `ai/AGENTS.md`. The
[control audit](DRONES_VS_DROIDS.md) records the nonconforming current implementation.

## Lesson layout

Each lesson must have its own small guide, scene, training command, inference command,
observation/action description, rewards, promotion test, and qualified checkpoint.
The same environment implementation must serve standalone and curriculum execution.
The walkthrough shows prerequisites, current lesson, training metrics, validation,
checkpoint transfer, and failures. It must link to each independently runnable scene.
The final match is itself a trainable environment and a frozen-policy spectator example.

A shared robot body and actuator contract must survive each transition. Transfer
learned weights and retain optimizer state when compatible; reset episode and recurrent
state at lesson boundaries. Record any deliberate fresh optimizer or architecture
migration. Reject incompatible checkpoints rather than silently dropping parameters.
Keep a replay of earlier evaluation tasks to detect loss of previously learned skills.

## Drone lessons

1. `drone-hover`: balance an intact body above a landing pad with four learned thrust
   outputs. Show individual actuator commands and the simulated body. Existing calm
   hover training can supply the starting implementation; replace the constant-thrust
   tutorial before publishing this as a learned example.
2. `drone-recovery`: recover from randomized tilt, velocity, and physical disturbances.
   Transfer hover weights. Existing calm-to-disturbed RL evidence is reusable after
   source and checkpoint verification; it does not establish the later lessons.
3. `drone-travel`: reach sampled positions and headings using learned control. Gradually
   increase distance and approach speed. Do not reuse the imitation-only tracking pilot.
4. `drone-clearance`: pass openings, brake short of walls, and travel around cover using
   local range observations. Gradually narrow openings and vary layouts. No planner
   may supply waypoints or obstacle-avoidance actions.
5. `drone-perception`: locate passive targets through limited sight and sound, retain
   observations in memory, and reacquire occluded targets. Policies choose search and alert state.
   Passive targets are task objects, not scripted opponents. Render sensor limitations.
6. `drone-fire`: aim at passive targets and request shots through the real spin-up,
   projectile, and cooldown rules. Transfer flight and perception. Policies choose aim
   and trigger; the environment never automatically fires on sight.
7. `drone-damage`: recover after actuator damage, then combine damaged travel and combat.
   Curriculum steps vary impairment severity, affected rotor, and failure time.
   Evaluate every rotor separately. Allow yaw rotation where the actuator geometry
   prevents stationary heading; record uncontrollable cases without inventing stability.

The [clearance guide](DRONE_CLEARANCE.md) now provides a goal-conditioned environment
with body-frame ranges over the shared collision world. Geometry curriculum, actor
encoding, training, qualification and the scene remain unfinished.

The existing sixteen-input damage trainer covers intact hover, a fixed complete
front-left failure, and scheduled failures. Its seed-7 run failed the fixed failure
lesson. That is not a qualified implementation of the graded-damage curriculum above.

## Droid lessons

1. `droid-stand`: balance an articulated body through bounded joint actuation. Establish
   mass, inertia, joint limits, contacts, and the licensed mannequin-to-body mapping.
2. `droid-recover`: regain balance after pushes and recover from feasible fallen poses.
   Transfer standing weights; measure falls, recovery time, and joint-limit violations.
3. `droid-travel`: learn forward, reverse, sideways, and turning locomotion, then running
   and sprinting. Measure speed tracking, energy, falls, and foot sliding.
4. `droid-low-cover`: learn crouching, prone transitions, and crawling under obstacles.
   Clearance must come from the articulated body, not a resized capsule or animation.
5. `droid-jump`: jump and land across progressively larger feasible gaps and obstacles.
   Transfer balance and travel. Test landing recovery and transitions into movement.
6. `droid-aim`: hold the physical weapon, aim, and fire at passive targets while standing
   and moving. Keep visible hands, weapon pose, projectile origin, and recoil consistent.
7. `droid-evade`: combine cover, low posture, travel, and aiming against frozen qualified
   drone policies. Policies decide when to hide, move, and shoot at rotor weak points.

Current death ragdolls are useful integration evidence but do not implement living
joint control. Authored locomotion playback cannot satisfy any droid locomotion gate.

No droid lesson above is qualified. The [standing guide](DROID_STANDING.md) now provides
a thirteen-segment, twenty-six-actuator environment, PPO training and frozen evaluation.
The retained seed-7 trial uses fixed gates; its first selection failed. Qualification
remains unfinished. The physical mannequin scene runs frozen inference with an
unqualified candidate.

## Shared competition

Start with one learned drone and one learned droid after their prerequisites pass.
Then train two-versus-two and three-versus-three matches with identical control rules.
The final scene defaults to three-versus-three. Each size is a curriculum configuration
of the same shared-world implementation, not independent duels combined visually.

Collect all living agents' actions from one observation snapshot before stepping
physics. Batch trajectories across arenas. Share weights within a team and retain
separate memory per agent. Update both teams between rollouts; keep policy versions
fixed while collecting a rollout. Use frozen historical opponents for evaluation.

Retain the typed action, sensor isolation, inference failure, death, and reset checks
in [Drones vs. Droids](DRONES_VS_DROIDS.md). Centralized critics are training-only.
Use mixed continuous/categorical action distributions with correct log probabilities.
Do not replace sampled discrete decisions with undocumented continuous thresholds.

Rewards should measure valid task outcomes and physical control. Publish all shaping
terms and inspect reward exploitation. Never inject teacher actions to repair training.
Separate selection seeds from final evaluation seeds; never train on final evaluation.
Publish both teams' results against frozen opponents, not just self-play win rate.

## Promotion and evidence

Freeze each lesson's measurable thresholds before its first qualification run.
Specify episode horizon, seeds, success conditions, physical bounds, and per-case
requirements beside the environment. Missing measurements fail qualification.
An update limit stops the lesson with failure; it never grants promotion.

For each checkpoint, record the source revision, environment schema, model hash,
training algorithm, initialization, transferred parent, optimizer handling, rewards,
seed streams, updates, samples, runtime, selection results, and independent outcomes.
Save complete evaluation recordings and contact sheets, including failures.
Do not use reward curves or selected footage alone as proof of a learned skill.

## Implementation checkpoints

1. Enforce the repository instruction and record inherited violations. Keep root
   `AGENTS.md` and `CLAUDE.md` pointed at the canonical `ai/AGENTS.md` source.
2. Audit and reuse the valid RL hover/recovery foundation. Give each skill its own
   scene and inference entry point; remove constant-action and imitation-only agents
   from published examples. Verify policy-only action paths before qualification.
3. Implement compatible lesson selection, checkpoint transfer, evaluation-only
   promotion, and the curriculum walkthrough. Test rejection and budget exhaustion.
4. Implement and qualify physical droid standing/recovery, then travel, low cover,
   jumping, and weapon handling. Use separate tested milestones for each skill.
5. Implement and qualify drone travel, clearance, perception, firing, and graded damage.
6. Implement shared competitive training, grow team size, and qualify 3v3 inference.
7. Inspect all scenes in the browser, publish evidence and checkpoints, and report
   the curriculum results. Keep the deferred examples roadmap separate.

Each milestone requires focused failing tests before changed behaviour, native and
WASM checks, strict/personal lint, measured coverage with exact gaps, documentation,
and a non-force push to main. Inspect rendered desktop and narrow views for changed UI.
Keep source files and checkpoints when a run fails; fix the cause before claiming success.

## Current run

An independent existing damage-curriculum run uses seed 11 and 600 updates per lesson.
It starts from random weights and uses PPO with eight lanes of 64 transitions.
This bounded run tests the existing recipe; it does not train droids or the new lessons.

- Unit: `bevy-gym-rl-curriculum-seed11-20261009.service`.
- Log: `/home/sagan/.cache/bevy-gym-quality-validation/rl-curriculum-seed11.log`.
- Output: `runs/rl-curriculum/seed11-20261009`.
- Source at launch: `4bee8d6`, with only instruction/documentation edits in progress.
- Status at launch: compiling, then training. Completion and qualification pending.

Inspect unit state and evaluation records before reporting the result. The seeded
run must not be presented as implementation of the complete requested curriculum.
