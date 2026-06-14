# Sprint Change Proposal: Burn-Backed Bevy Gym Training Core

Date: 2026-06-13
Project: bevy-gym
Workflow: bmad-correct-course
Mode: Batch
Status: Approved for implementation planning
Approved: 2026-06-13 by Sagan

## 1. Issue Summary

The current bevy-gym redesign removed Burn-backed training from the product core. That is incorrect for the revised product direction: this crate should be a Bevy-native reinforcement-learning training crate powered by Burn, not only an environment runner.

The small `Env` / `Reset` / `Step` / `EpisodeStatus` API remains valuable, but only as the environment contract feeding the trainer. Burn must become a first-class required dependency, and the crate must train useful policies out of the box.

Evidence from current artifacts:

- `ai/bmad-output/specs/spec-bevy-gym-api-redesign/SPEC.md` currently says a minimal environment succeeds without importing Burn and lists DQN, PPO, replay buffers, policy optimization, and checkpointing as non-goals.
- `ai/bmad-output/specs/spec-bevy-gym-api-redesign/architecture-and-migration.md` currently puts DQN/PPO/Burn integration in optional examples/adapters and says Layer 1 must not mention Burn.
- `ai/bmad-output/implementation-artifacts/spec-bevy-gym-api-redesign.md` currently says "Never" add Burn trainer integration, replay buffers, policy optimizer code, or first-class trainer adapters.
- `README.md` currently says training algorithms and replay buffers stay outside the core crate.
- `docs/examples/cartpole.md` and `examples/classic-control/cart_pole.rs` currently describe CartPole as a deterministic heuristic runner example with no optimizer or learning algorithm.
- Current crates.io metadata checked on 2026-06-13 lists `burn` 0.21.0, `burn-rl` 0.21.0, and `ember-rl` 0.3.4; Burn remains the current Rust-native deep learning dependency family for this direction.

## 2. Impact Analysis

### Epic Impact

No configured PRD or epic document was found under `ai/bmad-output/planning-artifacts/`. The current implementation artifact is acting as the approved work item, and it must be reopened or superseded.

Affected current work:

- `spec-bevy-gym-api-redesign` cannot be accepted as written because its acceptance criteria require the crate to build without Burn.
- The already implemented runner API can be retained, but the completed "training-free" docs and example direction must be revised.
- New training stories are required before implementation continues.

Recommended epic structure:

- Replace the current "API Redesign" epic framing with "Burn-Backed Bevy Gym Training Core".
- Keep a sub-story for the small environment contract and Bevy runner.
- Add first-slice DQN training stories for CartPole.
- Add second-slice PPO stories for parallel Bevy environments.
- Add documentation and acceptance stories around checkpoints, metrics, deterministic eval, and out-of-box training commands.

### Story Impact

Current in-review story:

- `spec-bevy-gym-api-redesign` should move from `in-review` to `needs-replan` or be superseded.
- Its completed code may be partially preserved, but its "no Burn/training" acceptance must be removed.

New stories needed:

- `BG-TRAIN-001`: Add required Burn dependency and training module layout.
- `BG-TRAIN-002`: Define Burn model interfaces and default MLP Q-network for discrete control.
- `BG-TRAIN-003`: Implement DQN trainer with optimizer, target network, epsilon schedule, and replay buffer.
- `BG-TRAIN-004`: Add metrics, run directory layout, checkpoint save/load, and `best.mpk` export.
- `BG-TRAIN-005`: Convert CartPole from heuristic demo to train/eval example with deterministic checkpoint evaluation.
- `BG-TRAIN-006`: Add acceptance tests or smoke scripts proving CartPole DQN reward improves and checkpoint eval succeeds.
- `BG-TRAIN-007`: Add PPO rollout storage and policy/value model as the second slice using Bevy parallel environments.

### Artifact Conflicts

Conflicting artifacts:

- SPEC: capabilities, constraints, non-goals, success signal, and open questions.
- API contract: the environment API remains valid, but trainer integration is no longer optional composition outside the crate.
- Architecture: optional adapters/examples must become required training engine layers.
- Implementation artifact: "Ask First", "Never", tasks, acceptance, and verification all conflict.
- README and CartPole docs: product positioning and example behavior conflict.
- Cargo manifest: Burn must be required, not absent.

Optional or missing artifacts:

- PRD: missing and should be created or restored.
- Epics/stories: missing and should be created or restored.
- UX: not applicable for this headless library change.
- `sprint-status.yaml`: not found, so no sprint status update can be made yet.

### Technical Impact

The crate needs a required Burn-backed training engine:

- Required Burn dependency in `Cargo.toml`.
- Model definitions for discrete DQN first, then actor-critic PPO.
- Optimizer setup and learning-rate configuration.
- Replay storage for DQN and rollout storage for PPO.
- Metrics API and JSONL or equivalent append-only run metrics.
- Checkpoint save/load with a stable run layout and `best.mpk`.
- Deterministic evaluation from a saved checkpoint.
- CartPole DQN training and eval commands that work out of the box.
- Tests or smoke gates that prove reward improvement, checkpoint creation, and eval success.

The small environment contract should stay:

- `Env`, `Reset`, `Step`, and `EpisodeStatus` remain the environment boundary.
- The Bevy runner continues to own parallel environment stepping.
- Training consumes runner transitions instead of replacing the runner contract.

## 3. Recommended Approach

Recommended path: Hybrid of Direct Adjustment and targeted rollback.

Direct adjustment:

- Keep the newly simplified environment contract and typed Bevy runner.
- Add Burn-backed trainer modules on top of that contract.
- Update docs and examples to present training as core product behavior.

Targeted rollback:

- Remove all "no Burn", "no replay buffer", "no optimizer", "no checkpointing", and "training outside the core crate" instructions from the current spec and implementation artifact.
- Replace the training-free CartPole acceptance target with CartPole DQN training acceptance.

Scope classification: Major.

Rationale:

- The change redefines product core, dependency policy, architecture layers, examples, acceptance criteria, and verification.
- It does not require throwing away the small environment API; that part is still the right contract.
- It does require a full replan of the current in-review implementation artifact before further implementation should be accepted.

Effort estimate: High.

Risk level: Medium-high.

Primary risks:

- Burn API and serialization choices must be validated against current `burn` 0.21.0 behavior.
- DQN training acceptance can be flaky unless seeds, evaluation protocol, thresholds, and CI/runtime budgets are carefully bounded.
- PPO should wait until DQN proves the trainer, metrics, and checkpoint surfaces.

Stop condition for this correction:

- The planning artifacts no longer frame Burn/training as optional or forbidden.
- The first implementation slice has clear acceptance: CartPole DQN trains, reward improves, `best.mpk` is saved, and deterministic eval from checkpoint succeeds.

## 4. Detailed Change Proposals

### Proposal A: SPEC Product Intent

Artifact: `ai/bmad-output/specs/spec-bevy-gym-api-redesign/SPEC.md`

Section: Why

OLD:

```text
The opportunity is a smaller repo-owned RL environment API that is beginner-simple, algorithm-agnostic, rendering-agnostic, and still usable for batched training in Bevy.
```

NEW:

```text
The opportunity is a Bevy-native reinforcement-learning training crate powered by Burn. The crate should keep a small repo-owned environment API, but that API is the contract feeding the required trainer. A new user should be able to define an environment, run Burn-backed training, save the best checkpoint, and evaluate the saved policy from the same crate.
```

Rationale: Restores Burn training as the product core while preserving the simplified environment contract.

### Proposal B: SPEC Capabilities

Artifact: `ai/bmad-output/specs/spec-bevy-gym-api-redesign/SPEC.md`

Section: Capabilities

OLD:

```text
CAP-1 success: A minimal environment compiles and runs without importing `rl-traits`, `ember-rl`, Burn, Bevy rendering, replay buffers, or policy abstractions.
```

NEW:

```text
CAP-1 success: A minimal environment compiles with `Env`, `Reset`, `Step`, and `EpisodeStatus`, and the same environment can feed the crate's required Burn-backed trainer without an adapter crate.
```

Add:

```text
- id: CAP-7
  intent: Users can train useful policies out of the box with a required Burn-backed training engine.
  success: CartPole DQN trains from the crate's example or CLI, improves deterministic evaluation reward over the initial/random policy, saves `best.mpk`, and can evaluate successfully from that checkpoint.

- id: CAP-8
  intent: Training artifacts are first-class and inspectable.
  success: Each run writes metrics, config/seed metadata, checkpoints, and a best-policy artifact in a stable run directory.

- id: CAP-9
  intent: PPO becomes the second training slice and uses Bevy parallel environments effectively.
  success: PPO rollout collection uses multiple Bevy environment entities and deterministic eval can load a saved PPO checkpoint.
```

Rationale: Adds training capabilities with explicit DQN first-slice and PPO second-slice acceptance.

### Proposal C: SPEC Constraints and Non-Goals

Artifact: `ai/bmad-output/specs/spec-bevy-gym-api-redesign/SPEC.md`

Section: Constraints / Non-goals

OLD:

```text
Own the core environment vocabulary in this crate; base public API must not depend on `rl-traits`, `ember-rl`, Burn, replay buffers, policy traits, or experience abstractions.
```

NEW:

```text
Own the core environment vocabulary in this crate; the minimal environment trait must stay small and must not depend on `rl-traits` or `ember-rl`. Burn, replay/rollout storage, model definitions, optimizers, metrics, and checkpoints are required trainer-layer concerns in this crate.
```

OLD:

```text
Implement DQN, PPO, SAC, replay buffers, policy optimization, checkpointing, or other trainer logic in the core crate.
```

NEW:

```text
Implement SAC or broad algorithm coverage before the DQN and PPO slices are accepted.
```

Rationale: Removes the direct contradiction with the course correction while limiting algorithm scope.

### Proposal D: Architecture Layers

Artifact: `ai/bmad-output/specs/spec-bevy-gym-api-redesign/architecture-and-migration.md`

Section: Layer 1 and Layer 3

OLD:

```text
This layer must not mention rendering, Burn, `ember-rl`, replay buffers, policy traits, Bevy schedules, or vector runner concerns.
```

NEW:

```text
This layer remains the small environment contract and should not contain model, optimizer, replay, or checkpoint code. Burn is required by the crate, but Burn-specific code belongs in the trainer layer rather than inside `Env`.
```

OLD:

```text
Examples or separate adapters can show:
- DQN/PPO/Burn integration
```

NEW:

```text
Layer 3: Burn Training Engine

The crate must provide required Burn-backed training modules:
- model definitions for DQN first and PPO second
- optimizer setup
- replay buffer for DQN
- rollout storage for PPO
- metrics and run metadata
- checkpoint save/load and `best.mpk`
- deterministic eval from checkpoint

Examples demonstrate the trainer, but the trainer is not merely an example or adapter.
```

Rationale: Keeps `Env` clean while making training a required architectural layer.

### Proposal E: Implementation Artifact Boundaries

Artifact: `ai/bmad-output/implementation-artifacts/spec-bevy-gym-api-redesign.md`

Section: Boundaries & Constraints

OLD:

```text
Ask before adding new external dependencies, deleting `ref/` reference directories, or adding a first-class trainer/algorithm adapter beyond the repo-owned runner.

Never: Do not add compatibility shims, deprecated aliases, old event names, old render features, `rl-traits`, `ember-rl`, Burn trainer integration, replay buffers, policy optimizer code, or a generic rendering abstraction.
```

NEW:

```text
Always: Keep the small environment contract as the trainer input boundary. Burn is a required dependency. The crate must include trainer-layer model definitions, optimizers, replay/rollout storage, metrics, checkpoint save/load, and deterministic eval.

Never: Do not reintroduce `rl-traits` as the public contract, make `ember-rl` the product core, hide training in examples only, or expand `Env` until it contains trainer internals.
```

Rationale: Removes the explicit blocker to the new product core.

### Proposal F: First Acceptance Target

Artifact: `ai/bmad-output/implementation-artifacts/spec-bevy-gym-api-redesign.md`

Section: Tasks & Acceptance / Acceptance Criteria

OLD:

```text
Given a clean checkout, when `cargo check` runs, then the crate builds without `rl-traits`, `ember-rl`, or Burn.
```

NEW:

```text
Given a clean checkout, when `cargo check` runs, then the crate builds with required Burn support and without `rl-traits` or `ember-rl`.

Given the CartPole DQN example is run with fixed seeds, then training improves deterministic evaluation reward over the initial/random policy baseline.

Given CartPole DQN training completes, then `best.mpk` is written in the run directory with accompanying metrics and config metadata.

Given `best.mpk` exists, then deterministic eval can load the checkpoint and report a successful CartPole evaluation.
```

Rationale: Aligns acceptance with the user's first target.

### Proposal G: README Positioning

Artifact: `README.md`

Section: Intro / Design goals

OLD:

```text
Bevy ECS plugin for parallelised reinforcement-learning environment simulation.

Training algorithms, replay buffers, and rendering stay outside the core crate.
```

NEW:

```text
Bevy-native reinforcement-learning training crate powered by Burn.

Bring a Rust environment, get a Bevy runner and Burn-backed trainer. `Env`, `Reset`, `Step`, and `EpisodeStatus` define the environment contract; the crate provides the trainer, replay/rollout storage, metrics, checkpoints, and deterministic evaluation.
```

Rationale: Product positioning must lead with training, not only simulation.

### Proposal H: CartPole Example and Docs

Artifacts:

- `docs/examples/cartpole.md`
- `examples/classic-control/cart_pole.rs`

Section: Notes / example purpose

OLD:

```text
The heuristic policy is intentionally small and deterministic. It is not a learning algorithm; it is there to prove the environment/runner API without pulling trainer code into this crate.
```

NEW:

```text
CartPole is the first training acceptance example. It supports DQN training and deterministic checkpoint evaluation. A short heuristic or random policy may remain as a baseline, but the primary path trains a Burn-backed policy, writes metrics and `best.mpk`, and verifies eval from the saved checkpoint.
```

Rationale: CartPole must prove useful training out of the box.

### Proposal I: New Story Set

Artifact: new or restored epics/stories under `ai/bmad-output/planning-artifacts/`

OLD:

```text
No PRD or epic/story file was found in the configured planning artifacts.
```

NEW:

```text
Epic: Burn-Backed Bevy Gym Training Core

Story BG-TRAIN-001: Required Burn dependency and trainer module skeleton
Acceptance: `cargo check` includes required Burn support; public exports expose trainer-layer modules without expanding `Env`.

Story BG-TRAIN-002: DQN model and optimizer
Acceptance: A default Burn MLP Q-network and optimizer config can be constructed for discrete environments.

Story BG-TRAIN-003: DQN replay and training loop
Acceptance: Replay storage, target updates, epsilon schedule, and batch optimization run against Bevy-collected CartPole transitions.

Story BG-TRAIN-004: Metrics and checkpoints
Acceptance: Training writes metrics, config metadata, checkpoints, and `best.mpk`.

Story BG-TRAIN-005: Deterministic eval
Acceptance: Eval loads `best.mpk` and reports deterministic CartPole reward for fixed seeds.

Story BG-TRAIN-006: PPO second slice
Acceptance: PPO rollout storage uses multiple Bevy env entities and can train/eval a saved policy.
```

Rationale: The workflow cannot route implementation safely without explicit planning artifacts.

## 5. Checklist Status

| Item | Status | Notes |
| --- | --- | --- |
| 1.1 Triggering story identified | [x] | Current in-review `spec-bevy-gym-api-redesign` implementation artifact revealed the issue. |
| 1.2 Core problem defined | [x] | Misunderstanding of original/current product requirement: training was removed from core. |
| 1.3 Evidence gathered | [x] | Conflicts found in SPEC, architecture, implementation artifact, README, CartPole docs/example, and Cargo. |
| 2.1 Current epic/story evaluated | [x] | Cannot complete as originally planned; must be superseded or reopened. |
| 2.2 Epic-level changes determined | [x] | Add Burn-backed training core epic/story set. |
| 2.3 Remaining epics reviewed | [!] | No epic file found under configured planning artifacts. |
| 2.4 Future epic invalidation checked | [!] | Current artifacts are too thin; likely future trainer work must be reprioritized ahead of environment-only examples. |
| 2.5 Epic priority checked | [x] | DQN first, PPO second. |
| 3.1 PRD conflicts checked | [!] | No PRD found; PRD must be created/restored to encode Burn training as product core. |
| 3.2 Architecture conflicts checked | [x] | Required trainer layer must replace optional adapter framing. |
| 3.3 UX conflicts checked | [N/A] | No UI/UX artifact and no UI surface in this correction. |
| 3.4 Other artifacts checked | [x] | Cargo, README, docs, example, verification commands impacted. |
| 4.1 Direct adjustment evaluated | [x] | Viable with high effort and medium-high risk. |
| 4.2 Rollback evaluated | [x] | Full rollback not needed; targeted rollback of anti-Burn constraints is needed. |
| 4.3 MVP review evaluated | [x] | MVP must be redefined around DQN training acceptance. |
| 4.4 Path selected | [x] | Hybrid: direct adjustment plus targeted rollback. |
| 5.1 Issue summary created | [x] | Included above. |
| 5.2 Epic/artifact impacts documented | [x] | Included above. |
| 5.3 Recommended path documented | [x] | Included above. |
| 5.4 MVP action plan defined | [x] | DQN first slice, PPO second slice. |
| 5.5 Handoff plan established | [x] | See below. |
| 6.1 Checklist reviewed | [x] | Action-needed items are missing PRD/epics/sprint status. |
| 6.2 Proposal accuracy reviewed | [x] | Based on local artifacts plus current crates.io metadata. |
| 6.3 User approval obtained | [x] | Approved by Sagan on 2026-06-13. |
| 6.4 Sprint status updated | [N/A] | `sprint-status.yaml` was not found. |

## 6. Implementation Handoff

Scope: Major.

Route to:

- Product Manager / Architect: update or create PRD, epics, architecture, and acceptance criteria around Burn-backed training as product core.
- Developer: after approval, revise the current implementation artifact/spec and implement the DQN first slice.
- Test Architect / QA: define deterministic training/eval thresholds that are meaningful but stable enough for CI.

Developer handoff tasks after approval:

1. Update the SPEC, architecture, implementation artifact, README, and CartPole docs with the approved text direction.
2. Add required Burn dependencies and trainer module layout.
3. Implement DQN CartPole training with model, optimizer, replay buffer, target updates, metrics, checkpointing, and deterministic eval.
4. Save `best.mpk` and verify eval can load it.
5. Add a fast smoke gate for CI and a longer training command for release/manual validation.
6. Start PPO only after the DQN checkpoint/eval path is accepted.

Success criteria:

- Burn is a required first-class dependency.
- `Env` / `Reset` / `Step` / `EpisodeStatus` stay small and remain the environment contract.
- CartPole DQN trains with fixed seeds and improves deterministic eval reward.
- `best.mpk` is saved.
- Eval from checkpoint succeeds.
- PPO is explicitly queued as the second slice using Bevy parallel environments.

## 7. Approval Gate

Approved by Sagan on 2026-06-13.

Decision:

- Accept this course correction.
- Route as a major change to Product Manager / Architect for planning artifact updates.
- Hand Developer the approved proposal for the Burn-backed DQN first slice after planning artifacts are updated.
- Leave `sprint-status.yaml` unchanged because no sprint status file was found.

## 8. Workflow Completion Log

- Issue addressed: Burn-backed training was incorrectly removed from the bevy-gym product core.
- Change scope: Major.
- Artifacts modified by this workflow: `ai/bmad-output/planning-artifacts/sprint-change-proposal-2026-06-13.md`.
- Artifacts requiring follow-up updates: SPEC, architecture companion, implementation artifact, README, CartPole docs/example, Cargo dependency policy, PRD, epics/stories, and any future sprint status file.
- Routed to: Product Manager / Architect for planning rework; Developer for implementation after replan; Test Architect / QA for deterministic train/eval thresholds.
