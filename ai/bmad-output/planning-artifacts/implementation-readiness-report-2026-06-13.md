---
stepsCompleted: [1, 2, 3, 4, 5, 6]
workflowType: 'implementation-readiness'
project_name: 'bevy-gym'
user_name: 'Sagan'
date: '2026-06-13'
status: 'complete'
completedAt: '2026-06-13'
inputDocuments:
  - prd-burn-backed-bevy-gym-training-core-2026-06-13.md
  - architecture.md
  - epics-burn-backed-bevy-gym-training-core-2026-06-13.md
  - implementation-readiness-report-2026-06-13.md
---

# Implementation Readiness Assessment Report

**Date:** 2026-06-13
**Project:** bevy-gym
**Assessment:** Revalidation after Burn-backed training-core story refinements

## Document Discovery

### PRD Files Found

**Whole Documents:**

- `prd-burn-backed-bevy-gym-training-core-2026-06-13.md` (10,169 bytes, modified 2026-06-13 08:15)

**Sharded Documents:**

- None found.

### Architecture Files Found

**Whole Documents:**

- `architecture.md` (26,582 bytes, modified 2026-06-13 08:38)

**Sharded Documents:**

- None found.

### Epics And Stories Files Found

**Whole Documents:**

- `epics-burn-backed-bevy-gym-training-core-2026-06-13.md` (18,038 bytes, modified 2026-06-13 09:45)

**Sharded Documents:**

- None found.

### UX Design Files Found

**Whole Documents:**

- None found.

**Sharded Documents:**

- None found.

### Discovery Issues

- No duplicate whole-vs-sharded document formats found.
- UX design document not found. This is non-blocking for the current headless Rust library and example-command scope.
- The prior readiness report was intentionally updated in place at the same dated path.

## PRD Analysis

### Functional Requirements

FR-1: Define Env With Repo-Owned Types. Users can define environments with `Env`, `Reset`, `Step`, and `EpisodeStatus`. A minimal environment compiles without `rl-traits` or `ember-rl`, and the same environment can feed the crate's trainer once trainer-required specs/tensorization are supplied.

FR-2: Preserve Terminated Versus Truncated. The system preserves terminated and truncated episode status from environment step through trainer storage. DQN and PPO target/advantage logic can distinguish natural termination from time-limit truncation, and auto-reset does not emit stale post-step action requests after terminal/truncated states.

FR-3: Spawn And Step Parallel Environments. Users can spawn many environment instances from a factory and step them through `FixedUpdate`. A runner example creates at least 16 environments, and transitions include env id, entity, observation, action, reward, next observation, status, and info.

FR-4: Provide Typed Action Request/Response Flow. Manual policies can respond to typed action requests without mutating `PendingAction`. Docs show `ActionRequest<E>` and `ActionResponse<E>` as the beginner manual path, while low-level ECS access remains an advanced escape hatch.

FR-5: Add Required Burn Dependency And Trainer Modules. The crate builds with Burn as a first-class dependency and exposes trainer-layer modules. `cargo check` succeeds with Burn support, and trainer code does not expand the minimal `Env` trait.

FR-6: Provide DQN Model, Optimizer, And Replay Storage. The crate provides a default Burn MLP Q-network, optimizer setup, epsilon schedule, target network policy, and replay buffer for discrete control. CartPole DQN can optimize from Bevy-collected transitions, and replay storage lives outside ECS components.

FR-7: Provide Metrics And Checkpoints. Training writes metrics, config/seed metadata, checkpoints, and `best.mpk`. Each run has a stable directory layout, and `best.mpk` is updated based on deterministic eval improvement.

FR-8: Provide Deterministic Eval From Checkpoint. Eval loads a checkpoint from disk and runs fixed-seed episodes without exploration. Eval from `best.mpk` succeeds after CartPole DQN training, and eval does not rely on the in-memory training model.

FR-9: Make CartPole DQN The First Acceptance Example. CartPole is the first train/eval acceptance target. Training reward or deterministic eval reward improves over initial/random baseline, and docs/examples present DQN training as primary, with heuristic/random only as baselines.

FR-10: Add PPO Rollout Storage. The crate records observations, actions, log probabilities, values, rewards, statuses, and env ids from multiple Bevy entities. Rollout storage supports advantage/return computation, and terminated/truncated states are handled correctly.

FR-11: Add PPO Actor-Critic Training And Eval. The crate trains a Burn actor-critic policy and evaluates a saved PPO checkpoint. PPO uses multiple Bevy environment entities, and PPO checkpoint/eval follows the same artifact conventions as DQN.

Total FRs: 11

### Non-Functional Requirements

NFR-1: Keep training and eval deterministic enough to reproduce evaluation from a saved checkpoint using fixed eval seeds.

NFR-2: Keep the crate headless-first; rendered dashboards and rendering demos must not be proof of training correctness.

NFR-3: Keep the public product contract Rust-native and crate-owned; do not require `rl-traits`, `ember-rl`, or Python as the public contract.

NFR-4: Keep the single-environment `Env` contract intentionally small; model definitions, optimizers, replay/rollout storage, metrics, checkpointing, and deterministic eval live above it in the required trainer layer.

NFR-5: Keep training artifacts inspectable: users can inspect run metrics, configs, seeds, and checkpoint artifacts.

NFR-6: Keep replay storage outside ECS components; PPO rollout storage must also avoid placing large trainer tensors in ordinary ECS component state.

NFR-7: Keep first-slice scope narrow; DQN and checkpoint/eval correctness come before broad algorithm coverage.

NFR-8: Preserve robotics boundary; `bevy-gym` does not claim robotics-grade sim-to-real dynamics validation from Bevy alone.

Total NFRs: 8

### Additional Requirements

- CartPole DQN is the first acceptance slice and must prove reward improvement, `best.mpk`, and deterministic eval from disk.
- Burn is required and first-class, with trainer internals above the small `Env` contract.
- PPO is the second slice and uses Bevy parallel environments.
- `rl-traits` and `ember-rl` must not become public/product-core dependencies.
- Architecture resolves the earlier PRD open questions for Burn backend, autodiff backend, recorder, run directory layout, train/eval command shape, CI smoke versus manual validation, and `src/training/*` module boundaries.
- Release/deprecation policy for already-completed runner-only changes remains a product-management cleanup item, not a blocker for implementation of the Burn-backed core.

### PRD Completeness Assessment

The PRD is complete enough for implementation planning. It clearly replaces the runner-only interpretation with a Burn-backed training core, preserves the small environment contract, defines DQN acceptance behavior, and sequences PPO second. Remaining ambiguity around release/deprecation policy does not block the first implementation sprint.

## Epic Coverage Validation

### Epic FR Coverage Extracted

- FR-1: BG-TRAIN-001.
- FR-2: BG-TRAIN-001, BG-TRAIN-004, BG-TRAIN-007.
- FR-3: BG-TRAIN-001, BG-TRAIN-007.
- FR-4: BG-TRAIN-001.
- FR-5: BG-TRAIN-001.
- FR-6: BG-TRAIN-002, BG-TRAIN-004.
- FR-7: BG-TRAIN-003, BG-TRAIN-005.
- FR-8: BG-TRAIN-005.
- FR-9: BG-TRAIN-004, BG-TRAIN-006.
- FR-10: BG-TRAIN-007.
- FR-11: BG-TRAIN-008.

Total FRs in epics: 11

### Coverage Matrix

| FR Number | PRD Requirement | Epic Coverage | Status |
| --- | --- | --- | --- |
| FR-1 | Define Env with repo-owned `Env`, `Reset`, `Step`, and `EpisodeStatus` types. | BG-TRAIN-001 | Covered |
| FR-2 | Preserve terminated versus truncated episode status through trainer storage. | BG-TRAIN-001, BG-TRAIN-004, BG-TRAIN-007 | Covered |
| FR-3 | Spawn and step parallel environments through `FixedUpdate`. | BG-TRAIN-001, BG-TRAIN-007 | Covered |
| FR-4 | Provide typed `ActionRequest<E>` / `ActionResponse<E>` flow. | BG-TRAIN-001 | Covered |
| FR-5 | Add required Burn dependency and trainer modules. | BG-TRAIN-001 | Covered |
| FR-6 | Provide DQN model, optimizer, and replay storage. | BG-TRAIN-002, BG-TRAIN-004 | Covered |
| FR-7 | Provide metrics and checkpoints. | BG-TRAIN-003, BG-TRAIN-005 | Covered |
| FR-8 | Provide deterministic eval from checkpoint. | BG-TRAIN-005 | Covered |
| FR-9 | Make CartPole DQN the first acceptance example. | BG-TRAIN-004, BG-TRAIN-006 | Covered |
| FR-10 | Add PPO rollout storage. | BG-TRAIN-007 | Covered |
| FR-11 | Add PPO actor-critic training and eval. | BG-TRAIN-008 | Covered |

### Missing Requirements

No missing PRD functional requirement coverage found.

### Coverage Statistics

- Total PRD FRs: 11
- FRs covered in epics: 11
- Coverage percentage: 100%

## UX Alignment Assessment

### UX Document Status

Not found. No whole or sharded UX document exists under the configured planning artifacts path.

### UX/UI Implied By PRD Or Architecture

No web, mobile, dashboard, or interactive graphical UI is implied by the accepted PRD or architecture. The user-facing surfaces are:

- Rust library API.
- Example command paths, especially `cartpole_dqn train` and `cartpole_dqn eval`.
- README, crate docs, plugin docs, and CartPole docs.
- Training artifacts in run directories.

The PRD excludes rendered training dashboards from MVP scope. The architecture confirms this is an existing Rust library crate, not a greenfield web, mobile, or API application.

### Alignment Issues

No UX-to-PRD or UX-to-architecture alignment issue found.

### Warnings

- UX artifact absent. This is not blocking for the current headless Rust library scope.
- Documentation and command ergonomics are captured in FR-9, architecture command-shape decisions, and BG-TRAIN-006.

## Epic Quality Review

### Review Summary

The refined epics and stories now meet implementation-readiness expectations for sprint planning. The previous story-quality blockers have been addressed:

- BG-TRAIN-001 can no longer pass with empty trainer modules because it requires concrete exported trainer-boundary types, Burn backend defaults, a compile proof, unchanged `Env` internals, and product-surface scans for `rl-traits` and `ember-rl`.
- BG-TRAIN-002 now has deterministic DQN initialization, forward-pass, tensorization, loss, optimizer-update, finite-value, and parameter-change acceptance.
- BG-TRAIN-003 now explicitly depends on BG-TRAIN-002 for concrete model checkpoint save/load and defines run layout, JSONL parseability, recorder contract, and checkpoint error requirements.
- BG-TRAIN-002 through BG-TRAIN-006 now have explicit backward dependencies.
- The epics artifact now includes NFR coverage for determinism, headless training, trainer storage boundaries, artifact stability, CI smoke, docs positioning, no `rl-traits`/`ember-rl` product-core dependency, and small `Env` preservation.

### Epic Structure Validation

| Epic | User Value Focus | Independence | Finding |
| --- | --- | --- | --- |
| Epic 1: Trainable Crate Foundation And Artifact Contract | Acceptable enabling value | Stands alone as concrete crate-boundary foundation | Pass. It is still enabling infrastructure, but acceptance now requires visible, testable trainer API outcomes rather than scaffolding only. |
| Epic 2: CartPole DQN Acceptance Slice | Strong | Depends only on Epic 1 outputs | Pass. It delivers the first user-visible training result: train, improve, save `best.mpk`, and eval from checkpoint. |
| Epic 3: PPO Parallel-Environment Slice | Strong | Depends on accepted DQN/checkpoint/eval foundation | Pass. It is properly sequenced after DQN and does not create a forward dependency for Epic 2. |

### Story Quality Findings For BG-TRAIN-001 Through BG-TRAIN-006

| Story | Dependency Status | FR Coverage | Acceptance Quality | Readiness Finding |
| --- | --- | --- | --- | --- |
| BG-TRAIN-001 Required Burn Dependency And Trainer Skeleton | None | FR-1, FR-2, FR-3, FR-4, FR-5 | Concrete exported types, Burn defaults, `Env` preservation, docs/product-surface scan | Ready |
| BG-TRAIN-002 DQN Model And Optimizer | Depends on BG-TRAIN-001 | FR-6 | Deterministic init, Q-values, tensorization, one loss/update, finite outputs, parameter change | Ready |
| BG-TRAIN-003 Metrics, Run Directory, And Checkpoint Contract | Depends on BG-TRAIN-001 and BG-TRAIN-002 | FR-7 | Stable run layout, JSONL metrics/eval records, `best.mpk`, Burn named MessagePack recorder, load-from-disk proof | Ready |
| BG-TRAIN-004 DQN Replay And Training Loop | Depends on BG-TRAIN-001 and BG-TRAIN-002 | FR-2, FR-6, FR-9 | Bevy transition ingestion, replay outside ECS, terminated/truncated target semantics, deterministic exploration, target updates | Ready |
| BG-TRAIN-005 Deterministic Eval From best.mpk | Depends on BG-TRAIN-002, BG-TRAIN-003, BG-TRAIN-004 | FR-7, FR-8 | Eval loads from disk, repeated fixed-seed eval tolerance, missing-checkpoint error | Ready |
| BG-TRAIN-006 CartPole Train/Eval Example And Smoke Gate | Depends on BG-TRAIN-003, BG-TRAIN-004, BG-TRAIN-005 | FR-9 | Out-of-box command shape, reward improvement, `best.mpk`, metadata, CI smoke, docs positioning | Ready |

### Dependency Analysis

No forward dependency violations were found.

- BG-TRAIN-001 has no dependencies and anchors the public trainer boundary.
- BG-TRAIN-002 depends only on BG-TRAIN-001.
- BG-TRAIN-003 depends on BG-TRAIN-001 and BG-TRAIN-002, with the concrete model checkpoint dependency now explicit.
- BG-TRAIN-004 depends on BG-TRAIN-001 and BG-TRAIN-002.
- BG-TRAIN-005 depends on BG-TRAIN-002, BG-TRAIN-003, and BG-TRAIN-004.
- BG-TRAIN-006 depends on BG-TRAIN-003, BG-TRAIN-004, and BG-TRAIN-005.
- BG-TRAIN-007 waits for BG-TRAIN-006, preserving PPO as the second slice.
- BG-TRAIN-008 depends on BG-TRAIN-007.

### NFR Coverage Assessment

| NFR Area | Coverage Status | Evidence |
| --- | --- | --- |
| Deterministic train/eval seeds | Covered | NFR map ties seed metadata, replay sampling, repeated eval, and fixed CI smoke seeds to BG-TRAIN-001, 003, 004, 005, 006, and 007. |
| Headless training | Covered | BG-TRAIN-001, 004, and 006 require headless/MinimalPlugins behavior and reject rendering as correctness proof. |
| Trainer storage boundaries | Covered | BG-TRAIN-001, 003, 004, and 007 keep replay/rollout/checkpoint/tensor storage under `src/training/*`, outside ordinary ECS components. |
| Stable artifacts and `best.mpk` | Covered | BG-TRAIN-003, 005, 006, and 008 require stable run layout, checkpoint creation, and eval from disk. |
| CI smoke stability | Covered | BG-TRAIN-004, 005, and 006 require same-run baseline, positive reward improvement, checkpoint creation, and checkpoint eval. |
| Docs/product positioning | Covered | BG-TRAIN-001 and BG-TRAIN-006 require Burn training to be product core and runner/heuristic behavior to be baseline-only. |
| No `rl-traits`/`ember-rl` product-core dependency | Covered | BG-TRAIN-001 and BG-TRAIN-006 include scan/product-surface gates. |
| Small `Env` contract | Covered | BG-TRAIN-001, 002, 004, and 007 require trainer requirements to stay outside the `Env` trait. |

### Best Practices Compliance Checklist

| Item | Status | Notes |
| --- | --- | --- |
| Epic delivers user value | Pass | Epic 1 is enabling but now has concrete user-visible crate-boundary value; Epics 2 and 3 deliver training outcomes. |
| Epic can function independently | Pass | Each epic depends only on prior accepted output. |
| Stories appropriately sized | Pass with calibration risk | BG-TRAIN-004 and BG-TRAIN-006 are substantial but bounded by clear DQN acceptance. |
| No forward dependencies | Pass | No story depends on a later story. |
| Database/entity creation timing | N/A | No database scope. |
| Clear acceptance criteria | Pass | BG-TRAIN-001 through BG-TRAIN-006 now have concrete, testable ACs. |
| Traceability to FRs maintained | Pass | FR coverage map covers all 11 PRD FRs. |
| NFR traceability maintained | Pass | NFR coverage map is explicit. |

### Residual Risks

- CI reward threshold still requires calibration during implementation. The architecture permits changing the exact threshold before dev starts if `+10.0` proves flaky, while preserving the invariant that CI proves positive improvement, `best.mpk`, and eval-from-disk.
- Full release/deprecation policy for earlier runner-only redesign work remains open. This is a documentation/product cleanup task, not an implementation-readiness blocker for the Burn training core.
- BG-TRAIN-004 and BG-TRAIN-006 may need careful sprint sizing because each touches the behaviorally important DQN vertical slice.

## Architecture And Handoff Alignment

### Required Architecture Decisions Resolved

- Default Burn backend/autodiff: Burn `0.21`, `burn::backend::Flex` for inference, `burn::backend::Autodiff<burn::backend::Flex>` for training.
- Recorder: `NamedMpkFileRecorder<FullPrecisionSettings>` or equivalent Burn named MessagePack wrapper behind `training::checkpoint`.
- Run directory architecture: `runs/<env>-<algorithm>/<run-id>/` with `config.json`, `seeds.json`, `metrics.jsonl`, `eval.jsonl`, `summary.json`, `checkpoints/latest.mpk`, optional `checkpoints/step-*.mpk`, and root `best.mpk`.
- Train/eval surface: library API plus `cargo run --example cartpole_dqn --release -- train ...` and `... eval --checkpoint ...`.
- CI/manual split: deterministic CI smoke proves wiring, reward improvement, `best.mpk`, and eval-from-disk; longer manual/release preset proves useful CartPole policy.
- Module boundaries: Burn trainer code lives under `src/training/*`; core `Env` and runner modules remain small and algorithm-agnostic.

### Developer Handoff Status

The developer handoff is aligned with the PRD, architecture, and refined epics. First implementation priority is BG-TRAIN-001, followed by the DQN foundation and checkpoint/run scaffolding needed for the CartPole acceptance slice. PPO remains second and should not start until DQN, metrics, checkpoint, and deterministic eval are accepted.

## Summary And Recommendations

### Overall Readiness Status

READY FOR SPRINT PLANNING.

The planning set now has complete FR coverage, explicit NFR coverage, concrete BG-TRAIN-001 through BG-TRAIN-006 acceptance criteria, and backward-only dependencies. No critical or major blockers remain before sprint planning.

### Critical Issues Requiring Immediate Action

None.

### Major Issues Requiring Action Before Sprint Planning

None.

### Minor Issues And Calibration Items

1. Calibrate the deterministic CI smoke reward threshold during implementation. Keep the invariant: positive reward improvement over same-run baseline, `best.mpk` creation, and eval-from-disk.
2. Keep BG-TRAIN-004 and BG-TRAIN-006 sprint sizing conservative because they carry the behavioral DQN acceptance proof.
3. Decide release/deprecation wording for the earlier runner-only redesign before public release notes or docs finalization.

### Recommended Next Steps

1. Run `bmad-sprint-planning` using the refined epics artifact.
2. Plan the first sprint around BG-TRAIN-001 through BG-TRAIN-003 or a tightly scoped BG-TRAIN-001 plus BG-TRAIN-002 slice, depending on sprint capacity.
3. Carry CI threshold calibration as an implementation risk, not a planning blocker.
4. Keep PPO stories out of the first implementation sprint unless the DQN acceptance chain is already complete.

### Final Note

This revalidation found 0 blockers across PRD, architecture, epics/stories, UX alignment, FR coverage, and NFR coverage. The previous story-quality blockers have been closed by the refined epics artifact. Proceed to sprint planning.
