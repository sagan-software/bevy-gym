---
id: SPEC-bevy-gym-api-redesign
companions:
  - api-contract.md
  - architecture-and-migration.md
  - framework-patterns.md
  - training-engine.md
  - ../../planning-artifacts/prd-burn-backed-bevy-gym-training-core-2026-06-13.md
  - ../../planning-artifacts/epics-burn-backed-bevy-gym-training-core-2026-06-13.md
sources:
  - ../../planning-artifacts/research/technical-modern-reinforcement-learning-frameworks-custom-environment-apis-research-2026-06-12.md
  - ../../planning-artifacts/sprint-change-proposal-2026-06-13.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability only - consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Bevy Gym Burn-Backed Training Core

## Why

`bevy-gym` should be a Bevy-native reinforcement-learning training crate powered by Burn. The simplified repo-owned `Env` / `Reset` / `Step` / `EpisodeStatus` API remains the right environment contract, but it feeds a required trainer layer instead of defining an environment-runner-only product. A user should be able to define a Rust environment, collect Bevy rollouts, train a useful Burn-backed policy, save `best.mpk`, and evaluate that checkpoint from this crate.

## Capabilities

- id: CAP-1
  intent: Users can define reinforcement-learning environments with repo-owned core types for reset, step, episode status, transition, observation, action, and info.
  success: A minimal environment compiles with `Env`, `Reset`, `Step`, and `EpisodeStatus`, and the same environment can feed the crate's required Burn-backed trainer without an adapter crate.

- id: CAP-2
  intent: Manual policy systems can answer typed action requests with typed action responses without querying and mutating `PendingAction` as the beginner path.
  success: The documented manual policy example consumes `ActionRequest<E>` and writes `ActionResponse<E>` using observation and info carried in the request.

- id: CAP-3
  intent: The Bevy runner can spawn and step many independent environment instances while keeping batching, scheduling, reset policy, and event emission outside the single-environment trait.
  success: A runner example creates at least 16 instances from a factory, steps them in `FixedUpdate`, emits typed transitions, and exposes transitions in the shape required by the trainer.

- id: CAP-4
  intent: Advanced users can opt into action/observation specs, sampling, validation, and low-level ECS access without making those concepts mandatory for the first environment.
  success: Optional `HasSpaces` or equivalent helpers support discrete and box-style examples plus an env checker; DQN and PPO integrations can require specs where training needs tensor shapes.

- id: CAP-5
  intent: Episode-end handling can distinguish continuing, terminated, and truncated steps and issue exactly one next-action request for each state that expects an action.
  success: Automated reset after a terminal or truncated step emits one post-reset action request, does not emit a stale post-step request for the same state, and preserves bootstrap-relevant status for trainer targets.

- id: CAP-6
  intent: Public docs and examples present `bevy-gym` as "bring a Rust environment, get a Bevy runner and Burn trainer" instead of a required `rl-traits` to `ember-rl` stack or an environment-runner-only crate.
  success: README, crate docs, plugin docs, CartPole docs, and examples match exported APIs, show Burn-backed training as first-class, remove missing `GymStatsPlugin` claims unless implemented, and show rendering as ordinary Bevy example code.

- id: CAP-7
  intent: Users can train useful policies out of the box with a required Burn-backed training engine.
  success: CartPole DQN trains from the crate's example or CLI, improves deterministic evaluation reward over the initial or random policy baseline, saves `best.mpk`, and can evaluate successfully from that checkpoint.

- id: CAP-8
  intent: Training artifacts are first-class and inspectable.
  success: Each run writes metrics, config and seed metadata, checkpoints, and a best-policy artifact in a stable run directory.

- id: CAP-9
  intent: PPO becomes the second training slice and uses Bevy parallel environments effectively.
  success: PPO rollout collection uses multiple Bevy environment entities, trains a Burn actor-critic policy, saves a checkpoint, and deterministic eval can load that checkpoint.

## Constraints

- Own the environment vocabulary in this crate; the minimal `Env` trait must stay small and must not contain model, optimizer, replay, rollout, metrics, or checkpoint responsibilities.
- Burn is a required first-class dependency for the crate's trainer layer, not an optional example-only integration.
- The public contract must not depend on `rl-traits` or make `ember-rl` the product core.
- Preserve typed `Observation`, `Action`, and `Info` associated types with `Clone + Send + Sync + 'static` bounds suitable for Bevy ECS.
- Step output must be atomic: observation, reward, episode status, and info are produced together.
- Preserve terminated versus truncated semantics because value-target and bootstrap behavior depend on the distinction.
- Keep batching/vectorization as a runner and trainer orchestration concern; do not expand the single-environment trait to know about batches.
- Store replay buffers, rollout buffers, large tensors, model state, metrics, and checkpoints in explicit trainer/run storage, not as ordinary ECS components.
- Deterministic train/eval paths must fix seeds, separate training and eval policy randomness, and make reward improvement testable without relying on rendering.
- Keep rendering out of training correctness; visualization belongs in normal Bevy systems, examples, or optional debugging surfaces.
- Do not make dynamic spaces/specs mandatory for a basic environment; training algorithms may require specs or tensorization adapters at the trainer boundary.
- The runner must not emit duplicate action requests around auto-reset.

## Non-goals

- Implement SAC, TD3, imitation learning, offline RL, multi-agent APIs, or broad algorithm coverage before the DQN and PPO slices are accepted.
- Provide a general graphics/rendering abstraction for environments.
- Clone Gymnasium, TorchRL, RLlib, Jumanji, Brax, PettingZoo, SB3, Burn RL crates, or any Rust crate wholesale.
- Treat `bevy-gym` as a robotics-grade dynamics simulator or make sim-to-real claims without external simulator validation.
- Guarantee backwards compatibility with the current public API until the release/deprecation policy is chosen.

## Success signal

A developer can run the CartPole DQN training path from a clean checkout with fixed seeds, see deterministic evaluation reward improve over the initial or random policy baseline, find metrics/config/checkpoints plus `best.mpk` in the run directory, and run deterministic eval that loads `best.mpk` successfully. PPO is planned next and explicitly uses multiple Bevy environment entities for rollout collection.

## Assumptions

- The existing simplified environment contract and typed Bevy runner are retained as the foundation rather than rolled back.
- The current spec slug remains `bevy-gym-api-redesign` because this is an approved course correction to the same redesign work, not an unrelated feature.
- `best.mpk` names the required best-policy artifact; exact Burn recorder/backend details are resolved during implementation against the current Burn API.

## Open Questions

- What exact CartPole DQN reward-improvement threshold and runtime budget should be used for CI versus longer manual/release validation?
- Which Burn backend, autodiff backend, and recorder configuration should be the default for CPU/headless training?
- Should this breaking change ship as the next minor release, a major release, or behind a short-lived migration feature?
