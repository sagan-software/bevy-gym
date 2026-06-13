---
id: SPEC-bevy-gym-api-redesign
companions:
  - api-contract.md
  - architecture-and-migration.md
  - framework-patterns.md
sources:
  - ../../planning-artifacts/research/technical-modern-reinforcement-learning-frameworks-custom-environment-apis-research-2026-06-12.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability only - consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Bevy Gym API Redesign

## Why

`bevy-gym` should keep its strong Bevy ECS runner idea - independent environment entities stepped in parallel - while replacing the current public surface that couples users to `rl-traits`, `ember-rl`, rendering traits, verbose action mutation, and ambiguous reset/request semantics. The opportunity is a smaller repo-owned RL environment API that is beginner-simple, algorithm-agnostic, rendering-agnostic, and still usable for batched training in Bevy.

## Capabilities

- id: CAP-1
  intent: Users can define reinforcement-learning environments with repo-owned core types for reset, step, episode status, transition, observation, action, and info.
  success: A minimal environment compiles and runs without importing `rl-traits`, `ember-rl`, Burn, Bevy rendering, replay buffers, or policy abstractions.

- id: CAP-2
  intent: Policy systems can answer typed action requests with typed action responses without querying and mutating `PendingAction` as the beginner path.
  success: The documented policy example consumes `ActionRequest<E>` and writes `ActionResponse<E>` using observation and info carried in the request.

- id: CAP-3
  intent: The Bevy runner can spawn and step many independent environment instances while keeping batching, scheduling, reset policy, and event emission outside the core environment contract.
  success: A runner example creates at least 16 instances from a factory, steps them in `FixedUpdate`, and emits typed transitions while the `Env` trait remains free of Bevy scheduling, rendering, and algorithm concepts.

- id: CAP-4
  intent: Advanced users can opt into action/observation specs, sampling, validation, and low-level ECS access without making those concepts mandatory for the first environment.
  success: Optional `HasSpaces` or equivalent helpers support discrete and box-style examples plus an env checker, while a basic environment only implements reset and step.

- id: CAP-5
  intent: Episode-end handling can distinguish continuing, terminated, and truncated steps and issue exactly one next-action request for each state that expects an action.
  success: Automated reset after a terminal or truncated step emits one post-reset action request and does not also emit a stale post-step request for the same environment state.

- id: CAP-6
  intent: Public docs and examples present `bevy-gym` as "bring a Rust environment, get a Bevy runner" instead of a required `rl-traits` to `ember-rl` stack.
  success: README, crate docs, plugin docs, and examples match exported APIs, remove missing `GymStatsPlugin` claims unless implemented, and show rendering as ordinary Bevy example code.

## Constraints

- Own the core environment vocabulary in this crate; base public API must not depend on `rl-traits`, `ember-rl`, Burn, replay buffers, policy traits, or experience abstractions.
- Preserve typed `Observation`, `Action`, and `Info` associated types with `Clone + Send + Sync + 'static` bounds suitable for Bevy ECS.
- Step output must be atomic: observation, reward, episode status, and info are produced together.
- Preserve terminated versus truncated semantics because bootstrapping algorithms need different value-target behavior.
- Keep batching/vectorization as a runner concern; do not expand the single-environment trait to know about batches.
- Keep rendering out of the core API; visualization belongs in normal Bevy systems, examples, or a future optional adapter outside the core contract.
- Optimize the default API for a user's first five minutes; low-level ECS components may remain available as an advanced escape hatch, not as the primary docs path.
- Do not make dynamic spaces/specs mandatory; specs, sampling, and validation are optional capability layers.
- The redesigned runner must not emit duplicate action requests around auto-reset.

## Non-goals

- Implement DQN, PPO, SAC, replay buffers, policy optimization, checkpointing, or other trainer logic in the core crate.
- Provide a general graphics/rendering abstraction for environments.
- Clone Gymnasium, TorchRL, RLlib, Jumanji, Brax, PettingZoo, SB3, or any Rust crate wholesale.
- Solve multi-agent environment APIs in this pass unless they fall out of the same core abstractions without expanding the beginner surface.
- Guarantee backwards compatibility with the current public API until the release/deprecation policy is chosen.

## Success signal

A new user can implement a tiny `Env`, add `GymPlugin::new(factory).with_envs(16).autoreset()` to a Bevy app, respond to typed action requests, and observe typed transitions without touching `rl-traits`, `ember-rl`, rendering traits, or `PendingAction`. Tests or examples demonstrate correct terminated/truncated behavior, optional space validation, and exactly one action request per actionable environment state.

## Assumptions

- The requested spec target is the `bevy-gym` API redesign implied by the research goals, not a generic survey of RL frameworks.
- Removing current couplings may require a breaking release or temporary compatibility shim; the source chose the direction but not the release policy.

## Open Questions

- Should removal of `rl-traits`, `GymRender`, render features, and current event names happen as one breaking release, or through deprecations and compatibility shims first?
- Should `ember-rl` and Burn integration live only in examples for this pass, or should a separate adapter crate/module be part of the first redesign milestone?
