---
title: Architecture And Migration
type: companion
spec: SPEC-bevy-gym-api-redesign
---

# Architecture And Migration

## Layer 1: Core RL Vocabulary

Core types are owned by `bevy-gym` and should avoid Bevy and Burn dependencies where practical:

- `Env`
- `Reset`
- `Step`
- `EpisodeStatus`
- `Transition`
- optional `Space`, `DiscreteSpace`, `BoxSpace`, and `HasSpaces`

This layer remains the small environment contract. It should not contain model, optimizer, replay, rollout, metrics, or checkpoint code. Burn is required by the crate, but Burn-specific code belongs in the trainer layer rather than inside `Env`.

## Layer 2: Bevy Runner

The runner owns Bevy integration:

- `BevyGymPlugin<E>`
- `GymSet`
- `EnvId`
- `EnvComponent<E>`
- `CurrentObservation<E>`
- `ActionRequest<E>`
- `ActionResponse<E>`
- `TransitionEvent<E>` or `StepEvent<E>`
- `EpisodeEndEvent<E>`
- reset and auto-reset policy

The runner remains rendering-agnostic. It is no longer product-complete by itself; it feeds the required trainer.

Suggested manual-control app shape:

```rust
App::new()
    .add_plugins(MinimalPlugins)
    .add_plugins(BevyGymPlugin::new(CartPole::new).with_envs(16).autoreset())
    .add_systems(FixedUpdate, cartpole_policy.after(GymSet::RequestActions))
    .run();
```

## Layer 3: Burn Training Engine

The crate must provide required Burn-backed training modules:

- model definitions for DQN first and PPO second;
- optimizer setup and learning-rate configuration;
- replay buffer for DQN;
- rollout storage for PPO;
- metrics and run metadata;
- checkpoint save/load and `best.mpk`;
- deterministic eval from checkpoint.

Examples demonstrate the trainer, but the trainer is not merely an example or adapter.

## Layer 4: Examples, Docs, And Optional Integrations

Examples and docs should show:

- CartPole DQN train and eval as the first acceptance slice;
- manual typed policy systems as a small runner demonstration, not the product endpoint;
- PPO rollout collection across multiple Bevy environment entities as the second slice;
- rendering with plain Bevy systems when useful for inspection;
- Gymnasium-compatible classic-control semantics where they help users compare results.

`ember-rl`, Python/SB3 bridges, robotics simulators, and rendering tools can remain references or future integrations. They must not displace the required Burn trainer or become the public contract.

## Migration Plan

1. Keep the local core module with `Env`, `Reset`, `Step`, `EpisodeStatus`, and `Transition`.
2. Keep first-party systems on `crate::Env`; do not reintroduce `rl_traits::Environment`.
3. Keep crate-owned transition events and typed action request/response messages.
4. Fix or preserve auto-reset request semantics so terminal/truncated steps produce one next-action request after reset and no stale post-step request for the same state.
5. Add required Burn dependency and trainer module layout without expanding `Env`.
6. Add trainer config, seed handling, run directory conventions, metrics writer, and checkpoint recorder/loader.
7. Add DQN model definitions, optimizer setup, replay storage, target-network updates, epsilon schedule, and batch optimization for discrete control.
8. Convert CartPole from heuristic-only demo to train/eval acceptance example; a random or heuristic baseline may remain only as a baseline.
9. Add deterministic eval that loads `best.mpk` and reports CartPole reward under fixed seeds.
10. Add a fast smoke gate proving reward improvement, `best.mpk` creation, and checkpoint eval; keep longer training as manual or release validation if CI budget is tight.
11. Add PPO rollout storage and actor-critic training after the DQN path is accepted, using multiple Bevy environment entities for rollout collection.
12. Rewrite README, crate docs, plugin docs, and CartPole docs around "bring a Rust env, get a Bevy runner and Burn trainer."

## Current Findings To Preserve

- The current source tree has the simplified core and runner files: `src/core.rs`, `src/components.rs`, `src/events.rs`, `src/plugin.rs`, `src/systems/reset.rs`, and `src/systems/step.rs`.
- `Cargo.toml` currently depends on Bevy only; Burn must be added as a required dependency during implementation.
- The current CartPole example is a deterministic heuristic runner demonstration, not a trainer.
- `GymRender` has been removed from the redesigned source; rendering should remain ordinary Bevy example code.
- The old review and implementation artifacts rejected Burn, replay buffers, optimizer code, and checkpointing; those constraints are superseded by the approved sprint change proposal.
- The runner's typed transition stream is the bridge between environment stepping and training storage.
- `headless` or `uncapped` behavior controls simulation throughput, while Bevy app composition still controls window/render setup.
- Historical research recommended a Burn-agnostic core; that recommendation is superseded for product scope, but the small environment contract portion remains valid.

## Validation Targets

- Compile a minimal `Counter` environment using only crate-owned core types.
- Compile a Bevy runner example using typed `ActionRequest<E>` and `ActionResponse<E>` without direct `PendingAction` mutation.
- Verify many environments can be spawned and stepped independently from a factory.
- Verify transitions contain previous observation, action, reward, next observation, status, info, env id, and entity.
- Verify terminated and truncated statuses remain distinct and expose bootstrap-relevant semantics.
- Verify auto-reset produces exactly one next-action request per environment state that expects an action.
- Verify optional spaces/spec helpers can be omitted by a basic environment and used by discrete/box training examples.
- Verify `cargo check` builds with required Burn support and without `rl-traits` or `ember-rl`.
- Verify CartPole DQN training improves deterministic eval reward over the initial or random policy baseline.
- Verify CartPole DQN writes metrics/config metadata, checkpoints, and `best.mpk`.
- Verify deterministic eval loads `best.mpk` and reports a successful CartPole evaluation.
- Verify PPO rollout storage uses multiple Bevy environment entities before PPO is treated as accepted.
- Verify README, crate docs, plugin docs, and CartPole docs only document APIs that exist.
