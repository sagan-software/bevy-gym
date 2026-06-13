---
title: Architecture And Migration
type: companion
spec: SPEC-bevy-gym-api-redesign
---

# Architecture And Migration

## Layer 1: Core RL Vocabulary

Core types are owned by `bevy-gym` and should avoid Bevy dependencies where practical:

- `Env`
- `Reset`
- `Step`
- `EpisodeStatus`
- `Transition`
- optional `Space`, `Discrete`, `BoxSpace`, and `HasSpaces`

This layer must not mention rendering, Burn, `ember-rl`, replay buffers, policy traits, Bevy schedules, or vector runner concerns.

## Layer 2: Bevy Runner

The runner owns Bevy integration:

- `GymPlugin<E>` or renamed `BevyGymPlugin<E>`
- `GymSet`
- `EnvId`
- `EnvComponent<E>`
- `CurrentObservation<E>`
- `ActionRequest<E>`
- `ActionResponse<E>`
- `TransitionEvent<E>` or `StepEvent<E>`
- `EpisodeEndEvent<E>` or an untyped event with typed extras
- reset and auto-reset policy

The runner remains algorithm-agnostic and rendering-agnostic.

Suggested app shape:

```rust
App::new()
    .add_plugins(MinimalPlugins)
    .add_plugins(GymPlugin::new(CartPole::new).with_envs(16).autoreset())
    .add_systems(FixedUpdate, cartpole_policy.after(GymSet::RequestActions))
    .run();
```

## Layer 3: Optional Adapters And Examples

Examples or separate adapters can show:

- random action sampling from `HasSpaces`
- DQN/PPO/Burn integration
- `ember-rl` integration if still useful
- rendering with plain Bevy systems
- Gymnasium-compatible classic-control environments

These examples must not make trainer crates or rendering traits part of the core API story.

## Migration Plan

1. Add a local core module with `Env`, `Reset`, `Step`, `EpisodeStatus`, and `Transition`.
2. Port current systems from `rl_traits::Environment` to `crate::Env`.
3. Replace `ExperienceEvent<O, A>` and `rl_traits::Experience` with crate-owned transition events.
4. Add `ActionRequest<E>` and `ActionResponse<E>` messages that carry the observation and action types directly.
5. Keep `PendingAction<E>` internally or as an advanced escape hatch, but remove it from the beginner path.
6. Remove `src/render.rs` and the core `GymRender`/`GymRenderPlugin` public API; move CartPole visualization into example-owned Bevy systems.
7. Rename `headless()` to the behavior it actually controls, such as `uncapped()`, or remove it and let Bevy app composition decide graphical versus headless execution.
8. Fix auto-reset request semantics so terminal/truncated steps produce one next-action request after reset and no stale post-step request for the same state.
9. Add a small env checker for optional spaces/specs: reset returns a valid observation, sampled action can step, status/reward/info are sane, and terminal/truncated handling emits exactly one next request.
10. Rewrite README, crate docs, plugin docs, and examples around "bring a Rust env, get a Bevy runner"; remove the `rl-traits` to `ember-rl` stack as the expected architecture.

## Current Findings To Preserve

- `Cargo.toml` depends on `rl-traits`; public components, events, plugin, render, and step/reset systems use `rl_traits::Environment`.
- `ember-rl` is a dev-dependency, but README and event docs make it feel central to the architecture.
- `GymRender` forces the environment type to know how to spawn and sync Bevy visuals.
- The CartPole example uses a newtype wrapper because `GymRender` and `CartPoleEnv` are both foreign to the example.
- README beginner policy code requires `ActionRequestEvent`, `CurrentObservation`, `PendingAction`, entity lookup, and mutation of `pending.action`.
- `step_system` emits `ActionRequestEvent` unconditionally after every step.
- `auto_reset_system` resets after `EpisodeEndEvent` and emits another `ActionRequestEvent`.
- Documented policy systems run after `GymSet::ManualReset`, so they can observe both stale post-step and post-reset requests in one tick.
- `headless()` sets uncapped tick rate and a config flag, while the Bevy app still decides runner/window composition.
- README documents `GymStatsPlugin` and `GymStats`, but current `src` does not define/export those names.

## Validation Targets

- Compile a minimal `Counter` environment using only crate-owned core types.
- Compile a Bevy example using typed `ActionRequest<E>` and `ActionResponse<E>` without direct `PendingAction` mutation.
- Verify many environments can be spawned and stepped independently from a factory.
- Verify transitions contain previous observation, action, reward, next observation, status, info, env id, and entity.
- Verify terminated and truncated statuses remain distinct and expose bootstrap-relevant semantics.
- Verify auto-reset produces exactly one next-action request per environment state that expects an action.
- Verify optional spaces/spec helpers can be omitted by a basic environment and used by discrete/box examples.
- Verify README, crate docs, and plugin docs only document APIs that exist.
