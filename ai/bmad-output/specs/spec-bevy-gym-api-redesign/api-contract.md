---
title: API Contract
type: companion
spec: SPEC-bevy-gym-api-redesign
---

# API Contract

## Core Environment Types

The crate owns the environment contract:

```rust
pub trait Env {
    type Observation: Clone + Send + Sync + 'static;
    type Action: Clone + Send + Sync + 'static;
    type Info: Default + Clone + Send + Sync + 'static;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info>;
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeStatus {
    Continuing,
    Terminated,
    Truncated,
}

#[derive(Debug, Clone)]
pub struct Reset<O, I = ()> {
    pub observation: O,
    pub info: I,
}

#[derive(Debug, Clone)]
pub struct Step<O, I = ()> {
    pub observation: O,
    pub reward: f64,
    pub status: EpisodeStatus,
    pub info: I,
}
```

`EpisodeStatus::Terminated` means natural MDP termination and zero bootstrap value. `EpisodeStatus::Truncated` means external cutoff and non-zero next-state bootstrap value. `EpisodeStatus::Continuing` means the episode remains actionable.

## Transition Events

Replace `ExperienceEvent<O, A>` carrying `rl_traits::Experience` with a crate-owned transition event:

```rust
pub struct Transition<O, A, I = ()> {
    pub env_id: usize,
    pub entity: Entity,
    pub observation: O,
    pub action: A,
    pub reward: f64,
    pub next_observation: O,
    pub status: EpisodeStatus,
    pub info: I,
}
```

The event name can be `TransitionEvent<E>` or `StepEvent<E>`, but it must not mention `ember-rl` or `rl-traits` in the core contract. Replay and rollout storage consume transitions at the trainer layer.

## Action Request Flow

The beginner manual-control path is typed message in, typed message out:

```rust
pub struct ActionRequest<E: Env> {
    pub env_id: usize,
    pub entity: Entity,
    pub observation: E::Observation,
    pub info: E::Info,
}

pub struct ActionResponse<E: Env> {
    pub entity: Entity,
    pub action: E::Action,
}
```

Manual policy systems should look like this:

```rust
fn policy(
    mut requests: MessageReader<ActionRequest<Counter>>,
    mut actions: MessageWriter<ActionResponse<Counter>>,
) {
    for request in requests.read() {
        actions.write(ActionResponse {
            entity: request.entity,
            action: true,
        });
    }
}
```

The low-level component path may remain public for advanced users who need zero-copy observations, custom batching, or direct ECS coordination. It must not be the primary README or quick-start path.

## Trainer Boundary

The required Burn trainer consumes the same environment contract and runner transitions. Burn types, model modules, optimizer state, replay buffers, rollout buffers, metrics, and checkpoint recorders belong in trainer modules, not in `Env`.

The trainer boundary may add algorithm-specific requirements such as:

- a discrete action spec for DQN;
- tensorization for observations and actions;
- a box or discrete action spec for PPO;
- fixed seed/config metadata for reproducible train/eval runs.

These requirements are trainer concerns. A basic environment that only implements `Env` remains valid even if it cannot be trained until it supplies the specs or tensorization required by a chosen algorithm.

## Optional Specs And Spaces

Specs are optional for the minimal `Env` trait and required only when a trainer or checker needs them:

- `HasSpaces` or `EnvSpec` can describe action shape, observation shape, sampling, and validation.
- `DiscreteSpace` and `BoxSpace` helpers should cover beginner examples.
- An env checker can validate reset observation, sampled-action stepping, reward/status/info sanity, and terminal/truncated handling.
- Environments that do not need spaces must still be valid first-class environments.

## Beginner Example Bias

The first environment should read like this:

```rust
struct Counter {
    n: u32,
}

impl Env for Counter {
    type Observation = u32;
    type Action = bool;
    type Info = ();

    fn reset(&mut self, _seed: Option<u64>) -> Reset<u32> {
        self.n = 0;
        Reset { observation: self.n, info: () }
    }

    fn step(&mut self, action: bool) -> Step<u32> {
        if action {
            self.n += 1;
        }

        Step {
            observation: self.n,
            reward: f64::from(self.n),
            status: if self.n >= 10 {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}
```

Everything beyond this - spaces, render helpers, Burn trainer modules, and low-level ECS access - is layered composition. Burn training is required product surface in the crate, but the single-environment contract stays this small.
