---
title: API Contract
type: companion
spec: SPEC-bevy-gym-api-redesign
---

# API Contract

## Core Types

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

The event name can be `TransitionEvent<E>` or `StepEvent<E>`, but it must not mention `ember-rl`, replay buffers, or `rl-traits` in the core contract.

## Action Request Flow

The beginner path is typed message in, typed message out:

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

Policy systems should look like this:

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

## Optional Specs And Spaces

Specs are optional, not part of the minimal `Env` trait:

- `HasSpaces` or `EnvSpec` can describe action shape, observation shape, sampling, and validation.
- `Discrete` and `BoxSpace` helpers should cover beginner examples.
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

Everything beyond this - spaces, render helpers, trainer adapters, and low-level ECS access - is optional composition.
