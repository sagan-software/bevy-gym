# bevy-gym

[![crates.io](https://img.shields.io/crates/v/bevy-gym.svg)](https://crates.io/crates/bevy-gym)
[![docs.rs](https://docs.rs/bevy-gym/badge.svg)](https://docs.rs/bevy-gym)
[![CI](https://github.com/vidarrio/bevy-gym/actions/workflows/ci.yml/badge.svg)](https://github.com/vidarrio/bevy-gym/actions/workflows/ci.yml)

Bevy ECS plugin for parallelised reinforcement-learning environment simulation and Burn-backed
training.

Bring a Rust environment, get a Bevy runner and a first-class Burn trainer boundary. `bevy-gym`
owns the small RL vocabulary it needs: `Env`, `Reset`, `Step`, `EpisodeStatus`, typed action
requests/responses, and typed transitions. Trainer internals live above that contract under
`bevy_gym::training` so environments do not absorb model, optimizer, replay, metrics, or
checkpoint responsibilities.

## Design goals

**Small environment contract.** Environments implement `reset` and `step`. Step results return
observation, reward, status, and info atomically.

**Free parallelism.** N environment instances live as separate Bevy entities. The runner steps
ready environments in parallel each `FixedUpdate` tick.

**Typed policy integration.** `ActionRequest<E>` carries observation and info to policy systems.
`ActionResponse<E>` carries the selected action back.

**Correct episode-end semantics.** `EpisodeStatus` distinguishes `Terminated` from `Truncated` so
value-learning code can bootstrap correctly.

**Runner-owned batching.** Batching is a Bevy runner concern. Single environments do not know they
are part of a batch.

**First-class training boundary.** Burn-backed trainer modules own backend aliases, run metadata,
seed streams, tensorization requirements, metrics, checkpoints, and algorithm entry points.

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
bevy-gym = "0.3"
bevy = { version = "0.18", default-features = false, features = ["multi_threaded"] }
```

### Minimal environment

```rust
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

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

### Runner integration

```rust
use bevy::prelude::*;
use bevy_gym::{ActionRequest, ActionResponse, BevyGymPlugin, GymSet};

App::new()
    .add_plugins(MinimalPlugins)
    .add_plugins(BevyGymPlugin::new(|_| Counter { n: 0 }).with_envs(16).uncapped())
    .add_systems(FixedUpdate, policy_system.in_set(GymSet::RequestActions))
    .run();

fn policy_system(
    mut requests: MessageReader<ActionRequest<Counter>>,
    mut responses: MessageWriter<ActionResponse<Counter>>,
) {
    for request in requests.read() {
        responses.write(ActionResponse {
            entity: request.entity,
            action: true,
        });
    }
}
```

### Manual resets

Add `ResetRequested` to any environment entity to trigger a reset on the next fixed update:

```rust
commands.entity(env_entity).insert(ResetRequested { seed: Some(42) });
```

## Feature flags

| Feature        | Description                                                       |
| -------------- | ----------------------------------------------------------------- |
| `fast-compile` | Enables Bevy dynamic linking for local iteration. Do not ship it. |
| `render`       | Default. Enables 2D rendering and Inspector-egui example HUDs.     |
| `bevy-mcp`     | Adds `bevy_brp_extras` for screenshots, shutdown, and MCP tooling. |
| `bevy_remote`  | Alias expected by Bevy BRP MCP launch tooling.                    |

## Training boundary

The training API is available under `bevy_gym::training`. The initial boundary exports the default
Burn `Flex` inference backend, `Autodiff<Flex>` training backend, run/config and seed types,
tensorization errors/specs, metrics writer types, checkpoint path/error types, and DQN/PPO module
roots. CartPole DQN train/eval is the first accepted behavioral path.

## Examples

| Example                                 | Notes                                                    |
| --------------------------------------- | -------------------------------------------------------- |
| [`cartpole`](docs/examples/cartpole.md) | CartPole Burn DQN trainer, eval, BRP/MCP visualizer, and screenshots |
| [`ecosystem`](examples/ecosystem/README.md) | Four visual-first recurrent PPO curriculum examples             |

## Plugin docs

- [BevyGymPlugin](docs/plugins/bevy_gym_plugin.md) -- core plugin, factory, runner messages, system
  ordering

## Development

This crate was developed with the assistance of AI coding tools.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
