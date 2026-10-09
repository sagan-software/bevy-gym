# bevy-gym

[![crates.io](https://img.shields.io/crates/v/bevy-gym.svg)](https://crates.io/crates/bevy-gym)
[![docs.rs](https://docs.rs/bevy-gym/badge.svg)](https://docs.rs/bevy-gym)
[![CI](https://github.com/sagan-software/bevy-gym/actions/workflows/ci.yml/badge.svg)](https://github.com/sagan-software/bevy-gym/actions/workflows/ci.yml)

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

## Drone flight

Run the first robot lesson from this checkout:

```sh
nix develop --command cargo run --no-default-features --features robots --example drone-hover
```

The [hover guide](examples/robots/hover.rs) applies four validated motor commands
in a private Rapier world and prints the resulting position. It runs without a
window. Open the visual lesson with:

```sh
nix develop --command cargo run --features robots --example drone-flight
```

Select Disturbed start, Bundled policy, and Run to watch recovery.
Select Fly east and Run to watch
the [waypoint pilot](docs/DRONE_TRACKING.md) fly eight metres. Manual controls
provide hover, climb, power-off, and tilt for comparison. The
[recovery guide](examples/robots/recovery.rs) adds initial tilt and
velocity through `DroneHover::disturbed()`. The
[browser build guide](robot-web/README.md) uses the same Bevy example in WebAssembly.

The [learning guide](docs/DRONE_LEARNING.md) trains a recovery policy and records
its native and browser qualification. The bundled policy survived all 32 native
held-out episodes. The [inference guide](docs/DRONE_INFERENCE.md) includes a browser
recording. The [browser training guide](docs/DRONE_BROWSER_TRAINING.md) covers
training, checkpoint export, and playback. See the
[physics contract](docs/ROBOT_ENVIRONMENT.md) and [execution status](docs/EXAMPLE_STATUS.md).

The [waypoint guide](docs/DRONE_TRACKING.md) uses a bundled imitation-trained pilot
to fly eight metres east and face east. Its heading and waypoint tests pass
natively and in Chrome/WASM. The playable arena still uses the hover controller.

The [actuator-failure guide](docs/DRONE_DAMAGE.md) disables a named motor during
flight. It demonstrates the resulting crash under constant commands. Learned
damaged-flight recovery and detached parts remain unfinished.

## Train in your browser

[Open the Gymnasium demo](https://sagan-software.github.io/bevy-gym/).
CartPole-v1 and MountainCar-v0 support fresh DQN training, frozen-policy inference, episode-return
and learning-rate charts, pause, single-step, and 1× through 16× speed controls.
MountainCarContinuous-v0 uses PPO with separate actor and critic learning-rate plots.
Download a trained policy and load it for inference. Training runs in a Rust
WebAssembly worker on your device.

The bundled CartPole policy scored 500 across 200 held-out episodes. A separate
browser training run reached the same score after 70,000 transitions.
See [setup and qualification evidence](gymnasium-web/README.md) and
[the remaining environment plan](GYMNASIUM_BROWSER_PLAN.md).

MountainCar's bundled browser-trained policy scored −102.41 with 200/200 goals.
Continuous MountainCar's selected policy scored 98.44 with 200/200 goals.
Other Gymnasium environments are not yet implemented in the browser.
Gymnasium builds do not enable Avian by default.

## WASM ecosystem demo

The static demo runs inference for all six ecosystem stages. It loads the
curated checkpoint manifest by default and accepts local `.mpk` uploads without
sending checkpoint data to a server.

Run the live-reloading loopback server:

```sh
nix run .#web-serve
```

Run the server on this machine's Tailscale IPv4 address:

```sh
nix run .#web-serve-tailnet
```

For Tailscale HTTPS, configure the proxy once and run the loopback server with
secure WebSocket reloads. Port 8443 avoids this host's existing Traefik listener
on port 443:

```sh
sudo tailscale serve --bg --https=8443 8080
nix run .#web-serve-tailscale-https
```

Build the reproducible static artifact or run the browser suite:

```sh
nix build .#web-dist
nix run .#web-check
```

Use `export-web-checkpoint` to validate a native checkpoint, copy its
content-addressed policy files, and update the manifest:

```sh
cargo run -p bevy-gym-web --features native-export \
  --bin export-web-checkpoint -- \
  --stage survival \
  --run-dir runs/example \
  --qualification best-compatible-available
```

## Feature flags

| Feature               | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `fast-compile`        | Enables Bevy dynamic linking for local iteration. Do not ship it.     |
| `render`              | Enables 2D rendering and Inspector-egui example HUDs by default.      |
| `ecosystem-inference` | Opts into Avian ecosystem examples and the portable inference facade. |
| `bevy-mcp`            | Adds `bevy_brp_extras` for screenshots, shutdown, and MCP tooling.    |
| `bevy_remote`         | Alias expected by Bevy BRP MCP launch tooling.                        |

## Training boundary

The training API is available under `bevy_gym::training`. The initial boundary exports the default
Burn `Flex` inference backend, `Autodiff<Flex>` training backend, run/config and seed types,
tensorization errors/specs, metrics writer types, checkpoint path/error types, and DQN/PPO module
roots. CartPole DQN train/eval is the first accepted behavioral path.

## Examples

| Example                                                           | Notes                                                              |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| [`classic-control`](examples/classic-control/README.md)           | Five Gymnasium-inspired control examples with recorded checkpoints |
| [`toy-text`](examples/toy-text/README.md)                         | Four finite-state tabular examples                                 |
| [`box2d-approximation`](examples/box2d-approximation/README.md)   | Three experimental approximations awaiting faithful Box2D ports    |
| [`mujoco-approximation`](examples/mujoco-approximation/README.md) | Eleven experimental approximations awaiting faithful MuJoCo ports  |
| [`mujoco`](examples/mujoco/README.md)                             | Three examples backed by official Gymnasium XML and MuJoCo 3.9     |
| [`ecosystem`](examples/ecosystem/README.md)                       | Four visual-first recurrent PPO curriculum examples                |

## Plugin docs

- [BevyGymPlugin](docs/plugins/bevy_gym_plugin.md) -- core plugin, factory, runner messages, system
  ordering
- [BevyGymRecorderPlugin](docs/plugins/bevy_gym_recorder_plugin.md) -- GIF and dynamic-checkpoint
  MP4 recording

## Development

This crate was developed with the assistance of AI coding tools.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
