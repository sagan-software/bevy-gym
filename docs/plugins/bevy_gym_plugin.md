# BevyGymPlugin

The core plugin. It spawns N environment entities and steps environments that receive valid action
responses each `FixedUpdate` tick.

## Setup

```rust
App::new()
    .add_plugins(BevyGymPlugin::new(|i| MyEnv::new(i)).with_envs(16).uncapped())
    .add_systems(FixedUpdate, policy_system.in_set(GymSet::RequestActions))
    .run();
```

The factory closure receives the environment index (`0..num_envs`). Use it to seed environments
differently or give them distinct configs.

## Builder API

```rust
BevyGymPlugin::new(factory)   // factory: Fn(usize) -> E
    .with_envs(16)            // parallel environment instances
    .with_tick_rate(120.0)    // fixed tick rate in Hz, default 60
    .uncapped()               // do not configure Time<Fixed>
    .autoreset()              // default: reset ended episodes automatically
```

Use `.without_autoreset()` when terminal environments should idle until a system inserts
`ResetRequested`.

## System ordering (`GymSet`)

All runner systems run in `FixedUpdate` in this order:

```text
GymSet::Step -> GymSet::AutoReset -> GymSet::ManualReset -> GymSet::RequestActions
```

Policy systems should usually run in `GymSet::RequestActions` so they see observations after any
same-tick auto or manual reset.

## Message types

| Message              | When fired                         | Key fields                                      |
| -------------------- | ---------------------------------- | ----------------------------------------------- |
| `ActionRequest<E>`   | Startup, continuing steps, resets | `env_id`, `entity`, `observation`, `info`       |
| `ActionResponse<E>`  | User policy output                 | `entity`, `action`                              |
| `TransitionEvent<E>` | After every successful step        | `env_id`, `entity`, full `Transition`           |
| `EpisodeEndEvent`    | When an episode ends               | `env_id`, `status`, `total_reward`, `episode_steps`, `extras` |

Missing action responses leave an environment idle for that tick. Duplicate responses for the same
entity in one tick are ignored for that entity.

## ECS components

Each environment entity carries:

| Component               | Description                              |
| ----------------------- | ---------------------------------------- |
| `EnvId(usize)`          | Index `0..num_envs`                      |
| `EnvComponent<E>`       | The environment itself                   |
| `CurrentObservation<E>` | Latest observation and info              |
| `EnvStats`              | Internal per-env step/episode counters   |

## Manual resets

Insert `ResetRequested` on any environment entity to trigger a reset on the next fixed update:

```rust
commands.entity(env_entity).insert(ResetRequested { seed: Some(42) });
```

The reset system removes the marker after handling it and emits a fresh `ActionRequest<E>`.

## Design notes

- Parallel step uses `par_iter_mut()`. Simulation runs across all CPU cores; message dispatch is
  serialised after stepping.
- `E: Env + Send + Sync + 'static` is required so the environment can live as a Bevy component.
- The factory closure is wrapped in `Arc<dyn Fn(usize) -> E>` so it can be cloned into the Startup
  system.
- Environment randomness and policy randomness are separate concerns. The runner only passes reset
  seeds.
