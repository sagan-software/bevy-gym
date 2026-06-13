---
title: Framework Patterns
type: companion
spec: SPEC-bevy-gym-api-redesign
---

# Framework Patterns

## Common Pattern

Modern reinforcement-learning environment APIs keep the core small:

- reset
- step
- atomic step result
- explicit episode-end cause
- optional metadata/specs around the environment

Batching, wrappers, validation, rendering, registration, and training algorithms are layered outside the core environment contract.

## Framework Signals

| Framework | Load-bearing signal for this spec |
| --- | --- |
| Gymnasium | Core learning contract is reset plus step; v0.26 separates terminated and truncated because bootstrapping needs the distinction. |
| TorchRL | Custom environments center on reset, step, seed, and specs; heavier TensorDict machinery organizes nested/batched data rather than expanding the basic concept. |
| Jumanji | Reset and step return explicit state/timestep values; state and RNG are explicit for JAX `jit`/`vmap`. |
| Brax | Environment state is explicit and suited for JAX vectorization; batching is external to the single-env contract. |
| Gymnasium VectorEnv and RLlib | Vectorization and large-scale runners wrap or orchestrate environments instead of complicating the single-env interface. |
| PettingZoo | Multi-agent APIs are distinct surfaces and should not be silently mixed into the beginner single-agent contract. |
| Stable-Baselines3 | Spaces and env checkers are useful for compatibility and validation, but they should be a support layer rather than the whole beginner story. |
| Rust `gymnasia` | Simulation can be separated from renderability; render wrappers are optional. |
| Rust `gmgn` | Rust-native Gymnasium-style crates include spaces, checkers, registries, vector envs, and wrappers, but the Rust ecosystem remains less mature and more fragmented than Python/JAX. |

## Design Consequences

- Own a small local `Env` trait instead of importing someone else's policy, replay buffer, agent, or experience abstractions.
- Return observation, reward, status, and info together so downstream systems cannot see split or out-of-sync transition state.
- Preserve terminated/truncated semantics even if naming is simplified elsewhere.
- Add optional specs/spaces for validation, examples, and algorithm compatibility; do not require a dynamic space system for every beginner environment.
- Keep vectorization and batching in `GymPlugin`/runner code.
- Treat rendering as Bevy application code, not RL environment contract.

## Source URLs

- Gymnasium custom env tutorial: <https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/>
- Gymnasium Env API: <https://gymnasium.farama.org/api/env/>
- Gymnasium VectorEnv API: <https://gymnasium.farama.org/api/vector/>
- TorchRL environment API: <https://docs.pytorch.org/rl/stable/reference/envs_api.html>
- TorchRL custom Pendulum tutorial: <https://docs.pytorch.org/rl/main/tutorials/pendulum.html>
- Jumanji Env API: <https://instadeepai.github.io/jumanji/api/env/>
- Jumanji TimeStep types: <https://instadeepai.github.io/jumanji/api/types/>
- Jumanji advanced vectorization: <https://instadeepai.github.io/jumanji/guides/advanced_usage/>
- Brax Env base source: <https://github.com/google/brax/blob/main/brax/envs/base.py>
- PettingZoo Parallel API: <https://pettingzoo.farama.org/api/parallel/>
- PettingZoo AEC API: <https://pettingzoo.farama.org/api/aec/>
- RLlib environments: <https://docs.ray.io/en/latest/rllib/rllib-env.html>
- RLlib multi-agent environments: <https://docs.ray.io/en/latest/rllib/multi-agent-envs.html>
- Stable-Baselines3 custom env guide: <https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html>
- Stable-Baselines3 env checker: <https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html>
- Rust gymnasia docs: <https://docs.rs/gymnasia/latest/gymnasia/>
- Rust gmgn docs: <https://docs.rs/gmgn/latest/gmgn/>
