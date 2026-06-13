---
stepsCompleted: [1]
inputDocuments:
  - src/lib.rs
  - src/plugin.rs
  - src/components.rs
  - src/events.rs
  - src/render.rs
  - src/systems/step.rs
  - src/systems/reset.rs
  - README.md
  - docs/plugins/bevy_gym_plugin.md
  - examples/classic-control/cart_pole.rs
  - ref/rl-traits/src/environment.rs
  - ref/rl-traits/src/episode.rs
  - ref/rl-traits/src/experience.rs
workflowType: 'research'
lastStep: 1
research_type: 'technical'
research_topic: 'modern reinforcement learning custom environment APIs'
research_goals: 'Compare modern RL framework environment APIs, critique the bevy-gym src API, and recommend a simpler repo-owned API that removes rl-traits, ember-rl coupling, and rendering concerns.'
user_name: 'Sagan'
date: '2026-06-12'
web_research_enabled: true
source_verification: true
---

# Research Report: Modern RL Custom Environment APIs

**Date:** 2026-06-12
**Author:** Sagan
**Research Type:** technical

---

## Executive Summary

The ergonomic center of modern RL environment APIs is small: `reset`, `step`, an atomic step result,
explicit episode-end cause, and optional metadata/specs around the environment. Batching, wrappers,
validation, rendering, registration, and training algorithms are generally layered outside that
core.

The current `bevy-gym` API gets one important thing right: it treats each environment instance as
independent state that can be stepped in parallel by Bevy. But the public surface is more coupled
than it needs to be:

- It delegates the core environment vocabulary to `rl-traits` instead of owning it in this crate.
- It documents and imports `ember-rl` as part of the public story even though the runner can be
  algorithm-agnostic.
- It includes a `GymRender` trait and plugin that forces rendering into the environment type and
  even causes an example-only newtype workaround.
- It asks users to handle `ActionRequestEvent`, query `CurrentObservation`, and mutate
  `PendingAction`, which is flexible but verbose for beginners.
- It has a likely duplicate-action-request issue at episode end: `step_system` emits an action
  request unconditionally, and auto-reset emits another request for the same environment in the same
  tick.

Recommendation: replace `rl-traits` and `ember-rl` with repo-owned core types, remove rendering from
the crate API, keep Bevy ECS integration as the runner layer, and expose a higher-level typed
request/action message flow while leaving low-level components public for advanced users.

## Methodology

Sources inspected:

- Local crate source and docs in `src`, `README.md`, and `examples/classic-control/cart_pole.rs`.
- Local reference copies of `rl-traits`.
- Current public docs for Gymnasium, TorchRL, Jumanji, Brax, PettingZoo, RLlib, Stable-Baselines3,
  and Rust-native Gymnasium-inspired crates.

Confidence:

- High for the local API critique: it is based on current files in this checkout.
- High for common RL API patterns: they appear consistently across multiple maintained frameworks.
- Medium for Rust ecosystem direction: Rust RL crates are less mature and more fragmented than
  Python/JAX, but the same design themes show up.

## Modern API Commonalities

### 1. The core environment contract stays small

Gymnasium's primary interface is `reset()` plus `step(action)`, where `step` returns observation,
reward, terminated, truncated, and info. Gymnasium docs explicitly frame `step`, `reset`, `render`,
and `close` as main methods, but the learning contract is the reset/step pair.

TorchRL also centers the custom environment on `_reset`, `_step`, `_set_seed`, and specs. Its
heavier TensorDict layer exists to organize arbitrary nested/batched tensor data, not to expand the
conceptual environment contract.

Jumanji and Brax simplify even further for JAX: `reset(key)` returns state plus timestep/state, and
`step(state, action)` returns the next state plus timestep/state. Their key difference is explicit
state and RNG so the environment can be `jit`/`vmap` friendly.

Design implication for `bevy-gym`: own a small Rust `Env` trait locally. Avoid importing someone
else's policy, replay buffer, agent, or experience abstractions into the base environment boundary.

### 2. Step results are atomic

Modern frameworks return observation, reward, termination state, and info together. This matters
because splitting reward/done/observation across separate setters, components, or user obligations
makes invalid states easy.

Current `bevy-gym` internally keeps atomicity during stepping, but the public event flow exposes
pieces separately:

- `CurrentObservation<E>` stores latest observation and info (`src/components.rs:45-57`).
- `PendingAction<E>` stores the next action as an `Option` (`src/components.rs:27-42`).
- `ExperienceEvent` wraps an `rl_traits::Experience` transition (`src/events.rs:24-35`).
- `EpisodeEndEvent` separately carries final episode stats (`src/events.rs:37-60`).

This is workable, but the ergonomic API should present one typed transition/result for most users.

### 3. Termination and truncation are now first-class

Gymnasium changed from a single `done` flag to `terminated` and `truncated` in v0.26 because
bootstrapping algorithms need to distinguish natural terminal states from externally cut-off
episodes.

The local `rl-traits` dependency captures this with
`EpisodeStatus::{Continuing, Terminated, Truncated}` (`ref/rl-traits/src/episode.rs:21-35`) and
`Experience::bootstrap_mask()` (`ref/rl-traits/src/experience.rs:56-67`). This is worth preserving,
but it should be preserved in repo-owned types.

### 4. Specs/spaces are useful, but should not dominate the beginner path

Gymnasium and SB3 require action and observation spaces. TorchRL makes specs central and validates
shapes. Jumanji exposes action, observation, reward, and discount specs. Rust crates such as `gmgn`
expose spaces, env checkers, registries, vector envs, and wrappers.

The current `rl-traits` design has `sample_action` but no explicit action/observation space
(`ref/rl-traits/src/environment.rs:87-92`). That is a weak compromise: it forces random sampling
into every env but does not give enough metadata for validation, vector batching, UI, or algorithm
compatibility checks.

Recommendation: make specs optional:

- `Env` is the minimum reset/step contract.
- `HasSpaces` or `EnvSpec` is an optional trait for action/observation shape, sampling, validation,
  and examples.
- Beginner examples can implement `Discrete` or `BoxSpace` helpers without forcing all environments
  through a full dynamic space system.

### 5. Vectorization belongs around the environment

Gymnasium has `VectorEnv` and `make_vec`. RLlib scales with EnvRunner actors and vector envs.
Jumanji and Brax rely on `vmap`/`scan` wrappers. The common pattern is: a single env contract
remains simple; batching/running is a wrapper or runner concern.

Current `bevy-gym` already follows this direction by spawning `num_envs` environment entities from a
factory (`src/plugin.rs:176-206`). Keep that idea, but make the runner independent of external RL
crates.

### 6. Rendering is optional and framework-specific

Gymnasium includes `render`, but that is a Python API legacy and convenience. Jumanji/Brax keep
render optional around state. Rust `gymnasia` explicitly separates `core::Env` simulation from
`core::Renderable` and a render wrapper so simulation has no graphics dependencies.

For Bevy, rendering is already the application framework's native job. A `bevy-gym` rendering trait
adds little and constrains users.

## Current `bevy-gym` API Critique

### Coupling to `rl-traits`

`Cargo.toml` depends directly on `rl-traits` (`Cargo.toml:37-38`). The public plugin, components,
systems, and render module all use `rl_traits::Environment` (`src/plugin.rs:5`,
`src/components.rs:2`, `src/render.rs:40`, `src/systems/step.rs:4`).

That means the crate does not own its own central contract. If you do not trust `rl-traits` or want
different semantics, every public type is downstream of the wrong abstraction.

Suggested replacement:

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

This is basically the useful part of `rl-traits`, but local, auditable, and shaped for this crate.

### Coupling to `ember-rl`

`ember-rl` is a dev-dependency (`Cargo.toml:57-60`), but the README and docs make it feel like part
of the architecture. The README ecosystem table presents `rl-traits -> ember-rl -> bevy-gym` as the
expected stack (`README.md:13-19`), and `ExperienceEvent` docs explicitly describe pushing into
ember-rl (`src/events.rs:6-23`).

This is the wrong public story if the goal is an ergonomic Bevy RL runner. Training algorithms
should consume transitions, but the environment runner should not point users toward a specific
algorithm crate.

Suggested replacement:

- Rename `ExperienceEvent` to a crate-owned `TransitionEvent<E>` or `StepEvent<E>`.
- Store plain fields, not `rl_traits::Experience`.
- Move any ember integration into an example or separate adapter crate.

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

### Rendering should leave `src`

The render module defines `GymRender` and `GymRenderPlugin` as public API (`src/render.rs:58-127`).
The trait requires the environment type to know how to spawn and sync Bevy visuals
(`src/render.rs:67-90`).

This creates three problems:

1. It makes rendering feel like part of the RL environment contract.
2. It assumes visualization can be derived from observation plus a `Visuals` component.
3. It causes the CartPole example to wrap `CartPoleEnv` in a local `CartPoleViz` newtype because the
   render trait and env type are otherwise both foreign
   (`examples/classic-control/cart_pole.rs:165-188`).

In Bevy, users can already render however they want with normal systems:

```rust
fn sync_cartpole_visuals(
    query: Query<(&EnvId, &CurrentObservation<CartPole>, &CartPoleVisuals)>,
    mut transforms: Query<&mut Transform>,
) {
    // User-owned rendering logic.
}
```

Recommendation: remove `src/render.rs`, `render`, `winit`, `x11`, and `wayland` feature flags from
the library API. Keep live visualization only in examples. If a helper is later useful, make it a
tiny example-local utility or optional companion crate, not an environment trait.

### The action-request flow is verbose

The README's beginner policy example requires users to:

1. Read `ActionRequestEvent`.
2. Query `CurrentObservation<MyEnv>` and `PendingAction<MyEnv>`.
3. Look up the requested entity.
4. Mutate `pending.action` (`README.md:65-74`).

That is idiomatic Bevy but not beginner-simple. A better default API is message in, message out:

```rust
fn policy(
    mut requests: MessageReader<ActionRequest<CartPole>>,
    mut actions: MessageWriter<ActionResponse<CartPole>>,
) {
    for request in requests.read() {
        actions.write(request.respond(my_policy(&request.observation)));
    }
}
```

The low-level component path can remain public for users who need maximum control.

Suggested message shape:

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

For very large observations, provide an advanced zero-copy/query path. Optimize the default for
comprehension.

### Episode-end currently appears to emit duplicate action requests

`step_system` emits an `ActionRequestEvent` after every step, unconditionally
(`src/systems/step.rs:124-127`). When a step ends an episode, it also emits `EpisodeEndEvent`
(`src/systems/step.rs:114-121`). The auto-reset system reads that event, resets the env, and emits
another `ActionRequestEvent` for the same env (`src/systems/reset.rs:28-48`).

Because user policy systems are documented to run after `GymSet::ManualReset`
(`docs/plugins/bevy_gym_plugin.md:18-24`), they can observe both action requests in the same tick.
Since `ActionRequestEvent` only carries `env_id` and `entity`, both requests will query the same
latest post-reset observation. That wastes policy work and makes the request semantics harder to
reason about.

Recommendation: if auto-reset is enabled, do not emit the post-step request for terminal/truncated
steps. Emit exactly one request per environment state that expects an action.

### `headless()` is a confusing name

`BevyGymPlugin::headless()` sets `headless = true` and `tick_rate = None` (`src/plugin.rs:128-135`),
but it does not add `ScheduleRunnerPlugin`; the README example still adds
`ScheduleRunnerPlugin::run_loop(Duration::ZERO)` manually (`README.md:59-61`). The
`GymConfig.headless` field is inserted (`src/plugin.rs:145-149`), but the core build path does not
otherwise use it.

Recommendation: remove "headless" from the runner API or rename it to what it actually controls,
such as `uncapped()` or `without_fixed_timestep()`. Let Bevy app composition decide whether the app
is graphical or headless.

### README documents missing API

The README describes `GymStatsPlugin` and `GymStats` (`README.md:120-140`), but `src` does not
export or define them. That is not part of the desired refactor, but it is a signal that the public
surface is ahead of implementation and should be tightened during redesign.

## Proposed Simplified Architecture

Use three layers.

### Layer 1: Core RL vocabulary

Owned by this crate, no Bevy dependency if practical:

- `Env`
- `Reset`
- `Step`
- `EpisodeStatus`
- `Transition`
- Optional `Space`, `Discrete`, `BoxSpace`, and `HasSpaces`

This layer should not mention rendering, Burn, ember, replay buffers, or Bevy schedules.

### Layer 2: Bevy runner

Owns Bevy integration:

- `GymPlugin<E>`
- `GymSet`
- `EnvId`
- `EnvComponent<E>`
- `CurrentObservation<E>`
- `ActionRequest<E>`
- `ActionResponse<E>`
- `TransitionEvent<E>`
- `EpisodeEndEvent<E>` or `EpisodeEndEvent` with typed extras
- reset/autoreset policy

The runner should be algorithm-agnostic and rendering-agnostic.

Suggested app usage:

```rust
App::new()
    .add_plugins(MinimalPlugins)
    .add_plugins(GymPlugin::new(CartPole::new).with_envs(16).autoreset())
    .add_systems(FixedUpdate, cartpole_policy.after(GymSet::RequestActions))
    .run();
```

### Layer 3: Optional adapters and examples

Examples or companion modules can show:

- Random action from `HasSpaces`.
- DQN/PPO integration.
- Burn integration.
- Rendering with plain Bevy systems.
- Gymnasium-compatible classic control envs.

Keep these out of the core API.

## Suggested Migration Plan

1. Add local `core` module with `Env`, `Reset`, `Step`, `EpisodeStatus`, and `Transition`.
2. Port current systems from `rl_traits::Environment` to `crate::Env`.
3. Replace `ExperienceEvent<O, A>` with crate-owned transition events.
4. Replace `ActionRequestEvent` plus direct `PendingAction` mutation with optional
   `ActionResponse<E>` messages.
5. Keep `PendingAction<E>` internally or as an advanced escape hatch, but stop making it the
   beginner path.
6. Delete `src/render.rs` and move CartPole visualization into the example as ordinary Bevy systems.
7. Rename `headless()` to `uncapped()` or remove it.
8. Fix autoreset/request semantics so terminal steps emit one next-action request after reset, not
   two.
9. Add a tiny env checker for optional spaces/specs:
   - reset returns valid observation
   - sampled action can step
   - status/reward/info are sane
   - terminal/truncated handling emits exactly one next request
10. Rewrite README around "bring a Rust env, get a Bevy runner" instead of the
    `rl-traits`/`ember-rl` ecosystem.

## Recommended API Bias

Optimize the first five minutes:

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
            status: if self.n >= 10 { EpisodeStatus::Terminated } else { EpisodeStatus::Continuing },
            info: (),
        }
    }
}
```

Then optimize the first Bevy integration:

```rust
fn policy(
    mut requests: MessageReader<ActionRequest<Counter>>,
    mut actions: MessageWriter<ActionResponse<Counter>>,
) {
    for request in requests.read() {
        actions.write(request.respond(true));
    }
}
```

Everything beyond that should be optional composition.

## Source Notes

- Gymnasium custom env tutorial:
  <https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/>
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
- Stable-Baselines3 custom env guide:
  <https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html>
- Stable-Baselines3 env checker:
  <https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html>
- Rust gymnasia docs: <https://docs.rs/gymnasia/latest/gymnasia/>
- Rust gmgn docs: <https://docs.rs/gmgn/latest/gmgn/>
