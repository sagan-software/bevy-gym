---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - docs/plugins/bevy_gym_plugin.md
  - docs/plugins/gym_render.md
  - docs/examples/cartpole.md
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Bevy physics, Avian, Rapier, best practices, common mistakes/issues, idiomatic patterns'
research_goals: 'Assess the current Bevy physics ecosystem for bevy-gym, compare Avian and Rapier, identify idiomatic ECS integration patterns, common mistakes, risks, and practical spike criteria.'
user_name: 'Sagan'
date: '2026-06-12'
web_research_enabled: true
source_verification: true
---

# Bevy Physics, Avian, and Rapier: Technical Research Report

**Date:** 2026-06-12
**Author:** Sagan
**Research Type:** technical

---

## Research Overview

This report researches the current Bevy physics ecosystem for `bevy-gym` as of 2026-06-12, with
emphasis on Avian, Rapier, idiomatic Bevy 0.18 integration, common physics mistakes, backend
adoption strategy, and implementation patterns that preserve headless RL throughput. It uses local
project inspection plus current source verification from Bevy documentation, docs.rs, crates.io
metadata, Avian documentation/repository material, Rapier Bevy plugin documentation, and Cargo
documentation.

The primary conclusion is architectural: keep the `bevy-gym` core crate physics-agnostic, run an
Avian-first implementation spike in a real example, keep Rapier as a controlled fallback/comparison
path, and do not extract a shared physics abstraction until both backends have been implemented
against the same task. The recommended first task is `examples/avian-3d/inverted_pendulum.rs` for a
minimal 3D physics-control loop, or `examples/avian-2d/lunar_lander.rs` if contact-heavy behavior is
the higher priority.

The full final synthesis appears in the "Research Synthesis and Final Recommendations" section. The
most important implementation requirement is measurable proof: fixed-step headless smoke tests,
finite-state checks, deterministic resets, explicit collision/contact assertions, feature-gated
backend compilation, and release/profile-optimized throughput measurements.

## Table of Contents

1. Research Overview
2. Technical Research Scope Confirmation
3. Technology Stack Analysis
4. Integration Patterns Analysis
5. Architectural Patterns and Design
6. Implementation Approaches and Technology Adoption
7. Technical Research Recommendations
8. Research Synthesis and Final Recommendations
9. Source Verification and Reference Appendix

---

<!-- Content will be appended sequentially through research workflow steps -->

## Technical Research Scope Confirmation

**Research Topic:** Bevy physics, Avian, Rapier, best practices, common mistakes/issues, idiomatic
patterns

**Research Goals:** Assess the current Bevy physics ecosystem for `bevy-gym`, compare Avian and
Rapier, identify idiomatic ECS integration patterns, common mistakes, risks, and practical spike
criteria.

**Technical Research Scope:**

- Architecture Analysis - design patterns, frameworks, system architecture
- Implementation Approaches - development methodologies, coding patterns
- Technology Stack - languages, frameworks, tools, platforms
- Integration Patterns - APIs, protocols, interoperability
- Performance Considerations - scalability, optimization, patterns

**Research Methodology:**

- Current web data with rigorous source verification
- Multi-source validation for critical technical claims
- Confidence level framework for uncertain information
- Comprehensive technical coverage with architecture-specific insights

**Scope Confirmed:** 2026-06-12

## Technology Stack Analysis

### Web Search Analysis

This step verified the current stack against live primary or near-primary sources on 2026-06-12:

- Bevy docs.rs and official Bevy examples for the engine version and fixed-timestep physics
  scheduling.
- Avian docs.rs and Avian 0.6 release notes for ECS-native physics features, scheduling,
  interpolation, diagnostics, character-controller status, and maturity caveats.
- Rapier official Bevy plugin docs and docs.rs for plugin structure, character-controller support,
  joints, and Bevy compatibility.
- crates.io API metadata, fetched with an explicit `User-Agent`, for current published crate
  versions and Bevy dependency requirements.
- Local `bevy-gym` docs and `Cargo.toml` for repo-specific integration constraints.

**Confidence:** High for current versions and Bevy dependency compatibility; medium-high for
practical recommendations because physics feel, solver behavior, and RL stability still require
local spikes.

### Programming Languages

The relevant implementation language is **Rust**. Bevy, Avian, Rapier, Tnua, and `bevy-gym` are Rust
crates; the project already uses Rust 2021 edition and keeps Bevy dependency features minimal for
headless simulation. Bevy itself describes the engine as an open-source modular game engine built in
Rust, with a focus on developer productivity and performance.

_Popular Languages:_ Rust is the only practical language for first-class Bevy physics integration.

_Emerging Languages:_ None should be introduced for the physics layer. Python may remain useful
around RL training orchestration, analysis, or external notebooks, but it should not sit in the
fixed-step physics loop.

_Language Evolution:_ The current Bevy ecosystem is moving quickly. On 2026-06-12, crates.io API
reports `bevy` newest published version as `0.19.0-rc.3`, while docs.rs `latest` still documents
`bevy 0.18.1`. For this repo, 0.18.1 remains the stable baseline and 0.19 remains migration-spike
material until final release and plugin compatibility are verified.

_Performance Characteristics:_ Rust is suitable for deterministic, high-throughput ECS simulation.
Keep allocation-heavy RL observation building, debug rendering, inspectors, and optional controller
tooling outside the hot physics step.

_Sources:_

- Bevy docs.rs: <https://docs.rs/bevy/latest/bevy/>
- Bevy fixed timestep example: <https://bevy.org/examples/movement/physics-in-fixed-timestep/>
- crates.io API: <https://crates.io/api/v1/crates/bevy>
- Local: `Cargo.toml`, `Cargo.lock`, `docs/plugins/bevy_gym_plugin.md`

### Development Frameworks and Libraries

The core framework is **Bevy ECS** plus a physics plugin. The important technology decision is not
"physics in Bevy" generically; it is which physics backend owns the simulation state and how much
Bevy-native ECS surface the project needs.

#### Bevy

Bevy provides the ECS, schedules, app/plugin model, time resources, fixed timestep schedules,
optional rendering, and feature gating surface. Official Bevy examples emphasize fixed-timestep
simulation, accumulated input before the fixed loop, and visual interpolation after the fixed loop.

**Repo fit:** `bevy-gym` already matches the right shape: `default = []`, minimal Bevy features,
optional `render`, explicit display backends, and `FixedUpdate` RL stepping.

#### Avian 3D

Avian 0.6.1 is the strongest ECS-native physics candidate. docs.rs identifies `avian3d 0.6.1` and
shows a normal dependency on `bevy ^0.18.0`. Its documentation describes Avian as an ECS-driven
2D/3D physics engine for Bevy and documents rigid body dynamics, collision detection,
constraints/joints, spatial queries, scheduling, debugging/profiling, and architecture. It is
designed around `PhysicsPlugins`, modular plugin configuration, physics diagnostics, fixed-timestep
scheduling, and built-in interpolation support.

**Best use in `bevy-gym`:**

- ECS-native RL environments.
- Component-oriented sensors, contacts, rewards, and reset logic.
- Spatial queries for ray sensors and shaped sensors.
- Procedural curricula and headless simulation.
- Physics examples or companion crates, not core `bevy-gym`.

**Important status:** Avian's own docs say Rapier is currently the more mature and feature-rich Rust
physics option, while Avian has the more native Bevy/ECS feel and less separate-world
synchronization overhead. Avian also explicitly notes that it is still young and may have missing
features and fewer third-party resources.

#### Rapier / bevy_rapier3d

bevy_rapier3d 0.34.0 is the mature fallback. docs.rs identifies `bevy_rapier3d 0.34.0`, with a
normal dependency on `bevy ^0.18.1` and `rapier3d ^0.32.0`. The crate is the official Rapier
integration for Bevy and exposes modules for character control, dynamics, geometry, pipeline, plugin
systems, debug rendering, reflection, and utility helpers.

**Best use in `bevy-gym`:**

- Mature kinematic character controller experiments.
- Joint-heavy prototypes and ragdoll-style fallback spikes.
- Cases where existing Rapier documentation and historical use matter more than ECS-native
  implementation internals.
- Cross-checking Avian behavior when contact stability or controller behavior is suspect.

#### Tnua

Tnua 0.31.0 is a Bevy character-controller crate, not a general physics engine. docs.rs shows a
`bevy ^0.18` dependency and documents support for Rapier and Avian integration crates. It is useful
if the project needs dynamic-style floating character control rather than writing a custom
controller from raw physics primitives.

_Major Frameworks:_ Bevy ECS, Avian, Rapier, Tnua.

_Micro-frameworks:_ Physics backend-specific helpers: Avian `MoveAndSlide`, Rapier
`KinematicCharacterController`, Rapier `move_shape`, Avian/Rapier debug rendering, Avian
diagnostics.

_Evolution Trends:_ Avian 0.6 added move-and-slide utilities, joint motors for revolute and
prismatic joints, BVH broad phase improvements, and spatial query optimizations. Rapier remains the
mature Bevy plugin with explicit KCC and joint docs.

_Ecosystem Maturity:_ Rapier is more mature; Avian is more idiomatic for Bevy ECS. Treat both as
viable, with Avian first for this repo's ECS/RL design and Rapier as the control/fallback
implementation.

_Sources:_

- Avian docs.rs: <https://docs.rs/avian3d/latest/avian3d/>
- Avian 0.6 release notes: <https://joonaa.dev/blog/12/avian-0-6>
- Rapier Bevy plugin docs: <https://rapier.rs/docs/user_guides/bevy_plugin/getting_started_bevy/>
- Rapier character controller docs:
  <https://rapier.rs/docs/user_guides/bevy_plugin/character_controller/>
- Rapier joints docs: <https://rapier.rs/docs/user_guides/bevy_plugin/joints/>
- bevy_rapier3d docs.rs: <https://docs.rs/bevy_rapier3d/latest/bevy_rapier3d/>
- Tnua docs.rs: <https://docs.rs/bevy-tnua/latest/bevy_tnua/>
- crates.io API: <https://crates.io/api/v1/crates/avian3d>,
  <https://crates.io/api/v1/crates/bevy_rapier3d>, <https://crates.io/api/v1/crates/bevy-tnua>

### Database and Storage Technologies

Physics itself does not require a database. For `bevy-gym`, storage should stay outside the physics
backend and serve RL reproducibility, telemetry, and regression analysis.

_Relational Databases:_ Not needed in the simulation loop. SQLite can be considered later for
experiment catalogs, but it should not be coupled to Bevy systems or physics plugin APIs.

_NoSQL Databases:_ Not needed for the physics integration. Avoid introducing document stores for
local training runs unless there is a separate experiment-management requirement.

_In-Memory Databases:_ Not needed. In-memory ECS components and resources already represent
simulation state.

_Data Warehousing:_ Not needed at this stage. Use append-only local run artifacts first: JSONL, CSV,
MessagePack, or Parquet if large-scale offline analysis becomes a real requirement.

**Repo recommendation:** keep physics state in ECS components/resources, keep RL experience events
in the current `bevy-gym` event flow, and serialize training outputs through the training layer
rather than through the physics plugin.

_Sources:_

- Local event and component model: `docs/plugins/bevy_gym_plugin.md`
- Local CartPole training output pattern: `docs/examples/cartpole.md`

### Development Tools and Platforms

The development toolchain should support three separate lanes:

1. **Headless correctness lane:** no rendering, no physics plugin in core, deterministic enough for
   CI and fast policy iteration.
2. **Physics spike lane:** Avian and Rapier examples or companion crates, each isolated enough to
   compare.
3. **Visual/debug lane:** optional rendering, debug renderers, inspectors, and physics diagnostics.

_IDE and Editors:_ Standard Rust tooling is enough. A Bevy scene editor is not required for this
research topic. For physics debugging, prefer debug render plugins, ECS inspection, and focused test
scenes.

_Version Control:_ Keep physics experiments in separate examples or branches. Avoid merging both
Avian and Rapier into the core crate before a spike proves the abstraction boundary.

_Build Systems:_ Cargo is sufficient. Current crate metadata confirms all key candidates target Bevy
0.18. Keep feature gates strict:

| Crate           |                        Current checked version | Bevy dependency | Role                             |
| --------------- | ---------------------------------------------: | --------------: | -------------------------------- |
| `bevy`          | 0.18.1 docs baseline; 0.19.0-rc.3 newest crate |             n/a | ECS, schedules, app/plugin model |
| `avian3d`       |                                          0.6.1 |  `bevy ^0.18.0` | ECS-native physics               |
| `bevy_rapier3d` |                                         0.34.0 |  `bevy ^0.18.1` | mature Rapier integration        |
| `bevy-tnua`     |                                         0.31.0 |    `bevy ^0.18` | character controller layer       |

_Testing Frameworks:_ Use Rust unit tests and integration tests for action/observation/reward
determinism; use Bevy app schedule tests for ECS ordering; use scenario smoke tests for physics
backend behavior. Avoid treating visual debug output as the primary correctness signal.

_Sources:_

- Bevy docs.rs: <https://docs.rs/bevy/latest/bevy/>
- Avian docs.rs: <https://docs.rs/avian3d/latest/avian3d/>
- bevy_rapier3d docs.rs: <https://docs.rs/bevy_rapier3d/latest/bevy_rapier3d/>
- Tnua docs.rs: <https://docs.rs/bevy-tnua/latest/bevy_tnua/>
- Local `Cargo.toml`

### Cloud Infrastructure and Deployment

Cloud infrastructure is not central to the Bevy physics backend decision. The key deployment
question is whether training and evaluation can run headless, repeatably, and cheaply.

_Major Cloud Providers:_ Any Linux CPU/GPU runner can execute Rust headless binaries. Cloud choice
should be deferred until the project has a real training-throughput target.

_Container Technologies:_ Containers are useful for reproducible training and CI, but they should
not drive the physics API design. Keep display and rendering dependencies optional so headless
containers stay small.

_Serverless Platforms:_ Not appropriate for fixed-step physics training loops.

_CDN and Edge Computing:_ Not relevant to the physics stack.

**Repo recommendation:** treat headless Linux as the default deployment target. Keep `render`,
`winit`, `x11`, and `wayland` optional, as the current `Cargo.toml` does.

_Sources:_

- Local `Cargo.toml`
- Local `docs/plugins/bevy_gym_plugin.md`

### Technology Adoption Trends

**Trend 1: Stable Bevy 0.18 for implementation, 0.19 RC for migration watch.** The current repo
resolves Bevy 0.18.1, docs.rs documents 0.18.1, and crates.io API shows 0.19.0-rc.3 as the newest
published crate. Use 0.18.1 for this research and treat 0.19 as a later compatibility spike.

**Trend 2: Physics remains plugin-led.** Bevy does not provide a full first-party physics engine.
The practical choices are plugin crates and domain-specific controller code.

**Trend 3: Avian is the idiomatic Bevy/RL candidate.** Avian's ECS-driven model fits `bevy-gym`
because observations, actions, contacts, sensors, reward calculations, and reset state can remain
component-oriented.

**Trend 4: Rapier remains the mature fallback.** Rapier has explicit Bevy plugin docs for KCC,
joints, debug rendering, and simulation plugin setup. Use it when maturity, KCC behavior, or
joint-heavy behavior matters more than a native ECS feel.

**Trend 5: Character controllers should be treated as their own decision.** Avian 0.6 added
move-and-slide utilities and future KCC plans, but its docs still say it does not have a built-in
character controller. Rapier has a built-in KCC, while Tnua provides a higher-level floating
character controller with Avian/Rapier integration crates.

_Migration Patterns:_ Keep `bevy-gym` core physics-agnostic. Create separate Avian and Rapier
examples; compare them with identical task definitions before introducing a common physics
abstraction.

_Emerging Technologies:_ Avian 0.6's move-and-slide, joint motors, BVH broad phase, and spatial
query optimization are directly relevant to future RL embodied-control examples.

_Legacy Technology:_ Older `bevy_xpbd` guidance should be treated as historical because Avian is the
successor line.

_Community Trends:_ The ecosystem appears split along a useful boundary: Avian for Bevy-native ECS
ergonomics and Rapier for mature physics middleware capabilities.

_Sources:_

- Bevy fixed timestep example: <https://bevy.org/examples/movement/physics-in-fixed-timestep/>
- Avian docs.rs FAQ and feature list: <https://docs.rs/avian3d/latest/avian3d/>
- Avian 0.6 release notes: <https://joonaa.dev/blog/12/avian-0-6>
- Rapier Bevy plugin docs: <https://rapier.rs/docs/user_guides/bevy_plugin/getting_started_bevy/>
- Rapier character controller docs:
  <https://rapier.rs/docs/user_guides/bevy_plugin/character_controller/>
- Rapier joints docs: <https://rapier.rs/docs/user_guides/bevy_plugin/joints/>

## Integration Patterns Analysis

### Web Search Analysis

This step verified integration surfaces against live sources on 2026-06-12:

- Bevy 0.18 docs.rs for `FixedUpdate`, `FixedMain`, `MessageReader`, and observer-style `Event`.
- Bevy migration docs for the Bevy 0.17+ split between buffered `Message` and observer-triggered
  `Event`.
- Avian 0.6.1 docs.rs for collision events, collision hooks, collision layers, spatial queries,
  physics schedules, diagnostics, and transform synchronization.
- Rapier official Bevy plugin docs for scene queries, active events, collision groups, KCC, joints,
  and collision-detection phases.
- bevy_rapier3d docs.rs for `RapierContext`, `QueryFilter`, and the debug render plugin.
- Local `bevy-gym` source for current ECS/message integration patterns.

**Quality Assessment:** High confidence for API names and integration mechanisms; medium confidence
for backend-selection guidance until local Avian/Rapier parity examples are implemented and
benchmarked.

### API Design Patterns

This topic is in-process ECS integration, not REST or public web API design. The relevant API
boundary is:

```text
RL policy/training system
  -> PendingAction<E>
  -> EnvironmentComponent<E>::env.step(action)
  -> CurrentObservation<E>, EnvStats, ExperienceEvent, EpisodeEndEvent, ActionRequestEvent
  -> policy/training system
```

`bevy-gym` already implements the right core API pattern: environment state is a typed component,
action input is a typed component, policy/training communication is buffered Bevy `Message`, and
fixed-step ordering is controlled with `GymSet`.

#### Local Bevy-Gym API Baseline

Local source establishes this integration contract:

- `BevyGymPlugin` registers `ExperienceEvent`, `EpisodeEndEvent`, and `ActionRequestEvent` as Bevy
  messages with `app.add_message`.
- `GymSet::Step -> GymSet::AutoReset -> GymSet::ManualReset` is chained inside `FixedUpdate`.
- `step_system` consumes `PendingAction`, updates `CurrentObservation`, records `EnvStats`, collects
  outcomes from `par_iter_mut`, and writes messages serially afterward.
- `auto_reset_system` reads `EpisodeEndEvent` with `MessageReader`, resets the environment in the
  same fixed tick, clears `PendingAction`, and writes a fresh `ActionRequestEvent`.

This is a strong pattern for RL because it preserves type safety and lets policy systems batch
action requests without coupling the policy to a specific physics backend.

#### Avian API Pattern

For Avian, prefer a Bevy-native component API:

- Put Avian rigid bodies, colliders, joints, velocities, forces, sensors, and physics state on
  environment-owned entities.
- Use systems scheduled relative to Avian physics sets for action application, observation
  extraction, reward calculation, and reset.
- Use `CollisionStart` / `CollisionEnd` messages or observers only when a transition needs
  edge-triggered contact state.
- Use `Collisions` or contact graph access when rewards need current contact state rather than only
  start/end events.
- Use `SpatialQuery`, `RayCaster`, or `ShapeCaster` for sensors and controller probes.

Avian collision docs state that collision events are only sent or triggered for entities with
`CollisionEventsEnabled`, which is directly relevant to foot-contact rewards and trigger sensors.

#### Rapier API Pattern

For Rapier, prefer the documented Bevy plugin interface:

- Use Bevy components for rigid bodies, colliders, collision groups, solver groups, active events,
  joints, and KCC.
- Use `ReadRapierContext` / `RapierContext` methods for scene queries and low-level inspection.
- Use `KinematicCharacterController` or `RapierContext::move_shape` for KCC experiments.
- Use `ActiveEvents::COLLISION_EVENTS` when collision events are required.
- Use `QueryFilter` for ray/shape query exclusions, groups, sensors, solids, dynamic/static
  filtering, and custom predicates.

Rapier's official scene-query docs expose ray casts, shape casts, point projection, intersection
tests, and query filters through the Bevy plugin.

_RESTful APIs:_ Not applicable to the physics loop. Avoid HTTP inside `FixedUpdate`.

_GraphQL APIs:_ Not applicable.

_RPC and gRPC:_ Not applicable inside the core crate. If a remote trainer is added later, use a
separate transport layer outside physics systems.

_Webhook Patterns:_ Not applicable. For in-process eventing, use Bevy messages and observers.

_Sources:_

- Local `src/plugin.rs`, `src/systems/step.rs`, `src/systems/reset.rs`, `src/events.rs`
- Bevy `MessageReader`: <https://docs.rs/bevy/latest/bevy/prelude/struct.MessageReader.html>
- Bevy `Event`: <https://docs.rs/bevy/latest/bevy/ecs/event/trait.Event.html>
- Bevy 0.16 to 0.17 migration guide: <https://bevy.org/learn/migration-guides/0-16-to-0-17/>
- Avian collision module: <https://docs.rs/avian3d/latest/avian3d/collision/index.html>
- Rapier scene queries: <https://rapier.rs/docs/user_guides/bevy_plugin/scene_queries/>

### Communication Protocols

The core communication protocol is Bevy ECS scheduling plus typed messages.

#### Bevy Message vs Event Naming

Bevy 0.17 split buffered cross-system communication into `Message`, while `Event` now means
observer-triggered behavior. Bevy 0.18 `MessageReader` docs describe ordered message reading with
per-reader tracking, and the migration guide explicitly calls out the `Event` to `Message` split.

**Implication for this repo:** new docs and examples should say `MessageReader` / `MessageWriter`,
not `EventReader` / `EventWriter`, unless they intentionally use observer-triggered `Event`s. Local
source already uses `MessageReader` and `MessageWriter`; `src/lib.rs` still has a quick-start
snippet using `EventReader<ActionRequestEvent>`, which is a documentation drift risk.

#### Fixed Schedule Protocol

Bevy `FixedUpdate` is documented for fixed-rate gameplay logic such as physics, AI, networking, and
game rules. `FixedMain` is run by `RunFixedMainLoop`, and variable-timestep systems can be ordered
before or after fixed logic with `RunFixedMainLoopSystems`.

**Implication for physics integration:**

- Apply actions before the physics step that should consume them.
- Extract observations after physics has advanced.
- Render from interpolated or latest observation data in `Update` or after fixed-loop interpolation,
  not by coupling render framerate to physics stepping.
- Keep `bevy-gym`'s `GymSet` ordering explicit and documented.

#### Avian Schedule Protocol

Avian has a `schedule` module that sets up default scheduling, system set configuration, and time
resources for physics. It exposes high-level `PhysicsSystems` sets so systems can be scheduled
before or after physics without depending on implementation details. Avian docs also state that by
default physics runs at a fixed timestep in `FixedPostUpdate`.

**Integration point:** in a real Avian environment, define local sets like:

```text
ApplyRlAction -> AvianPhysics -> ExtractObservationAndReward -> ResetIfDone
```

Then wire those sets against Avian's `PhysicsSystems` rather than hiding physics stepping inside
`Environment::step` unless `Environment::step` owns a dedicated nested physics app.

#### Rapier Schedule Protocol

Rapier's Bevy plugin owns a simulation pipeline and synchronizes between Bevy ECS components and
Rapier's internal world/context. The docs.rs plugin module describes the plugin as responsible for
setting up the Rapier physics simulation pipeline and resources.

**Integration point:** use Rapier's component surface and query/context APIs from systems scheduled
around Rapier's plugin systems. Avoid manually mutating both Bevy transforms and Rapier state in the
same tick without understanding sync order.

_HTTP/HTTPS Protocols:_ Not relevant to the physics loop.

_WebSocket Protocols:_ Optional future debugger or remote trainer transport only; not part of
physics integration.

_Message Queue Protocols:_ Bevy messages are the in-process queue. External queues should not be
introduced for environment stepping.

_gRPC and Protocol Buffers:_ Possible future remote policy boundary, but out of scope for local
physics integration.

_Sources:_

- Bevy `FixedUpdate`: <https://docs.rs/bevy/latest/bevy/app/struct.FixedUpdate.html>
- Bevy `FixedMain`: <https://docs.rs/bevy_app/latest/bevy_app/struct.FixedMain.html>
- Bevy `MessageReader`: <https://docs.rs/bevy/latest/bevy/prelude/struct.MessageReader.html>
- Avian schedule module: <https://docs.rs/avian3d/latest/avian3d/schedule/index.html>
- bevy_rapier3d plugin module:
  <https://docs.rs/bevy_rapier3d/latest/bevy_rapier3d/plugin/index.html>

### Data Formats and Standards

The primary data format should be typed Rust data in ECS components and RL trait types. Do not
serialize within the hot physics path unless crossing a real process boundary.

#### In-Process Data

Use strongly typed Rust structures:

- `E::Observation` for policy input.
- `E::Action` for policy output.
- `Experience<O, A>` for transition records.
- `EnvStats` for per-env counters.
- typed info/extras maps for scalar metrics where the environment cannot statically enumerate every
  value.

#### Physics State

Do not expose physics backend internals as the public environment API. Normalize observation
extraction into domain-level fields:

- position, rotation, velocity, angular velocity,
- joint angle and joint velocity,
- contact booleans,
- contact force summaries,
- ray/sensor hits,
- reward terms and done/truncation flags.

This matters because Avian and Rapier expose different state and query surfaces. A policy should not
learn against a `RapierContext` or Avian contact graph directly.

#### External Data

For training output and reproducibility:

- JSONL is suitable for scalar run logs.
- MessagePack is suitable for checkpoints when the training layer already uses it.
- CSV is acceptable for small tabular diagnostics.
- Parquet can be deferred until large-scale offline analysis exists.

The local CartPole docs already describe JSONL logging and `best.mpk` checkpoint output through
`ember-rl`, so physics examples should reuse that pattern instead of inventing a physics-specific
storage format.

_JSON and XML:_ JSONL for logs; XML only if importing MuJoCo-style environment definitions later.

_Protobuf and MessagePack:_ MessagePack is already part of the local checkpoint story through
`best.mpk`.

_CSV and Flat Files:_ Good for small benchmark summaries.

_Custom Data Formats:_ Avoid until scenario/curriculum authoring needs a schema. Prefer Rust structs
plus optional RON/TOML/JSON config.

_Sources:_

- Local `docs/examples/cartpole.md`
- Local `src/events.rs`
- Avian spatial queries: <https://docs.rs/avian3d/latest/avian3d/spatial_query/index.html>
- Rapier scene queries: <https://rapier.rs/docs/user_guides/bevy_plugin/scene_queries/>

### System Interoperability Approaches

The main interoperability question is how to let `bevy-gym`, RL policies, renderers, and physics
plugins cooperate without locking the core crate to one physics backend.

#### Recommended Boundary

Keep `bevy-gym` core as:

```text
Environment trait adapter + ECS scheduling + typed messages + optional rendering
```

Put physics backend code in:

```text
examples, feature-gated companion modules, or separate crates
```

The physics environment can still use Bevy ECS internally, but the public `bevy-gym` contract should
remain `Environment::reset`, `Environment::step`, observations, actions, messages, and stats.

#### Backend-Specific Adapter Pattern

Use one adapter per backend:

- `AvianWalkerEnv`, `AvianLanderEnv`, `AvianHumanoidEnv`
- `RapierWalkerEnv`, `RapierLanderEnv`, `RapierHumanoidEnv`

Each adapter should own:

- backend plugin setup,
- entity spawning,
- action application,
- physics stepping and schedule order,
- observation extraction,
- reward/done calculation,
- reset/despawn/rebuild or state restore.

Only after both backend examples exist should a shared physics abstraction be introduced. Otherwise
the abstraction will likely encode one backend's assumptions.

#### Local Gap

This repo currently has many files under `examples/avian-2d` and `examples/avian-3d`, but targeted
inspection found they are reference/spec stubs made of documentation comments, not implemented
Bevy/Avian environments. They are useful requirements artifacts, not integration proof.

_Point-to-Point Integration:_ Good for the first spike: one environment wired directly to Avian, one
wired directly to Rapier.

_API Gateway Patterns:_ Not applicable.

_Service Mesh:_ Not applicable.

_Enterprise Service Bus:_ Not applicable.

_Sources:_

- Local `src/plugin.rs`
- Local `examples/avian-2d/*`, `examples/avian-3d/*`
- Avian docs.rs: <https://docs.rs/avian3d/latest/avian3d/>
- bevy_rapier3d docs.rs: <https://docs.rs/bevy_rapier3d/latest/bevy_rapier3d/>

### Microservices Integration Patterns

Microservices patterns mostly do not apply to this local Rust ECS crate. The analogs that matter are
isolation, circuit breakers, and service-discovery-like entity lookup.

_API Gateway Pattern:_ Replace with a local facade: `BevyGymPlugin` is already the app-facing facade
for environment spawning, stepping, messages, and reset behavior.

_Service Discovery:_ Use entity IDs and typed components, not string names. `ActionRequestEvent`
already includes both `env_id` and `entity`, which is better than requiring policy systems to scan
or match by name.

_Circuit Breaker Pattern:_ For RL, this maps to invalid-state handling:

- terminate/truncate episodes when state is non-finite,
- reset when bodies fall out of bounds,
- clamp or reject invalid actions,
- detect physics explosions and emit an explicit episode end,
- collect backend-specific diagnostics before reset.

_Saga Pattern:_ Not relevant. For multi-step reset, prefer a deterministic reset system or state
machine rather than distributed transaction concepts.

**Recommendation:** do not introduce service architecture vocabulary into the code. Use Bevy terms:
plugin, resource, component, message, schedule, system set, observer, query, and app.

_Sources:_

- Local `src/plugin.rs`, `src/systems/reset.rs`, `src/events.rs`
- Bevy `FixedUpdate`: <https://docs.rs/bevy/latest/bevy/app/struct.FixedUpdate.html>
- Bevy `MessageReader`: <https://docs.rs/bevy/latest/bevy/prelude/struct.MessageReader.html>

### Event-Driven Integration

Event-driven integration is central, but use the Bevy 0.18 terminology precisely:

- **Message:** buffered cross-system queue. Best for `ExperienceEvent`, `EpisodeEndEvent`, and
  `ActionRequestEvent`.
- **Event:** observer-triggered behavior. Useful for immediate reactions and entity-scoped
  observers, but not the default for RL transition streams.

#### Current Bevy-Gym Message Flow

```text
spawn_environments
  -> ActionRequestEvent
policy system
  -> PendingAction
GymSet::Step
  -> ExperienceEvent
  -> EpisodeEndEvent if done
  -> ActionRequestEvent
GymSet::AutoReset
  -> reads EpisodeEndEvent
  -> reset env
  -> ActionRequestEvent
GymSet::ManualReset
  -> reads ResetRequested component
  -> reset env
  -> ActionRequestEvent
```

This is idiomatic for `bevy-gym` because it decouples training/policy systems from environment
internals while preserving deterministic fixed-step ordering.

#### Physics Collision Events

Avian and Rapier both require opt-in for collision event emission:

- Avian: collision events are sent or triggered only for entities with `CollisionEventsEnabled`.
- Rapier: collision events are generated only when at least one collider has
  `ActiveEvents::COLLISION_EVENTS`.

This is a common source of "collisions work but events do not" bugs. For RL, do not depend solely on
edge-triggered collision events for persistent contact observations. Use collision/contact state or
scene queries when the observation needs "is touching now" each step.

#### Publish-Subscribe Patterns

Use Bevy messages for one-to-many consumers:

- training buffer consumes `ExperienceEvent`,
- logger consumes `EpisodeEndEvent`,
- policy consumes `ActionRequestEvent`,
- render/debug UI may consume stats or observe ECS state.

#### Event Sourcing

Do not use event sourcing as the authoritative physics state. `ExperienceEvent` can be logged for
replay/training, but the live state remains ECS plus physics backend state.

#### Message Broker Patterns

Use Bevy's in-process message resources. External brokers add latency and nondeterminism and should
be outside the fixed-step environment loop.

#### CQRS Patterns

A lightweight CQRS-like split is useful:

- Commands/actions: `PendingAction`, reset marker components, controller target components.
- Queries/observations: `CurrentObservation`, physics query APIs, stats, messages.

Keep this conceptual; do not add CQRS infrastructure.

_Sources:_

- Bevy `MessageReader`: <https://docs.rs/bevy/latest/bevy/prelude/struct.MessageReader.html>
- Bevy `Event`: <https://docs.rs/bevy/latest/bevy/ecs/event/trait.Event.html>
- Bevy migration guide: <https://bevy.org/learn/migration-guides/0-16-to-0-17/>
- Avian collision events: <https://docs.rs/avian3d/latest/avian3d/collision/index.html>
- Rapier active events: <https://rapier.rs/docs/user_guides/bevy_plugin/collider_active_events/>

### Integration Security Patterns

Security is not the central risk for local physics integration, but isolation still matters.

_OAuth 2.0 and JWT:_ Not applicable unless a remote trainer or web dashboard is added.

_API Key Management:_ Not applicable to local physics. Do not add secrets to examples or training
configs.

_Mutual TLS:_ Not applicable unless remote policy inference is introduced.

_Data Encryption:_ Not applicable to local run artifacts unless logs contain sensitive data.

#### Practical Safety and Isolation Rules

- Keep physics plugins out of core `bevy-gym` dependencies until a backend is selected.
- Keep debug renderers and inspectors behind example/dev features.
- Do not expose Bevy Remote Protocol, MCP, or any remote mutation surface in training binaries by
  default.
- Treat remote control/debugging as localhost-only dev tooling.
- Keep physics reset deterministic and do not let a remote/debug tool mutate training state during
  benchmark runs.

_Sources:_

- Local `Cargo.toml`
- Local `docs/plugins/bevy_gym_plugin.md`
- Bevy docs.rs: <https://docs.rs/bevy/latest/bevy/>

### Integration Recommendation

For `bevy-gym`, use this integration sequence:

1. **Fix terminology drift in docs when implementation work resumes.** Source uses
   `MessageReader`/`MessageWriter`; public examples should not show `EventReader` for message types.
2. **Implement one minimal Avian physics environment first.** Use an environment with simple contact
   and force/impulse semantics, such as Lunar Lander or Inverted Pendulum, before humanoids.
3. **Implement one equivalent Rapier environment second.** Keep task definition and observations
   identical enough to compare backend behavior.
4. **Compare integration costs before abstracting.** Measure API complexity, schedule clarity, reset
   determinism, contact-event reliability, sensor-query ergonomics, and headless throughput.
5. **Only then design a shared backend abstraction.** The abstraction should be driven by observed
   common needs, not guessed from docs.

**Decision trigger:** If Avian's ECS-native API makes action, observation, reset, contact, and
sensor integration simple enough for the first two environments, keep Avian as the default physics
path. If KCC, joints, contact stability, or documentation gaps slow progress materially, build the
Rapier fallback before broadening the environment set.

## Architectural Patterns and Design

### Web Search Analysis

This step verified architectural claims against current sources on 2026-06-12:

- Bevy official ECS, App, and Plugin documentation for the engine's architectural model.
- Bevy ECS docs.rs for parallel scheduling, components, resources, schedules, messages, and
  observers.
- Avian repository design notes and docs.rs modules for ECS-native physics, modular plugin
  architecture, schedules, diagnostics, and physics component design.
- Rapier official Bevy plugin docs for plugin architecture, multiple physics contexts, common
  mistakes, scene queries, and debug rendering.
- Local `bevy-gym` source for current plugin boundaries, message flow, feature gating, and render
  separation.

**Quality Assessment:** High confidence for architectural facts and local code shape; medium
confidence for final backend architecture until one Avian and one Rapier environment are implemented
against the same task and measured.

### System Architecture Patterns

The best architecture for `bevy-gym` is a
**physics-agnostic core with backend-specific environment adapters**.

#### Current Core Pattern

`bevy-gym` already has a clean core architecture:

```text
bevy-gym core crate
  - BevyGymPlugin<E>
  - EnvironmentComponent<E>
  - PendingAction<E>
  - CurrentObservation<E>
  - EnvStats
  - Bevy messages: ActionRequestEvent, ExperienceEvent, EpisodeEndEvent
  - fixed-step systems: Step, AutoReset, ManualReset
  - optional render plugin behind feature flags
```

This is effectively a **plugin facade** over an RL environment adapter. Bevy's official docs
describe plugins as collections of app-modifying code, and Bevy ECS docs emphasize componentized
data, systems, resources, schedules, and parallel execution. The local architecture aligns with
that: `BevyGymPlugin` configures messages, resources, systems, and startup spawning; the environment
type remains generic.

#### Recommended Target Pattern

Use this layered architecture:

```text
Core bevy-gym
  -> RL schedule, messages, environment pool, optional render sync

Physics adapter examples/crates
  -> AvianLanderEnv, AvianPendulumEnv, AvianWalkerEnv
  -> RapierLanderEnv, RapierPendulumEnv, RapierWalkerEnv

Training layer
  -> policy, replay buffer, checkpointing, metrics

Debug/render layer
  -> optional visualizers, debug physics rendering, inspectors
```

Keep the core crate narrow. Put Avian/Rapier dependencies in examples, feature-gated modules, or
companion crates until one backend clearly deserves first-class support.

#### Avian Architecture Fit

Avian is architected for Bevy. Its repository states core design principles: built with Bevy, no
wrapper around an existing engine, heavy ECS use, no separate physics world, and modular plugin
architecture. That is the strongest architectural match for `bevy-gym` because RL observations,
actions, contacts, sensors, rewards, and resets can be component-oriented.

The tradeoff is maturity. Avian docs also describe it as young and still evolving, so it should be
treated as the default **architecture candidate**, not a proven foundation until spikes pass.

#### Rapier Architecture Fit

Rapier uses a more middleware-style architecture. The Bevy plugin sets up a full Rapier simulation
pipeline and resources; docs.rs exposes plugin systems that copy Bevy components into Rapier,
propagate Rapier transforms, and expose `RapierContext`/`ReadRapierContext` style APIs.

Rapier's multiple-context docs are directly relevant to RL: they explicitly mention AI-training
projects simulating multiple physics contexts in parallel. The caveat is important: Rapier contexts
share the same `TimestepMode` resource and execute at the same time/rate. That means multi-context
Rapier can isolate environment collisions, but it does not automatically give independent
per-environment clocks.

_Source:_

- Bevy ECS docs: <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- Bevy plugin docs: <https://bevy.org/learn/quick-start/getting-started/plugins/>
- Bevy `Plugin`: <https://docs.rs/bevy/latest/bevy/app/trait.Plugin.html>
- Avian repository design: <https://github.com/avianphysics/avian>
- Rapier multiple contexts: <https://rapier.rs/docs/user_guides/bevy_plugin/multiple_contexts/>
- bevy_rapier3d plugin docs: <https://docs.rs/bevy_rapier3d/latest/bevy_rapier3d/plugin/index.html>
- Local `src/plugin.rs`, `src/components.rs`, `src/events.rs`, `src/systems/step.rs`,
  `src/render.rs`

### Design Principles and Best Practices

#### Principle 1: Keep Core Physics-Agnostic

The core library should not know whether an environment is implemented with Avian, Rapier,
hand-coded dynamics, or an external simulator. Its job is to schedule typed `Environment` instances
and coordinate policy/training messages.

This preserves:

- fast headless builds,
- small dependency surface,
- stable public API,
- clean docs.rs builds,
- freedom to compare physics backends honestly.

#### Principle 2: Make Backend Choice Observable, Not Implicit

Do not hide backend behavior behind a generic `PhysicsBackend` trait too early. Physics engines
differ in contacts, joints, character controllers, scheduling, transforms, precision, sleeping, CCD,
and scene queries. A premature abstraction will either leak backend-specific concepts or omit the
features that matter.

Use explicit environment types first:

```text
AvianInvertedPendulumEnv
RapierInvertedPendulumEnv
```

Only extract a shared trait after the second backend example proves the repeated shape.

#### Principle 3: Preserve Deterministic Step Boundaries

For RL, the key architectural invariant is:

```text
action_n -> physics_step -> observation_n+1, reward_n, done_n
```

Avoid architecture that reads observations before physics has settled, writes actions after the
physics step that should consume them, or lets render-frame systems mutate training state.

#### Principle 4: Treat Physics Units as Architecture, Not Tuning

Rapier's common-mistakes docs warn that using pixels as physics units can make simulation appear to
run in slow motion; it recommends SI-style units and scaling between graphics and physics. This
applies to Avian too as a general physics architecture rule.

For `bevy-gym`, define units per environment:

- meters, seconds, radians, kilograms when possible,
- documented action scaling,
- documented reward scaling,
- explicit conversion between render pixels and physics units.

#### Principle 5: Separate Contact State from Contact Events

Collision events are useful for edge transitions. RL observations often need persistent contact
state, contact force summaries, or "is touching now" booleans. Architect foot sensors, lander-leg
contacts, and contact-cost rewards around queryable contact state or contact accumulation, not only
event edges.

_Source:_

- Bevy ECS docs: <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- Avian repository design: <https://github.com/avianphysics/avian>
- Avian collision docs: <https://docs.rs/avian3d/latest/avian3d/collision/index.html>
- Rapier common mistakes: <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Rapier active events: <https://rapier.rs/docs/user_guides/bevy_plugin/collider_active_events/>

### Scalability and Performance Patterns

#### Current Scalability Pattern: Parallel Environment Entities

The local `step_system` uses Bevy `par_iter_mut()` over independent environment entities, then
serially writes messages after the parallel phase. This is a strong pattern for CPU-heavy pure-Rust
environments because each `EnvironmentComponent<E>` is independent.

#### Physics Plugin Scaling Caveat

Physics plugins can change the scaling story:

- If each `EnvironmentComponent<E>` owns its own internal Bevy app/world, `par_iter_mut()` may still
  parallelize independent environment stepping, but each environment pays nested-app overhead.
- If one Bevy app owns one shared physics world, all environments share that world's broad phase,
  solver, schedules, and collision space unless carefully isolated.
- If Rapier uses multiple contexts, environments can be isolated by context, but all contexts share
  the same `TimestepMode` and execute at the same time/rate.
- If Avian uses one Bevy world, isolation must be done by spatial separation, collision layers, or
  separate app/world instances.

This is the core architectural spike: **environment-per-entity** works today, but
**physics-world-per-environment** may be needed for clean RL batching.

#### Recommended Scaling Sequence

1. Implement one single-environment Avian example.
2. Implement the same task with Rapier.
3. Add N parallel environments with isolation.
4. Measure throughput and correctness:
   - steps/sec,
   - reward regression,
   - contact event reliability,
   - reset latency,
   - memory growth,
   - determinism across seeds,
   - render/debug overhead off vs on.

#### Performance Guardrails

- Run physics benchmarks in release or with optimized dependency profiles.
- Use debug render only for diagnosis, not throughput measurement.
- Keep rendering and training lanes separate.
- Prefer simple colliders and documented unit scaling.
- Treat high-contact scenes, many joints, and humanoids as late-stage tests.

_Source:_

- Local `src/systems/step.rs`
- Bevy ECS scheduling docs: <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- Rapier multiple contexts: <https://rapier.rs/docs/user_guides/bevy_plugin/multiple_contexts/>
- Rapier common mistakes: <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Avian schedule docs: <https://docs.rs/avian3d/latest/avian3d/schedule/index.html>

### Integration and Communication Patterns

#### Message-Driven RL Boundary

Keep the local Bevy message pattern:

```text
ActionRequestEvent -> policy writes PendingAction
PendingAction -> step system consumes action
ExperienceEvent -> trainer/replay/logging
EpisodeEndEvent -> stats/reset/logging
```

This is the right architectural boundary because it decouples policy systems from physics
implementation.

#### Schedule-Oriented Physics Boundary

Physics integration should be designed around schedules, not around ad hoc function calls hidden
inside random systems.

For Avian:

- use `PhysicsSystems` to order action application and observation extraction around physics,
- use Avian's `PhysicsSchedule`/substep model when needed,
- use component and system sets rather than a separate service object.

For Rapier:

- understand where Bevy components are copied into Rapier,
- understand where Rapier transforms are copied back,
- schedule action and observation systems around the plugin's physics sets,
- use `ReadRapierContext` for queries, not direct mutation of internal state.

#### Render Separation

`GymRenderPlugin` already uses a separate render sync pattern: setup in `First`, sync in `Update`,
while simulation runs in `FixedUpdate`. Preserve this separation. Visual interpolation/debug
rendering should observe simulation state, not own it.

_Source:_

- Local `src/plugin.rs`, `src/systems/step.rs`, `src/render.rs`
- Bevy `FixedUpdate`: <https://docs.rs/bevy/latest/bevy/app/struct.FixedUpdate.html>
- Bevy fixed timestep example: <https://bevy.org/examples/movement/physics-in-fixed-timestep/>
- Avian schedule docs: <https://docs.rs/avian3d/latest/avian3d/schedule/index.html>
- bevy_rapier3d plugin docs: <https://docs.rs/bevy_rapier3d/latest/bevy_rapier3d/plugin/index.html>

### Security Architecture Patterns

Security is mostly about keeping training state local, reproducible, and protected from accidental
mutation.

Recommended architecture:

- no remote mutation surface in default training binaries,
- no BRP/MCP/debug inspector in benchmark runs unless explicitly enabled,
- feature-gate render/debug/inspector/physics-debug dependencies,
- keep secrets out of environment examples and run logs,
- make benchmark presets immutable during a run,
- log backend version, feature flags, seed, timestep, unit scale, and physics config.

If remote control is introduced later, keep it as a dev-only plugin with localhost binding and
explicit startup logs.

_Source:_

- Local `Cargo.toml`
- Local `docs/plugins/bevy_gym_plugin.md`
- Bevy plugin docs: <https://bevy.org/learn/quick-start/getting-started/plugins/>

### Data Architecture Patterns

#### Domain Observation Model

Architect observations as domain data, not raw backend data. For example:

```text
LanderObservation {
  position,
  linear_velocity,
  angle,
  angular_velocity,
  left_leg_contact,
  right_leg_contact,
}
```

This should be computed from Avian or Rapier state, but not expose their internal handles or
contexts.

#### Physics Metadata Model

Each physics environment should carry explicit metadata:

- backend (`avian3d 0.6.1`, `bevy_rapier3d 0.34.0`),
- Bevy version,
- timestep and substep count,
- gravity,
- unit scale,
- solver/iteration settings where applicable,
- collision layers/groups,
- action scaling,
- observation scaling,
- reset seed.

This metadata is necessary because RL results are not meaningful without environment
reproducibility.

#### Scenario/Fixture Model

Use small scenario fixtures before full Gymnasium parity:

- spawn layout,
- body/collider definitions,
- reward settings,
- termination thresholds,
- reset distribution,
- debug camera/render config.

Prefer Rust structs first. Move to RON/TOML/JSON only when scenario authoring needs external
editing.

_Source:_

- Local `src/events.rs`
- Local `docs/examples/cartpole.md`
- Avian collision/spatial query docs: <https://docs.rs/avian3d/latest/avian3d/collision/index.html>
- Rapier scene queries: <https://rapier.rs/docs/user_guides/bevy_plugin/scene_queries/>

### Deployment and Operations Architecture

#### Validation Lanes

Use separate operational lanes:

| Lane          | Purpose                       | Expected dependencies                   |
| ------------- | ----------------------------- | --------------------------------------- |
| core headless | library correctness           | Bevy minimal features only              |
| render smoke  | visual sync                   | `render` plus explicit display backend  |
| Avian spike   | ECS-native physics validation | Avian dependency in example/feature     |
| Rapier spike  | fallback physics validation   | Rapier dependency in example/feature    |
| benchmark     | steps/sec and determinism     | release or optimized dependency profile |

#### Observability

Each physics spike should emit:

- steps/sec,
- episode return,
- reset count,
- non-finite state count,
- contact event count,
- persistent contact count,
- collision/query timing if available,
- physics backend diagnostics if available.

Avian documents diagnostics support for physics timers and counters, and Rapier provides debug
render and plugin resources for inspecting the physics world. Use those for diagnosis, not as the
only correctness proof.

#### Architecture Decision Records

Before merging a backend into core, create a small ADR recording:

- task implemented,
- backend version,
- Bevy version,
- architecture used,
- throughput,
- failure modes,
- remaining risks,
- reason for choosing or rejecting backend.

_Source:_

- Avian diagnostics/schedule docs: <https://docs.rs/avian3d/latest/avian3d/>
- Rapier debug renderer docs: <https://rapier.rs/docs/user_guides/bevy_plugin/getting_started_bevy/>
- Rapier common mistakes: <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Local `Cargo.toml`, `docs/plugins/bevy_gym_plugin.md`, `docs/plugins/gym_render.md`

### Architectural Decision

Recommended architecture for the next implementation phase:

1. **Do not add Avian or Rapier to core `bevy-gym` dependencies yet.**
2. **Create a first implemented Avian example for a simple task**, preferably Inverted Pendulum or
   Lunar Lander, because it exercises fixed timestep, reset, action application, observation
   extraction, contacts, and rendering without humanoid complexity.
3. **Keep the example explicit rather than generic.** Let the first implementation teach the
   required abstraction.
4. **Build a Rapier equivalent only after the Avian example is honest and measured.**
5. **Compare before extracting shared physics traits.**

The architectural posture is therefore:
**Avian-first spike, Rapier-controlled fallback, core crate remains physics-agnostic.**

## Implementation Approaches and Technology Adoption

### Technology Adoption Strategies

The practical adoption model is gradual and evidence-driven:

1. Keep the `bevy-gym` core crate physics-agnostic.
2. Implement Avian in a single explicit example first.
3. Measure correctness, throughput, reset behavior, contact behavior, and render sync.
4. Implement a Rapier equivalent only after the Avian example has honest metrics.
5. Extract a shared physics abstraction only after both implementations reveal stable duplication.

This avoids the main adoption failure mode for this repo: locking the library into a backend
abstraction before there is proof of what the abstraction must preserve. Cargo features are the
right mechanism for optional physics dependencies, but they should remain additive and should not
silently change core behavior. A reasonable future layout is:

- `default = []`
- `render = [...]`
- `avian3d = ["dep:avian3d"]`
- `rapier3d = ["dep:bevy_rapier3d"]`
- explicit example targets that require their matching backend feature

The repo already follows this pattern for rendering and display backends: `render`, `winit`, `x11`,
and `wayland` are optional, while the default crate stays headless.

_Source:_

- Cargo feature documentation: <https://doc.rust-lang.org/cargo/reference/features.html>
- Local `Cargo.toml`
- Local `docs/plugins/gym_render.md`

### Development Workflows and Tooling

The existing workflow is a good thin baseline:

- `cargo check --features x11`
- `cargo clippy --features x11 -- -D warnings`

Physics adoption needs wider lanes because feature-gated examples can compile or fail independently.
Recommended workflow:

| Lane     | Command shape                                                            | Purpose                                             |
| -------- | ------------------------------------------------------------------------ | --------------------------------------------------- |
| core     | `cargo check --no-default-features`                                      | prove the library remains headless and physics-free |
| render   | `cargo check --features render,x11`                                      | prove visual sync APIs compile                      |
| examples | `cargo test --examples --features render,x11` or targeted example checks | prove examples do not rot                           |
| Avian    | `cargo check --example <avian-example> --features avian3d,render,x11`    | prove Avian backend compiles                        |
| Rapier   | `cargo check --example <rapier-example> --features rapier3d,render,x11`  | prove fallback backend compiles                     |
| lint     | `cargo clippy --all-targets --features ... -- -D warnings`               | catch common Rust mistakes                          |
| perf     | `cargo bench` or release-mode smoke                                      | measure steps/sec and reset throughput              |

For local development, prefer a simple loop:

1. implement headless physics step,
2. add deterministic reset/finite-state smoke,
3. add render sync,
4. add instrumentation,
5. compare backend behavior.

Avoid starting with a generalized physics adapter crate. The fastest route to a correct abstraction
is to write two explicit implementations and delete duplication later.

_Source:_

- Cargo test documentation: <https://doc.rust-lang.org/cargo/commands/cargo-test.html>
- Cargo clippy documentation: <https://doc.rust-lang.org/cargo/commands/cargo-clippy.html>
- Cargo bench documentation: <https://doc.rust-lang.org/cargo/commands/cargo-bench.html>
- Local `.github/workflows/ci.yml`

### Testing and Quality Assurance

Testing should cover three different claims:

1. `bevy-gym` scheduling is correct.
2. The physics environment produces finite, reproducible transitions.
3. The rendered view follows the latest simulated observation without changing simulation behavior.

Recommended test layers:

- **Core unit tests:** verify `PendingAction` is consumed once, `CurrentObservation` updates,
  `ExperienceEvent` is emitted, and episode reset behavior is ordered after `GymSet::Step`.
- **Fixed-step app tests:** construct a minimal `App`, add the plugin, control `Time<Fixed>`, run
  fixed steps, and assert observation/reward invariants.
- **Physics smoke tests:** run 100-1000 headless steps with a fixed seed and assert finite
  observations, finite rewards, bounded body transforms, and no unexpected panics.
- **Reset determinism tests:** reset with the same seed and compare initial observations and physics
  metadata.
- **Contact tests:** assert expected contact state separately from event delivery.
- **Feature compile tests:** compile each physics example with only the feature set it declares.
- **Performance tests:** measure release-mode steps/sec for 1, 4, 16, and 64 environments.

Bevy's fixed-timestep model matters here. Systems in `FixedUpdate` may run zero, one, or multiple
times relative to a rendered frame, so tests should drive fixed time deliberately and avoid coupling
correctness to frame rate. Bevy's `Time<Fixed>` docs also expose methods intended for test control
of fixed overstep.

For collisions:

- In Avian, collision events require `CollisionEventsEnabled`; they can be read as messages or
  observed with observers.
- In Rapier, collision event generation is disabled by default unless the relevant `ActiveEvents`
  bit is enabled.

These should become explicit test fixture assertions instead of implicit assumptions.

_Source:_

- Bevy fixed timestep example: <https://bevy.org/examples/movement/physics-in-fixed-timestep/>
- Bevy `Time<Fixed>` docs: <https://docs.rs/bevy/latest/bevy/prelude/struct.Time.html>
- Avian collision docs: <https://docs.rs/avian3d/latest/avian3d/collision/index.html>
- Rapier active events docs:
  <https://rapier.rs/docs/user_guides/bevy_plugin/collider_active_events/>
- Local `src/systems/step.rs`, `src/events.rs`, `docs/plugins/bevy_gym_plugin.md`

### Deployment and Operations Practices

For this project, "deployment" mostly means repeatable local and CI execution lanes:

- headless training,
- visual evaluation,
- deterministic smoke tests,
- backend comparison benchmarks,
- release-ready crate publication.

Headless training should be the default operational mode. Rendering should be a
diagnostic/evaluation mode. The repo already models this split through `BevyGymPlugin::headless()`
and `GymRenderPlugin`, where simulation runs in `FixedUpdate` and rendering syncs from
`CurrentObservation` in `Update`.

Each physics run should emit a compact run manifest:

- backend and backend version,
- Bevy version,
- feature set,
- timestep and substep count,
- gravity,
- unit scale,
- body/collider counts,
- collision layers/groups,
- reset seed,
- solver settings where exposed,
- action scaling,
- observation scaling,
- render enabled/disabled.

Each run should also emit operational metrics:

- steps/sec,
- episode return,
- episode length,
- reset count,
- non-finite observation count,
- contact event count,
- persistent contact count,
- physics diagnostic counters where available.

Avian provides diagnostics support for physics timers and counters when diagnostics features/plugins
are enabled. Rapier's Bevy docs call out release/profile settings as a major performance factor, so
benchmark lanes must use optimized builds or explicit dependency profile overrides.

_Source:_

- Avian diagnostics docs: <https://docs.rs/avian3d/latest/avian3d/diagnostics/index.html>
- Rapier common mistakes and performance note:
  <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Local `docs/plugins/bevy_gym_plugin.md`
- Local `docs/plugins/gym_render.md`

### Team Organization and Skills

The implementation path needs a small set of focused capabilities:

- Rust generics, feature flags, and optional dependencies.
- Bevy ECS scheduling, plugins, resources, components, messages, and queries.
- Bevy fixed timestep and render/simulation separation.
- Avian components, collision events, spatial queries, scheduling, and diagnostics.
- Rapier contexts, active events, scene queries, units, and performance profiles.
- RL environment design: action scaling, observation scaling, reward shaping, reset distribution,
  termination/truncation semantics.
- Benchmarking discipline: release-mode comparisons and reproducible run metadata.

The first physics example should be written by someone comfortable staying explicit. The goal is not
a framework rewrite; the goal is to learn the shape of a correct physics-backed environment under
the existing `Environment` trait and `BevyGymPlugin` schedule.

_Source:_

- Bevy plugin docs: <https://bevy.org/learn/quick-start/getting-started/plugins/>
- Avian crate docs: <https://docs.rs/avian3d/latest/avian3d/>
- Rapier multiple contexts docs: <https://rapier.rs/docs/user_guides/bevy_plugin/multiple_contexts/>

### Cost Optimization and Resource Management

The main costs are compile time, CI time, runtime throughput, and abstraction maintenance.

Recommended controls:

- keep default features empty,
- keep physics backends optional,
- keep render dependencies out of headless lanes,
- run most CI on `cargo check` and targeted examples,
- reserve release-mode benchmarks for scheduled or pre-merge lanes,
- use dependency profile overrides when debug builds make physics unusably slow,
- avoid simulating all environments in one physical world until isolation and collision layers are
  proven,
- record body/collider counts and step timing per backend.

Rapier explicitly warns that non-optimized local builds can be much slower than demos and recommends
release/profile optimization. This matters for RL because training throughput is a product
requirement, not just an implementation detail.

_Source:_

- Rapier common mistakes: <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Cargo bench documentation: <https://doc.rust-lang.org/cargo/commands/cargo-bench.html>
- Local `.github/workflows/ci.yml`

### Risk Assessment and Mitigation

| Risk                                                | Likelihood | Impact | Mitigation                                                                     |
| --------------------------------------------------- | ---------: | -----: | ------------------------------------------------------------------------------ |
| Physics abstraction added too early                 |       High |   High | implement Avian and Rapier examples before extracting traits                   |
| Unit-scale mistakes                                 |       High |   High | use SI-style physics units, document render-to-physics scale                   |
| Collision events silently missing                   |     Medium |   High | fixture tests for `CollisionEventsEnabled`/`ActiveEvents`                      |
| Non-finite transforms or rewards                    |     Medium |   High | finite-state smoke tests and panic regression tests                            |
| Debug-build performance misread as backend weakness |       High | Medium | benchmark release/profile-optimized builds                                     |
| Multi-env collision leakage                         |     Medium |   High | use collision layers, spatial separation, or separate contexts; test isolation |
| Reset nondeterminism                                |     Medium |   High | seed reset fixtures and compare initial observations                           |
| Feature matrix rot                                  |     Medium | Medium | add targeted CI lanes for core/render/backend examples                         |
| Render code influencing simulation                  | Low-Medium |   High | keep render sync read-only from `CurrentObservation`                           |

_Source:_

- Rapier common mistakes: <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Rapier multiple contexts docs: <https://rapier.rs/docs/user_guides/bevy_plugin/multiple_contexts/>
- Avian collision docs: <https://docs.rs/avian3d/latest/avian3d/collision/index.html>
- Local `docs/plugins/gym_render.md`

## Technical Research Recommendations

### Implementation Roadmap

Recommended implementation sequence:

1. Fix documentation terminology drift first: update stale `EventReader` references in the crate
   quick-start to Bevy 0.18 `MessageReader`.
2. Convert the chosen Avian stub into a real environment. Prefer
   `examples/avian-3d/inverted_pendulum.rs` for first scope, or `examples/avian-2d/lunar_lander.rs`
   if contact-heavy behavior is more important.
3. Add an optional `avian3d` feature and keep it out of default/core dependencies.
4. Implement headless physics first: spawn bodies, apply actions, step fixed time, read
   observations, compute reward/status, reset deterministically.
5. Add a deterministic smoke test for finite observations/rewards and reset behavior.
6. Add render sync only after headless behavior is stable.
7. Add metrics and a run manifest.
8. Record an ADR with version, feature set, task, throughput, failure modes, and remaining risks.
9. Implement the same task in Rapier.
10. Compare implementation complexity, throughput, determinism, contact/query ergonomics, and
    maintenance risk.
11. Extract common traits or helpers only if both implementations justify them.

### Technology Stack Recommendations

Current stack recommendation:

- **Bevy:** stay on `0.18.1` stable for implementation.
- **Avian:** use `avian3d 0.6.1` or `avian2d 0.6.1` for the first physics spike.
- **Rapier:** keep `bevy_rapier3d 0.34.0` as fallback/comparison.
- **Core crate:** no mandatory physics dependency.
- **Rendering:** keep `render` plus explicit display backend features.
- **Testing:** use `cargo test`, targeted examples, and fixed-step app tests.
- **Benchmarking:** use optimized smoke/benchmark lanes; do not judge physics throughput from
  debug-only runs.

### Skill Development Requirements

Before or during the first spike, document the local conventions for:

- Bevy `MessageReader`/`MessageWriter` policy loops,
- fixed-timestep stepping and test control,
- `GymSet` ordering,
- environment reset semantics,
- unit scaling,
- collision event enablement,
- contact-state vs contact-event differences,
- feature-gated example compilation,
- run metadata and metric logging.

This should live near the example docs, not only in the research report.

### Success Metrics and KPIs

The first physics spike is successful only if it produces measurable evidence:

- compiles with a targeted feature set,
- runs headless without render dependencies,
- produces finite observations and rewards for at least 1000 fixed steps,
- resets deterministically under a fixed seed,
- logs backend/version/timestep/scale/seed metadata,
- records steps/sec in release or optimized profile,
- emits contact metrics if contact is part of the task,
- renders without changing simulation behavior,
- has a short ADR documenting what worked, what failed, and whether Rapier comparison is still
  needed.

## Research Synthesis and Final Recommendations

### Executive Summary

`bevy-gym` already has the right core shape for physics-backed reinforcement learning: a small Bevy
plugin, a fixed-step RL schedule, message-driven policy integration, parallel per-environment
stepping, and optional rendering separated from headless training. Physics should extend that
architecture, not replace it.

The current Bevy physics landscape has two realistic backend candidates. Avian is the better
architectural fit because it is explicitly ECS-driven and designed for Bevy-native use. Rapier
remains the stronger fallback when maturity, broad engine history, scene queries, and multiple
physics contexts matter. For this repository, the practical answer is not "choose one forever"; it
is to build one honest Avian example, measure it, then build a Rapier mirror only if the evidence
requires a fallback or comparison.

The decision should be made on observed repo-local behavior: deterministic reset behavior, stable
fixed-step observations, contact/query ergonomics, feature-gated compile cost, headless throughput,
render/debug usability, and maintenance burden. Until that evidence exists, a generic physics
abstraction would be speculative.

**Key technical findings:**

- Bevy 0.18.1 is the stable implementation baseline for this repo; Bevy 0.19 release candidates
  should remain migration-spike material.
- Avian 0.6.1 and `bevy_rapier3d` 0.34.0 both target Bevy 0.18-era APIs, so both are viable
  candidates for local spikes.
- Bevy 0.18 uses `MessageReader`/`MessageWriter` for buffered system-to-system communication; local
  quick-start documentation still contains stale `EventReader` terminology.
- Physics simulation should stay in fixed-step systems; rendering should read the latest observation
  and must not drive simulation correctness.
- Collision/contact behavior must be tested explicitly because both Avian and Rapier require opt-in
  event configuration in common cases.
- Rapier performance must be judged in release/profile-optimized builds; debug-build physics
  performance can be misleading.
- Multi-environment physics can be modeled with spatial/layer isolation, separate contexts, or
  separate app/world lanes, but the right approach must be proven by a spike.

**Top recommendations:**

1. Keep core `bevy-gym` physics-free and preserve `default = []`.
2. Add Avian as an optional backend feature only when the first real Avian example is implemented.
3. Implement `inverted_pendulum` first unless contact-heavy behavior is the immediate target, in
   which case use `lunar_lander`.
4. Add fixed-step headless smoke tests before rendering polish.
5. Record a small ADR after the Avian spike, then decide whether a Rapier mirror is still necessary.

### 1. Technical Research Introduction and Methodology

This research was scoped around a concrete repo question: how should `bevy-gym` adopt physics
without damaging its current headless, ECS-driven RL architecture? The methodology combined:

- local inspection of `Cargo.toml`, CI, plugin docs, render docs, core systems, message types, and
  examples;
- source verification from Bevy, Avian, Rapier, Cargo, docs.rs, and crates.io;
- version compatibility checks for Bevy, Avian, Rapier, and related ecosystem crates;
- architectural evaluation against `bevy-gym`'s current schedule and feature model;
- implementation risk analysis focused on RL-specific correctness.

The research intentionally prioritizes primary and near-primary sources over general blog guidance.
Where recommendations depend on local performance or physics behavior, this report marks them as
spike criteria rather than settled facts.

_Source:_

- Bevy fixed timestep example: <https://bevy.org/examples/movement/physics-in-fixed-timestep/>
- Bevy 0.18 release notes: <https://bevy.org/news/bevy-0-18/>
- Avian crate docs: <https://docs.rs/avian3d/latest/avian3d/>
- Avian repository: <https://github.com/avianphysics/avian>
- Rapier Bevy plugin docs: <https://rapier.rs/docs/user_guides/bevy_plugin/getting_started_bevy/>
- Local `Cargo.toml`, `src/plugin.rs`, `src/systems/step.rs`, `docs/plugins/bevy_gym_plugin.md`,
  `docs/plugins/gym_render.md`

### 2. Technical Landscape and Architecture Analysis

The current architecture should be preserved:

- `BevyGymPlugin<E>` owns environment spawning and RL schedule registration.
- Environment entities carry `EnvironmentComponent<E>`, `PendingAction<E>`, `CurrentObservation<E>`,
  `EnvStats`, and `EnvId`.
- `GymSet::Step`, `GymSet::AutoReset`, and `GymSet::ManualReset` form the fixed-step RL order.
- Buffered Bevy messages carry action requests, experiences, and episode endings.
- Optional render sync reads `CurrentObservation` in render-frame schedules.

Physics should enter at the environment/example layer. The core crate should not know whether a
particular environment uses Avian, Rapier, hand-written dynamics, or an external simulator. That
separation protects crate compile time, preserves headless training, and keeps backend-specific
failure modes out of the library surface.

Avian fits Bevy's ECS model closely: its docs describe it as ECS-driven, and the repository
emphasizes a Bevy-native design rather than a wrapper around a separate engine. Rapier's Bevy plugin
provides a mature physics bridge with explicit docs for contexts, scene queries, active events,
common mistakes, and performance tuning. This makes Avian the preferred first spike and Rapier the
right comparison path.

_Source:_

- Avian design/repository: <https://github.com/avianphysics/avian>
- Rapier multiple contexts: <https://rapier.rs/docs/user_guides/bevy_plugin/multiple_contexts/>
- Rapier common mistakes: <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Bevy ECS docs: <https://docs.rs/bevy_ecs/latest/bevy_ecs/>

### 3. Implementation Approaches and Best Practices

The first implementation should be explicit:

1. Add a backend feature only for the concrete example.
2. Spawn physics bodies and colliders from the example, not from core.
3. Apply actions in a fixed-step system.
4. Extract observations after physics has advanced.
5. Compute reward and termination from a stable state snapshot.
6. Reset deterministically with a recorded seed.
7. Emit run metadata and metrics.

Avoid introducing a `PhysicsBackend` trait at the start. The first Avian example will expose
concrete needs: where actions are applied, how observations are extracted, whether collision state
is needed, how reset should clear bodies, and how render sync should visualize simulation state. A
shared abstraction should only be extracted after a Rapier mirror confirms that duplication is real
and stable.

Common implementation mistakes to avoid:

- using render-scale units as physics-scale units;
- relying on collision events without enabling them;
- reading transient collision events when persistent contact state is required;
- measuring debug builds and treating poor throughput as backend failure;
- coupling policy systems to `Update` rather than `FixedUpdate` state;
- letting render systems mutate simulation state;
- hiding backend dependencies behind default features.

_Source:_

- Avian collision docs: <https://docs.rs/avian3d/latest/avian3d/collision/index.html>
- Avian schedule docs: <https://docs.rs/avian3d/latest/avian3d/schedule/index.html>
- Rapier active events: <https://rapier.rs/docs/user_guides/bevy_plugin/collider_active_events/>
- Rapier scene queries: <https://rapier.rs/docs/user_guides/bevy_plugin/scene_queries/>
- Cargo features: <https://doc.rust-lang.org/cargo/reference/features.html>

### 4. Technology Stack Evolution and Current Trends

The implementation baseline should remain conservative:

| Layer        | Recommendation                                        | Rationale                                                |
| ------------ | ----------------------------------------------------- | -------------------------------------------------------- |
| Bevy         | `0.18.1` stable                                       | current local lockfile and plugin compatibility baseline |
| Core crate   | minimal Bevy features                                 | protects headless RL throughput                          |
| Avian        | `avian3d 0.6.1` or `avian2d 0.6.1`                    | Bevy-native ECS physics candidate                        |
| Rapier       | `bevy_rapier3d 0.34.0`                                | mature fallback/comparison backend                       |
| Rendering    | existing `render`, `x11`, `wayland` features          | keeps visual dependencies optional                       |
| Testing      | `cargo test`, targeted examples, fixed-step app tests | validates schedule and feature matrix                    |
| Benchmarking | release/profile optimized smoke or bench lanes        | prevents debug-performance false negatives               |

Bevy 0.19 release candidates exist, but this report does not recommend moving the implementation
baseline until 0.19 final and backend compatibility are verified. The physics spike should reduce
variables, not add a Bevy migration at the same time.

_Source:_

- Bevy crates.io metadata: <https://crates.io/api/v1/crates/bevy>
- Avian docs.rs: <https://docs.rs/avian3d/latest/avian3d/>
- bevy_rapier3d docs.rs: <https://docs.rs/crate/bevy_rapier3d/latest>
- Local `Cargo.lock`

### 5. Integration and Interoperability Patterns

The idiomatic integration pattern is message-driven and schedule-driven:

- `ActionRequestEvent` requests policy output after step/reset.
- Policy systems write actions into `PendingAction<E>`.
- Step systems consume `PendingAction<E>` once.
- `ExperienceEvent` carries transition data to learning/logging systems.
- `EpisodeEndEvent` carries episode-level metrics.
- Policy, learning, and logging systems run after `GymSet::ManualReset`.

Physics should not alter this contract. The physics environment may contain backend-specific
components and resources, but the policy-facing integration should remain the existing `Environment`
trait plus Bevy messages/components. This preserves interoperability with `ember-rl`, local docs,
and the current CartPole example.

For backend-specific integration:

- Avian systems should be scheduled relative to Avian physics sets where needed.
- Rapier contexts should be used explicitly if multiple isolated worlds are needed.
- Collision layers/groups should be part of the environment fixture.
- Contact state and event streams should be separate concepts in the environment API.

_Source:_

- Local `src/events.rs`, `src/systems/step.rs`, `docs/plugins/bevy_gym_plugin.md`
- Avian schedule docs: <https://docs.rs/avian3d/latest/avian3d/schedule/index.html>
- Rapier multiple contexts: <https://rapier.rs/docs/user_guides/bevy_plugin/multiple_contexts/>

### 6. Performance and Scalability Analysis

The current `bevy-gym` architecture already parallelizes independent environment stepping with
`par_iter_mut()` and serializes message writing afterward. That pattern is appropriate for pure Rust
environments. Physics-backed environments need a spike because backend simulation may centralize
state or schedule work differently.

Performance should be measured across:

- number of environments: 1, 4, 16, 64;
- render off vs render on;
- debug vs release/profile-optimized builds;
- Avian vs Rapier;
- one shared physics world vs isolated contexts/layers where applicable;
- collision-heavy vs non-contact tasks.

The most important performance KPI is not raw engine speed in isolation. It is stable headless
training throughput under the `bevy-gym` schedule with reproducible results and low maintenance
cost.

_Source:_

- Rapier common mistakes and profile guidance:
  <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Cargo bench docs: <https://doc.rust-lang.org/cargo/commands/cargo-bench.html>
- Local `src/systems/step.rs`

### 7. Security and Compliance Considerations

This research did not identify special regulatory requirements for Bevy physics in `bevy-gym`. The
relevant engineering controls are dependency, feature, and CI discipline:

- keep default features empty;
- keep backend dependencies optional;
- avoid adding network/runtime services to physics examples;
- compile each feature lane explicitly;
- keep generated run metadata free of secrets;
- keep CI publish behavior guarded by existing token scoping;
- avoid broad dependency activation through convenience features.

Security impact is mainly supply-chain and build-surface management. Avian and Bevy use dual
MIT/Apache-style licensing in their public docs/repositories; Rapier's Bevy plugin is Apache-2.0.
License compatibility should still be checked before publication if new backend dependencies are
added.

_Source:_

- Cargo feature docs: <https://doc.rust-lang.org/cargo/reference/features.html>
- Avian repository license section: <https://github.com/avianphysics/avian>
- bevy_rapier repository: <https://github.com/dimforge/bevy_rapier>
- Local `.github/workflows/ci.yml`

### 8. Strategic Technical Recommendations

The strategic decision framework is:

| Question                       | Decision rule                                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Should core depend on physics? | No, not until multiple examples prove a stable abstraction is needed.                                                      |
| Which backend first?           | Avian, because the architecture is Bevy-native and ECS-driven.                                                             |
| Is Rapier still valuable?      | Yes, as a fallback and comparison backend after Avian is measured.                                                         |
| Which task first?              | `inverted_pendulum` for minimal 3D control; `lunar_lander` for contact-heavy 2D validation.                                |
| When to abstract?              | Only after Avian and Rapier implementations duplicate stable concepts.                                                     |
| What proves success?           | Deterministic fixed-step smoke, finite state, reset reproducibility, contact assertions, and optimized throughput metrics. |

The near-term technical investment should be a real example and test harness, not broad framework
design.

### 9. Implementation Roadmap and Risk Assessment

Recommended sequence:

1. Update stale crate quick-start docs from `EventReader` to `MessageReader`.
2. Choose the first physics example target.
3. Add optional Avian dependency and feature behind the example.
4. Implement headless physics stepping.
5. Add deterministic fixed-step smoke tests.
6. Add collision/contact assertions if the task uses contact.
7. Add render sync from observation state only.
8. Add run metadata and metrics.
9. Run release/profile-optimized throughput checks.
10. Write a short ADR.
11. Decide whether to build the Rapier mirror.

Highest risks:

- premature backend abstraction;
- unit-scale mismatch;
- collision events not enabled;
- nondeterministic reset;
- debug-build performance misinterpretation;
- feature matrix rot;
- render/simulation coupling.

These risks are manageable with narrow scope, explicit fixtures, and evidence gates.

### 10. Future Technical Outlook and Innovation Opportunities

Near term, the Bevy physics ecosystem is still moving quickly. Bevy 0.18 is stable today for this
repo, while Bevy 0.19 release candidates should be tracked but not mixed into the first physics
spike. Avian is the more interesting long-term architectural bet because it aligns with Bevy ECS and
plugin patterns. Rapier remains valuable because it is mature, documented, and battle-tested in Rust
physics usage.

The main innovation opportunity for `bevy-gym` is not simply adding physics. It is producing a clean
pattern for high-throughput, headless, reproducible, feature-gated, physics-backed RL environments
in Bevy. If the first spike succeeds, the repo can define a useful idiom for:

- Gym-style Bevy physics examples,
- repeatable fixed-step RL simulation,
- optional visualization without simulation coupling,
- benchmarked backend comparison,
- controlled physics metadata for reproducible training.

### 11. Technical Research Methodology and Source Verification

Source verification covered:

- Bevy official release notes and examples,
- Bevy docs.rs API docs,
- Avian docs.rs and repository documentation,
- Rapier official Bevy plugin docs,
- Cargo book documentation,
- crates.io/API metadata for current crate versions,
- local repository files.

Confidence levels:

- **High:** current local architecture, Cargo feature model, Bevy 0.18 message/fixed-step patterns,
  Avian/Rapier documented API behavior, common mistakes explicitly documented by Rapier.
- **Medium-high:** Avian-first recommendation, because it follows architecture fit but still needs
  repo-local spike proof.
- **Medium:** multi-environment physics isolation strategy, because the right answer depends on
  backend behavior and throughput measurements.
- **Low until measured:** exact steps/sec, determinism under parallel physics, and long-run RL
  stability for each backend.

Research limitations:

- No physics spike was implemented during this research workflow.
- No benchmarks were run.
- No rendered physics scene was visually inspected.
- No final Bevy 0.19 migration assessment was performed.

### 12. Source Verification and Reference Appendix

Primary references:

- Bevy 0.18 release notes: <https://bevy.org/news/bevy-0-18/>
- Bevy fixed timestep example: <https://bevy.org/examples/movement/physics-in-fixed-timestep/>
- Bevy plugin docs: <https://bevy.org/learn/quick-start/getting-started/plugins/>
- Bevy ECS docs: <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- Bevy `Time<Fixed>` docs: <https://docs.rs/bevy/latest/bevy/prelude/struct.Time.html>
- Avian crate docs: <https://docs.rs/avian3d/latest/avian3d/>
- Avian repository: <https://github.com/avianphysics/avian>
- Avian collision docs: <https://docs.rs/avian3d/latest/avian3d/collision/index.html>
- Avian schedule docs: <https://docs.rs/avian3d/latest/avian3d/schedule/index.html>
- Avian diagnostics docs: <https://docs.rs/avian3d/latest/avian3d/diagnostics/index.html>
- Rapier Bevy getting started:
  <https://rapier.rs/docs/user_guides/bevy_plugin/getting_started_bevy/>
- Rapier multiple contexts: <https://rapier.rs/docs/user_guides/bevy_plugin/multiple_contexts/>
- Rapier common mistakes: <https://rapier.rs/docs/user_guides/bevy_plugin/common_mistakes/>
- Rapier active events: <https://rapier.rs/docs/user_guides/bevy_plugin/collider_active_events/>
- Rapier advanced collision detection:
  <https://rapier.rs/docs/user_guides/bevy_plugin/advanced_collision_detection/>
- Cargo features: <https://doc.rust-lang.org/cargo/reference/features.html>
- Cargo test: <https://doc.rust-lang.org/cargo/commands/cargo-test.html>
- Cargo bench: <https://doc.rust-lang.org/cargo/commands/cargo-bench.html>

Local references:

- `Cargo.toml`
- `Cargo.lock`
- `.github/workflows/ci.yml`
- `src/plugin.rs`
- `src/components.rs`
- `src/events.rs`
- `src/systems/step.rs`
- `src/systems/reset.rs`
- `src/render.rs`
- `docs/plugins/bevy_gym_plugin.md`
- `docs/plugins/gym_render.md`
- `docs/examples/cartpole.md`
- `examples/classic-control/cart_pole.rs`
- `examples/avian-3d/inverted_pendulum.rs`
- `examples/avian-2d/lunar_lander.rs`

### Technical Research Conclusion

The research answer is conditional but actionable: adopt physics through a narrow Avian-first spike,
not a core rewrite. Preserve `bevy-gym`'s current headless fixed-step architecture, add backend
dependencies only where needed, and let measurable examples drive abstraction. Rapier should stay in
the plan as a comparison and fallback, but the first implementation should test whether the
Bevy-native Avian model can satisfy `bevy-gym`'s RL requirements with less integration friction.

**Technical Research Completion Date:** 2026-06-12 **Research Period:** current comprehensive
technical analysis **Source Verification:** current source-backed verification with local repo
inspection **Technical Confidence Level:** high for architecture direction, medium-high for backend
priority, pending implementation evidence for performance and determinism
