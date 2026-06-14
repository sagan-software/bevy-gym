---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - ref/bevy/examples
  - ref/bevy/examples/README.md
  - ref/bevy/Cargo.toml
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Bevy examples best practices and techniques'
research_goals: 'Review all Bevy examples in ref/bevy/examples and compile reusable best practices and techniques for bevy-gym.'
user_name: 'Sagan'
date: '2026-06-12'
web_research_enabled: true
source_verification: true
---

# Bevy Examples Best Practices and Techniques

**Date:** 2026-06-12
**Author:** Sagan
**Research Type:** technical
**Local Bevy baseline:** 0.18.1
**Local Bevy ref:** `ref/bevy` at `f667c28` / `latest`

---

## Research Overview

This report reviews the local Bevy examples tree at `ref/bevy/examples` to extract practices worth
reusing in `bevy-gym`. The review combined:

- a structural scan of all 375 Rust example files under `ref/bevy/examples`,
- focused reads of representative examples across application setup, ECS, messages, observers,
  states, timing, assets, input, UI, rendering, diagnostics, stress tests, and async work,
- current public Bevy sources for context, especially the official examples README, Bevy ECS docs,
  docs.rs API docs, and Bevy 0.17 to 0.18 migration notes.

The local source is the authority for code patterns. The public sources are used to validate
terminology and version-sensitive guidance.

## Technical Research Scope Confirmation

**Research Topic:** Bevy examples best practices and techniques

**Research Goals:** Review all Bevy examples in `ref/bevy/examples` and compile a practical list of
techniques to use in `bevy-gym`.

**Technical Research Scope:**

- Architecture Analysis: ECS shape, plugins, schedules, states, feature boundaries.
- Implementation Approaches: idiomatic systems, messages, observers, resources, queries, asset
  readiness, timing, input.
- Technology Stack: Bevy 0.18.1 local examples, Bevy ECS, Bevy app/plugins, Bevy
  asset/render/UI/input modules.
- Integration Patterns: headless execution, custom runners, async tasks, custom assets, diagnostics,
  optional rendering.
- Performance Considerations: fixed timestep, parallel scheduling, nonblocking tasks, stress
  profiles, diagnostics.

**Research Methodology:**

- Local examples scanned as executable source material.
- Critical patterns verified with representative file reads.
- Current public Bevy documentation checked for version and terminology alignment.

**Scope Confirmed:** 2026-06-12

---

## Coverage Inventory

The examples tree contains 375 Rust examples and about 74k source lines. Largest local categories:

| Category          | Rust files |
| ----------------- | ---------: |
| `3d`              |         63 |
| `ui`              |         47 |
| `ecs`             |         31 |
| `2d`              |         26 |
| `shader`          |         17 |
| `stress_tests`    |         16 |
| `app`             |         15 |
| `asset`           |         14 |
| `animation`       |         13 |
| `input`           |         13 |
| `window`          |         12 |
| `reflection`      |         10 |
| `shader_advanced` |         10 |
| `camera`          |          9 |
| `gltf`            |          9 |
| `games`           |          7 |
| `audio`           |          7 |
| `math`            |          6 |

Common patterns across all Rust examples:

| Pattern                       | Files | Meaning                                                    |
| ----------------------------- | ----: | ---------------------------------------------------------- |
| `add_systems`                 |   360 | Examples are schedule-driven first.                        |
| `DefaultPlugins`              |   343 | Full app examples usually opt into standard Bevy plugins.  |
| `commands.spawn`              |   313 | Entity composition is the default construction style.      |
| `AssetServer`                 |   189 | Asset handles and async loading are widespread.            |
| `Assets<T>`                   |   187 | Runtime asset stores are the normal mutation surface.      |
| Query filters                 |   181 | `With`, `Without`, `Changed`, `Added` keep systems narrow. |
| `#[derive(Component)]`        |   168 | Behavior is modeled through typed components.              |
| `ButtonInput` / input helpers |   132 | Input is read as state and edges.                          |
| `#[derive(Resource)]`         |   125 | Shared app state is typed as resources.                    |
| Time delta usage              |   110 | Frame-rate-independent behavior is expected.               |
| `Single<T>`                   |    96 | Singleton ECS access is explicit where expected.           |
| `children![]`                 |    73 | Entity hierarchies are composed declaratively.             |
| custom materials/render paths |    54 | Rendering extensions are plugin-backed.                    |
| `Local<T>`                    |    49 | Per-system state is preferred for small local memory.      |
| messages                      |    48 | Buffered communication uses Bevy messages.                 |
| observers                     |    41 | Triggered/lifecycle reactions use observers.               |
| `run_if`                      |    36 | System execution is gated declaratively.                   |
| `chain`                       |    35 | Required ordering is explicit and local.                   |
| timers                        |    30 | Timed behavior is data, usually resources/components.      |
| custom plugins                |    27 | Reusable behavior is packaged as plugins.                  |
| diagnostics                   |    23 | Runtime measurement is first-class.                        |
| states                        |    16 | Higher-level app modes use `States`.                       |
| fixed update                  |    12 | Deterministic or physics-like logic uses fixed schedules.  |

---

## Best Practices To Use

### 1. Treat the examples as versioned source, not generic snippets

The official examples README warns that Bevy `main` can differ significantly from crates.io releases
and recommends using release-matched examples. The local checkout is Bevy `0.18.1`, so use these
examples as Bevy 0.18 examples, not future-main guidance.

**Use in `bevy-gym`:** keep the stable Bevy 0.18 baseline unless doing an explicit migration spike.

**Evidence:** `ref/bevy/examples/README.md:1-25`, `ref/bevy/Cargo.toml:1-18`

### 2. Keep the core app/plugin surface minimal and feature-gated

Bevy examples distinguish full app examples from headless/minimal setups. `headless.rs` shows
`ScheduleRunnerPlugin` and disabled defaults for non-windowed execution; `without_winit.rs` disables
`WinitPlugin`; `no_renderer.rs` keeps a window but disables GPU backend creation for tests/CI.

**Use in `bevy-gym`:** keep the library headless by default. Put render, window, inspector, capture,
and BRP/MCP tooling behind features or examples.

**Evidence:** `ref/bevy/examples/app/headless.rs:1-45`,
`ref/bevy/examples/app/without_winit.rs:1-13`, `ref/bevy/examples/app/no_renderer.rs:1-24`

### 3. Package reusable behavior as small plugins

The plugin example says plugins should be scoped sets of components, resources, and systems,
generally smaller in scope. Plugin groups are for sets of plugins registered together and can be
modified by disabling or inserting plugins around others.

**Use in `bevy-gym`:** prefer small plugins such as `GymCorePlugin`, `GymStatsPlugin`,
`GymRenderPlugin`, and dev-only tooling plugins rather than one large plugin that always enables
everything.

**Evidence:** `ref/bevy/examples/app/plugin.rs:1-39`, `ref/bevy/examples/app/plugin_group.rs:1-34`

### 4. Model behavior with typed components and resources

The ECS guide frames components as scoped pieces of functionality, resources as shared global data,
and systems as logic over components/resources. It uses typed components for player data and typed
resources for game state/rules.

**Use in `bevy-gym`:** represent environment state, action buffers, rewards, done flags, episode
counters, observation caches, and curriculum parameters as typed components/resources. Avoid
stringly typed or monolithic world state.

**Evidence:** `ref/bevy/examples/ecs/ecs_guide.rs:35-87`

### 5. Use `Commands` for deferred world mutation; reserve exclusive systems

The ECS guide explains that normal systems run in parallel and should queue world mutation through
`Commands`. Exclusive systems provide immediate world access but block parallel execution and should
be avoided unless necessary.

**Use in `bevy-gym`:** spawn/reset/despawn environments through commands or scheduled state
transitions. Use exclusive systems only for narrow test harness or integration cases that genuinely
need `&mut World`.

**Evidence:** `ref/bevy/examples/ecs/ecs_guide.rs:214-244`

### 6. Order only the systems that need order

Bevy examples rely on parallel scheduling by default, then use `chain`, `before`/`after`, and
`SystemSet` only where sequencing matters. The ECS guide demonstrates ordered sets for
before/round/after phases.

**Use in `bevy-gym`:** define explicit sets for action collection, environment stepping,
reward/experience emission, reset, and rendering sync. Do not serialize unrelated systems.

**Evidence:** `ref/bevy/examples/ecs/ecs_guide.rs:283-365`,
`ref/bevy/examples/ecs/message.rs:119-142`

### 7. Use messages for buffered, multi-reader communication

The message example uses `#[derive(Message)]`, `add_message`, `MessageWriter`, `MessageReader`, and
`MessageMutator`. It shows that writers should run before readers when same-frame processing
matters, often via `chain`.

**Use in `bevy-gym`:** keep action requests, experience records, and episode-end notifications as
Bevy messages. Use `MessageReader`/`MessageWriter` terminology in docs and examples.

**Evidence:** `ref/bevy/examples/ecs/message.rs:7-20`, `ref/bevy/examples/ecs/message.rs:37-87`,
`ref/bevy/examples/ecs/message.rs:110-142`

### 8. Use `ParamSet` or `Local<MessageCursor>` for same-message read/write

Bevy rejects overlapping `MessageReader<T>` and `MessageWriter<T>` borrows in one system. The
examples show two remedies: temporally separated access through `ParamSet`, or manual cursor state
through `Local<MessageCursor<T>>` and `ResMut<Messages<T>>`.

**Use in `bevy-gym`:** if an adapter both consumes and re-emits the same message type, split systems
first. If one system is materially cleaner, use `ParamSet`; use manual cursors only when you need
explicit queue control.

**Evidence:** `ref/bevy/examples/ecs/send_and_receive_messages.rs:1-18`,
`ref/bevy/examples/ecs/send_and_receive_messages.rs:107-164`

### 9. Use observers for triggered reactions and component lifecycle, not routine dataflow

Observer examples show `On<T>`, `commands.trigger`, entity-targeted events, and lifecycle observers
for `Add`/`Remove`. They are good for immediate reactions, cascades, picking, UI activation, and
component lifecycle indexing.

**Use in `bevy-gym`:** observers are a good fit for dev UI, pointer interactions, debug overlays, or
maintaining indexes on component add/remove. Routine RL stepping should remain
schedule/message-driven for determinism.

**Evidence:** `ref/bevy/examples/ecs/observers.rs:16-39`,
`ref/bevy/examples/ecs/observers.rs:61-74`, `ref/bevy/examples/ecs/observers.rs:93-116`,
`ref/bevy/examples/ecs/observers.rs:119-153`

### 10. Gate systems declaratively with run conditions

Run conditions are normal read-only systems or closures returning `bool`, and can be composed with
`and`, `or`, and `not`. Examples use them for resource existence, input, elapsed time, and states.

**Use in `bevy-gym`:** gate policy stepping on training mode, render sync on render feature/state,
reset cleanup on state, and debug sampling on dev configuration.

**Evidence:** `ref/bevy/examples/ecs/run_conditions.rs:11-52`,
`ref/bevy/examples/ecs/run_conditions.rs:61-89`

### 11. Use change detection intentionally

`Changed<T>`, `Added<T>`, `Ref<T>`, and `is_changed` let systems react only to data changes. The
examples warn that mutable dereference marks a value changed even if equal, and show `set_if_neq` to
avoid false change detection.

**Use in `bevy-gym`:** use `Changed` filters for render sync, UI labels, debug visualizers, and
statistics displays. Use `set_if_neq` for resources/components where spurious changes cause work.

**Evidence:** `ref/bevy/examples/ecs/change_detection.rs:35-45`,
`ref/bevy/examples/ecs/change_detection.rs:71-105`

### 12. Use custom query types when tuple queries get brittle

The custom query example shows `#[derive(QueryData)]` and `#[derive(QueryFilter)]` to replace large
tuple queries with named, reusable query shapes. This avoids destructuring maintenance and bypasses
tuple component limits.

**Use in `bevy-gym`:** define custom query data for environment stepping if the tuple grows beyond a
few components. This keeps training systems readable as environment data expands.

**Evidence:** `ref/bevy/examples/ecs/custom_query_param.rs:1-13`,
`ref/bevy/examples/ecs/custom_query_param.rs:48-125`

### 13. Use `States`, `OnEnter`, and `OnExit` for mode lifecycle

State examples use `init_state`, `NextState`, `OnEnter`, `OnExit`, and `in_state` to separate setup,
teardown, and active logic. Substates exist only while a source state exists, and computed states
derive simpler marker states from more complex source states.

**Use in `bevy-gym`:** use states for environment lifecycle, playback/demo mode, paused/render-only
mode, or benchmark phases. Use `DespawnOnExit` for state-scoped debug/render entities.

**Evidence:** `ref/bevy/examples/state/states.rs:1-27`, `ref/bevy/examples/state/states.rs:94-123`,
`ref/bevy/examples/state/sub_states.rs:21-58`, `ref/bevy/examples/state/computed_states.rs:46-77`,
`ref/bevy/examples/state/computed_states.rs:167-213`

### 14. Put deterministic simulation in fixed schedules

The fixed timestep movement example explains why variable frame deltas are unsuitable for
physics-like simulation. It stores accumulated input, advances physics in `FixedUpdate`, separates
physical state from visual `Transform`, and interpolates visuals after the fixed loop.

**Use in `bevy-gym`:** environment stepping should live in fixed or externally driven ticks. Keep
training state separate from rendering transforms. If rendering is enabled, interpolate only
presentation state.

**Evidence:** `ref/bevy/examples/movement/physics_in_fixed_timestep.rs:20-38`,
`ref/bevy/examples/movement/physics_in_fixed_timestep.rs:54-83`,
`ref/bevy/examples/movement/physics_in_fixed_timestep.rs:100-141`,
`ref/bevy/examples/movement/physics_in_fixed_timestep.rs:370-408`

### 15. Distinguish real, virtual, and fixed time

The virtual time example shows `Time<Real>` for unscaled wall-clock time, `Time<Virtual>` for
pause/speed control, and notes that default `Time` resolves to virtual time in regular schedules and
fixed time in fixed schedules.

**Use in `bevy-gym`:** use fixed time for simulation, real time for diagnostics/wall-clock
throughput, and virtual time only for interactive demos or pause/speed controls.

**Evidence:** `ref/bevy/examples/time/virtual_time.rs:1-31`,
`ref/bevy/examples/time/virtual_time.rs:121-149`, `ref/bevy/examples/time/virtual_time.rs:156-199`

### 16. Treat asset loading as asynchronous

The asset loading example notes that `AssetServer` loads in parallel without blocking; assets appear
later in `Assets<T>`. Folder loads use `LoadedFolder`, and dependency completion can be observed
through `AssetEvent::LoadedWithDependencies`.

**Use in `bevy-gym`:** render examples should not assume textures/models are ready at startup. Use
states or readiness systems before spawning dependent visual content.

**Evidence:** `ref/bevy/examples/asset/asset_loading.rs:18-55`,
`ref/bevy/examples/2d/texture_atlas.rs:32-48`

### 17. Store asset handles in resources/components to keep them alive and addressable

Examples keep handles in resources (`RpgSpriteFolder`, `OneHundredThings`, custom asset `State`) and
use `Assets<T>` to retrieve loaded values.

**Use in `bevy-gym`:** keep policy visualization assets, debug meshes, fonts, and environment render
assets in typed resources. Avoid reloading the same asset path in hot loops.

**Evidence:** `ref/bevy/examples/2d/texture_atlas.rs:29-58`,
`ref/bevy/examples/asset/multi_asset_sync.rs:55-82`,
`ref/bevy/examples/asset/custom_asset.rs:105-158`

### 18. Use load barriers or states for multi-asset readiness

`multi_asset_sync.rs` demonstrates sync polling through a run condition and async readiness through
`AsyncComputeTaskPool`, with a loading state and `OnExit` cleanup.

**Use in `bevy-gym`:** for visual examples with multiple assets, introduce a `Loading` state and
only enter `Ready` once all asset handles are ready. Keep headless training independent of render
asset readiness.

**Evidence:** `ref/bevy/examples/asset/multi_asset_sync.rs:16-42`,
`ref/bevy/examples/asset/multi_asset_sync.rs:92-168`,
`ref/bevy/examples/asset/multi_asset_sync.rs:213-228`

### 19. Use custom assets for scenario/config data

The custom asset example defines `#[derive(Asset, TypePath)]`, implements `AssetLoader`, registers
assets/loaders, and handles loader errors explicitly.

**Use in `bevy-gym`:** if environment scenarios, curricula, or reward configs become data files,
prefer Bevy custom assets over ad hoc file reads in systems.

**Evidence:** `ref/bevy/examples/asset/custom_asset.rs:11-54`,
`ref/bevy/examples/asset/custom_asset.rs:92-121`

### 20. Use loader settings and atlases deliberately

Image examples use `.meta` files or `load_with_settings` to select samplers, and texture atlas
examples use padding/sampling to avoid bleeding.

**Use in `bevy-gym`:** if adding sprites or grid renderers, choose nearest/linear sampling
explicitly and use atlas padding where scaled sprites can bleed.

**Evidence:** `ref/bevy/examples/asset/asset_settings.rs:21-77`,
`ref/bevy/examples/2d/texture_atlas.rs:1-18`, `ref/bevy/examples/2d/texture_atlas.rs:60-95`

### 21. Use embedded assets for bootstrap/loading UI

Embedded assets can be placed into the executable and loaded through `embedded://` paths.

**Use in `bevy-gym`:** only use embedded assets for tiny, always-needed dev UI/bootstrap visuals. Do
not embed large training assets by default.

**Evidence:** `ref/bevy/examples/asset/embedded_asset.rs:1-8`,
`ref/bevy/examples/asset/embedded_asset.rs:22-57`

### 22. Read input as edge/level state, then translate it into domain data

Keyboard examples distinguish `KeyCode` for physical key location and `Key` for character meaning.
Gamepad input is queried through `Gamepad` components and analog values. Fixed timestep movement
accumulates input before simulation.

**Use in `bevy-gym`:** convert human/controller input into typed action components/messages before
stepping environments. Accumulate input before fixed ticks if interactive examples reuse simulation
systems.

**Evidence:** `ref/bevy/examples/input/keyboard_input.rs:12-42`,
`ref/bevy/examples/input/gamepad_input.rs:12-29`,
`ref/bevy/examples/movement/physics_in_fixed_timestep.rs:66-83`

### 23. Convert viewport/screen coordinates through cameras

Viewport examples use `Camera::viewport_to_world_2d`, `Camera::world_to_viewport`,
`GlobalTransform`, and window cursor position. They also clamp dynamic viewports to window bounds.

**Use in `bevy-gym`:** debug picking, click-to-spawn, and observation overlays should convert
through Bevy camera APIs rather than assuming screen coordinates are world coordinates.

**Evidence:** `ref/bevy/examples/2d/2d_viewport_to_world.rs:22-39`,
`ref/bevy/examples/2d/2d_viewport_to_world.rs:79-123`

### 24. Keep UI data binding explicit

The standard widgets example keeps external widget state in a resource, updates widget components
when that resource changes, and uses observers for widget value changes.

**Use in `bevy-gym`:** debug UI should bind to training/render stats resources, not become the
source of training truth. Let UI write commands or messages that update typed resources.

**Evidence:** `ref/bevy/examples/ui/standard_widgets.rs:27-55`,
`ref/bevy/examples/ui/standard_widgets.rs:100-138`,
`ref/bevy/examples/ui/standard_widgets.rs:146-193`

### 25. Use render layers and explicit camera targets for multi-pass visualizations

Render-to-texture examples create a target `Image`, assign a camera `RenderTarget::Image`, use
`Camera.order`, and isolate passes with `RenderLayers`.

**Use in `bevy-gym`:** observation capture, mirrors, minimaps, and render-to-texture examples should
use separate cameras/layers rather than special-casing the main camera.

**Evidence:** `ref/bevy/examples/3d/render_to_texture.rs:24-39`,
`ref/bevy/examples/3d/render_to_texture.rs:48-81`,
`ref/bevy/examples/3d/render_to_texture.rs:86-106`

### 26. Build render assets through `Assets<T>` and wrapper components

3D examples add meshes/materials to `Assets<Mesh>` and `Assets<StandardMaterial>`, then attach
`Mesh3d` and `MeshMaterial3d` components. 2D examples use analogous wrappers. This prevents handle
ambiguity and keeps assets centralized.

**Use in `bevy-gym`:** render examples should spawn visual entities from handles and wrappers, not
store raw mesh/material data in environment components.

**Evidence:** `ref/bevy/examples/3d/3d_scene.rs:13-42`, `ref/bevy/examples/3d/pbr.rs:14-59`

### 27. Use custom materials/shaders through Bevy material plugins

Shader examples define an `Asset + TypePath + AsBindGroup` material, register `MaterialPlugin::<T>`,
and override only the material trait methods needed.

**Use in `bevy-gym`:** use standard materials for most visualization. If observation channels need
special shaders, keep them in a render feature/example and register a narrow material plugin.

**Evidence:** `ref/bevy/examples/shader/shader_material.rs:10-15`,
`ref/bevy/examples/shader/shader_material.rs:42-62`

### 28. Use diagnostics and stress profiles for performance claims

Diagnostics examples register custom metrics and use built-in frame/entity/system diagnostics.
Stress tests explicitly require the `stress-test` profile because dev profile is too slow to
represent production behavior.

**Use in `bevy-gym`:** measure training throughput, step latency, entity count, and render cost with
diagnostics. Run performance comparisons in release or an explicit stress/profiled build, not dev.

**Evidence:** `ref/bevy/examples/diagnostics/custom_diagnostic.rs:10-29`,
`ref/bevy/examples/diagnostics/log_diagnostics.rs:25-54`,
`ref/bevy/examples/stress_tests/README.md:1-12`

### 29. Offload heavy work without blocking the main thread

Async examples use `AsyncComputeTaskPool`, store `Task<CommandQueue>` as a component, poll
completion with `check_ready`, and append returned command queues. They explicitly warn against
blocking poll patterns.

**Use in `bevy-gym`:** background work such as dataset loading, visualization preprocessing, or
expensive evaluation should run off-thread and communicate through tasks/channels/messages. Do not
block the main ECS schedule.

**Evidence:** `ref/bevy/examples/async_tasks/async_compute.rs:1-13`,
`ref/bevy/examples/async_tasks/async_compute.rs:56-117`,
`ref/bevy/examples/async_tasks/async_compute.rs:119-140`

### 30. Treat dynamic ECS and unsafe APIs as advanced tooling only

Stress tests use dynamic component registration, `ComponentDescriptor`, dynamic query builders, and
unsafe pointer access to model extreme ECS workloads. This is useful for engine testing, not normal
app architecture.

**Use in `bevy-gym`:** prefer static Rust component types. Reach for dynamic ECS only for tooling,
testbeds, or reflection-driven editors where the flexibility is worth the complexity.

**Evidence:** `ref/bevy/examples/stress_tests/many_components.rs:15-29`,
`ref/bevy/examples/stress_tests/many_components.rs:79-127`,
`ref/bevy/examples/stress_tests/many_components.rs:129-160`

---

## Techniques By Bevy Area

### Application Setup

- `ScheduleRunnerPlugin::run_once()` for one-shot headless runs.
- `ScheduleRunnerPlugin::run_loop()` for fixed-frequency headless loops.
- `set_runner` for externally driven apps.
- Disable `WinitPlugin` or renderer creation for tests/CI.
- Tune `TaskPoolPlugin` when thread count matters.

### ECS and Scheduling

- Components for per-entity state.
- Resources for shared state and config.
- `Commands` for deferred world changes.
- `SystemSet` plus `configure_sets` for phase ordering.
- `chain` for small ordered groups.
- `before`/`after` for targeted dependencies.
- `Local<T>` for per-system memory.
- `Single<T>` for expected singleton entities.
- `QueryData` / `QueryFilter` for complex reusable queries.

### Communication

- Bevy messages for buffered communication.
- Observers for triggered events and lifecycle.
- `ParamSet` for disjoint borrow sequencing.
- `MessageCursor` only when manual message cursor state is needed.

### State and Lifecycle

- `States` for high-level modes.
- `SubStates` for modes that only exist under a parent state.
- `ComputedStates` for derived marker states.
- `OnEnter` / `OnExit` for setup and teardown.
- `DespawnOnExit` / `DespawnOnEnter` for scoped entities.

### Time and Simulation

- `FixedUpdate` for deterministic simulation.
- `RunFixedMainLoop` before/after fixed loop for input accumulation and visual interpolation.
- `Time<Real>` for wall-clock diagnostics.
- `Time<Virtual>` for pause/speed controls.
- `Timer` resources/components for timed behavior.

### Assets

- `AssetServer` returns handles immediately; loading is async.
- Keep handles in resources/components.
- Use `AssetEvent::LoadedWithDependencies` or load barriers for readiness.
- Use custom assets/loaders for scenario/config data.
- Use `.meta` or `load_with_settings` for loader settings.
- Use embedded assets sparingly for tiny bootstrap assets.

### Input and UI

- `ButtonInput<KeyCode>` for physical key locations.
- `ButtonInput<Key>` for character/symbol meaning.
- Gamepad input lives on `Gamepad` components.
- Convert viewport coordinates through `Camera` APIs.
- Keep UI widgets bound to external resources; do not make widgets the deep source of truth.
- Use observers for widget activation/value changes.

### Rendering and Visualization

- Add meshes/materials to `Assets<T>`.
- Attach render wrapper components like `Mesh3d` / `MeshMaterial3d`.
- Use `RenderLayers` for camera/pass isolation.
- Use explicit `RenderTarget::Image` for render-to-texture.
- Register custom materials with `MaterialPlugin::<T>`.
- Keep advanced render and shader code outside the core headless library.

### Diagnostics and Performance

- Register custom diagnostics for domain metrics.
- Use built-in frame/entity/system diagnostics for runtime visibility.
- Run stress/performance comparisons under `stress-test` or release profiles.
- Use async tasks and readiness polling for background work.

---

## Recommended `bevy-gym` Application

1. Keep `bevy-gym` core headless and stable on Bevy 0.18.
2. Define explicit system sets: action request, action apply, fixed environment step,
   reward/experience emit, reset cleanup, render sync.
3. Use Bevy messages for RL action/experience/episode communication.
4. Use observers only for dev UI, picking, lifecycle indexing, and triggered debug actions.
5. Use `FixedUpdate` or externally driven `app.update()` for training determinism.
6. Keep simulation state separate from render state; render state can lag/interpolate.
7. Use states for demos, benchmark phases, loading screens, and render-only modes.
8. Store assets and handles in typed resources for visualization examples.
9. Add custom diagnostics for steps/sec, episodes/sec, mean reward, reset count, action latency, and
   render frame time.
10. Put render, diagnostics overlays, inspectors, BRP/MCP, and capture behind features or examples.

## Source Register

### Local source

- `ref/bevy/examples/README.md`
- `ref/bevy/Cargo.toml`
- `ref/bevy/examples/app/headless.rs`
- `ref/bevy/examples/app/without_winit.rs`
- `ref/bevy/examples/app/no_renderer.rs`
- `ref/bevy/examples/app/custom_loop.rs`
- `ref/bevy/examples/app/plugin.rs`
- `ref/bevy/examples/app/plugin_group.rs`
- `ref/bevy/examples/app/thread_pool_resources.rs`
- `ref/bevy/examples/ecs/ecs_guide.rs`
- `ref/bevy/examples/ecs/message.rs`
- `ref/bevy/examples/ecs/send_and_receive_messages.rs`
- `ref/bevy/examples/ecs/observers.rs`
- `ref/bevy/examples/ecs/run_conditions.rs`
- `ref/bevy/examples/ecs/change_detection.rs`
- `ref/bevy/examples/ecs/custom_query_param.rs`
- `ref/bevy/examples/state/states.rs`
- `ref/bevy/examples/state/sub_states.rs`
- `ref/bevy/examples/state/computed_states.rs`
- `ref/bevy/examples/movement/physics_in_fixed_timestep.rs`
- `ref/bevy/examples/time/virtual_time.rs`
- `ref/bevy/examples/asset/asset_loading.rs`
- `ref/bevy/examples/asset/multi_asset_sync.rs`
- `ref/bevy/examples/asset/custom_asset.rs`
- `ref/bevy/examples/asset/asset_settings.rs`
- `ref/bevy/examples/asset/embedded_asset.rs`
- `ref/bevy/examples/2d/texture_atlas.rs`
- `ref/bevy/examples/2d/2d_viewport_to_world.rs`
- `ref/bevy/examples/input/keyboard_input.rs`
- `ref/bevy/examples/input/gamepad_input.rs`
- `ref/bevy/examples/ui/standard_widgets.rs`
- `ref/bevy/examples/3d/3d_scene.rs`
- `ref/bevy/examples/3d/pbr.rs`
- `ref/bevy/examples/3d/render_to_texture.rs`
- `ref/bevy/examples/shader/shader_material.rs`
- `ref/bevy/examples/diagnostics/custom_diagnostic.rs`
- `ref/bevy/examples/diagnostics/log_diagnostics.rs`
- `ref/bevy/examples/stress_tests/README.md`
- `ref/bevy/examples/stress_tests/many_components.rs`
- `ref/bevy/examples/async_tasks/async_compute.rs`

### Public sources

- Bevy examples README: <https://github.com/bevyengine/bevy/blob/main/examples/README.md>
- Bevy ECS quick start: <https://bevy.org/learn/quick-start/getting-started/ecs/>
- `bevy_ecs` docs: <https://docs.rs/bevy_ecs/0.18.1/bevy_ecs/>
- Bevy 0.17 to 0.18 migration guide: <https://bevy.org/learn/migration-guides/0-17-to-0-18/>

## Confidence

High for local example patterns, file counts, and Bevy 0.18.1 source behavior because they were
verified from the checked-out `ref/bevy` tree. Medium for future-facing ecosystem implications
because Bevy and plugins move quickly; re-check before dependency upgrades.
