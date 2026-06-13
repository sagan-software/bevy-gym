---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - docs/plugins/bevy_gym_plugin.md
  - docs/plugins/gym_render.md
  - docs/examples/cartpole.md
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Latest Bevy engine practices'
research_goals: 'Research the latest version of Bevy and current best practices for linting, performance, compile times, UI, hot reloading, 2D/3D physics, particles, editors, debugging, unit testing, MCP/AI manipulation, AI controllers, character controllers, input mapping, video recording, and screenshots.'
user_name: 'Sagan'
date: '2026-06-12'
web_research_enabled: true
source_verification: true
---

# Latest Bevy Engine Practices: Technical Research Report

**Date:** 2026-06-12
**Author:** Sagan
**Research Type:** technical
**Primary baseline:** Bevy 0.18.1 stable
**Watch item:** Bevy 0.19.0-rc.3 pre-release

---

## Research Overview

This report researches Bevy as of June 12, 2026, with emphasis on the practical engineering surface needed by `bevy-gym`: fast headless ECS simulation, optional rendering, development tooling, current plugin compatibility, debugging, testing, and AI/remote-control workflows. It uses primary sources where possible: Bevy official docs and release notes, docs.rs crate documentation, crates.io API/version metadata, GitHub release pages, and local project docs.

The most important finding is version discipline. `bevy-gym` is already pinned to the right practical baseline: `bevy = "0.18"` resolves to 0.18.1 in the local lockfile. Crates.io and GitHub show 0.19.0 release candidates, including `0.19.0-rc.3`, but those should be treated as migration-spike material until 0.19 final lands and Bevy-adjacent plugins catch up.

For this repository, the recommended strategy is to keep the library small and feature-gated: preserve the current minimal Bevy dependency for headless RL throughput, keep rendering optional, add tooling only behind dev/example features, and use Bevy Remote Protocol plus MCP adapters as opt-in development surfaces rather than production dependencies.

---

## Table of Contents

1. Executive Summary
2. Version Baseline and Compatibility
3. Best Practices for Bevy Architecture
4. Linting, CI, and Code Quality
5. Compilation Time Strategy
6. Runtime Performance Strategy
7. UI Development
8. Hot Reloading and Live Iteration
9. 2D and 3D Physics
10. Character Controllers
11. Particle Effects and VFX
12. Editors and Authoring Workflow
13. Debugging and Inspection
14. Unit, Integration, and Headless Testing
15. MCP and AI Manipulation
16. AI Controllers and RL Controllers
17. Input Mapping
18. Screenshots and Video Recording
19. Recommendation Matrix
20. Implementation Roadmap for `bevy-gym`
21. Source Register

---

## 1. Executive Summary

### Key Findings

- **Use Bevy 0.18.1 as the stable target.** Official Bevy setup docs show `bevy = "0.18.1"` as the current stable dependency, and the local `Cargo.lock` resolves to 0.18.1. Crates.io currently lists `0.19.0-rc.3` as the newest crate version, but it is a release candidate.
- **The repo is already aligned with Bevy best practice for a library crate.** `Cargo.toml` uses `default-features = false`, enables only `default_app` and `multi_threaded`, and gates rendering behind a `render` feature. Keep that shape.
- **Bevy 0.18 materially improves tool/UI/dev workflows.** Release notes call out standard widget work, automatic directional navigation, first-party camera controllers, cargo feature collections, glTF extension support, and easier screenshot/video recording.
- **Physics remains plugin-led.** Bevy does not ship a full first-party physics engine. Avian is the strongest ECS-native fit for Bevy 0.18; Rapier is the mature fallback when kinematic controllers, joints, and production-proven physics behavior matter more than ECS-native ergonomics.
- **MCP/AI control has a concrete route.** The standard term is **remote ECS control**: Bevy Remote Protocol can inspect and mutate ECS state over JSON-RPC, and MCP adapters such as `bevy_brp_mcp` can expose that surface to AI tools. Keep it dev-only and localhost-bound.
- **For `bevy-gym`, prioritize headless determinism over visual tooling.** Rendering, inspectors, editors, BRP, and capture tools should be optional layers around the core fixed-step RL simulation.

### Top Recommendations

1. **Stay on Bevy 0.18.1 for normal development.** Create a separate `codex/bevy-0-19-spike` or equivalent branch only after 0.19 final lands.
2. **Adopt a two-lane validation matrix:** headless lane (`cargo test`, examples, no default features) and render/tool lane (`--features render` plus smoke examples).
3. **Use Avian first for ECS-native RL physics spikes; keep Rapier as fallback.** Treat active ragdolls and physics-based locomotion as spike risks, not assumptions.
4. **Use Leafwing Input Manager as the stable action-mapping default for Bevy 0.18.** Consider `bevy_enhanced_input` only if its release-candidate status and API direction are acceptable.
5. **Use BRP/MCP only behind explicit dev features.** Register `Reflect` for components you want AI/tools to inspect, never expose BRP transports in a release profile.
6. **Use official screenshot/video APIs first.** Use `bevy_capture` only after compatibility verification; its latest crate is older than Bevy 0.18-era plugin versions.

---

## 2. Version Baseline and Compatibility

### Current Bevy State

| Item | Current finding | Confidence | Source |
|---|---:|---|---|
| Local repo dependency | `bevy = "0.18"`, `default-features = false` | High | `Cargo.toml` |
| Local resolved Bevy | `0.18.1` | High | `Cargo.lock` |
| Stable baseline for this report | `0.18.1` | High | Bevy setup docs and local lockfile |
| Newest crates.io version | `0.19.0-rc.3` | High | crates.io API, checked 2026-06-12 |
| Recommendation | Stable work on 0.18.1; spike 0.19 after final | High | Version + ecosystem compatibility |

Bevy 0.18 was released on January 13, 2026, and official setup documentation currently shows `bevy = "0.18.1"` as the latest dependency example. GitHub release pages show `v0.18.1` and `v0.19.0-rc.3`; crates.io API reports `0.19.0-rc.3` as the newest published crate version. Because `0.19.0-rc.3` is explicitly an RC, it should not be the default for this repo unless the goal is migration testing.

### Ecosystem Version Snapshot

Crate metadata was checked through crates.io API on 2026-06-12 with an explicit User-Agent.

| Area | Crate / tool | Latest observed version | Practical status for Bevy 0.18 |
|---|---|---:|---|
| ECS-native physics | `avian3d` | 0.6.1 | Good Bevy 0.18 candidate |
| Rapier physics | `bevy_rapier3d` | 0.34.0 | Strong fallback, mature controller/joint docs |
| Character control | `bevy-tnua` | 0.31.0 | Good dynamic-controller candidate |
| Input mapping | `leafwing-input-manager` | 0.20.0 | Stable recommendation |
| Input mapping | `bevy_enhanced_input` | 0.26.0-rc.1 | Interesting, but RC |
| Inspection | `bevy-inspector-egui` | 0.36.0 | Good dev-tool fit |
| Particles | `bevy_hanabi` | 0.18.0 | Strong GPU particle fit |
| Schedule debugging | `bevy_mod_debugdump` | 0.15.0 | Good debugging fit |
| MCP / BRP | `bevy_brp_mcp` | 0.20.0-rc.1 | Use 0.18.x line for Bevy 0.18 |
| MCP debugger | `bevy_debugger_mcp` | 0.1.8 | Experimental / likely stale |
| Capture | `bevy_capture` | 0.4.1 | Verify compatibility before adopting |
| Behavior trees | `bevy_behave` | 0.5.0 | Worth evaluating |
| Utility AI | `big-brain` | 0.22.0 | Older; use cautiously |
| Editor | `jackdaw` | 0.4.1 | Real editor candidate, still early |

### Version Policy for This Repo

- Keep `bevy = "0.18"` in library dependencies unless there is a specific migration task.
- Avoid depending on pre-release ecosystem crates in core `bevy-gym`.
- Put tools, inspectors, BRP, particle demos, and editors in examples, dev-dependencies, or feature-gated companion crates.
- Use a migration spike branch for Bevy 0.19 final:
  - `cargo update -p bevy`
  - no-feature `cargo test`
  - `cargo test --features render`
  - render example smoke test
  - BRP/inspector/physics compatibility pass

---

## 3. Best Practices for Bevy Architecture

### ECS and Plugin Structure

Bevy is a data-driven Rust engine built around ECS, schedules, systems, resources, plugins, and explicit app configuration. For `bevy-gym`, the existing architecture is directionally right:

- one environment entity per RL environment,
- fixed-step simulation in `FixedUpdate`,
- explicit `GymSet` ordering,
- events/messages for action requests, experience, and episode ends,
- optional render sync from latest observation,
- headless mode using minimal plugins and schedule runner.

Recommended Bevy structure:

- Use small plugins grouped by feature:
  - `BevyGymPlugin` for simulation,
  - `GymStatsPlugin` for metrics,
  - `GymRenderPlugin` for visual sync,
  - separate dev-only plugins for inspector, BRP, capture, and diagnostics.
- Use typed `SystemSet` labels for ordering.
- Keep policy/learning systems after environment reset/step reconciliation.
- Separate simulation state from presentation state.
- Register reflectable components only when a tool actually needs them.

### Data and Asset Architecture

For game-like projects, prefer data-driven scene construction:

- Use glTF or Blender exports for 3D content.
- Use LDtk/Tiled only where 2D level authoring is the dominant workflow.
- Use RON/TOML/JSON assets for tunable policies, curriculum settings, controller parameters, and scenario definitions.
- Avoid hard-coding authored scene state in systems.

For `bevy-gym`, this translates to:

- environment definitions remain Rust-first for training speed,
- visualizers are optional and observation-driven,
- authoring metadata can be externalized later as curriculum and scenario files.

---

## 4. Linting, CI, and Code Quality

### Recommended Checks

For a Bevy library with optional render support:

```sh
cargo fmt --check
cargo clippy --all-targets --no-default-features -- -D warnings
cargo clippy --all-targets --features render -- -D warnings
cargo test --no-default-features
cargo test --features render
cargo test --doc --all-features
```

For examples:

```sh
cargo run --example cartpole --release
cargo run --example cartpole --features render --release -- --render
```

The local project already documents headless and render paths for CartPole. Keep both paths tested because Bevy render features pull in a much larger dependency and runtime surface than pure ECS scheduling.

### Bevy-Specific Linting Guidance

- Avoid global `allow` blocks for borrow/workflow pain; prefer query decomposition, `ParamSet`, or smaller systems.
- Prefer explicit system ordering over relying on incidental insertion order.
- Keep `default-features = false` for library crates.
- Avoid adding dev tools as normal dependencies.
- Add `Reflect`/`Serialize` only where tooling, save data, or BRP needs it.
- Use feature-gated dev plugins for inspectors, debug cameras, and BRP.

### CI Matrix

Minimum CI matrix:

| Lane | Purpose |
|---|---|
| `--no-default-features` | core headless library correctness |
| `--features render` | optional Bevy rendering support |
| examples build | catches API drift in docs/examples |
| docs build | catches public API documentation regressions |
| release example smoke | catches performance/optimization-only issues |

Optional matrix:

- Linux x11,
- Linux wayland,
- wasm build if web is a target,
- `cargo deny` or equivalent dependency policy,
- screenshot/capture smoke for visual examples.

---

## 5. Compilation Time Strategy

### Official Bevy Guidance

Bevy's setup guide recommends:

- small optimization in `[profile.dev]`,
- high optimization for dependencies in `[profile.dev.package."*"]`,
- dynamic linking for faster iteration,
- alternative linkers such as LLD or mold,
- release profile tuning only for shipping,
- nightly/Cranelift only when the team accepts instability.

Recommended local profile for app/examples, not necessarily library publication:

```toml
[profile.dev]
opt-level = 1

[profile.dev.package."*"]
opt-level = 3
```

For local developer machines:

```sh
cargo run --features bevy/dynamic_linking
```

Do not ship with dynamic linking enabled; use it only for local iteration.

### Repo-Specific Compile Strategy

`bevy-gym` should keep compile time low by design:

- preserve `default-features = false`,
- keep `render` optional,
- avoid adding physics/render/editor crates to the core library,
- move heavy demos into examples or workspace members,
- test minimal and render lanes separately,
- avoid generic public API churn in hot code unless it buys real ergonomics.

### 0.18 Feature Collections

Bevy 0.18 introduced cargo feature collections for common scenarios such as 2D, 3D, and UI. For applications, these are a good way to compile only the engine surface needed by the game. For `bevy-gym`, precise minimal features are still preferable because the crate is a library and its default mode is headless training.

---

## 6. Runtime Performance Strategy

### Headless Simulation

For RL/training workloads:

- use `MinimalPlugins` and `ScheduleRunnerPlugin`,
- run simulation in `FixedUpdate`,
- use virtual time or uncapped runner for throughput,
- keep rendering off by default,
- batch environment entities and use `par_iter_mut()` when environment state is independent,
- emit compact experience events/messages after step computation.

The local docs already follow this shape: `.headless()` sets uncapped tick rate and skips window/rendering setup, while `GymRenderPlugin` syncs visuals separately in `Update`.

### Rendered Performance

For rendered game/demo workloads:

- use release builds for profiling,
- use Bevy diagnostics for frame-time visibility,
- use low-overhead logs in release,
- avoid spawning/despawning every frame when pooling or visibility toggles are enough,
- avoid excessive `Commands` churn inside hot loops,
- use instancing/meshes/material reuse for many repeated objects,
- use visibility/culling and LOD for large worlds,
- keep physics and rendering schedules clear enough to debug.

### Profiling and Diagnostics

Use:

- Bevy diagnostics plugins for FPS/frame-time visibility,
- `bevy_mod_debugdump` to visualize schedules and render graphs,
- `bevy-inspector-egui` for ECS inspection,
- RenderDoc/wgpu tooling when rendering bugs require GPU capture,
- flamegraph/tracing/tracy-style tooling when CPU time is the issue.

For `bevy-gym`, track:

- env steps/sec,
- policy inference time,
- environment step time,
- reset frequency,
- event/message volume,
- render sync time when `render` is enabled.

---

## 7. UI Development

### Bevy UI in 0.18

Bevy 0.18 release notes call out several UI/tooling improvements:

- more standard logical widgets,
- experimental Bevy Feathers tooling widgets,
- automatic directional navigation,
- font variations,
- pick-able text sections,
- UI scroll/parent-position improvements,
- interpolation for colors and layout.

This makes Bevy UI more credible for in-game HUDs, menus, debugging panels, and tool overlays than in older versions.

### Recommended UI Stack

| Need | Recommendation |
|---|---|
| In-game HUD/menu | First-party Bevy UI |
| Gamepad/keyboard menu navigation | Bevy 0.18 automatic directional navigation |
| Dev inspector / quick tools | `bevy-inspector-egui` or `bevy_egui` |
| Dense editor UI | Evaluate Jackdaw, egui, or external tool |
| Full product-like desktop UI | Consider egui/iced/tauri outside game runtime |

For `bevy-gym`, first UI should stay utilitarian:

- live reward/episode/steps/sec overlay,
- pause/speed controls for render examples,
- selected environment inspector,
- capture/screenshot buttons in dev builds.

---

## 8. Hot Reloading and Live Iteration

### Asset Hot Reloading

Use Bevy assets for live-tuned data:

- materials,
- textures,
- glTF scenes,
- RON/TOML/JSON configuration,
- shader files,
- controller tuning parameters.

This gives a safer hot-reload path than runtime code mutation.

### Code Hot Reloading

Bevy and the Rust ecosystem have experimental hotpatching directions, but code hot reload should not be treated as a core dependency for `bevy-gym`. Use it as an app/dev workflow only if the team is comfortable with instability.

### Practical Live-Iteration Pattern

Recommended pattern:

1. Keep simulation systems deterministic and testable.
2. Move tuning knobs into assets/resources.
3. Hot reload assets and config.
4. Use inspector/BRP for live observation and small state edits.
5. Restart app for code changes.

This is more reliable for RL and physics than trying to hot-reload compiled controller logic.

---

## 9. 2D and 3D Physics

### First-Party Status

Bevy does not currently provide a full first-party physics engine. Physics selection is a plugin decision.

### Avian

Avian is the strongest ECS-native physics candidate for Bevy 0.18-era work:

- 2D and 3D crates exist,
- integrates naturally with Bevy ECS,
- supports rigid bodies, colliders, forces, impulses, spatial queries, collision events, and joints,
- is attractive for RL because physics state can remain component-oriented.

Best fit:

- RL environments,
- custom sensors,
- ECS-native controller experimentation,
- procedural worlds,
- component damage systems,
- simple ragdoll/building-block spikes.

Risks:

- active ragdoll and physics-based locomotion remain spike items,
- advanced character-controller workflows may require extra code or companion crates,
- ecosystem maturity is lower than Rapier.

### Rapier

Rapier remains the mature fallback:

- strong documentation for Bevy plugin usage,
- documented joints,
- documented kinematic character controller,
- broader historical use in Rust game projects.

Best fit:

- mature collision/rigid-body needs,
- kinematic character controllers,
- joint-heavy prototypes,
- ragdoll fallback if Avian stalls.

Tradeoff:

- less Bevy-native in feel than Avian,
- integration model can feel more middleware-like.

### Physics Recommendation

For `bevy-gym`:

1. Use Avian first for ECS-native RL physics spikes.
2. Use Rapier when mature KCC/joint behavior is the deciding factor.
3. Do not commit to active ragdoll locomotion without a spike.
4. Keep physics integration out of core `bevy-gym`; use examples or companion crates.

---

## 10. Character Controllers

### Controller Categories

| Controller type | Best use |
|---|---|
| Kinematic controller | FPS/platformer capsule movement, stable authored movement |
| Dynamic controller | physics-influenced movement, slopes, moving platforms |
| Custom RL controller | observations/actions directly drive forces/targets |
| Camera controller | dev navigation and scene inspection |

### Current Options

- **Bevy 0.18 first-party camera controllers:** good for dev cameras, not gameplay character bodies.
- **Rapier KCC:** mature kinematic controller choice.
- **Tnua:** Bevy character controller ecosystem option for dynamic-style movement.
- **Avian custom controller:** best when RL and ECS-native physics need full control.

### Recommendation

For game characters, do not confuse camera controllers with character controllers. For `bevy-gym`, create an explicit controller abstraction:

- observation extraction,
- action decoding,
- force/impulse/target application,
- safety limits,
- telemetry,
- reset handling.

This keeps RL policy control separate from player input and from physics plugin details.

---

## 11. Particle Effects and VFX

### Bevy Native Rendering Direction

Bevy 0.18 improves rendering and visual tooling in areas such as atmospheric scattering, Solari, PBR fixes, fullscreen materials, and screenshot/video capture ergonomics. That makes Bevy more credible for visual demos, but particles are still mostly plugin-led.

### Particle Options

| Need | Recommendation |
|---|---|
| GPU particles | `bevy_hanabi` |
| Simple 2D particles | simple ECS sprites or 2D particle plugin |
| Deterministic RL visual markers | custom ECS particles/lines/gizmos |
| Editor-authored VFX | evaluate per project; do not assume maturity |

`bevy_hanabi` is the best primary candidate for GPU particle effects on Bevy 0.18.

### Recommendation for `bevy-gym`

Use particles sparingly and outside the training lane:

- contact markers,
- reward/penalty pulses,
- sensor rays/gizmos,
- episode end effects in render demos.

Do not add a particle dependency to the core crate.

---

## 12. Editors and Authoring Workflow

### Bevy Editor Reality

Bevy has strong programmatic ECS ergonomics and improving editor-adjacent tooling, but it is not yet Godot/Unity-like for authoring. There are real editor candidates, especially Jackdaw, but they should be treated as early tools.

### Tooling Options

| Tool | Role | Recommendation |
|---|---|---|
| Blender + glTF | 3D authoring | Default for 3D assets |
| LDtk / Tiled | 2D level data | Use for tile/2D games |
| `bevy-inspector-egui` | ECS inspection | Add as dev-only |
| Jackdaw | Bevy editor candidate | Spike, not baseline |
| Bevy Remote Protocol | remote ECS inspection/editing | Dev-only |
| Custom scenario editor | domain-specific authoring | Good long-term for RL curricula |

### Recommendation for `bevy-gym`

The right editor is probably not a general-purpose Bevy editor. For RL, build or script a domain-specific scenario/curriculum authoring layer:

- scenario resources,
- reset distributions,
- obstacle/layout assets,
- controller parameter assets,
- benchmark presets,
- inspector/BRP hooks for live inspection.

Use Jackdaw only if the task is 3D scene layout and the spike proves it helps more than Blender/glTF plus custom config.

---

## 13. Debugging and Inspection

### Core Debugging Stack

Recommended Bevy debugging stack:

- `RUST_LOG` / Bevy logging filters,
- `FrameTimeDiagnosticsPlugin` and log diagnostics,
- first-party gizmos for visual debugging,
- first-party camera controllers for scene navigation,
- `bevy-inspector-egui` for runtime ECS inspection,
- `bevy_mod_debugdump` for schedule/render graph debugging,
- BRP for remote tool access.

### Debugging Practices

- Add debug visualizations as systems behind a `debug_tools` feature.
- Render physics shapes, sensors, reward regions, and contact normals.
- Log episode-level summaries rather than every frame in hot loops.
- Keep system ordering explicit and dump schedules when behavior is surprising.
- Use deterministic seeds for reproducible bugs.

### `bevy-gym` Debug Checklist

- Does headless mode produce stable steps/sec and rewards?
- Do render visuals read only the latest observation?
- Are policy systems scheduled after reset reconciliation?
- Are all environment entities independent enough for parallel stepping?
- Are reset seeds explicit in failing tests?
- Is render-only state excluded from training correctness?

---

## 14. Unit, Integration, and Headless Testing

### Unit Testing Bevy Systems

Common pattern:

```rust
let mut app = App::new();
app.add_plugins(MinimalPlugins);
app.insert_resource(MyResource::default());
app.add_systems(Update, my_system);
app.world_mut().spawn(MyComponent);
app.update();
```

For `bevy-gym`, prefer tests that:

- build an `App`,
- add `BevyGymPlugin`,
- spawn deterministic environments,
- run one or more updates,
- inspect `CurrentObservation`, `EnvStats`, and emitted messages/events.

### FixedUpdate Testing

For fixed-step simulation:

- control time explicitly,
- avoid wall-clock expectations,
- use deterministic seeds,
- test one-env and many-env cases,
- test auto-reset and manual-reset boundaries,
- test render sync separately from simulation correctness.

### Test Layers

| Layer | Purpose |
|---|---|
| pure environment unit tests | validate `rl-traits` behavior |
| Bevy app tests | validate plugin systems/order |
| example smoke tests | validate integration and docs |
| release-mode training smoke | catch throughput/perf regressions |
| render smoke | catch feature-gated Bevy render drift |

---

## 15. MCP and AI Manipulation

### Standard Term

The more precise term for "MCP/AI manipulation" in Bevy is **remote ECS control**: an external tool inspects, queries, spawns, removes, or mutates entities/components through a remote API.

### Bevy Remote Protocol

Bevy 0.18 includes Bevy Remote Protocol documentation. BRP is a JSON-RPC surface for remote control of a Bevy app. It includes built-in methods such as:

- `world.get_components`,
- `world.query`,
- `world.spawn_entity`,
- `world.despawn_entity`,
- `world.insert_components`,
- `world.mutate_components`,
- `world.list_components`,
- resource mutation/listing,
- event triggering,
- registry schema discovery.

This maps directly to AI and MCP workflows: an MCP server can translate tool calls into BRP JSON-RPC requests.

### MCP Options

| Option | Status | Recommendation |
|---|---|---|
| `bevy_brp_mcp` | active crate with 0.18.x versions and newer RCs | Best candidate |
| `bevy_debugger_mcp` | older, experimental | Research only |
| custom BRP MCP server | straightforward if scope is narrow | Good for `bevy-gym` |

### Safety Rules

- Compile BRP/MCP only behind a dev feature.
- Bind to localhost by default.
- Do not expose mutation endpoints in production.
- Register only the components/resources you want tools to see.
- Prefer read-only query tools before mutation tools.
- Add audit logging for AI mutations.

### Recommendation for `bevy-gym`

Build a small BRP/MCP spike with three tools:

1. list environment entities and stats,
2. query one environment observation/state,
3. request reset or set a pending action.

This is enough to prove AI-assisted debugging without giving broad mutation access.

---

## 16. AI Controllers and RL Controllers

### RL Controller Architecture

For this repo, the best AI-controller shape is already implied by the existing plugin:

```text
Observation -> Policy System -> PendingAction -> Environment Step -> ExperienceEvent
```

Best practices:

- keep policy inference as a system scheduled after reset reconciliation,
- separate exploration RNG from environment RNG,
- use fixed tick rate for reproducibility,
- export experience tuples through typed events/messages,
- keep training state as resources,
- keep render state out of training correctness.

### Game AI Plugins

For non-RL game AI:

- behavior trees: evaluate `bevy_behave`,
- utility AI: `big-brain` exists but is older,
- pathfinding/navigation: evaluate current plugins per project,
- simple finite-state machines: often better as plain Bevy state/components.

### Recommendation

For `bevy-gym`, do not add a behavior-tree or utility-AI crate unless there is a specific game-agent use case. RL policy systems and environment components are the primary AI-controller abstraction.

---

## 17. Input Mapping

### Bevy Built-In Input

Bevy's first-party input APIs are good for low-level keyboard, mouse, touch, and gamepad handling. For game action mapping, a higher-level crate is usually better.

### Plugin Options

| Option | Status | Best use |
|---|---|---|
| Bevy built-in input | first-party | low-level input |
| `leafwing-input-manager` 0.20.0 | stable ecosystem option | action mapping |
| `bevy_enhanced_input` 0.26.0-rc.1 | newer / RC | Unreal-style contexts/actions |

### Recommendation

Use `leafwing-input-manager` as the default stable action-mapping layer for Bevy 0.18. Consider `bevy_enhanced_input` for a game/app prototype only if its RC status is acceptable and its context/action model is worth the migration risk.

For `bevy-gym`, keep player input separate from policy input:

- player input maps to debug/manual control actions,
- policy output maps to `PendingAction`,
- replay/eval mode should bypass live input entirely.

---

## 18. Screenshots and Video Recording

### Bevy 0.18 Improvements

Bevy 0.18 release notes include "Easy Screenshot and Video Recording," and the official example set includes a screenshot example. This should be the first place to look before adding external capture dependencies.

### Practical Capture Strategy

| Need | Recommendation |
|---|---|
| Single screenshot | use official Bevy screenshot APIs/examples |
| CI visual snapshot | deterministic camera + screenshot + image diff |
| Offline video | fixed-step render to image sequence, encode with ffmpeg |
| In-app capture | evaluate `bevy_capture` after compatibility check |
| Training video | separate evaluation render from training loop |

For deterministic RL demos:

- run evaluation with fixed seed,
- cap simulation/render timing,
- save frames or screenshots from render examples,
- encode outside the app where possible.

### Recommendation for `bevy-gym`

Add a capture example rather than a core dependency:

```text
examples/capture_cartpole.rs
```

It should run a trained/eval policy, use deterministic seeds, capture screenshots or frames, and leave the core crate unchanged.

---

## 19. Recommendation Matrix

| Area | Default recommendation | Confidence | Notes |
|---|---|---|---|
| Bevy version | 0.18.1 stable | High | 0.19.0-rc.3 is pre-release |
| Core dependency shape | minimal features, no default features | High | already implemented |
| Linting | fmt + clippy no-feature/render matrix | High | add docs/examples checks |
| Compile times | feature-gate, dev opt levels, dynamic linking locally, LLD/mold | High | official setup guidance |
| Runtime perf | headless MinimalPlugins + FixedUpdate + metrics | High | matches repo |
| UI | Bevy UI for HUD, egui/inspector for dev tools | Medium-high | 0.18 improves UI |
| Hot reload | assets/config first, code hotpatch experimental | Medium | keep restart loop for code |
| Physics | Avian first, Rapier fallback | Medium-high | spike active ragdoll |
| Character controller | Rapier KCC/Tnua/custom | Medium | depends on game feel |
| Particles | Hanabi for GPU particles | Medium-high | keep out of core |
| Editor | Blender/glTF + inspector; Jackdaw spike only | Medium | Jackdaw is real but early |
| Debugging | diagnostics + inspector + debugdump + gizmos | High | dev-feature only |
| Testing | App-based headless tests + render smoke | High | essential for Bevy upgrades |
| MCP/AI | BRP + `bevy_brp_mcp` dev-only | Medium-high | exact integration needs spike |
| AI controllers | RL system pipeline first | High | matches `bevy-gym` design |
| Input mapping | Leafwing stable default | Medium-high | enhanced input is RC |
| Capture | official screenshot/video first | Medium-high | external capture needs compat check |

---

## 20. Implementation Roadmap for `bevy-gym`

### Phase 1: Harden Current Baseline

- Keep Bevy 0.18.1.
- Add/confirm CI lanes:
  - `cargo test --no-default-features`,
  - `cargo test --features render`,
  - `cargo clippy --all-targets --no-default-features -- -D warnings`,
  - `cargo clippy --all-targets --features render -- -D warnings`.
- Add an example smoke command for headless CartPole.
- Add a render smoke command behind `render`.

### Phase 2: Dev Tooling Feature

Create a dev-only feature or example module for:

- diagnostics,
- inspector,
- debug camera,
- schedule dump,
- screenshot capture.

Keep these out of normal dependencies.

### Phase 3: BRP/MCP Spike

Add a separate example:

```text
examples/brp_debug.rs
```

Scope:

- register reflectable components for environment stats and observation wrapper,
- expose localhost-only BRP,
- prove read-only query,
- prove reset request mutation,
- document safety limits.

### Phase 4: Physics Spike

Create separate Avian and Rapier examples rather than mixing both into core:

```text
examples/avian_cartpole_like.rs
examples/rapier_character_control.rs
```

Success criteria:

- deterministic reset,
- stable fixed-step behavior,
- useful telemetry,
- no render dependency in training path,
- clear controller API.

### Phase 5: 0.19 Migration Spike

Only after Bevy 0.19 final:

- create branch,
- update Bevy,
- update plugin compatibility table,
- run full no-feature/render/test/example matrix,
- note API changes,
- decide whether migration buys enough value.

---

## 21. Source Register

### Primary Bevy Sources

- Bevy 0.18 release notes: https://bevy.org/news/bevy-0-18/
- Bevy setup guide: https://bevy.org/learn/quick-start/getting-started/setup/
- Bevy docs.rs latest: https://docs.rs/bevy/latest/bevy/
- Bevy 0.18.1 docs.rs: https://docs.rs/bevy/0.18.1/bevy/
- Bevy GitHub releases: https://github.com/bevyengine/bevy/releases
- Bevy 0.18.1 release: https://github.com/bevyengine/bevy/releases/tag/v0.18.1
- Bevy 0.19.0-rc.3 release: https://github.com/bevyengine/bevy/releases/tag/v0.19.0-rc.3
- Bevy crate versions: https://crates.io/crates/bevy/versions
- Bevy examples: https://bevy.org/examples/
- Bevy screenshot example: https://bevy.org/examples/window/screenshot/
- Bevy Remote Protocol docs: https://docs.rs/bevy/0.18.1/bevy/remote/index.html

### Rust Tooling Sources

- Cargo profiles: https://doc.rust-lang.org/cargo/reference/profiles.html
- Clippy docs: https://doc.rust-lang.org/clippy/
- rustfmt: https://github.com/rust-lang/rustfmt

### Physics and Character Controller Sources

- Avian 3D docs: https://docs.rs/avian3d/latest/avian3d/
- Rapier Bevy plugin docs: https://rapier.rs/docs/user_guides/bevy_plugin/getting_started_bevy/
- Rapier Bevy character controller docs: https://rapier.rs/docs/user_guides/bevy_plugin/character_controller/
- Rapier joints docs: https://rapier.rs/docs/user_guides/bevy_plugin/joints/
- `bevy_rapier3d`: https://docs.rs/bevy_rapier3d/latest/bevy_rapier3d/
- `bevy-tnua`: https://docs.rs/bevy-tnua/latest/bevy_tnua/

### Input, UI, Debug, VFX, Editor, MCP Sources

- `leafwing-input-manager`: https://docs.rs/leafwing-input-manager/latest/leafwing_input_manager/
- `bevy_enhanced_input`: https://docs.rs/bevy_enhanced_input/latest/bevy_enhanced_input/
- `bevy-inspector-egui`: https://docs.rs/bevy-inspector-egui/latest/bevy_inspector_egui/
- `bevy_hanabi`: https://docs.rs/bevy_hanabi/latest/bevy_hanabi/
- `bevy_mod_debugdump`: https://docs.rs/bevy_mod_debugdump/latest/bevy_mod_debugdump/
- Jackdaw repo: https://github.com/jbuehler23/jackdaw
- Jackdaw docs: https://jbuehler23.github.io/jackdaw/
- `bevy_brp_mcp` latest docs: https://docs.rs/bevy_brp_mcp/latest/bevy_brp_mcp/
- `bevy_brp_mcp` Bevy 0.18-compatible docs: https://docs.rs/bevy_brp_mcp/0.18.8/bevy_brp_mcp/
- `bevy_brp_mcp` versions: https://crates.io/crates/bevy_brp_mcp/versions
- `bevy_debugger_mcp`: https://docs.rs/bevy_debugger_mcp/latest/bevy_debugger_mcp/
- `bevy_behave`: https://docs.rs/bevy_behave/latest/bevy_behave/
- `big-brain`: https://docs.rs/big-brain/latest/big_brain/
- `bevy_capture`: https://docs.rs/bevy_capture/latest/bevy_capture/

### Local Project Sources

- `Cargo.toml`
- `Cargo.lock`
- `docs/plugins/bevy_gym_plugin.md`
- `docs/plugins/gym_render.md`
- `docs/examples/cartpole.md`

---

## Conclusion

Bevy is a credible foundation for `bevy-gym` precisely because the project leans into Bevy's strengths: ECS scheduling, fixed-step simulation, parallel entity processing, Rust-native tests, optional rendering, and composable plugins. The core mistake to avoid is turning a lean headless RL crate into a full game-engine distribution by adding physics, editors, capture, UI, MCP, and VFX directly to the core dependency surface.

The practical path is stable and incremental: stay on Bevy 0.18.1, keep core dependencies minimal, add separate examples/spikes for dev tooling and physics, use BRP/MCP as a controlled development surface, and only evaluate Bevy 0.19 once it is final and the ecosystem has caught up.
