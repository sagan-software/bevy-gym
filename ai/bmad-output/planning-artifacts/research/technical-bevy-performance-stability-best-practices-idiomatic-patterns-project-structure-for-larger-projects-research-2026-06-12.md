---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments: []
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Bevy performance, stability, best practices, idiomatic patterns, and project structure for larger projects'
research_goals: 'Research current Bevy performance, stability, idiomatic ECS patterns, and large-project structure guidance for bevy-gym and larger Bevy applications.'
user_name: 'Sagan'
date: '2026-06-12'
web_research_enabled: true
source_verification: true
---

# Research Report: technical

**Date:** 2026-06-12
**Author:** Sagan
**Research Type:** technical

---

## Research Overview

This research evaluates current Bevy performance, stability, idiomatic ECS usage, integration
patterns, and larger-project structure with a specific application target: `bevy-gym`, a
headless-first Bevy plugin for reinforcement-learning environments. The research combines current
web verification from Bevy, docs.rs, Cargo, Rust performance guidance, MCP documentation, and local
repository inspection of `Cargo.toml`, `Cargo.lock`, source modules, docs, CI, examples, and
development environment files.

The central finding is that `bevy-gym` should remain a small, stable, headless-first Bevy 0.18.1
library crate while rendering, physics, inspector, capture, BRP, and MCP integrations stay
feature-gated or isolated in examples/spikes until they prove compatibility and value. The full
executive synthesis appears in the Research Synthesis section, including source verification,
roadmap, risks, and success metrics.

---

## Technical Research Scope Confirmation

**Research Topic:** Bevy performance, stability, best practices, idiomatic patterns, and project
structure for larger projects **Research Goals:** Research current Bevy performance, stability,
idiomatic ECS patterns, and large-project structure guidance for bevy-gym and larger Bevy
applications.

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

Current-source research covered Bevy official release notes and quick-start documentation, docs.rs
API documentation for Bevy 0.18.1, Cargo documentation for
workspaces/features/profiles/dependencies, the Rust Performance Book, and this repository's
`Cargo.toml`, `Cargo.lock`, source, README, and plugin docs.

**Research Coverage:**

- Programming language and engine baseline: Rust + Bevy 0.18.1 stable, with Bevy 0.19.0 release
  candidates treated as migration-spike material.
- Framework and library surface: Bevy ECS/app/schedules/messages/states/assets/scenes, plus optional
  physics, input, inspector, and authoring plugins as feature-gated layers.
- Storage and data technologies: Bevy assets, glTF, scenes/reflection, external config/data files,
  and RL checkpoint/output storage rather than traditional app databases.
- Development tools: Cargo workspaces/features/profiles, Bevy feature collections, dynamic linking,
  alternative linkers, CI lanes, examples, and docs.
- Deployment surface: native binaries, headless runners, wasm/web builds, desktop render backends,
  and optional container/cloud packaging for simulation workers.

**Quality Assessment:** High confidence for Bevy 0.18.1, ECS, fixed timestep, feature, plugin,
asset, state, and Cargo recommendations because they are grounded in official Bevy/docs.rs/Cargo
sources and the local repo. Medium confidence for ecosystem plugin choices because compatibility
moves quickly across Bevy releases and each plugin must be rechecked before adoption.

### Programming Languages

Bevy's practical language stack is Rust first. The official Bevy crate describes Bevy as a Rust
data-driven game engine and app framework, with design goals around data-oriented ECS, modularity,
parallel app logic, and productive compile times. The current docs.rs `bevy` page is for **0.18.1**,
while docs.rs also lists **0.19.0-rc.3** as a release-candidate version dated 2026-06-10. For normal
production or library work today, the stable baseline should remain **0.18.1** unless a branch is
explicitly testing 0.19 migration.

For `bevy-gym`, the local dependency shape already matches the right language/engine posture:

- `Cargo.toml` uses `bevy = "0.18"` with `default-features = false`.
- `Cargo.lock` resolves Bevy and core Bevy crates to `0.18.1`.
- The crate exposes a Rust library/plugin surface around `rl-traits` environments and Bevy ECS
  scheduling.
- The core simulation runs without rendering, which keeps the Rust/Bevy dependency surface smaller
  and easier to benchmark.

_Popular Languages:_ Rust is the primary implementation language. WGSL is relevant only for custom
shaders. RON/TOML/JSON/serde-friendly formats are practical for tuning data, scenarios, and tool
metadata.

_Emerging Languages:_ No alternate language should enter the core stack. Scripting or data DSLs can
be considered later for content authoring, but they should sit behind assets/configs and should not
replace Rust for hot simulation loops.

_Language Evolution:_ Bevy tracks recent Rust capabilities closely; the Bevy crate notes that its
MSRV is generally close to the latest stable Rust. This is a stability consideration for larger
projects: engine upgrades can imply Rust toolchain upgrades.

_Performance Characteristics:_ Rust is a strong fit for deterministic, CPU-heavy ECS simulation.
Bevy ECS system parameters encode data access, allowing Bevy to run non-conflicting systems in
parallel. In `bevy-gym`, `query.par_iter_mut()` is correctly used for parallel environment stepping,
while message emission is serialized afterward because writers are not used inside the parallel
closure.

_Sources:_

- <https://docs.rs/crate/bevy/latest>
- <https://bevy.org/learn/quick-start/getting-started/>
- Local: `Cargo.toml`, `Cargo.lock`, `src/plugin.rs`, `src/systems/step.rs`

### Development Frameworks and Libraries

The core framework stack for larger Bevy projects should be organized around
**plugins, schedules, systems, resources, components, messages/events, states, and assets**.

Bevy's official ECS guide frames ECS as a design pattern that encourages decoupled data/logic and
improves memory access and parallelism. The `bevy_ecs` docs confirm that systems are ordinary Rust
functions and that Bevy uses their declared data access to decide what can run in parallel. For
larger projects, this means the architecture should not be "one global game loop"; it should be a
set of small domain plugins with explicit schedule/set ordering.

For `bevy-gym`, the current framework choices are idiomatic:

- `BevyGymPlugin` owns setup and schedule registration.
- `GymSet` provides named ordering in `FixedUpdate`.
- `ActionRequestEvent`, `ExperienceEvent`, and `EpisodeEndEvent` are Bevy messages.
- Environment state is stored on environment entities as components.
- Rendering is feature-gated through `render`, `winit`, `x11`, and `wayland`.

For larger projects, the recommended plugin layering is:

- `core` / simulation plugin: pure ECS, deterministic, no rendering dependency.
- `domain` plugins: gameplay/RL/physics/content subsystems with their own components and system
  sets.
- `presentation` plugins: rendering, UI, audio, camera, gizmos.
- `tooling` plugins: inspector, BRP/MCP, capture, diagnostics, debug UI.
- `platform` plugins: desktop, wasm, Steam Deck, headless worker, replay/screenshot harnesses.

Bevy 0.18 also improves larger-project feature selection through high-level feature collections
(`2d`, `3d`, `ui`) and mid-level collections. This is directly relevant to stability and compile
times: do not enable all default features in reusable libraries when a smaller feature set is
sufficient.

_Major Frameworks:_ Bevy App/ECS/Schedule/Plugin system; Cargo workspaces for multi-crate projects;
optional Bevy crates/features for assets, render, UI, state, and platform.

_Micro-frameworks:_ Bevy plugins should be small, domain-owned modules. In library crates, prefer
generic plugins and minimal features over monolithic app dependencies.

_Evolution Trends:_ Bevy 0.17 split buffered `Event` usage into `Message` while reserving `Event`
for observer-triggered event semantics. Bevy 0.18 added feature collections and continued
schedule/asset/API cleanup. Large codebases should expect migration work each release and keep
compatibility tables.

_Ecosystem Maturity:_ Bevy is usable but still fast-moving. The Bevy crate warns that important
features may be missing, docs may be sparse, and breaking releases are frequent. Treat the core
engine as strong for Rust-native ECS, while treating editor/workflow/plugin choices as
compatibility-managed dependencies.

_Sources:_

- <https://bevy.org/learn/quick-start/getting-started/ecs/>
- <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- <https://bevy.org/news/bevy-0-18/>
- <https://bevy.org/learn/migration-guides/0-16-to-0-17/>
- <https://bevy.org/learn/quick-start/plugin-development/>
- Local: `docs/plugins/bevy_gym_plugin.md`

### Database and Storage Technologies

Traditional databases are not part of the default Bevy stack. For larger Bevy projects, "storage"
usually means a mix of asset storage, authored scene/content storage, save-game or run-output
storage, telemetry, and build artifacts.

Use Bevy-native asset flow for game/runtime content:

- Asset handles are the main reference mechanism for Bevy assets.
- Asset hot reloading is available for desktop development when the `file_watcher` feature is
  enabled.
- Procedural assets can be added to `Assets<T>`, but repeated procedural insertion can duplicate
  asset data and consume memory if not managed.
- glTF is Bevy's primary 3D scene/model interchange path. Bevy 0.18 adds better support for glTF
  extensions and custom processing hooks.
- Dynamic scenes are reflection-powered serializable representations of
  entities/components/resources. They are useful for tool output and content snapshots, but they are
  not a substitute for carefully versioned game state.

For `bevy-gym`, the practical storage strategy should be:

- Keep environment definitions in Rust for performance-critical loops.
- Use lightweight external config files for curricula, seeds, run presets, and benchmark
  definitions.
- Store RL checkpoints, telemetry, videos, screenshots, and run metadata outside ECS as normal
  files.
- Keep Bevy scenes/glTF out of the headless core unless the environment genuinely depends on
  authored world geometry.
- If a future project needs cloud persistence, put it behind a simulation-output or telemetry
  boundary, not inside normal Bevy systems.

_Relational Databases:_ Not a default part of Bevy. Use SQL only for analytics, leaderboards,
account data, or persistent services outside the game/simulation runtime.

_NoSQL Databases:_ Not a default part of Bevy. Use document/object storage for telemetry or content
pipelines only if file-based artifacts stop scaling.

_In-Memory Databases:_ Usually unnecessary inside a Bevy app because ECS is already the in-memory
runtime store. External caches are relevant only for services around the app.

_Data Warehousing:_ Relevant for RL experiments and telemetry analysis, not for the engine loop.
Persist compact run records and process them outside Bevy.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/asset/index.html>
- <https://docs.rs/bevy/latest/bevy/gltf/index.html>
- <https://docs.rs/bevy_scene/latest/bevy_scene/>
- <https://bevy.org/news/bevy-0-18/>

### Development Tools and Platforms

The development platform is Cargo plus Bevy's optional feature system. The most important
larger-project rule is to keep feature boundaries additive, explicit, and documented.

Cargo's feature model supports conditional compilation and optional dependencies. The Cargo docs
warn that dependency default features are enabled unless `default-features = false` is specified,
which matters heavily for Bevy because render, audio, windowing, UI, and asset features can pull in
large compile and runtime surfaces. Cargo also documents that dev-dependencies are used for
tests/examples/benchmarks and are not propagated to dependent packages, which is exactly where
larger Bevy projects should place inspectors, capture tools, benchmarks, and compatibility fixtures
when possible.

Cargo workspaces are the right structure once a Bevy project grows beyond a single app crate.
Official Rust documentation describes workspaces as a way to manage related packages developed
together with a shared lockfile and output directory. For a larger Bevy project, split when a
boundary has a different dependency profile or validation lane:

- `crates/core`: pure game/simulation model, no Bevy render stack.
- `crates/bevy_app`: Bevy app/plugin integration.
- `crates/render`: rendering and presentation.
- `crates/tools`: editor/dev tooling.
- `crates/content`: asset processors/importers if needed.
- `examples/`: runnable usage and smoke tests.

Bevy official plugin-development guidance reinforces this: keep plugin crate size small, enable only
the Bevy features needed, avoid large dependencies, use `cargo tree`/`cargo-deny` to detect
duplication, put optional functionality behind Cargo features, set up CI, document system
sets/components, and publish a Bevy-version compatibility table.

For compile time, Bevy's setup docs recommend dynamic linking for faster local iteration and
alternative linkers such as LLD/mold. The same docs caution against shipping with dynamic linking
enabled. Cargo profiles and the Rust Performance Book support using profile changes for optimization
tradeoffs, but performance options should be benchmarked one at a time.

_IDE and Editors:_ Use rust-analyzer-compatible Rust module/workspace structure. For Bevy
authoring/editor candidates, treat each tool as optional and version-sensitive.

_Version Control:_ Git plus a documented migration branch policy. Keep Bevy release upgrades in
explicit branches because Bevy releases frequently contain breaking changes.

_Build Systems:_ Cargo workspaces, feature-gated Bevy dependencies, explicit dev/release profiles,
and optional dynamic linking/linker config for local iteration.

_Testing Frameworks:_ Standard Rust tests for pure logic, Bevy `App` schedule tests for ECS
behavior, example smoke tests for render/tooling features, and benchmark/replay harnesses for
performance-sensitive systems.

_Sources:_

- <https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html>
- <https://doc.rust-lang.org/cargo/reference/features.html>
- <https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html>
- <https://doc.rust-lang.org/cargo/reference/profiles.html>
- <https://nnethercote.github.io/perf-book/build-configuration.html>
- <https://bevy.org/learn/quick-start/getting-started/setup/>
- <https://bevy.org/learn/quick-start/plugin-development/>

### Cloud Infrastructure and Deployment

Bevy itself is not a cloud framework. Larger Bevy projects usually deploy along one of four paths:

- Native desktop game/app binary.
- Web/wasm build.
- Headless simulation/server/training worker.
- Tooling/editor binary for content or debug workflows.

For `bevy-gym`, the key deployment path is headless simulation: `MinimalPlugins` plus
`ScheduleRunnerPlugin`, no render/window backend, and a fixed or uncapped tick loop depending on
training needs. That design matches the current repo documentation and should remain the default
performance lane.

Cloud/container deployment is a packaging concern around the app, not an engine architecture
concern. Headless simulation workers can run in containers or CI, but the Bevy core should not
depend directly on cloud SDKs. Write run outputs to files, object storage adapters, or telemetry
sinks through a thin boundary.

For rendered deployments, keep platform features explicit:

- Desktop rendering needs the correct Bevy render/window backend features.
- Linux render paths should choose x11/wayland deliberately.
- Web builds have platform-specific dependency and configuration concerns; Bevy migration docs
  called out web `getrandom` changes during 0.17 migration.
- Dynamic linking is for local iteration, not release packaging.

_Major Cloud Providers:_ None should be directly coupled to Bevy gameplay/simulation code.

_Container Technologies:_ Useful for headless training, reproducible benchmarks, CI smoke tests, and
batch simulation. Keep GPU/window requirements explicit.

_Serverless Platforms:_ Poor fit for Bevy runtime loops; useful only for post-processing outputs or
lightweight metadata workflows.

_CDN and Edge Computing:_ Relevant for distributing web builds/assets, not for Bevy ECS
architecture.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/prelude/struct.FixedUpdate.html>
- <https://docs.rs/bevy/latest/bevy/prelude/struct.Fixed.html>
- <https://bevy.org/learn/quick-start/getting-started/setup/>
- <https://bevy.org/learn/migration-guides/0-16-to-0-17/>
- Local: `README.md`, `docs/examples/cartpole.md`, `examples/classic-control/cart_pole.rs`

### Technology Adoption Trends

The current Bevy trend is toward more modular, feature-selectable, ECS-native architecture:

- Bevy 0.18 introduced scenario-oriented feature collections such as `2d`, `3d`, and `ui`, reducing
  the need to manually curate large low-level feature lists.
- Bevy 0.17 split buffered messages from observer events, so new code should use
  `MessageWriter`/`MessageReader` for queued communication and observers/events for triggered
  reaction semantics.
- Bevy's official plugin-development guidance explicitly pushes smaller feature footprints, optional
  dependencies, CI, examples, and compatibility tables.
- Bevy's state system supports standard states, substates, computed states, transition schedules,
  run conditions, and state-scoped entity lifetime helpers, which is the idiomatic structure for
  large-scale app/game modes.
- Fixed timestep APIs remain central for physics, AI, networking, game rules, and deterministic
  simulation.

For stability, the larger-project rule is version discipline:

- Stay on 0.18.1 for the mainline until 0.19 final and critical plugins are verified.
- Use a migration branch for 0.19 release candidates.
- Run both no-render/headless and render/tooling CI lanes.
- Maintain a compatibility table for every Bevy-facing plugin.
- Avoid pre-release ecosystem crates in the core unless the project has explicitly accepted
  migration churn.

For performance, the trend is not "add more engine features"; it is "compile and run less":

- Disable Bevy default features in libraries.
- Use Bevy feature collections for app crates.
- Keep render/UI/audio/tooling out of headless simulation crates.
- Put expensive authoring/editor/debug tools behind features.
- Benchmark schedule/query/layout choices before treating style preferences as performance rules.

_Migration Patterns:_ Frequent Bevy releases make compatibility tables, migration branches, and
small plugin boundaries more important than in slower-moving engines.

_Emerging Technologies:_ Bevy feature collections, observer/event refinements, glTF extension
handling, improved UI/navigation, and BRP/MCP-style tooling are useful but should remain opt-in
until project needs justify them.

_Legacy Technology:_ Avoid older Bevy examples/tutorials unless version-matched. Bevy API patterns
changed substantially across releases, especially schedules, events/messages, and rendering
organization.

_Community Trends:_ The official Bevy Assets site is the discovery surface for community plugins and
tools, but adoption should be based on Bevy-version compatibility, maintenance status, and local
spike results.

_Sources:_

- <https://bevy.org/news/bevy-0-18/>
- <https://bevy.org/learn/migration-guides/0-16-to-0-17/>
- <https://docs.rs/bevy_state/latest/bevy_state/>
- <https://docs.rs/bevy/latest/bevy/prelude/struct.FixedUpdate.html>
- <https://docs.rs/bevy/latest/bevy/prelude/struct.Fixed.html>
- <https://bevy.org/assets/>

### Technology Stack Implications for `bevy-gym`

`bevy-gym` is already aligned with the strongest current Bevy guidance:

- It keeps Bevy defaults disabled.
- It exposes rendering as an optional feature.
- It uses `FixedUpdate` for simulation.
- It defines a named `SystemSet` for ordering.
- It uses Bevy messages rather than older event-writer terminology.
- It separates headless training from render visualization.
- It keeps RL environment stepping parallel and serializes only the message emission boundary.

The next stack-level improvements to consider are not broad rewrites:

1. Add a documented CI matrix for `--no-default-features`, `--features render`, docs, examples, and
   release-mode smoke runs.
2. Add a short compatibility table in docs for Bevy, `rl-traits`, `burn`, and `ember-rl`.
3. Add optional benchmark harnesses for environment count, step throughput, message overhead, and
   render sync overhead.
4. Keep Avian/Rapier/editor/BRP/MCP tooling out of the core crate until each has a feature-gated
   spike and compatibility note.
5. If the project grows, split a workspace only when dependency surfaces diverge; do not split
   merely for file organization.

## Integration Patterns Analysis

### Web Search Analysis

Current-source research covered Bevy 0.18.1 docs.rs pages for messages, observers, plugins,
schedules, reflection, BRP, and headless schedule running; official MCP protocol docs for JSON-RPC
transports and tool exposure; and local `bevy-gym` source/docs for action/experience messages,
rendering integration, and headless/render split.

**Research Coverage:**

- API design patterns: Rust trait/plugin APIs, ECS component/resource APIs, schedule/system-set
  APIs, and dev-only BRP APIs.
- Communication protocols: in-process ECS/message flow, JSON-RPC over HTTP for Bevy Remote Protocol,
  JSON-RPC over stdio/HTTP for MCP, and file/asset boundaries.
- Data formats: strongly typed Rust values, reflected type metadata, JSON/JSON-RPC, glTF, Bevy
  scenes, and run-output files.
- Interoperability: RL environment integration, policy/inference integration, rendering integration,
  tooling/editor integration, and external service boundaries.
- Security: local-only/dev-only posture for mutable remote ECS control, feature gates, loopback
  binding, authorization at external transports, and no cloud/service coupling in hot ECS systems.

**Quality Assessment:** High confidence for Bevy's in-process integration guidance because it is
directly supported by Bevy docs.rs and local source. High confidence that BRP uses JSON-RPC over
HTTP and can inspect/mutate ECS state. Medium confidence for Bevy MCP ecosystem choices because
specific crates move with Bevy release compatibility and must be rechecked before adoption.

### API Design Patterns

The primary Bevy API pattern is not REST or GraphQL. It is **typed, in-process Rust APIs built from
plugins, systems, components, resources, messages, traits, and schedules**. External APIs should be
adapters around that core, not the internal architecture.

For larger Bevy projects, the stable public API surface should usually be one of these:

- **Plugin API:** A plugin configures an `App`; Bevy's `Plugin` trait docs define plugins as
  collections of app logic/configuration whose `build` method configures the app.
- **Trait API:** Domain-specific traits let external crates implement behavior without depending on
  internal systems. `bevy-gym` uses this well with `GymRender` and `rl_traits::Environment`.
- **Message API:** Buffered, pull-based communication between systems. Bevy 0.18 `Message` docs
  state that messages are written with `MessageWriter`, read with `MessageReader`, and evaluated at
  fixed schedule points rather than immediately.
- **Observer/Event API:** Immediate reactive behavior. Bevy ECS docs distinguish observers from
  messages: observers run when an event is triggered, not at a later schedule point.
- **Reflect/BRP API:** Runtime inspection/mutation for tools. Bevy reflection provides runtime type
  metadata and serialization/deserialization support; BRP exposes reflected ECS state over JSON-RPC.

For `bevy-gym`, the current API design is strong:

- `BevyGymPlugin` is the main integration point.
- `GymSet` provides an ordering contract for user systems.
- `ExperienceEvent`, `EpisodeEndEvent`, and `ActionRequestEvent` are actually Bevy **messages** in
  0.18 terms.
- `GymRender` is an optional trait extension rather than a required dependency in the core.

The main documentation cleanup is naming consistency: local crate docs still show
`EventReader<ActionRequestEvent>` in one quick-start snippet, while the event types derive `Message`
and Bevy 0.18 docs use `MessageReader`/`MessageWriter` for buffered communication. The integration
pattern should be documented as **messages**, reserving "events" for observer-triggered
`Event`/`EntityEvent` APIs.

_RESTful APIs:_ Not a core Bevy pattern. Use REST only at external service boundaries such as
telemetry upload, experiment management, or account services.

_GraphQL APIs:_ Not a core Bevy pattern. Consider only for external dashboards or content
management, not for ECS runtime calls.

_RPC and gRPC:_ BRP is the current Bevy-native RPC-style option, but it uses JSON-RPC rather than
gRPC. Use gRPC/protobuf only when integrating with external high-throughput services that already
require it.

_Webhook Patterns:_ Useful outside the app for CI, experiment completion, or content pipeline
notifications; not appropriate inside the ECS loop.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/prelude/trait.Plugin.html>
- <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- <https://docs.rs/crate/bevy_ecs/latest>
- <https://docs.rs/bevy_reflect/latest/bevy_reflect/>
- Local: `src/events.rs`, `src/plugin.rs`, `src/render.rs`, `src/lib.rs`

### Communication Protocols

Bevy's strongest integration protocol is in-process ECS data access. Systems communicate through
queries, resources, commands, messages, and schedule ordering. This keeps hot-loop behavior typed,
parallelizable, and visible to Bevy's scheduler.

Use the following protocol hierarchy:

1. **Direct ECS access:** queries/resources for tight, same-schedule logic.
2. **Buffered messages:** cross-system communication where producer and consumer are decoupled but
   still in the same app.
3. **Observers/events:** immediate reactions to discrete triggers, especially local entity-level
   reactions.
4. **Assets/files:** durable content, config, run outputs, telemetry, checkpoints, scenes, and glTF.
5. **BRP JSON-RPC:** dev-only external inspection/mutation of ECS state.
6. **MCP:** AI/tooling bridge layered over BRP or custom commands, not a gameplay/runtime
   dependency.
7. **HTTP/gRPC/message queues:** external services only, outside the frame/tick critical path.

Bevy Remote Protocol is the most important current Bevy-specific external protocol. Its docs state
that remote clients can inspect and alter ECS state, and that BRP is based on JSON-RPC 2.0. The HTTP
transport accepts JSON requests over HTTP and defaults to port `15702`. This makes BRP useful for
inspectors, editor tooling, AI manipulation, and test harnesses, but risky as a production-exposed
interface.

MCP is relevant as a tooling bridge. The MCP architecture docs define a JSON-RPC data layer and
transports including stdio and Streamable HTTP. MCP tools are model-invoked capabilities with
schemas, which maps well to narrowly scoped Bevy dev tools such as "list entities", "query
component", "set transform", "capture screenshot", or "run one fixed tick."

_HTTP/HTTPS Protocols:_ Use for BRP HTTP, MCP Streamable HTTP, telemetry upload, dashboards, and
control planes. Keep HTTP off the hot ECS path.

_WebSocket Protocols:_ Possible for custom live dashboards, but not necessary if BRP/MCP and
file-based telemetry cover the workflow.

_Message Queue Protocols:_ Appropriate for distributed training orchestration or offline telemetry
ingestion; not a first choice for a single Bevy app.

_gRPC and Protocol Buffers:_ Use only for a service boundary that needs strongly typed
cross-language calls. It is not Bevy-native and should not shape ECS internals.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://docs.rs/bevy/latest/bevy/remote/http/index.html>
- <https://modelcontextprotocol.io/docs/learn/architecture>
- <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>
- <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>

### Data Formats and Standards

Inside Bevy, the best "format" is usually the Rust type system: components, resources, messages,
assets, and trait implementations. Crossing a boundary requires choosing a representation
deliberately.

Recommended data formats:

- **Rust structs/enums:** hot runtime state, messages, environment observations/actions, and policy
  integration.
- **Reflect metadata:** dev tooling, dynamic inspection, scenes, editor integration, BRP schemas,
  and limited save/load support.
- **JSON / JSON-RPC:** BRP, MCP, debug tools, lightweight telemetry, and external control planes.
- **glTF:** 3D content and scene/model import. Bevy's glTF docs describe the loader for glTF 2.0
  scenes and assets, and Bevy 0.18 release notes emphasize glTF extensions for adding custom data to
  authored content.
- **Bevy scenes / `.scn.ron`:** reflected entity/resource snapshots and tool-generated scene
  content. Use carefully because scene serialization is reflection-driven and should not be treated
  as a complete domain save format by default.
- **Flat files:** training run outputs, checkpoints, benchmark CSV/JSON, screenshots, and video.

For `bevy-gym`, avoid serializing the entire Bevy world for training data. Persist the domain
artifacts instead: seed/config, observations/actions/rewards/statuses, model checkpoints, episode
summaries, benchmark stats, and optional rendered captures.

_JSON and XML:_ JSON is useful for BRP/MCP/control/telemetry. XML has no obvious role unless
required by an external tool.

_Protobuf and MessagePack:_ Potentially useful for high-volume external telemetry or cross-language
services, but overkill for the current crate.

_CSV and Flat Files:_ Good for benchmark summaries, experiment comparison, and quick analysis.

_Custom Data Formats:_ Acceptable for compact RL replay/checkpoint formats, but keep schema/version
metadata with each artifact.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://docs.rs/bevy/latest/bevy/gltf/index.html>
- <https://docs.rs/bevy_scene/latest/bevy_scene/>
- <https://docs.rs/bevy_reflect/latest/bevy_reflect/>
- <https://bevy.org/news/bevy-0-18/>

### System Interoperability Approaches

The idiomatic Bevy interoperability boundary is
**plugin plus messages plus shared components/resources**, with schedule ordering as the contract.
This is exactly the pattern `bevy-gym` is already using.

For `bevy-gym`, system interoperability should be framed as four lanes:

- **Environment lane:** each environment entity owns `EnvironmentComponent<E>`,
  `CurrentObservation<E>`, `PendingAction<E>`, and `EnvStats`.
- **Policy lane:** policy systems listen for `ActionRequestEvent` messages and write
  `PendingAction<E>`.
- **Learning/telemetry lane:** learners/loggers listen for `ExperienceEvent` and `EpisodeEndEvent`.
- **Render lane:** optional `GymRenderPlugin` reads the latest observation in `Update`, while
  simulation runs independently in `FixedUpdate`.

This preserves important boundaries:

- The environment does not depend on the learner.
- The learner does not need direct access to the environment internals.
- Rendering never controls simulation correctness.
- Policy systems can batch action requests because action request and action application are
  decoupled.
- Headless training can skip all render/window features.

_Point-to-Point Integration:_ Direct query/resource access is fine inside a narrow domain plugin,
but cross-domain integration should use messages or explicit traits to avoid hidden coupling.

_API Gateway Patterns:_ For Bevy, the closest equivalent is a dev-only BRP/MCP adapter. Do not route
ordinary ECS behavior through an HTTP gateway.

_Service Mesh:_ Not relevant inside Bevy. Use service-mesh concepts only around distributed training
infrastructure if it exists.

_Enterprise Service Bus:_ Not relevant for the engine/app. Use a lightweight event/log sink instead.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- <https://docs.rs/bevy/latest/bevy/prelude/struct.MessageReader.html>
- <https://docs.rs/bevy/latest/bevy/prelude/struct.FixedUpdate.html>
- <https://docs.rs/bevy/latest/bevy/app/struct.ScheduleRunnerPlugin.html>
- Local: `src/plugin.rs`, `src/systems/step.rs`, `docs/plugins/bevy_gym_plugin.md`,
  `docs/plugins/gym_render.md`

### Microservices Integration Patterns

Microservices should not be imported into Bevy's internal architecture. A Bevy app or RL simulator
is a stateful runtime loop, not a request/response web backend. Use microservice patterns only
outside the runtime boundary:

- experiment orchestration,
- model registry,
- telemetry collection,
- artifact storage,
- replay analysis,
- multiplayer/backend services if the project becomes a networked game.

For larger Bevy projects, treat Bevy as one or more runtime processes with a narrow set of service
adapters. External services should communicate through command queues, run files, telemetry batches,
or explicit HTTP/RPC clients that feed resources/messages at controlled schedule points.

_API Gateway Pattern:_ Useful for external control planes, not internal ECS.

_Service Discovery:_ Relevant only when headless workers are distributed across machines.

_Circuit Breaker Pattern:_ Useful in service adapters so external outages do not stall frames or
training ticks.

_Saga Pattern:_ Not relevant unless the project manages distributed transactions outside Bevy.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/app/struct.ScheduleRunnerPlugin.html>
- <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- <https://doc.rust-lang.org/cargo/reference/features.html>
- Local: `README.md`, `docs/examples/cartpole.md`

### Event-Driven Integration

Bevy 0.18 requires precise terminology:

- **Messages** are buffered, pull-based, schedule-point communication. They are the right default
  for training transitions, telemetry, policy requests, UI notifications, and decoupled systems.
- **Events/observers** are immediate reactive triggers. They are better for local entity reactions,
  tool hooks, lifecycle reactions, and cases where immediate propagation is intended.

This distinction matters for performance and determinism. Bevy's `Message` docs note that messages
are evaluated at fixed points in the schedule and can be efficient for batch processing many
messages. `MessageReader` docs also note that multiple reader systems can run concurrently, although
not concurrently with writers/mutators for the same message type. For `bevy-gym`, this supports the
current architecture: generate training transitions in `FixedUpdate`, then let
policy/learning/logging systems consume them at explicit schedule points.

Event-driven recommendations:

- Use messages for `ExperienceEvent`, `EpisodeEndEvent`, and `ActionRequestEvent` semantics.
- Prefer `write_batch` or equivalent batching when high message volume becomes measurable.
- Use `ActionRequestEvent.entity` for direct lookup rather than scanning by `env_id` in hot policy
  systems.
- Use observers sparingly for lifecycle/tooling events where immediate behavior is intentional.
- Keep message types small enough to avoid excessive clone/memory pressure; large artifacts should
  be stored elsewhere and referenced by handle/path/id.

_Publish-Subscribe Patterns:_ Bevy messages are the in-process pub/sub workhorse.

_Event Sourcing:_ Useful conceptually for RL experience streams, but full event-sourced world state
is probably too heavy. Persist domain transitions, not every ECS mutation.

_Message Broker Patterns:_ External brokers are only for distributed runs or telemetry pipelines.

_CQRS Patterns:_ Not a core Bevy pattern. A lightweight version can exist as "write simulation
messages, read query/report state," but do not over-architect it.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- <https://docs.rs/bevy/latest/bevy/prelude/struct.MessageReader.html>
- <https://docs.rs/crate/bevy_ecs/latest>
- <https://bevy.org/learn/migration-guides/0-16-to-0-17/>
- Local: `src/events.rs`, `src/systems/step.rs`

### Integration Security Patterns

The most important Bevy integration security rule is to keep mutable remote-control tools behind
explicit development boundaries. BRP can inspect and alter ECS state. That is powerful enough to
treat as privileged access.

Recommended controls:

- Feature-gate BRP/MCP/debug inspectors and exclude them from release builds unless explicitly
  needed.
- Bind remote tools to loopback by default.
- Do not expose BRP HTTP directly to a network without an authentication, authorization, and
  transport-security layer.
- Register/reflect only the components and resources that tooling needs.
- Keep model/AI tools scoped to explicit commands such as inspect, screenshot, run tick, or set
  component; avoid broad arbitrary mutation in shared environments.
- Persist audit logs for remote mutation tools when they are used in collaborative workflows.
- Keep secrets out of ECS resources that are exposed to reflection/BRP.

For MCP, the official architecture separates data and transport layers and calls out transport
authorization. The transport spec says MCP uses JSON-RPC and supports stdio plus Streamable HTTP.
For local AI tooling, stdio is often the safer default because it is process-local; HTTP should get
normal service security treatment.

_OAuth 2.0 and JWT:_ Relevant for MCP Streamable HTTP or external control planes. Not part of Bevy
ECS.

_API Key Management:_ Relevant only for external services. Do not put service keys in reflected Bevy
resources.

_Mutual TLS:_ Appropriate if a remote Bevy control or telemetry endpoint crosses machine boundaries.

_Data Encryption:_ Apply at external transport/storage boundaries. ECS memory is not a secret vault.

_Sources:_

- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://docs.rs/bevy/latest/bevy/remote/http/index.html>
- <https://modelcontextprotocol.io/docs/learn/architecture>
- <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>

### Integration Implications for `bevy-gym`

`bevy-gym` should keep its integration model simple and typed:

1. Treat `BevyGymPlugin` as the primary API and keep it generic over `Environment`.
2. Standardize public docs on `MessageReader`/`MessageWriter` terminology for Bevy 0.18.
3. Add a hot-path policy example that uses `ActionRequestEvent.entity` for direct `Query::get_mut`
   lookup rather than scanning by `EnvId`.
4. Keep render synchronization one-way: simulation writes observations; rendering reads and displays
   them.
5. Add optional BRP/MCP tooling only as dev/example features, with a clear warning that remote
   mutation is privileged.
6. Keep experiment telemetry and checkpoints as explicit run artifacts, not implicit serialized Bevy
   worlds.

## Architectural Patterns and Design

### Web Search Analysis

Current-source research covered Bevy's official 0.18 release notes, Bevy plugin-development
guidance, docs.rs pages for Bevy ECS, app schedules, states, `FixedUpdate`, `Entity`, and Bevy's
local `bevy-gym` architecture. The generic architecture categories from the workflow were translated
into Bevy's actual design surface: ECS-first modular runtime architecture, plugin/module boundaries,
explicit schedules, feature-gated optional layers, and external service adapters outside the hot
simulation loop.

**Research Coverage:**

- System architecture: ECS, plugin architecture, app/schedule lifecycle, states, feature-gated
  layers.
- Design principles: data-oriented design, explicit schedule ordering, decoupled messages, small
  plugins, trait boundaries.
- Scalability/performance: entity-per-environment parallelism, `FixedUpdate`, component storage,
  headless scheduling, feature minimization.
- Integration/security/data architecture: messages/observers, BRP/MCP boundaries, reflected data,
  stable domain IDs, artifact persistence.
- Operations: CI lanes, Bevy-version compatibility tables, headless/render validation, migration
  branch discipline.

**Quality Assessment:** High confidence for ECS, schedules, states, feature minimization, storage,
and version-policy guidance because these are from Bevy/docs.rs and Cargo/Rust sources. Medium
confidence for larger-project organization beyond plugin boundaries because Bevy's official guidance
is stronger on APIs/features than on a single prescribed folder/workspace layout.

### System Architecture Patterns

The best Bevy architecture pattern for larger projects is a **modular ECS monolith**, not a
microservice architecture. In this pattern, the runtime remains one Bevy app/world for the hot path,
while major domains are isolated as plugins, system sets, components, messages, and optional Cargo
features.

Bevy ECS docs state that app logic uses the Entity Component System pattern and that ECS encourages
clean, decoupled designs by breaking data and logic into components and systems while improving
memory access and parallelism. This means the architecture boundary is not an object hierarchy or
service graph. It is the composition of:

- entities: runtime instances,
- components: typed data attached to entities,
- resources: singleton world state,
- systems: logic units over explicit data access,
- schedules/system sets: execution order and phase boundaries,
- plugins: app configuration and domain assembly,
- messages/events: decoupled cross-system communication.

For larger Bevy projects, prefer these layers:

- **Core/domain layer:** Rust types, components, resources, messages, domain rules, no
  render/window/editor dependency.
- **Simulation layer:** fixed-tick systems, physics/AI/RL/game rules, deterministic stepping where
  feasible.
- **Presentation layer:** rendering, UI, audio, camera, VFX, render sync.
- **Tooling layer:** inspectors, BRP, MCP, debug overlays, capture, editor helpers.
- **Platform layer:** desktop/window backend, wasm, headless runner, Steam Deck or target-specific
  integration.
- **Service boundary layer:** telemetry, experiment tracking, network services, model registry,
  artifact storage.

`bevy-gym` already follows the right system pattern:

- one environment instance per entity,
- environment state as components,
- simulation in `FixedUpdate`,
- ordered `GymSet` phases,
- messages for policy/learning/episode flow,
- optional render plugin reading observations in `Update`,
- headless-first default feature set.

The main future architecture decision is when to move from single crate to workspace. Use a
workspace only when dependency profiles diverge, for example separating `bevy-gym-core`,
`bevy-gym-render`, `bevy-gym-tools`, and heavy examples. Do not split crates just to create folders;
Bevy plugins and Rust modules are enough until dependency surfaces or validation lanes differ.

_Source:_

- <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- <https://docs.rs/bevy/latest/bevy/prelude/trait.Plugin.html>
- <https://docs.rs/bevy/latest/bevy/app/index.html>
- <https://bevy.org/learn/quick-start/plugin-development/>
- Local: `src/plugin.rs`, `src/components.rs`, `src/systems/step.rs`, `src/render.rs`

### Design Principles and Best Practices

The core design principle is **data ownership clarity**: put persistent runtime facts in
components/resources, use systems for behavior, and use schedules/system sets to make ordering
explicit. Avoid central "manager" resources that know every subsystem unless they are truly global
configuration or metrics.

Recommended Bevy design rules:

- Use plugins as domain assembly units, not as dumping grounds.
- Give each plugin a small public contract: components, messages, system sets, resources, and
  feature flags.
- Use explicit `SystemSet` labels for cross-system ordering that users need to compose with.
- Use `FixedUpdate` for gameplay/simulation work that must run at a fixed rate; use `Update` for
  render-frame presentation.
- Use states for large app flow such as loading, menu, playing, paused, evaluation, training,
  replay, and editor/tool modes.
- Use messages for decoupled scheduled communication; use observers/events only for intentionally
  immediate reactions.
- Keep render systems read-only with respect to simulation truth where possible.
- Keep public examples version-matched to Bevy 0.18 APIs.

Bevy's state docs define states as app-wide finite state machines for large-scale program structure,
including paused/loading/combat-style modes. For a larger Bevy app, states should control which
plugins/systems are active, while plugins own the behavior inside those states.

For `bevy-gym`, the near-term design improvement is documentation consistency:

- public examples should consistently use `MessageReader`/`MessageWriter` for Bevy 0.18 messages,
- policy examples should show `query.get_mut(req.entity)` to avoid hot-path scans,
- examples should show system placement after `GymSet::ManualReset` when policy systems need fully
  reconciled state.

_Source:_

- <https://docs.rs/bevy/latest/bevy/state/index.html>
- <https://docs.rs/crate/bevy_state/latest>
- <https://docs.rs/bevy/latest/bevy/app/prelude/struct.FixedUpdate.html>
- <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- <https://bevy.org/learn/migration-guides/0-10-to-0-11/>
- Local: `README.md`, `docs/plugins/bevy_gym_plugin.md`

### Scalability and Performance Patterns

In Bevy, scalability starts with ECS layout and scheduling rather than process distribution. The
performance architecture should ask:

- Can hot systems run in parallel?
- Are components laid out for the queries that run every tick?
- Are system dependencies explicit but not over-constraining?
- Are render/tool dependencies outside headless simulation?
- Are high-volume messages batched or kept compact?
- Are long-running services outside the fixed-tick path?

Bevy ECS docs describe Bevy ECS as fast, massively parallel, and data-oriented. They also document
component storage tradeoffs: table storage is default and optimized for fast/cache-friendly
iteration, while sparse sets favor fast add/remove but slower iteration. For `bevy-gym`, environment
components, observations, pending actions, and stats are hot per-tick data, so default table storage
is appropriate. Marker-like components such as `ResetRequested` could be sparse-set candidates only
if add/remove churn becomes measurable.

`FixedUpdate` is the correct architecture for RL simulation because Bevy docs identify it as the
schedule for fixed-rate gameplay logic, including physics, AI, networking, and game rules.
`bevy-gym` uses this correctly. Rendering belongs in `Update` because render sync is presentation,
not simulation truth.

Scalability recommendations for `bevy-gym`:

- Preserve one entity per environment instance.
- Keep `step_system` parallel over independent environment entities.
- Avoid adding shared mutable resources to the step phase unless they are sharded or read-only.
- Use direct entity lookup for policy responses.
- Consider batching message writes if profiling shows message dispatch overhead.
- Benchmark with `num_envs` sweeps and separate headless vs render lanes.
- Keep `render`, BRP/MCP, physics demos, and inspectors feature-gated.
- Keep large tensors, replay buffers, model checkpoints, and video outputs outside ECS components.

_Source:_

- <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- <https://docs.rs/crate/bevy_ecs/latest>
- <https://docs.rs/bevy/latest/bevy/app/prelude/struct.FixedUpdate.html>
- <https://docs.rs/bevy/latest/bevy/app/struct.ScheduleRunnerPlugin.html>
- <https://bevy.org/learn/quick-start/plugin-development/>
- Local: `src/systems/step.rs`, `src/systems/reset.rs`, `README.md`

### Integration and Communication Patterns

The architectural communication pattern should be **local messages before external protocols**:

- direct query/resource access inside tightly owned systems,
- messages for cross-domain scheduled communication,
- observers/events for immediate local reactions,
- trait boundaries for extension points,
- BRP/MCP for dev-only remote inspection and AI tooling,
- external HTTP/RPC/queues only around service boundaries.

In `bevy-gym`, the core communication architecture is already message-driven:

- `ActionRequestEvent`: simulation asks policy for a new action.
- `ExperienceEvent`: simulation emits `(s, a, r, s', status)` for learners/loggers.
- `EpisodeEndEvent`: simulation emits terminal/truncated episode summary.

That pattern keeps the RL learner, policy, renderer, and environment decoupled. It also supports
batched inference because action requests can be collected before actions are written back.

BRP/MCP should not become the normal integration path. BRP can inspect and mutate ECS over JSON-RPC,
and MCP can expose model-invoked tools over JSON-RPC transports. Those are powerful
development/control surfaces, but they should remain opt-in tooling layers.

_Source:_

- <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- <https://docs.rs/crate/bevy_ecs/latest>
- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://modelcontextprotocol.io/docs/learn/architecture>
- Local: `src/events.rs`, `src/render.rs`

### Security Architecture Patterns

The security architecture is mostly about keeping unsafe boundaries narrow:

- no secrets in reflected ECS resources/components,
- no remote mutable ECS control in default builds,
- no network exposure for BRP/MCP without explicit auth and transport security,
- no cloud SDKs or service credentials in hot simulation code,
- no long-term persistence of unstable Bevy identifiers.

Bevy's `Entity` docs explicitly warn that `Entity` should be treated as opaque and that direct
serialization makes no guarantee of long-term wire compatibility. This affects security and
correctness: persisted training artifacts should use stable domain identifiers (`env_id`, run ID,
episode ID, step index), not raw Bevy entity IDs.

Security boundaries for larger Bevy projects:

- **Internal ECS boundary:** Rust type safety, Bevy scheduler borrow checking, explicit system
  access.
- **Tooling boundary:** feature-gated BRP/MCP, loopback default, reflected components kept minimal.
- **Artifact boundary:** schema-versioned files and stable IDs.
- **Service boundary:** auth, TLS/mTLS, API keys, OAuth/JWT where externally exposed.

For `bevy-gym`, the highest-value security rule is: keep headless simulation deterministic and local
by default; add remote tooling only in examples/dev features with explicit documentation that remote
mutation is privileged.

_Source:_

- <https://docs.rs/bevy/latest/bevy/ecs/entity/struct.Entity.html>
- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://docs.rs/bevy/latest/bevy/remote/http/index.html>
- <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>

### Data Architecture Patterns

Bevy data architecture should separate **runtime ECS data** from **persistent domain data**.

Runtime ECS data:

- components: environment state, pending action, current observation, statistics, render handles,
- resources: global config, learners/sessions, asset collections, metrics,
- messages: transitions, episode ends, action requests,
- states: major app mode,
- assets: meshes, materials, scenes, external content.

Persistent domain data:

- experiment config,
- random seeds,
- environment IDs and stable episode/step IDs,
- observations/actions/rewards/statuses,
- model checkpoints,
- benchmark summaries,
- telemetry logs,
- screenshots/videos,
- scenario/content files.

Do not persist raw Bevy world state as the source of truth for RL experiments unless the goal is an
editor/tool snapshot. Bevy scenes and reflection are useful for tooling and content, but RL
reproducibility should be driven by stable domain configuration and environment seeds.

For large Bevy projects, use these data rules:

- Put high-frequency per-entity data in components.
- Put cross-cutting singleton data in resources, but avoid bloated "god resources."
- Use stable IDs for persisted domain facts.
- Keep asset handles in ECS, asset bytes/files outside ECS.
- Keep checkpoints and replay buffers in explicit storage, not hidden in components.
- Version all external artifact schemas.

_Source:_

- <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- <https://docs.rs/bevy_scene/latest/bevy_scene/>
- <https://docs.rs/bevy/latest/bevy/asset/index.html>
- <https://docs.rs/bevy/latest/bevy/ecs/entity/struct.Entity.html>
- Local: `src/components.rs`, `src/events.rs`

### Deployment and Operations Architecture

The operational architecture should be lane-based:

- **Core lane:** no default features, headless ECS only, unit/integration tests.
- **Render lane:** `render` plus platform backend (`x11`, `wayland`, or `winit`) and visual smoke
  tests.
- **Example lane:** runnable examples, including release-mode CartPole smoke runs.
- **Docs lane:** doctests and docs.rs feature set.
- **Benchmark lane:** environment throughput, message overhead, render sync overhead.
- **Migration lane:** Bevy version spike branch with compatibility table updates.

Bevy plugin-development guidance recommends CI even for plugins with few tests, small crate size,
minimal feature selection, optional dependencies behind Cargo features, and a visible Bevy-version
compatibility table. This fits `bevy-gym` directly because it is a reusable Bevy plugin crate.

Deployment patterns:

- Use `MinimalPlugins`/`ScheduleRunnerPlugin` for headless training and CI.
- Use `DefaultPlugins` only for rendered examples or apps.
- Keep dynamic linking as a local development optimization, not a release architecture.
- Use containers only around headless workers or reproducible benchmarks, not as an excuse to couple
  Bevy runtime code to infrastructure.
- Treat Bevy 0.19 release candidates as migration-spike input, not mainline, until 0.19 final and
  plugin compatibility are verified.

_Source:_

- <https://bevy.org/learn/quick-start/plugin-development/>
- <https://bevy.org/learn/quick-start/getting-started/setup/>
- <https://docs.rs/bevy/latest/bevy/app/struct.ScheduleRunnerPlugin.html>
- <https://bevy.org/news/bevy-0-18/>
- Local: `Cargo.toml`, `README.md`, `docs/examples/cartpole.md`

### Architecture Decision Candidates for `bevy-gym`

These are the architectural decisions worth recording before the project grows:

1. **Core architecture:** `bevy-gym` remains a headless-first Bevy plugin crate; rendering and
   tooling are optional layers.
2. **Scheduling:** environment stepping/resetting stays in `FixedUpdate`; rendering sync stays in
   `Update`.
3. **Ordering:** `GymSet` remains the public ordering contract for policy/learning systems.
4. **Communication:** use Bevy messages for scheduled policy/experience/episode flow; reserve
   observers for immediate local reactions.
5. **Persistence:** persist domain artifacts with stable IDs; do not persist raw Bevy entity IDs as
   durable identifiers.
6. **Features:** keep Bevy `default-features = false`; add new tools only behind additive Cargo
   features.
7. **Workspace split:** split crates only when dependency or validation lanes diverge.
8. **Remote tooling:** BRP/MCP are dev/example features, loopback by default, and documented as
   privileged mutation surfaces.

## Implementation Approaches and Technology Adoption

### Technology Adoption Strategies

Adopt Bevy incrementally. Keep `bevy-gym` on Bevy `0.18.1` for mainline work, treat `0.19` release
candidates as migration-spike material, and require a compatibility pass before moving the library
baseline. Bevy's own plugin-development guidance emphasizes small crate size, minimal Bevy features,
optional dependencies, CI, examples, and visible Bevy-version compatibility tables. Those are the
right adoption controls for this repo.

For `bevy-gym`, adoption should be lane-based rather than big-bang:

- mainline: Bevy 0.18.1, headless-first, stable crate API;
- migration spike: Bevy 0.19 final/RC branch, isolated from mainline until dependencies and examples
  pass;
- tooling spike: BRP/MCP/inspector/capture behind dev/example features;
- physics/editor spike: Avian/Rapier/Jackdaw experiments outside the core crate.

_Source:_

- <https://bevy.org/learn/quick-start/plugin-development/>
- <https://bevy.org/learn/contribute/project-information/release-process/>
- <https://bevy.org/news/bevy-0-18/>
- Local: `Cargo.toml`, `Cargo.lock`, `ref/bevy/Cargo.toml`

### Development Workflows and Tooling

The development workflow should prioritize fast headless feedback first, then render/tooling
compatibility. The current CI only runs `cargo check --features x11` and
`cargo clippy --features x11 -- -D warnings`; that misses the no-default-features core lane, tests,
docs, formatting, examples, and platform feature combinations.

Recommended workflow lanes:

- `cargo fmt --check`
- `cargo clippy --all-targets --no-default-features -- -D warnings`
- `cargo clippy --all-targets --features render,x11 -- -D warnings`
- `cargo test --no-default-features`
- `cargo test --features render,x11`
- `cargo test --doc --all-features`
- example build/run smoke tests, especially CartPole headless and render variants
- release-mode smoke for throughput-sensitive examples

Use the Nix dev shell to keep local native render dependencies reproducible. Keep Bevy dynamic
linking and alternative linkers as local developer-speed options, not release settings.

_Source:_

- <https://bevy.org/learn/quick-start/getting-started/setup/>
- <https://doc.rust-lang.org/cargo/commands/cargo-test.html>
- <https://doc.rust-lang.org/clippy/usage.html>
- <https://nnethercote.github.io/perf-book/build-configuration.html>
- Local: `.github/workflows/ci.yml`, `flake.nix`, `README.md`

### Testing and Quality Assurance

Use three levels of tests:

1. Pure Rust tests for environment-independent logic, policies, and data transforms.
2. Bevy `App` tests for plugin wiring, schedule order, messages, resets, and render feature
   compilation.
3. Example/release smoke tests for real user paths.

For `bevy-gym`, high-value Bevy tests include:

- plugin creates exactly `num_envs` environment entities;
- `GymSet` order is `Step -> AutoReset -> ManualReset`;
- a pending action produces an `ExperienceEvent` and next `ActionRequestEvent`;
- terminal/truncated episodes produce `EpisodeEndEvent` and reset in the same fixed tick;
- manual `ResetRequested` removes the marker and emits a new action request;
- render plugin setup runs in `First` and sync runs in `Update`;
- docs examples compile against Bevy 0.18 message names.

Use benchmarks for performance claims. Measure environment count scaling, `par_iter_mut()`
throughput, serial message write overhead, reset overhead, and render sync overhead.

_Source:_

- <https://docs.rs/bevy/latest/bevy/app/struct.ScheduleRunnerPlugin.html>
- <https://docs.rs/bevy/latest/bevy/struct.MinimalPlugins.html>
- <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- <https://doc.rust-lang.org/cargo/commands/cargo-test.html>
- Local: `src/plugin.rs`, `src/systems/step.rs`, `src/systems/reset.rs`, `src/render.rs`

### Deployment and Operations Practices

Deploy `bevy-gym` as a library and example suite, not as a monolithic application. Operations should
focus on repeatable checks, crates.io publishing safety, and deterministic example artifacts.

Recommended operations changes:

- run CI on docs-only PRs if the docs include Rust snippets or user-facing commands;
- make publish dependent on the fuller matrix, not only `x11`;
- avoid `cargo publish --allow-dirty` unless there is a specific packaging reason and a pre-publish
  verification step;
- add a release checklist with version bump, changelog/release notes, CI matrix,
  `cargo publish --dry-run`, and docs.rs feature check;
- keep run outputs under `runs/` out of source truth unless deliberately committed as fixtures.

Headless workers and benchmarks can run in CI or containers later, but core architecture should
remain independent of any one deployment platform.

_Source:_

- <https://doc.rust-lang.org/cargo/commands/cargo-publish.html>
- <https://doc.rust-lang.org/cargo/commands/cargo-package.html>
- <https://bevy.org/learn/quick-start/plugin-development/>
- Local: `.github/workflows/ci.yml`, `Cargo.toml`, `runs/`

### Team Organization and Skills

The necessary skill set for larger Bevy work is specific:

- Rust ownership/borrowing/generics and trait bounds;
- Bevy ECS query design, component storage, and schedules;
- message vs observer semantics in Bevy 0.18;
- fixed timestep simulation;
- feature-flag design and Cargo workspaces;
- headless testing and CI matrix design;
- profiling/benchmarking before optimization;
- Bevy migration-guide reading and plugin compatibility evaluation.

For solo or AI-assisted development, preserve small scoped tasks: one plugin boundary, one system
set, one example, one benchmark, or one migration spike at a time. Avoid broad rewrites where a
focused test or example can validate the pattern.

_Source:_

- <https://bevy.org/learn/quick-start/getting-started/ecs/>
- <https://bevy.org/learn/quick-start/plugin-development/>
- <https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html>
- Local: `README.md`, `docs/plugins/bevy_gym_plugin.md`

### Cost Optimization and Resource Management

The main costs are compile time, CI time, local iteration time, and simulation throughput.

Cost controls:

- keep Bevy default features disabled in library code;
- keep render/window/tooling dependencies optional;
- test core headless behavior without compiling render stacks;
- use `render,x11` or `render,wayland` only for render lanes;
- use dynamic linking/fast linkers locally when useful;
- measure before adding new dependencies;
- prefer examples/dev-dependencies for heavy integrations;
- avoid broad workspace splits until dependency boundaries justify them.

For runtime cost, keep hot data in ECS components, avoid unnecessary shared mutable resources in
`FixedUpdate`, and avoid large cloned payloads in messages. Persist large artifacts outside ECS and
reference them by stable IDs or paths.

_Source:_

- <https://doc.rust-lang.org/cargo/reference/features.html>
- <https://bevy.org/learn/quick-start/getting-started/setup/>
- <https://nnethercote.github.io/perf-book/build-configuration.html>
- <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- Local: `Cargo.toml`, `src/systems/step.rs`

### Risk Assessment and Mitigation

Primary risks:

- Bevy release churn: mitigate with version pinning, migration branches, and compatibility tables.
- Plugin compatibility drift: mitigate by feature-gated spikes and docs.rs/crates.io verification
  before adoption.
- CI blind spots: mitigate with no-default, render, docs, tests, examples, and release lanes.
- Performance regressions: mitigate with benchmark baselines and release-mode smoke runs.
- Remote tooling exposure: mitigate by feature gates, loopback defaults, auth/TLS at external
  boundaries, and no secrets in reflected ECS state.
- Over-architecture: mitigate by staying single-crate until dependency lanes justify a workspace
  split.
- Documentation/API drift: mitigate by doctests and consistency checks for Bevy 0.18 message
  terminology.

_Source:_

- <https://bevy.org/learn/quick-start/plugin-development/>
- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://docs.rs/bevy/latest/bevy/ecs/entity/struct.Entity.html>
- Local: `src/lib.rs`, `README.md`, `.github/workflows/ci.yml`

## Technical Research Recommendations

### Implementation Roadmap

1. Document the current Bevy 0.18.1 baseline and compatibility table.
2. Tighten CI: fmt, clippy no-default/render, tests no-default/render, docs, examples.
3. Add focused `App` tests for plugin setup, message flow, reset flow, and system ordering.
4. Add benchmark scaffolding for headless step throughput and message overhead.
5. Fix public doc terminology to consistently use `MessageReader`/`MessageWriter`.
6. Add release checklist and dry-run publish validation.
7. Create optional feature-gated spikes for BRP/MCP, Avian/Rapier, and render capture only after the
   core lanes are stable.
8. Consider workspace split only when tool/render/physics dependencies make the core crate slower or
   less stable.

### Technology Stack Recommendations

- Core: Rust + Bevy 0.18.1 + `rl-traits`.
- Simulation: `FixedUpdate`, `MinimalPlugins`, `ScheduleRunnerPlugin`, `par_iter_mut()`.
- Rendering: optional `render` plus platform backend.
- Tooling: optional BRP/MCP/inspector/capture features, not core.
- CI: stable Rust toolchain plus Nix dev shell for local native dependencies.
- Storage: stable run artifacts and checkpoints outside ECS.

### Skill Development Requirements

- Bevy ECS query/schedule design.
- Cargo features/workspaces/profiles.
- Bevy migration guide discipline.
- Rust test, doctest, and benchmark patterns.
- Profiling and performance measurement.
- Secure remote-tool boundary design.

### Success Metrics and KPIs

- CI covers no-default core, render, docs, tests, examples, and publish dry-run.
- Headless CartPole smoke remains fast and deterministic enough for regression detection.
- Benchmark tracks steps/sec across environment counts.
- Public docs compile and use Bevy 0.18 terminology.
- No core dependency is added without a feature gate or compatibility note.
- Bevy migration spikes produce a written compatibility result before mainline upgrade.

## Research Synthesis: Comprehensive Bevy Technical Research for Performance, Stability, Idiomatic Patterns, and Larger Project Structure

### Executive Summary

Bevy is a Rust data-driven game engine and app framework built around ECS, modular plugins, optional
feature sets, and parallel app logic. That makes it a strong fit for `bevy-gym` as a simulation
harness, but only if the project keeps a disciplined boundary between the small reusable core and
heavier application/tooling layers. As of 2026-06-12, the local repository resolves Bevy to
`0.18.1`; docs.rs lists `bevy/latest` as `0.18.1`, while `0.19.0-rc.3` is available as a release
candidate. The stable recommendation is therefore Bevy 0.18.1 for mainline work, with Bevy 0.19
tracked in a separate migration spike.

The architecture should stay headless-first. `MinimalPlugins`, `ScheduleRunnerPlugin`,
`FixedUpdate`, explicit `GymSet` ordering, and optional rendering match both Bevy's plugin model and
the needs of RL workloads. Bevy's plugin guidance reinforces this direction: keep plugin crates
small, use `default-features = false`, add only required Bevy features, put optional dependencies
behind Cargo features, run CI, provide examples, and document compatible Bevy versions.

The highest-value next work is not a broad rewrite. It is a stability pass: extend CI lanes, add
focused Bevy `App` tests, benchmark headless stepping, fix public docs to consistently use Bevy 0.18
message terminology, and publish a compatibility/release checklist. Large-project structure should
grow only when dependency or validation lanes force it.

**Key Technical Findings:**

- Bevy's design goals align with `bevy-gym`: ECS data orientation, modularity, optional features,
  and parallel app logic.
- Stability depends on version discipline because Bevy is still pre-1.0 and releases breaking
  changes regularly with migration guides.
- `bevy-gym` already has the right core shape: Bevy `default-features = false`, optional rendering,
  and headless simulation through ECS schedules.
- The current local CI is too narrow for a reusable plugin crate because it mainly checks the `x11`
  feature lane.
- Bevy Remote Protocol and MCP-style tooling are powerful but privileged mutation surfaces; they
  belong behind dev/example features, loopback defaults, and explicit documentation.

**Technical Recommendations:**

- Keep Bevy 0.18.1 as the mainline baseline until a Bevy 0.19 final migration branch passes
  compatibility checks.
- Preserve a headless core with optional render/tooling/physics features.
- Add `App` tests for plugin setup, message flow, reset behavior, and system ordering.
- Add benchmarks before optimizing message or stepping internals.
- Use Cargo features and, later, workspaces only when dependency boundaries justify the added
  structure.

### Table of Contents

1. Technical Research Introduction and Methodology
2. Bevy Technical Landscape and Architecture Analysis
3. Implementation Approaches and Best Practices
4. Technology Stack Evolution and Current Trends
5. Integration and Interoperability Patterns
6. Performance and Scalability Analysis
7. Security and Compliance Considerations
8. Strategic Technical Recommendations
9. Implementation Roadmap and Risk Assessment
10. Future Technical Outlook and Innovation Opportunities
11. Technical Research Methodology and Source Verification
12. Technical Appendices and Reference Materials

## 1. Technical Research Introduction and Methodology

### Technical Research Significance

Bevy's relevance to `bevy-gym` is not primarily "game engine rendering." It is Bevy as an ECS app
framework: deterministic scheduling, typed components/resources/messages, modular plugins, and
parallel system execution. For reinforcement-learning environments, those properties matter because
the runtime must step many environments, collect transitions, reset terminal episodes, expose
policy/learner hooks, and optionally visualize state without making rendering a core requirement.

_Technical Importance:_ Bevy's modular ECS design gives `bevy-gym` a way to build RL simulation
workflows using normal Bevy schedules and systems rather than a separate bespoke runtime.

_Project Impact:_ The right architecture can make `bevy-gym` useful both as a library and as an
example suite without forcing users to compile a full render/window stack for headless training.

_Source:_

- <https://docs.rs/crate/bevy/latest>
- <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- Local: `Cargo.toml`, `src/plugin.rs`, `src/systems/step.rs`

### Technical Research Methodology

The research used a source-priority method:

- **Technical Scope:** Bevy versioning, ECS patterns, schedules, messages, optional rendering,
  remote tooling, Cargo features/workspaces, CI, testing, benchmarking, and larger project
  structure.
- **Data Sources:** Bevy official release/setup/plugin documentation, docs.rs API pages, Cargo Book
  pages, Rust performance guidance, MCP specifications, and local repository files.
- **Analysis Framework:** Compare current authoritative documentation with the local `bevy-gym`
  implementation, then identify stable decisions, risk areas, and next implementation controls.
- **Time Period:** Current state as of 2026-06-12, with Bevy 0.18.1 stable and Bevy 0.19 release
  candidates treated as migration evidence only.
- **Technical Depth:** Practical engineering guidance for a reusable Bevy plugin crate rather than a
  generic Bevy tutorial.

### Technical Research Goals and Objectives

**Original Technical Goals:** Research current Bevy performance, stability, idiomatic ECS patterns,
and large-project structure guidance for `bevy-gym` and larger Bevy applications.

**Achieved Technical Objectives:**

- Identified the stable Bevy baseline and release-candidate boundary.
- Mapped local `bevy-gym` architecture to Bevy idioms.
- Defined headless, render, docs, examples, benchmark, publish, and migration validation lanes.
- Documented feature-gating and project-structure guidance for scaling without premature workspace
  complexity.
- Flagged API documentation drift around Bevy 0.18 `MessageReader`/`MessageWriter` terminology.

## 2. Bevy Technical Landscape and Architecture Analysis

### Current Technical Architecture Patterns

The dominant Bevy architecture pattern is a plugin-composed ECS app. Application logic is organized
into systems scheduled into app schedules; state is represented by components and resources;
communication can use messages or observers depending on timing semantics; features are controlled
through Cargo feature flags.

For `bevy-gym`, this maps cleanly:

- `BevyGymPlugin` configures app behavior.
- One ECS entity represents each environment instance.
- `FixedUpdate` owns simulation stepping and reset flow.
- `GymSet` exposes ordering for downstream policy/learner systems.
- Messages carry action requests, experience transitions, and episode endings.
- Rendering is optional and syncs visual state separately.

_Dominant Patterns:_ Plugin boundary, ECS data modeling, explicit system sets, feature-gated
optional capabilities.

_Architectural Evolution:_ Bevy 0.17/0.18 continued tightening ECS APIs, messages, observers,
relationships, and migration guides. These changes increase Bevy's power but reinforce the need for
version-specific docs and tests.

_Architectural Trade-offs:_ Bevy's flexibility makes it easy to overbuild. The safer pattern for
`bevy-gym` is to keep the core crate narrow and push uncertain tooling into examples or spike
branches.

_Source:_

- <https://docs.rs/crate/bevy/latest>
- <https://bevy.org/news/bevy-0-18/>
- <https://bevy.org/learn/migration-guides/0-17-to-0-18/>
- Local: `src/plugin.rs`, `src/events.rs`, `src/render.rs`

### System Design Principles and Best Practices

The key design principle is to model runtime facts where Bevy expects them:

- high-frequency per-env state as components;
- small global configuration as resources;
- scheduled cross-system communication as messages;
- app-level modes as states;
- visual/content data as assets and handles;
- durable RL artifacts outside ECS with stable domain IDs.

Use stable identifiers for persisted experiment facts. Do not persist raw Bevy entity IDs as durable
experiment identifiers. Entity IDs are runtime handles, not a long-term storage schema.

_Design Principles:_ Small plugin core, explicit schedules, typed data, stable external IDs,
optional features, measured performance.

_Best Practice Patterns:_ Add abstractions only around real repeated boundaries: environment
lifecycle, policy/action flow, episode accounting, rendering sync, and benchmark/test harnesses.

_Architectural Quality Attributes:_ Maintainability comes from narrow features and tests;
performance comes from ECS-friendly data and benchmarks; stability comes from pinned versions and
migration gates.

_Source:_

- <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- <https://docs.rs/bevy/latest/bevy/ecs/entity/struct.Entity.html>
- <https://docs.rs/bevy_scene/latest/bevy_scene/>
- Local: `src/components.rs`, `src/events.rs`

## 3. Implementation Approaches and Best Practices

### Current Implementation Methodologies

The recommended implementation method is incremental and lane-based:

- mainline stays on Bevy 0.18.1;
- Bevy 0.19 is evaluated in a migration spike;
- render examples stay behind `render` plus platform features;
- physics/editor/capture/remote tooling stays outside the core until proven;
- CI expands before the dependency graph expands.

This approach matches Bevy plugin-development guidance around small crates, minimal feature
activation, optional dependencies, examples, CI, and version compatibility tables.

_Development Approaches:_ One scoped change per plugin boundary, schedule, message flow, example,
benchmark, or migration spike.

_Code Organization Patterns:_ Keep the core library cohesive while grouping systems by lifecycle:
spawn, step, reset, render sync, stats, and examples.

_Quality Assurance Practices:_ Use pure Rust tests for domain logic, Bevy `App` tests for
plugin/schedule/message behavior, and example smoke tests for user workflows.

_Deployment Strategies:_ Treat `bevy-gym` as a library plus examples. Use publish dry runs and docs
checks before crates.io release.

_Source:_

- <https://bevy.org/learn/quick-start/plugin-development/>
- <https://doc.rust-lang.org/cargo/commands/cargo-test.html>
- <https://doc.rust-lang.org/cargo/commands/cargo-publish.html>
- Local: `.github/workflows/ci.yml`, `src/plugin.rs`

### Implementation Framework and Tooling

The implementation framework should be built around standard Rust and Bevy tools:

- Cargo features for optional render/tooling/physics integration;
- `cargo fmt` and Clippy for static checks;
- doctests for public examples;
- Bevy `App` tests for schedule behavior;
- release-mode examples and benchmarks for throughput-sensitive paths;
- Nix dev shell for repeatable native dependencies.

_Development Frameworks:_ Rust, Bevy 0.18.1, `rl-traits`, and optional Bevy render/window features.

_Tool Ecosystem:_ Cargo, rustfmt, Clippy, docs.rs, GitHub Actions, Nix, and optional Bevy tooling in
dev/example lanes.

_Build and Deployment Systems:_ Use CI feature matrices and `cargo publish --dry-run`; avoid relying
on `--allow-dirty` as the default publish posture.

_Source:_

- <https://doc.rust-lang.org/cargo/reference/features.html>
- <https://doc.rust-lang.org/clippy/usage.html>
- <https://doc.rust-lang.org/cargo/commands/cargo-publish.html>
- Local: `flake.nix`, `.github/workflows/ci.yml`

## 4. Technology Stack Evolution and Current Trends

### Current Technology Stack Landscape

The current stack should remain:

- **Language:** Rust.
- **Engine/App framework:** Bevy 0.18.1.
- **Core Bevy surface:** ECS, app schedules, messages, minimal plugins, optional render/window
  stack.
- **RL integration:** `rl-traits` plus local examples.
- **Optional integrations:** rendering, BRP/MCP, inspector/capture, physics, and editor tooling only
  when feature-gated or isolated.

Bevy's own docs emphasize modularity and feature control. Cargo features are additive; a plugin that
enables heavy Bevy features forces them on downstream users. That makes minimal feature activation a
compatibility and compile-time concern, not only a style preference.

_Programming Languages:_ Rust is primary. WGSL only matters for custom shader work.
TOML/RON/JSON/serde-friendly formats are practical for configuration and test fixtures.

_Frameworks and Libraries:_ Bevy 0.18.1, Bevy ECS/app/message/state/render crates, `rl-traits`,
optional physics/tooling crates after compatibility verification.

_Database and Storage Technologies:_ Use explicit files/artifacts for seeds, run metadata,
checkpoints, metrics, benchmark outputs, screenshots, and videos.

_API and Communication Technologies:_ Bevy messages internally; BRP/MCP externally only for
controlled tooling surfaces.

_Source:_

- <https://docs.rs/crate/bevy/latest>
- <https://doc.rust-lang.org/cargo/reference/features.html>
- <https://bevy.org/learn/quick-start/plugin-development/>
- Local: `Cargo.toml`, `Cargo.lock`

### Technology Adoption Patterns

Adoption should follow compatibility evidence:

- adopt Bevy stable releases after local tests/examples pass;
- test release candidates in branches before mainline upgrades;
- require plugin compatibility checks for ecosystem dependencies;
- keep new heavy dependencies out of the core by default;
- publish compatibility information in README/docs.

_Adoption Trends:_ Bevy continues active pre-1.0 evolution; migration guides are normal operating
input.

_Migration Patterns:_ Use branch-based migration spikes with written findings, not opportunistic
dependency bumps.

_Emerging Technologies:_ BRP, MCP tooling, advanced editor workflows, physics integrations, and
render capture can be valuable, but they should remain optional until core RL behavior is stable.

_Source:_

- <https://bevy.org/learn/contribute/project-information/release-process/>
- <https://bevy.org/learn/migration-guides/0-17-to-0-18/>
- <https://bevy.org/assets/>
- Local: `ref/bevy/Cargo.toml`

## 5. Integration and Interoperability Patterns

### Current Integration Approaches

The best internal integration pattern is Bevy-native scheduling. Policy systems should read
`ActionRequestEvent` messages, write `PendingAction` components, and rely on `GymSet` ordering to
avoid hidden timing contracts. Experience and episode-end messages are appropriate because they
describe buffered scheduled data flow.

External integrations should be thin:

- training frameworks communicate through examples or adapters;
- assets/config live outside ECS and enter through handles or parsed resources;
- BRP/MCP tooling runs only in dev/example feature lanes;
- physics integrations are environment-specific, not core dependencies.

_API Design Patterns:_ Typed Rust APIs for library code, Bevy messages for app flow, stable external
IDs for persisted data, optional remote APIs for tools.

_Service Integration:_ Avoid service-style coupling in the crate core. Use CLI/examples/workers
around the library when needed.

_Data Integration:_ Version external schemas and keep run artifacts outside ECS.

_Source:_

- <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>
- Local: `src/events.rs`, `src/systems/step.rs`

### Interoperability Standards and Protocols

For Bevy tooling, the main protocol-level surface is the Bevy Remote Protocol, which uses JSON-RPC
2.0 and can inspect or alter ECS state once transports are enabled. For AI/tool orchestration, MCP
exposes a model where clients connect to servers and invoke tools over defined transports. Both are
useful for diagnostics and automation, but both should be treated as privileged mutation surfaces.

_Standards Compliance:_ JSON-RPC for BRP, MCP for AI tool interfaces, Cargo semver/features for Rust
crate compatibility.

_Protocol Selection:_ Use direct Rust APIs inside the crate; use BRP/MCP only when remote inspection
or tooling is the actual requirement.

_Integration Challenges:_ The main challenge is avoiding accidental expansion of the core crate's
dependency and security surface.

_Source:_

- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>
- <https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html>

## 6. Performance and Scalability Analysis

### Performance Characteristics and Optimization

The most important performance rule is to measure before optimizing. `bevy-gym` has a plausible
performance shape today: many environments step in `FixedUpdate`, environment stepping uses
`par_iter_mut()`, and messages are written after step outcomes are collected. The likely bottlenecks
are environment count scaling, mutex collection overhead, serial message writes, reset costs, clone
costs in observations/actions, and optional render sync.

Recommended benchmark axes:

- environments: 1, 4, 16, 64, 256;
- headless fixed ticks per second;
- observation/action payload size;
- terminal/reset frequency;
- message volume per tick;
- render sync enabled vs disabled;
- debug vs release profiles.

_Performance Benchmarks:_ No durable benchmark baseline exists in this research artifact yet. Add it
before making throughput claims.

_Optimization Strategies:_ Keep hot data in components, avoid bloated global resources, reduce
clone-heavy messages where profiling supports it, keep rendering optional, and use release-mode
smoke runs for performance-sensitive examples.

_Monitoring and Measurement:_ Add benchmark output under a deliberate artifact path, track
steps/sec, resets/sec, message count, and per-system timing where possible.

_Source:_

- <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- <https://bevy.org/learn/quick-start/getting-started/setup/>
- <https://doc.rust-lang.org/cargo/reference/profiles.html>
- Local: `src/systems/step.rs`, `src/systems/reset.rs`

### Scalability Patterns and Approaches

Scale `bevy-gym` through ECS and feature discipline before crate splitting:

- keep per-env state as components;
- parallelize independent env stepping;
- keep core systems deterministic and narrow;
- separate render/tooling/physics features from headless core checks;
- split into a workspace only when dependency validation lanes diverge.

_Scalability Patterns:_ ECS data decomposition, optional feature lanes, explicit test/benchmark
lanes, and branch-based migration lanes.

_Capacity Planning:_ CI capacity should be planned by lane. Run cheap no-default checks first,
render/tooling lanes second, and release benchmarks on schedule or before releases.

_Elasticity and Auto-scaling:_ For future headless workers, scale at the process/job level rather
than coupling distributed execution into the library core.

_Source:_

- <https://doc.rust-lang.org/cargo/reference/workspaces.html>
- <https://doc.rust-lang.org/cargo/reference/features.html>
- <https://docs.rs/bevy/latest/bevy/prelude/struct.MinimalPlugins.html>

## 7. Security and Compliance Considerations

### Security Best Practices and Frameworks

The core library has a modest security surface if it stays local, headless, and dependency-light.
The risk increases when remote tooling is enabled. BRP can inspect and mutate ECS state. MCP tools
can expose executable operations to external clients. Those capabilities are useful for development
and automation, but not safe to enable casually in production or in shared networks.

Security controls:

- feature-gate BRP/MCP and inspector tooling;
- default remote transports to loopback;
- document mutation capability clearly;
- avoid exposing secrets or credentials as reflected ECS resources/components;
- require auth/TLS/reverse-proxy controls if a remote surface crosses a trust boundary;
- keep publish workflows free of embedded tokens and secret output.

_Security Frameworks:_ Least privilege, explicit opt-in features, local-only defaults, and
documented trust boundaries.

_Threat Landscape:_ Accidental remote ECS mutation, data exposure through reflected
components/resources, CI/publish token leakage, and dependency drift.

_Secure Development Practices:_ CI checks, dependency review, feature-gated optional dependencies,
dry-run publishing, and no secrets in examples.

_Source:_

- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>
- <https://doc.rust-lang.org/cargo/commands/cargo-publish.html>

### Compliance and Regulatory Considerations

There is no special regulatory compliance requirement inherent to Bevy or `bevy-gym` from the
evidence inspected. The relevant governance work is open-source hygiene: licensing, dependency
metadata, reproducible release checks, compatibility documentation, and safe handling of generated
artifacts.

_Industry Standards:_ Rust crate semver expectations, Cargo feature discipline, MIT/Apache-style
ecosystem conventions where applicable, and documented Bevy compatibility tables.

_Regulatory Compliance:_ Not directly applicable unless future environments ingest regulated data.

_Audit and Governance:_ Keep release checklists, CI logs, benchmark baselines, docs.rs results, and
`cargo publish --dry-run` output as release evidence.

_Source:_

- <https://bevy.org/learn/quick-start/plugin-development/>
- <https://doc.rust-lang.org/cargo/commands/cargo-package.html>

## 8. Strategic Technical Recommendations

### Technical Strategy and Decision Framework

Use this decision framework:

1. Does the change affect the headless core? If yes, require no-default tests and `App` tests.
2. Does it add render/window/physics/tooling dependencies? If yes, feature-gate or isolate it.
3. Does it depend on a Bevy release candidate or ecosystem plugin? If yes, run it as a spike.
4. Does it change public docs/API examples? If yes, run doctests or at least compile example
   snippets.
5. Does it claim performance improvement? If yes, require a benchmark.

_Architecture Recommendations:_ Headless-first single crate with optional feature lanes.

_Technology Selection:_ Rust + Bevy 0.18.1 + `rl-traits` in core; optional Bevy render/tooling only
outside the minimum path.

_Implementation Strategy:_ Tests and CI before dependency expansion; benchmark before optimization;
migrate through branches.

_Source:_

- <https://bevy.org/learn/quick-start/plugin-development/>
- <https://doc.rust-lang.org/cargo/reference/features.html>
- Local: `Cargo.toml`, `.github/workflows/ci.yml`

### Competitive Technical Advantage

`bevy-gym` can differentiate by being a Bevy-native RL harness rather than a thin wrapper around a
renderer. The advantage is ECS composability: policies, logging, visualization, resets, domain
randomization, curriculum, and tools can be normal systems plugged into a schedule.

_Technology Differentiation:_ Fast headless ECS runtime with optional visual/debug overlays.

_Innovation Opportunities:_ Bevy-native curriculum systems, batched environment visualization,
deterministic replay artifacts, BRP/MCP-controlled inspection, and integration examples for common
RL workflows.

_Strategic Technology Investments:_ Invest in tests, benchmarks, docs, compatibility tables, and
examples before heavy integrations.

_Source:_

- <https://docs.rs/crate/bevy/latest>
- <https://docs.rs/bevy/latest/bevy/prelude/trait.Plugin.html>
- Local: `README.md`, `docs/examples/cartpole.md`

## 9. Implementation Roadmap and Risk Assessment

### Technical Implementation Framework

Recommended phases:

1. **Stabilize baseline:** document Bevy 0.18.1 compatibility, fix message terminology, and add
   release checklist.
2. **Expand CI:** fmt, clippy no-default/render, tests no-default/render, doctests, example checks,
   publish dry-run.
3. **Add Bevy `App` tests:** plugin setup, env spawn count, `GymSet` ordering, action/experience
   messages, auto/manual reset, render schedule placement.
4. **Add benchmarks:** headless step throughput, message overhead, reset overhead, render sync
   overhead.
5. **Create optional spikes:** BRP/MCP, physics, capture, inspector, and Bevy 0.19 migration.
6. **Revisit structure:** split workspace crates only after dependency boundaries create measurable
   friction.

_Implementation Phases:_ Stabilize, validate, benchmark, spike, then restructure only if needed.

_Technology Migration Strategy:_ Use a migration branch with source-backed compatibility notes and
passing examples.

_Resource Planning:_ Focus effort on Rust/Bevy ECS testing, Cargo feature hygiene, benchmark design,
and docs.

_Source:_

- <https://doc.rust-lang.org/cargo/commands/cargo-test.html>
- <https://bevy.org/learn/migration-guides/0-17-to-0-18/>
- Local: `src/plugin.rs`, `src/systems/reset.rs`, `src/render.rs`

### Technical Risk Management

Primary risks and mitigations:

- **Bevy release churn:** pin stable baseline, use migration branches, maintain compatibility table.
- **Plugin compatibility drift:** verify ecosystem crates against exact Bevy version before
  adoption.
- **CI blind spots:** add no-default, render, docs, example, test, and publish lanes.
- **Performance regressions:** add benchmark baselines and release-mode smoke checks.
- **Remote tooling exposure:** feature-gate, loopback defaults, clear trust boundaries.
- **Over-architecture:** stay single-crate until dependencies and validation lanes justify workspace
  split.
- **Docs/API drift:** doctest examples and standardize on Bevy 0.18 message terminology.

_Technical Risks:_ Version churn, dependency bloat, unclear schedule ordering, remote mutation
surfaces, and unmeasured performance.

_Implementation Risks:_ Premature workspace split, broad refactors, and adopting release candidates
as defaults.

_Business Impact Risks:_ User confusion, slow compile times, fragile examples, and unreliable
release publishing.

_Source:_

- <https://docs.rs/crate/bevy/latest>
- <https://bevy.org/learn/quick-start/plugin-development/>
- <https://doc.rust-lang.org/cargo/commands/cargo-publish.html>

## 10. Future Technical Outlook and Innovation Opportunities

### Emerging Technology Trends

Near term, expect Bevy to continue improving ECS ergonomics, observers/messages, rendering, scenes,
relationships, and tooling while still requiring migration attention. The 0.19 release-candidate
line indicates ongoing active development; it should be tracked but not used as the default until
final release and local compatibility checks pass.

_Near-term Technical Evolution:_ Bevy 0.19 final migration evaluation, docs terminology cleanup, CI
expansion, and benchmark baseline.

_Medium-term Technology Trends:_ Better editor/tooling surfaces, richer BRP/MCP workflows, more
mature physics integration examples, and larger example suites.

_Long-term Technical Vision:_ `bevy-gym` can become a Bevy-native RL simulation framework with
headless throughput, optional visual debugging, reproducible artifacts, and tool-driven inspection.

_Source:_

- <https://docs.rs/crate/bevy/latest>
- <https://bevy.org/news/bevy-0-18/>
- <https://bevy.org/learn/contribute/project-information/release-process/>

### Innovation and Research Opportunities

High-value research opportunities:

- benchmark ECS-based RL stepping against equivalent non-Bevy harnesses;
- explore deterministic replay with stable domain artifacts;
- build feature-gated BRP/MCP inspection for running environments;
- prototype physics-backed environments in examples before core adoption;
- compare `Message` payload designs for large observation/action types;
- evaluate workspace split only after measuring compile/test friction.

_Research Opportunities:_ Performance baselines, deterministic artifacts, tool-driven environment
inspection, and feature-gated physics/render examples.

_Emerging Technology Adoption:_ Adopt through spikes with acceptance criteria, not direct dependency
upgrades.

_Innovation Framework:_ One isolated spike, one written finding, one validation lane, one decision.

_Source:_

- <https://docs.rs/bevy/latest/bevy/remote/index.html>
- <https://modelcontextprotocol.io/docs/learn/architecture>
- Local: `examples/`, `runs/`

## 11. Technical Research Methodology and Source Verification

### Comprehensive Technical Source Documentation

**Primary Technical Sources:**

- Bevy crate and version docs: <https://docs.rs/crate/bevy/latest>
- Bevy 0.18 release notes: <https://bevy.org/news/bevy-0-18/>
- Bevy plugin-development guidance: <https://bevy.org/learn/quick-start/plugin-development/>
- Bevy ECS docs: <https://docs.rs/bevy_ecs/latest/bevy_ecs/>
- Bevy messages: <https://docs.rs/bevy/latest/bevy/ecs/message/trait.Message.html>
- Bevy `MinimalPlugins`: <https://docs.rs/bevy/latest/bevy/prelude/struct.MinimalPlugins.html>
- Bevy `DefaultPlugins`: <https://docs.rs/bevy/latest/bevy/prelude/struct.DefaultPlugins.html>
- Bevy Remote Protocol: <https://docs.rs/bevy/latest/bevy/remote/index.html>
- Cargo features: <https://doc.rust-lang.org/cargo/reference/features.html>
- Cargo workspaces: <https://doc.rust-lang.org/cargo/reference/workspaces.html>
- Cargo publish/package: <https://doc.rust-lang.org/cargo/commands/cargo-publish.html> and
  <https://doc.rust-lang.org/cargo/commands/cargo-package.html>
- MCP tools/transports: <https://modelcontextprotocol.io/specification/2025-06-18/server/tools> and
  <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>

**Local Sources:**

- `Cargo.toml`
- `Cargo.lock`
- `src/plugin.rs`
- `src/systems/step.rs`
- `src/systems/reset.rs`
- `src/components.rs`
- `src/events.rs`
- `src/render.rs`
- `src/lib.rs`
- `README.md`
- `docs/examples/cartpole.md`
- `.github/workflows/ci.yml`
- `flake.nix`
- `ref/bevy/Cargo.toml`

**Technical Web Search Queries Used:**

- `site:docs.rs/crate/bevy/latest bevy 0.18.1 0.19.0-rc.3 docs.rs`
- `site:bevy.org Bevy 0.18 release notes relationships observers stable Rust 1.85`
- `site:bevy.org Bevy plugin development small crate size tests CI compatible versions optional dependencies`
- `site:doc.rust-lang.org/cargo Cargo features workspaces publish dry-run allow-dirty`

### Technical Research Quality Assurance

_Technical Source Verification:_ Current facts were verified against Bevy official pages, docs.rs,
the Cargo Book, MCP specification pages, and local source files.

_Technical Confidence Levels:_

- High: Bevy 0.18.1 local baseline, Bevy plugin feature guidance, Cargo feature/publish behavior,
  local CI gaps, local message/render/system architecture.
- Medium: Future Bevy 0.19 migration impact, ecosystem plugin compatibility, and exact performance
  bottlenecks until benchmarks exist.
- Low: Any specific throughput claim before benchmarks are added.

_Technical Limitations:_ This research did not run performance benchmarks, run the crate test suite,
or perform a Bevy 0.19 migration spike. It is a source-backed technical research artifact, not
empirical benchmark evidence.

_Methodology Transparency:_ The final recommendations are conservative because they favor current
stable sources, local implementation evidence, and measurable validation over speculative ecosystem
adoption.

## 12. Technical Appendices and Reference Materials

### Detailed Technical Data Tables

| Area              | Current Position                    | Recommendation                                                 | Confidence |
| ----------------- | ----------------------------------- | -------------------------------------------------------------- | ---------- |
| Bevy version      | Local resolves to 0.18.1            | Keep 0.18.1 mainline; spike 0.19 separately                    | High       |
| Core features     | `default-features = false`          | Preserve minimal Bevy core                                     | High       |
| Runtime mode      | Headless-first with optional render | Keep render optional                                           | High       |
| CI                | Narrow `x11` check/clippy lanes     | Add no-default, render, docs, tests, examples, publish dry-run | High       |
| Messages          | Local event types derive `Message`  | Use `MessageReader`/`MessageWriter` terminology                | High       |
| Performance       | No benchmark baseline in artifact   | Add benchmarks before optimization claims                      | High       |
| Project structure | Single crate                        | Split workspace only after dependency lanes justify it         | Medium     |
| Remote tooling    | BRP/MCP possible                    | Feature-gate and document as privileged                        | High       |

### Technical Resources and References

_Technical Standards:_

- Cargo features, workspaces, package/publish commands.
- JSON-RPC basis for Bevy Remote Protocol.
- MCP tool/transport specifications for AI-accessible tools.

_Open Source Projects and Communities:_

- Bevy official repository/docs/releases.
- Bevy Assets ecosystem directory.
- Rust/Cargo documentation.

_Research Papers and Publications:_

- None required for this workflow; practical source-backed engineering guidance was sufficient.

_Technical Communities:_

- Bevy docs, Bevy examples, Bevy Assets, docs.rs, and Cargo documentation.

## Technical Research Conclusion

### Summary of Key Technical Findings

`bevy-gym` is already oriented in the right direction: a Bevy plugin crate with headless simulation,
optional rendering, ECS components/messages, and explicit schedule sets. The main gap is not
conceptual architecture. It is operational hardening: CI coverage, tests, benchmarks, docs
consistency, release checklist, and migration discipline.

### Strategic Technical Impact Assessment

Keeping the core small and stable protects downstream users from unnecessary compile cost and
dependency risk. It also leaves room for richer Bevy tooling where it belongs: feature-gated
examples, dev-only inspection, and migration spikes with explicit acceptance criteria.

### Next Steps Technical Recommendations

1. Add a Bevy compatibility table and release checklist.
2. Expand CI to no-default, render, docs, tests, examples, and publish dry-run lanes.
3. Add `App` tests for plugin wiring, message flow, reset flow, and schedule ordering.
4. Add benchmark scaffolding for headless step throughput.
5. Fix public docs to use `MessageReader`/`MessageWriter` consistently.
6. Create separate spikes for Bevy 0.19, BRP/MCP, physics, and capture tooling.

**Technical Research Completion Date:** 2026-06-12  
**Research Period:** Current comprehensive technical analysis  
**Source Verification:** Technical facts cited with current sources and local repository evidence  
**Technical Confidence Level:** High for architectural and stability recommendations; medium for
future ecosystem/plugin migration impact; benchmark-dependent for specific performance claims.
