---
stepsCompleted: [1, 2, 3, 4]
inputDocuments: []
workflowType: 'research'
lastStep: 5
research_type: 'technical'
research_topic: 'Burn Rust crate'
research_goals: 'Research best practices, linting, performance, compilation time, reinforcement learning, feature flags, inference, WASM compatibility, training, batch training, optimizer algorithms, reputable third-party crates or official supporting crates, and shipping models.'
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

[Research overview and methodology will be appended here]

---

## Technical Research Scope Confirmation

**Research Topic:** Burn Rust crate
**Research Goals:** Research best practices, linting, performance, compilation time, reinforcement learning, feature flags, inference, WASM compatibility, training, batch training, optimizer algorithms, reputable third-party crates or official supporting crates, and shipping models.

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

---

## Technology Stack Analysis

### Web Search Analysis

Research for this step used current public sources, prioritizing official Burn documentation, Burn's GitHub organization, docs.rs metadata, and this repository's local dependency surface.

**Source coverage:**

- Official Burn crate documentation for current conceptual model, performance claims, backends, training/inference, and feature flags: https://burn.dev/docs/burn/
- Official Burn source-rendered rustdoc for feature-flag details and exported modules: https://burn.dev/docs/src/burn/lib.rs.html
- docs.rs feature metadata for current published Burn version and feature list: https://docs.rs/crate/burn/latest/features
- Official Burn GitHub README for training/inference examples, ONNX import notes, WASM/browser examples, no_std caveat, benchmark tooling, and recursion-limit warning: https://github.com/tracel-ai/burn
- Official Burn 0.21.0 release notes for current version direction, runtime config, dispatch, and compile-time intent: https://github.com/tracel-ai/burn/releases
- Official supporting repos for benchmarks, models, and large-model inference: https://github.com/tracel-ai/burn-bench, https://github.com/tracel-ai/models, https://github.com/tracel-ai/burn-lm
- Local repo context: `Cargo.toml`, `README.md`, `docs/plugins/bevy_gym_plugin.md`, `docs/examples/cartpole.md`, and `examples/gymnasium/cart_pole.rs`

**Version boundary:** Burn's current public docs and latest docs.rs metadata report `burn 0.21.0`; this repo's current demonstration stack is pinned to `burn = "0.20.1"` as a dev-dependency, with `ember-rl = "0.3.4"` providing DQN/PPO/SAC algorithms and `bevy-gym` providing the parallel Bevy ECS environment loop. Treat recommendations below as current ecosystem guidance, then verify API details against the exact pinned Burn version before editing code.

**Quality assessment:** High confidence for Burn's official feature/backend/training/inference surface. Medium confidence for third-party RL ecosystem maturity because the best public signals are crate docs, examples, and smaller community projects rather than a single dominant Rust RL standard.

### Programming Languages

Burn is a Rust-native tensor and deep-learning framework. Its main value for this project is that model definition, training, inference, environment integration, checkpointing, and packaging can remain in Rust instead of crossing into Python or C++ runtime layers. The official docs describe Burn as a dynamic deep-learning framework built in Rust with flexibility, compute efficiency, and portability as core goals, and the GitHub README positions it as both a tensor library and a deep-learning framework for numerical computing, inference, and training.

For `bevy-gym`, Rust is also the application/runtime language: environments are Bevy entities, stepping is scheduled in `FixedUpdate`, and policy/learning systems consume events. This makes Burn a natural fit when model inference or training should happen inside the same Rust process as Bevy simulation.

_Popular Languages:_ Rust is the primary implementation and integration language. Python remains relevant for interoperability only when importing/exporting models through ONNX, PyTorch, or SafeTensors workflows.

_Emerging Languages:_ No competing language is central to Burn itself. WASM is a compilation target rather than a source language; WGSL/CUDA/ROCm details are generally backend concerns.

_Language Evolution:_ Burn's current direction favors generic Rust APIs over backend-specific user code: most model code is generic over `Backend`, while `Autodiff`, `Fusion`, and concrete backends are composed as types or features.

_Performance Characteristics:_ Rust's static typing and monomorphization are useful for backend-generic model code, but can increase compile times when many backend combinations are enabled. For this repo, prefer narrow features and stable type aliases for training/inference backends.

_Sources:_ https://burn.dev/docs/burn/ ; https://github.com/tracel-ai/burn

### Development Frameworks and Libraries

Burn's stack is split across a top-level `burn` crate and supporting crates. Relevant first-party components include:

- `burn-core` and `burn-nn` for tensors, modules, layers, records, and neural-network building blocks.
- `burn-autodiff` as a backend decorator for training.
- `burn-optim` for optimizers, gradient clipping, learning-rate schedulers, and optimizer record state.
- `burn-train` for learner/checkpoint/logger/metric/renderer abstractions, plus RL loop types behind the `rl` feature.
- `burn-rl` for environment, policy, and transition-buffer abstractions.
- `burn-wgpu`, `burn-cuda`, `burn-rocm`, `burn-tch`, `burn-candle`, `burn-flex`, `burn-ndarray`, `burn-router`, `burn-remote`, and `burn-store` for backend, dispatch, remote execution, and storage options.
- `burn-bench` for backend/hardware benchmarking.
- `burn-lm` and the official `tracel-ai/models` repository for larger model and example ecosystems.

For this repo specifically, `bevy-gym` currently treats Burn indirectly through `ember-rl`: the CartPole example uses `type B = Autodiff<NdArray>` for training and `type InferB = NdArray` for inference, while `TrainingSession` handles checkpointing and `DqnPolicy` loads `best.mpk` for evaluation.

_Major Frameworks:_ Burn is the core ML framework; Bevy is the simulation/application framework; `ember-rl` is the local RL algorithm layer; `rl-traits` is the environment contract.

_Micro-frameworks:_ `burn-rl` is official but currently looks like foundational RL abstractions rather than a complete batteries-included RL suite. Community projects such as `burn-rl-examples` are useful references but should not be treated as authoritative for production architecture without local validation.

_Evolution Trends:_ Burn 0.21.0 introduced `burn-dispatch` and a project-level `burn.toml` path for runtime/backend tuning without recompiling, which is directly relevant to compile-time and backend-selection strategy.

_Ecosystem Maturity:_ Burn's tensor/backend/training ecosystem is active and official. Rust RL on Burn is less mature than Python RL stacks, so `bevy-gym` should keep algorithm integration behind clear traits and examples instead of assuming a settled ecosystem.

_Sources:_ https://docs.rs/crate/burn/latest/features ; https://docs.rs/burn-optim ; https://docs.rs/burn-train ; https://docs.rs/burn-rl ; https://github.com/tracel-ai/burn/releases

### Model, Dataset, and Storage Technologies

For Burn, "storage" primarily means model weights, checkpoints, optimizer state, datasets, logs, and import/export formats rather than relational or NoSQL databases. The official feature list includes `store` for model storage with SafeTensors format and PyTorch interoperability. The top-level `burn` crate exposes `store` only when the `store` feature is enabled, and `optim` state types implement records so optimizer state can be saved with model state when the recorder path supports it.

The Burn README states that ONNX import is available through `burn-onnx`, converting supported ONNX graphs into Rust code that uses Burn APIs and can run on Burn backends. The same README notes that ONNX support is active development with a limited operator set. It also documents direct loading of PyTorch or SafeTensors weights into Burn-defined models.

For `bevy-gym`, local examples currently use `TrainingSession` from `ember-rl` to write checkpoints, JSONL logs, and `best.mpk`. That is the right shape for reproducible RL runs: keep model artifacts and run metadata adjacent, and avoid requiring Bevy/renderer state to load a policy for headless evaluation.

_Relational Databases:_ Not part of the core Burn workflow. Use SQLite only if adopting Burn features or tooling that explicitly require it; Burn's current feature list exposes `sqlite` and `sqlite-bundled` under `burn-core`, but they should not be pulled into a minimal Bevy/RL runtime by default.

_NoSQL Databases:_ No direct role for Burn model execution. If experiment tracking grows, prefer file-backed run directories first, then evaluate an external tracker separately.

_In-Memory Storage:_ Replay buffers and transition buffers are the relevant RL data structures. Burn's official `burn-rl` crate exposes a transition-buffer module; `ember-rl` provides the concrete algorithm/session layer used locally.

_Data Warehousing:_ Out of scope for runtime training loops. JSONL metrics are sufficient for early Bevy Gym experiments; warehouse-style analytics can be a later reporting concern.

_Sources:_ https://burn.dev/docs/src/burn/lib.rs.html ; https://github.com/tracel-ai/burn ; https://docs.rs/burn-rl

### Development Tools and Platforms

Cargo feature hygiene is central. Burn's current `burn` crate exposes a large feature surface: backend features (`flex`, `ndarray`, `wgpu`, `cuda`, `rocm`, `tch`, `candle`), decorators (`autodiff`, `fusion`), deployment/storage features (`store`, `remote`, `server`, `network`), training features (`train`, `tui`, `metrics`, `distributed`, `rl`), and performance features (`autotune`, BLAS/SIMD variants). For `bevy-gym`, default to the smallest backend set that proves the workflow.

Recommended local development shape:

- Keep the library crate's default features empty, matching the existing `bevy-gym` default.
- Keep rendering behind `render`.
- Keep Burn in examples/dev-dependencies until the core crate needs direct model integration.
- Use CPU-only `ndarray`/`flex` for fast CI and deterministic headless examples.
- Gate WGPU/CUDA/ROCm behind explicit example or workspace features, because GPU backend dependencies and adapter discovery can make builds and runtime behavior less portable.
- Add `#![recursion_limit = "256"]` only in binaries/crates that actually hit WGPU recursive type evaluation; Burn's README documents this as a WGPU-specific compile issue, not a blanket rule.

_IDE and Editors:_ Standard Rust tools are enough: rust-analyzer, rustfmt, clippy, and Cargo. Burn's heavy generic types make rust-analyzer latency more likely when many backend features are enabled.

_Version Control:_ Pin Burn versions in examples and update intentionally; Burn 0.21.0 has current improvements, while this repo currently uses 0.20.1.

_Build Systems:_ Cargo is the build system. Use feature-specific CI jobs (`cpu`, `render`, and optional GPU smoke jobs) rather than one maximal feature build.

_Testing Frameworks:_ Use Rust unit/integration tests for environment contracts and model serialization, plus headless training/evaluation smoke tests that verify a saved policy can be loaded into an inference backend.

_Sources:_ https://docs.rs/crate/burn/latest/features ; https://github.com/tracel-ai/burn ; https://github.com/tracel-ai/burn/releases

### Cloud Infrastructure and Deployment

Burn's backend abstraction is designed for code that can train in one place and infer elsewhere. Official docs describe backend-generic code via the `Backend` trait and backend decorators such as `Autodiff` and `Fusion`; the docs also list WGPU, Candle, LibTorch, Flex, Autodiff, and Fusion among current backend/decorator concepts.

Deployment targets relevant to this project:

- **Local CPU/headless:** Best default for CI, examples, and reproducible RL training smoke tests.
- **Desktop GPU:** WGPU/Vulkan/Metal/CUDA/ROCm can improve throughput but should be opt-in and benchmarked with `burn-bench` or local environment workloads.
- **Browser/WASM:** Burn's README says Flex can run CPU execution in WASM and WGPU can accelerate via browser WebGPU; it links browser examples for MNIST and image classification.
- **Embedded/no_std:** Burn documents no_std support, with the current caveat that only Flex can be used in a no_std environment.
- **Remote/server:** Burn exposes `remote`/`server` features, but they are not necessary for the current Bevy ECS training loop unless training is split out of process.

For shipping models from `bevy-gym`, the practical path is: train headless, save a model bundle (`best.mpk` or a Burn store/SafeTensors-backed bundle), load with an inference backend in a minimal runtime, and keep any renderer optional.

_Major Cloud Providers:_ Burn itself is not tied to AWS/Azure/GCP. GPU backend availability and driver/runtime setup are the real cloud constraints.

_Container Technologies:_ Useful for CUDA/ROCm reproducibility, but not required for CPU examples.

_Serverless Platforms:_ Poor fit for training; may fit short CPU inference only if artifact size and cold-start constraints are acceptable.

_CDN and Edge Computing:_ WASM/browser inference and no_std/Flex are the meaningful edge story, not CDN compute by default.

_Sources:_ https://burn.dev/docs/burn/ ; https://burn.dev/docs/src/burn/lib.rs.html ; https://github.com/tracel-ai/burn

### Technology Adoption Trends

Burn is moving toward a broader first-party platform: the latest release notes emphasize distributed workflows, improved autotuning validation, project-level runtime config through `burn.toml`, reduced framework overhead, and `burn-dispatch` as a backend-selection layer that may reduce compile-time pressure. Official adjacent projects also show a larger ecosystem: `burn-bench` for benchmark sharing, `tracel-ai/models` for official model examples, and `burn-lm` for large-model inference/training experiments.

For `bevy-gym`, the adoption strategy should be conservative:

- Keep the core Bevy/RL environment loop independent of Burn.
- Keep Burn-backed algorithms in examples or an integration crate until the API stabilizes around this project.
- Prefer one CPU backend for default examples and one optional accelerated backend for demos.
- Separate training and inference backend types (`Autodiff<...>` for training, inner backend for inference).
- Treat Burn 0.21.0's `burn-dispatch`/`burn.toml` direction as promising for compile-time/backend hygiene, but verify migration cost from the current `0.20.1` stack before adopting.

_Migration Patterns:_ Move from direct backend proliferation toward narrow feature profiles and runtime dispatch where it helps compile time without hiding device errors.

_Emerging Technologies:_ `burn-dispatch`, Flex, WebGPU/WASM, SafeTensors/PyTorch import, ONNX import, and Burn-LM are the most relevant emerging pieces.

_Legacy Technology:_ `ndarray` is still used locally and remains useful, but Burn's own docs now label the `ndarray` backend as legacy and prefer `flex` for new projects.

_Community Trends:_ Burn has active first-party repos and examples, but Rust RL remains thinner than Python RL. Keep algorithm choices modular and measured by local training/eval proofs.

_Sources:_ https://github.com/tracel-ai/burn/releases ; https://github.com/tracel-ai/burn-bench ; https://github.com/tracel-ai/models ; https://github.com/tracel-ai/burn-lm

---

## Integration Patterns Analysis

### Web Search Analysis

Research for this step focused on Burn's public API boundaries, serialization/import/export paths, optional remote execution surface, WASM/browser support, and the local `bevy-gym` event bridge.

**Source coverage:**

- Burn rustdoc for backend-generic API design, training/inference, backends, feature flags, and exported modules: https://docs.rs/burn/latest/burn/
- Burn source-rendered docs for feature flags including `train`, `autodiff`, `store`, `server`, `network`, `flex`, `wgpu`, `webgpu`, `vulkan`, `cuda`, and `rocm`: https://burn.dev/docs/src/burn/lib.rs.html
- Burn GitHub README for ONNX, PyTorch/SafeTensors imports, WASM/browser inference, no_std, benchmarks, and model examples: https://github.com/tracel-ai/burn
- `burn-onnx` rustdoc for ONNX-to-Burn source generation: https://docs.rs/burn-onnx/latest/burn_onnx/
- `burn-store` rustdoc for Burnpack, SafeTensors, PyTorch compatibility, zero-copy loading, filtering, and remapping: https://docs.rs/burn-store/latest/burn_store/
- `burn-remote` and `burn-router` rustdocs for remote/backend routing integration surfaces: https://docs.rs/burn-remote/latest/burn_remote/ and https://docs.rs/burn-router/latest/burn_router/
- `burn-rl` and `burn-train` rustdocs for environment/policy/transition-buffer and learner/checkpoint/metric abstractions: https://docs.rs/burn-rl/latest/burn_rl/ and https://docs.rs/burn-train/latest/burn_train/
- Local repo source for Bevy ECS integration: `src/plugin.rs`, `src/events.rs`, `docs/plugins/bevy_gym_plugin.md`, and `examples/gymnasium/cart_pole.rs`

**Cross-integration analysis:** Burn is not primarily a web-service framework. Its strongest integration pattern is a typed, in-process Rust boundary: generic `Backend` model code, `Module`/record APIs for state, feature-gated backends for execution, and explicit artifact formats for import/export. For `bevy-gym`, this matches Bevy's ECS scheduling model: environments produce typed events, policy systems batch observations, Burn/ember-rl runs inference or learning, then actions are written back to ECS components.

**Quality assessment:** High confidence for Burn's official Rust APIs, features, and import/export formats. Medium confidence for `burn-remote` production-readiness because the rustdoc exposes the shape and dependencies but not a complete deployment/security guide. Low confidence that generic enterprise patterns such as GraphQL, OAuth, API gateways, service mesh, or ESBs are relevant to this repo unless a separate product/API server is introduced.

### API Design Patterns

Burn's primary API pattern is **backend-generic Rust composition**. Model code is usually written over `B: Backend`; training uses a backend decorator such as `Autodiff<B>`; inference uses the non-autodiff backend. The official docs state that most code is generic over the `Backend` trait, which allows swappable backends and backend decorators such as autodiff and fusion.

For `bevy-gym`, the strongest API boundary is:

- `rl-traits::Environment` defines environment state transitions.
- `BevyGymPlugin<E>` spawns `E` as parallel Bevy entities.
- `ActionRequestEvent` asks a policy system for actions.
- `ExperienceEvent` delivers transitions for replay buffers or learning.
- `EpisodeEndEvent` delivers episode-level metrics.
- `ember-rl::TrainingSession` and policy types keep Burn-backed algorithm code out of core `bevy-gym`.

This is a better local integration shape than a REST or GraphQL API because policy inference is on the hot path of the ECS tick loop, and the observation/action types are Rust generics rather than JSON schemas.

_RESTful APIs:_ Not a first-choice integration pattern inside `bevy-gym`. Use REST only for external dashboards, experiment registry APIs, or remote control surfaces outside the per-tick policy path.

_GraphQL APIs:_ Not relevant for Burn model execution. It may fit experiment metadata browsing later, but it should not sit between Bevy environments and policy inference.

_RPC and gRPC:_ A custom RPC boundary can make sense only if training/inference is moved out-of-process. Burn exposes `burn-remote`, whose rustdoc describes a `RemoteBackend`, `RemoteDevice`, and server module; treat it as the first official surface to evaluate before designing ad-hoc gRPC.

_Webhook Patterns:_ Not relevant to core Burn/Bevy integration. Webhooks may fit training-complete notifications but not synchronous action selection.

_Sources:_ https://docs.rs/burn/latest/burn/ ; https://docs.rs/burn-remote/latest/burn_remote/ ; local `src/events.rs` and `src/plugin.rs`

### Communication Protocols

The default Burn integration path has no network protocol: tensors, modules, records, and optimizer state are Rust values in one process. That is the recommended shape for early `bevy-gym` training because it preserves type safety, avoids serialization per action, and keeps batching under the policy system's control.

When an out-of-process boundary is necessary, prefer official or artifact-based channels:

- Artifact boundary: save/load model state through Burn records, Burn Store, SafeTensors, PyTorch-compatible stores, or existing `ember-rl` checkpoint directories.
- Import boundary: use ONNX import when starting from a TensorFlow/PyTorch model and Burn supports the needed operators.
- Browser boundary: compile inference to WASM with Flex or WGPU/WebGPU when the model and target browser capabilities fit.
- Remote compute boundary: evaluate `burn-remote`/`burn-router` rather than inventing a tensor RPC layer.

_HTTP/HTTPS Protocols:_ Burn itself does not require HTTP for local execution. `burn-remote` depends on web server/networking crates such as Axum and Tokio pieces, but a repo decision to expose HTTP should include explicit authentication, artifact integrity, and latency testing.

_WebSocket Protocols:_ `burn-remote` depends on `tokio-tungstenite` when optional server/networking features are enabled, which suggests WebSocket-style transport is part of the remote story. Verify the exact protocol contract before relying on it.

_Message Queue Protocols:_ AMQP/MQTT/Kafka are not native Burn integration points. They may fit asynchronous experiment orchestration, but not per-step RL control loops.

_gRPC and Protocol Buffers:_ No first-party Burn gRPC path appeared in current official docs. Use only if an external service boundary demands it, and keep model artifact formats separate from service request formats.

_Sources:_ https://docs.rs/burn-remote/latest/burn_remote/ ; https://docs.rs/burn-router/latest/burn_router/ ; https://github.com/tracel-ai/burn

### Data Formats and Standards

Burn integration is artifact-heavy. The critical formats are model records/checkpoints, tensor data, SafeTensors, PyTorch state dicts, Burnpack, ONNX input graphs, and run metrics. `burn-store` documents Burnpack as a native format with CBOR metadata and ParamId persistence, SafeTensors support, PyTorch compatibility, zero-copy loading via memory mapping/lazy materialization, filtering, remapping, and no_std support for core functionality.

For `bevy-gym`, the local example already demonstrates a practical artifact protocol: train with `TrainingSession`, save checkpoints and `best.mpk`, then run evaluation by loading the best policy into an inference backend. Keep this model-bundle boundary stable before adding additional interchange formats.

_JSON and XML:_ JSON/JSONL fits metrics, config, and lightweight metadata. XML has no clear role unless a third-party tool forces it.

_Protobuf and MessagePack:_ Current `burn-remote` dependencies include `rmp-serde`, which indicates MessagePack-style serialization is relevant to remote execution internals. Do not expose it as a public format unless the official remote protocol requires it.

_CSV and Flat Files:_ CSV is useful for simple datasets and metric export, but not for model weights or high-throughput trajectory exchange.

_Custom Data Formats:_ Prefer official Burn Store, SafeTensors, Burnpack, ONNX import, and the existing `best.mpk`/checkpoint run layout over custom binary formats.

_Sources:_ https://docs.rs/burn-store/latest/burn_store/ ; https://docs.rs/burn-onnx/latest/burn_onnx/ ; https://github.com/tracel-ai/burn

### System Interoperability Approaches

Burn interoperates best at three boundaries:

- **Backend interoperability:** generic `Backend` code lets the same model run over CPU/GPU/WASM-compatible backends when the operations are supported.
- **Artifact interoperability:** SafeTensors, PyTorch loading, Burnpack, and ONNX import move model state or model definitions across frameworks.
- **Runtime interoperability:** WASM/WebGPU and optional remote/router crates move execution to browser, embedded, or remote compute contexts.

For `bevy-gym`, this suggests a layered design:

- Core crate remains environment/scheduling only.
- Algorithm examples or integration crates own Burn and `ember-rl`.
- Saved policy bundles are the boundary between training and shipping.
- Runtime feature flags select CPU, browser, or GPU paths without changing environment code.

_Point-to-Point Integration:_ Best for early development: `bevy-gym` ECS systems call `TrainingSession`/policy objects directly. Keep this path fast and testable.

_API Gateway Patterns:_ Not applicable unless model serving becomes a web product.

_Service Mesh:_ Not applicable to in-process Bevy/Burn training.

_Enterprise Service Bus:_ Not applicable.

_Sources:_ https://docs.rs/burn/latest/burn/ ; https://burn.dev/docs/src/burn/lib.rs.html ; local `README.md`

### Microservices Integration Patterns

Microservices are not the default recommendation for `bevy-gym` Burn integration. A microservice split adds serialization, process management, device scheduling, and failure semantics to a loop where latency and determinism matter. It becomes worth considering only when one of these is true:

- multiple Bevy simulations need a shared model server,
- GPU training must run on a separate host,
- browser/edge clients need a centralized inference API,
- or experiment orchestration needs remote workers.

Before creating a custom service, evaluate Burn's official remote/router surface. `burn-router` documents a backend router that forwards tensor operations to appropriate backend runners and supports tensor transfer between backends; `burn-remote` documents a remote backend and remote device.

_API Gateway Pattern:_ Only for external HTTP inference or dashboards, not ECS-internal policy selection.

_Service Discovery:_ Only relevant if remote Burn workers or model-serving instances are introduced.

_Circuit Breaker Pattern:_ Necessary for remote inference/training; not needed for in-process training.

_Saga Pattern:_ Not relevant to tensor execution. Use explicit checkpoint and run-directory semantics instead of distributed transactions.

_Sources:_ https://docs.rs/burn-router/latest/burn_router/ ; https://docs.rs/burn-remote/latest/burn_remote/

### Event-Driven Integration

The local `bevy-gym` design is already event-driven in the right place. `ActionRequestEvent` separates action requests from action writes, which lets a policy system collect multiple requests and run batched inference. `ExperienceEvent` delivers complete transitions to learning systems without coupling Bevy internals to replay-buffer implementations. `EpisodeEndEvent` emits episode summaries for logging and training progress.

This aligns well with RL batch training:

- Bevy owns environment stepping and parallelism.
- The policy system batches observations across `ActionRequestEvent`s.
- Burn/ember-rl owns inference and learning.
- Actions are written to `PendingAction`.
- Transitions and episode summaries flow back through events.

_Publish-Subscribe Patterns:_ Bevy messages are the local pub/sub mechanism. Keep them typed and narrow.

_Event Sourcing:_ Do not event-source full tensor or environment state by default. Logging transitions can be useful for debugging, but replay-buffer persistence should be explicit.

_Message Broker Patterns:_ Kafka/RabbitMQ-style brokers are too heavy for the hot path; use them only for offline experiment telemetry if needed.

_CQRS Patterns:_ Not directly useful. Separate command/action writes from observation/metric reads through ECS components and events instead.

_Sources:_ local `src/events.rs`, `src/plugin.rs`, and `docs/plugins/bevy_gym_plugin.md`; https://docs.rs/burn-rl/latest/burn_rl/

### Integration Security Patterns

Security concerns depend on whether Burn stays in-process or crosses a trust boundary.

For in-process training/evaluation:

- Treat model artifacts as executable-adjacent inputs: validate paths, pin versions, and avoid loading untrusted model bundles blindly.
- Prefer SafeTensors/Burn Store paths for model interchange because they are designed for tensor serialization and framework interoperability.
- Keep training run directories self-contained with metadata, config, metrics, and checkpoints.

For remote/server use:

- Do not expose `burn-remote` or any custom inference endpoint without authentication, transport encryption, request limits, artifact integrity checks, and device-resource isolation.
- Assume browser/WASM models and weights are visible to users; use browser inference for client-side UX, not secret model protection.

_OAuth 2.0 and JWT:_ Relevant only for an external API server; not part of Burn core.

_API Key Management:_ Relevant only for remote inference/training services.

_Mutual TLS:_ Worth considering for private remote workers or model servers; unnecessary for local in-process Bevy/Burn loops.

_Data Encryption:_ Apply to model artifacts at rest and remote transport if artifacts or inference requests cross machine boundaries.

_Sources:_ https://docs.rs/burn-store/latest/burn_store/ ; https://docs.rs/burn-remote/latest/burn_remote/ ; https://github.com/tracel-ai/burn

---

## Architectural Patterns and Design

### Web Search Analysis

Research for this step used primary sources only: official Burn rustdocs/source docs, official Burn repository and releases, official Cargo/Rust documentation, official Clippy documentation, official Bevy rustdocs, and local `bevy-gym` source.

**Source coverage:**

- Burn architecture, performance, training/inference, backend composition, quantization, and feature flags: https://docs.rs/burn/latest/burn/ and https://burn.dev/docs/src/burn/lib.rs.html
- Burn 0.21.0 release architecture direction, including `burn-dispatch`, `burn.toml`, distributed workflows, autotuning validation, device-handle overhead reduction, and compile-time intent: https://github.com/tracel-ai/burn/releases
- Burn Store architecture for model serialization, SafeTensors, Burnpack, PyTorch compatibility, lazy materialization, and no_std core support: https://docs.rs/burn-store/latest/burn_store/
- Burn RL/train crate surfaces for environment/policy/transition buffers, learners, checkpoints, metrics, and RL loops: https://docs.rs/burn-rl/latest/burn_rl/ and https://docs.rs/burn-train/latest/burn_train/
- Cargo feature and profile architecture: https://doc.rust-lang.org/cargo/reference/features.html and https://doc.rust-lang.org/cargo/reference/profiles.html
- Clippy lint-group guidance: https://doc.rust-lang.org/stable/clippy/lints.html
- Bevy ECS scheduling and event/message architecture: https://docs.rs/bevy/latest/bevy/ecs/schedule/ and https://docs.rs/bevy/latest/bevy/ecs/event/trait.Event.html
- Local architecture: `src/lib.rs`, `src/plugin.rs`, `src/systems/step.rs`, `src/events.rs`, `src/components.rs`, and `examples/gymnasium/cart_pole.rs`

**Quality assessment:** High confidence for current official Burn/Cargo/Bevy architecture facts. Medium confidence for future Burn migration recommendations because `bevy-gym` is currently pinned to Burn 0.20.1 in examples while official docs now describe Burn 0.21.0.

### System Architecture Patterns

The best architecture for `bevy-gym` is a **Burn-agnostic environment core with optional Burn-backed algorithm integrations**. The local crate already follows this direction: `bevy-gym` depends on `rl-traits`, `rand`, and minimal Bevy, while Burn and `ember-rl` are dev-dependencies used by the CartPole example. Preserve that boundary.

Recommended layers:

- **Environment layer:** `rl-traits::Environment` implementations run as Bevy entities.
- **Simulation scheduler:** `BevyGymPlugin` owns parallel environment spawning, fixed-tick stepping, reset ordering, and headless/render mode.
- **Message bridge:** typed Bevy messages carry action requests, transitions, and episode summaries.
- **Policy/training adapter:** `ember-rl` or a future Burn-native adapter consumes observations/transitions and writes actions.
- **Artifact layer:** training run directories, checkpoints, Burn records, Burn Store, SafeTensors, or `best.mpk` policy bundles provide the train-to-ship boundary.
- **Optional deployment layers:** rendering, browser/WASM, GPU, remote compute, or model serving are feature-gated adapters, not core requirements.

Burn's own architecture supports this shape. Official docs emphasize backend-generic code over the `Backend` trait and composable backend decorators such as `Autodiff` and `Fusion`. That means the same model definition can be used for training and inference as long as code is careful about backend type aliases and feature availability.

_Source:_ https://docs.rs/burn/latest/burn/ ; https://burn.dev/docs/src/burn/lib.rs.html ; local `src/lib.rs`, `src/plugin.rs`, and `examples/gymnasium/cart_pole.rs`

### Design Principles and Best Practices

**Keep the core crate small and stable.** `bevy-gym` should not make Burn a normal dependency until the public API needs to expose model or trainer concepts. Burn-backed algorithms belong in examples, integration crates, or optional features.

**Use ports-and-adapters boundaries.** Treat `rl-traits::Environment`, Bevy messages, Burn policy/session types, and model stores as explicit boundaries. This prevents the ECS loop from depending on a specific optimizer, backend, checkpoint format, or renderer.

**Separate training and inference types.** The local example does this well with `type B = Autodiff<NdArray>` for training and `type InferB = NdArray` for inference. Keep this as an architectural convention, and migrate to `Flex` or another preferred backend only after proving the exact Burn version and `ember-rl` compatibility.

**Treat Cargo features as public architecture.** Cargo features are conditional compilation and optional dependency boundaries. The Cargo Book warns that defaults are enabled automatically unless `default-features = false` is used, so heavy backend features should stay opt-in. For this repo, maintain empty default features and keep `render`, GPU, WASM, and Burn integration features explicit.

**Lint for signal, not theater.** Clippy recommends applying `perf` suggestions, fixing or locally justifying suspicious/complex code, and cherry-picking `pedantic`, `restriction`, and `nursery` lints rather than enabling those groups wholesale. For Burn-heavy code, prioritize lints that catch clone/device-transfer mistakes, manifest quality, needless allocations, excessive type complexity, and missing error handling.

_Source:_ https://doc.rust-lang.org/cargo/reference/features.html ; https://doc.rust-lang.org/stable/clippy/lints.html ; local `Cargo.toml`

### Scalability and Performance Patterns

There are two separate performance planes:

- **Environment throughput:** Bevy ECS can step independent environment entities in parallel. Official Bevy schedule docs state that the multi-threaded executor runs non-conflicting systems in parallel, and local `step_system` uses `query.par_iter_mut()` for the expensive per-environment step phase.
- **Model throughput:** Burn can improve tensor/model performance through backend choice, automatic kernel fusion, asynchronous execution, memory management, automatic kernel selection, hardware-specific features, and custom backend extensions.

Architectural recommendations:

- Batch observations at the policy system boundary. `ActionRequestEvent` intentionally lets the policy system collect multiple pending observations and run one batched forward pass instead of one inference call per environment.
- Keep environment stepping CPU-parallel and model inference centralized unless a benchmark proves per-env model instances are better.
- Minimize tensor/device transfers. Convert observations into tensors once per batch, run inference, then convert actions back to environment action types.
- Use Burn's official benchmark tooling (`burn-bench`) or local training throughput measurements before choosing WGPU/CUDA/ROCm over CPU.
- Treat `fusion` and `autotune` as performance features to validate, not defaults to enable blindly.
- Preserve headless training mode as the baseline; rendering should stay independent from the fixed-tick training loop.

Compile-time architecture:

- Keep backend features narrow. Avoid building `wgpu`, `cuda`, `rocm`, `tch`, `candle`, `store`, `train`, `metrics`, and `tui` together unless the target explicitly needs them.
- Consider Burn 0.21.0's `burn-dispatch`/`burn.toml` direction when upgrading, because the release notes explicitly mention backend-selection simplification and future compile-time improvement.
- Use Cargo profiles deliberately. The Cargo Book defines profiles as compiler-setting groups; the Rust Performance Book notes that `codegen-units = 1` and more aggressive LTO can improve runtime speed but increase compile times. Do not apply release-max settings to inner-loop development profiles.

_Source:_ https://docs.rs/bevy/latest/bevy/ecs/schedule/ ; https://docs.rs/burn/latest/burn/ ; https://github.com/tracel-ai/burn/releases ; https://doc.rust-lang.org/cargo/reference/profiles.html ; https://nnethercote.github.io/perf-book/build-configuration.html

### Integration and Communication Patterns

Use **typed in-process integration** for the hot path and **artifact integration** for train-to-ship boundaries.

In-process pattern:

- `BevyGymPlugin` requests actions through `ActionRequestEvent`.
- Policy systems batch observations from `CurrentObservation`.
- Burn/ember-rl produces actions.
- Actions are written to `PendingAction`.
- `ExperienceEvent` and `EpisodeEndEvent` feed learning, logging, and checkpoint logic.

Artifact pattern:

- Training writes model/checkpoint artifacts.
- Evaluation and shipping load a minimal inference policy.
- Burn Store/SafeTensors/PyTorch/ONNX paths are used only when cross-framework or cross-runtime movement is needed.

Avoid adding HTTP, GraphQL, or a service mesh to the inner RL loop. If remote compute becomes necessary, evaluate `burn-remote` and `burn-router` first. Their official rustdocs expose a remote backend/device and backend router model, which is closer to Burn's tensor/backend architecture than generic web-service APIs.

_Source:_ https://docs.rs/burn-remote/latest/burn_remote/ ; https://docs.rs/burn-router/latest/burn_router/ ; https://docs.rs/burn-store/latest/burn_store/ ; local `src/events.rs` and `docs/plugins/bevy_gym_plugin.md`

### Security Architecture Patterns

For local training, the main security boundary is artifact trust:

- Treat checkpoints, SafeTensors, Burnpack, PyTorch files, and ONNX files as untrusted inputs unless they come from the current run or a trusted source.
- Keep model artifacts and metadata together so the runtime can verify expected model name, version, backend, observation/action space, and environment compatibility before loading.
- Prefer explicit paths and run manifests over "latest file in directory" behavior for shipping.

For remote/browser architecture:

- Browser/WASM inference does not protect model weights. Assume the model can be inspected by users.
- Remote training/inference endpoints need authentication, transport encryption, request limits, artifact integrity checks, and device isolation. Burn's `remote`/`server` feature surface should not be exposed directly without a product-level security wrapper.
- GPU backends should fail closed when no compatible device is available; they should not silently change training semantics without metadata.

_Source:_ https://docs.rs/burn-store/latest/burn_store/ ; https://github.com/tracel-ai/burn ; https://docs.rs/burn-remote/latest/burn_remote/

### Data Architecture Patterns

Use **run-directory data architecture** for RL experiments:

- `config` or manifest: algorithm, backend, feature flags, environment ID/version, observation/action shape, seed policy, and device class.
- `metrics`: JSONL or similar append-only training/evaluation metrics.
- `checkpoints`: periodic training checkpoints with optimizer/model state when resuming is required.
- `best policy`: minimal inference artifact for shipping/evaluation.
- `normalization/statistics`: observation normalizer or reward/stat sidecars if used.

Burn Store is the relevant official model-data architecture for broader interoperability. Its rustdoc documents Burnpack, SafeTensors, PyTorch loading, zero-copy loading, flexible filtering/remapping, and lazy materialization. That is the right direction for model shipping or cross-framework import/export, while local `ember-rl` `best.mpk` remains acceptable for current examples.

For RL batch training, separate data planes:

- live transition flow through typed messages,
- replay/trajectory buffers inside the algorithm/session,
- persistent checkpoints and metrics on disk,
- model shipping bundles for inference.

_Source:_ https://docs.rs/burn-store/latest/burn_store/ ; https://docs.rs/burn-rl/latest/burn_rl/ ; https://docs.rs/burn-train/latest/burn_train/ ; local `docs/examples/cartpole.md`

### Deployment and Operations Architecture

Deployment should be target-specific:

- **CI/default examples:** CPU-only, headless, minimal features, deterministic smoke tests.
- **Desktop demos:** optional `render` feature and optional accelerated backend feature.
- **Training runs:** release build, headless mode, explicit artifact directory, metrics, checkpoint retention, and best-policy export.
- **Inference shipping:** inference-only backend, no autodiff, no trainer dashboard, no replay buffers unless needed.
- **Browser/WASM:** Flex CPU or WGPU/WebGPU only after target-browser and model-size validation; keep training out of the browser unless a specific demo requires it.
- **Embedded/no_std:** Flex only, per official Burn caveat, and only for small inference models.
- **Remote/GPU workers:** explicit feature profile, explicit device selection, benchmark proof, and operational safeguards.

Recommended feature architecture for `bevy-gym`:

- Core: default empty, Bevy ECS scheduling only.
- `render`: Bevy render/window features only.
- `burn-cpu-example` or integration crate: Burn CPU inference/training examples.
- `burn-gpu-example`: opt-in WGPU/CUDA/ROCm demonstration after benchmarks.
- `wasm-inference`: opt-in browser inference if the Bevy/WASM surface and Burn backend are both proven.
- `store`: only when SafeTensors/Burn Store/PyTorch interop is needed.

Shipping model rule: ship the smallest artifact that can run inference, plus a manifest that proves model version, environment/action contract, backend expectations, and evaluation evidence. Do not ship the full training stack unless resuming training is a product requirement.

_Source:_ https://burn.dev/docs/src/burn/lib.rs.html ; https://github.com/tracel-ai/burn ; https://docs.rs/crate/burn/latest/features ; local `Cargo.toml`

---

<!-- Content will be appended sequentially through research workflow steps -->
