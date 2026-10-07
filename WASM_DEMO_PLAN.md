# Plan: static WASM checkpoint demo

- Status: implemented; tailnet HTTPS verified on port 8443
- Research date: 2026-08-01
- Repository: `/home/sagan/Sync/playground/bevy-gym`
- Scope: browser inference and playback only; training remains native
- Verified remote endpoint: `https://nixos.tail87afcc.ts.net:8443/`

## Outcome

Build one static site that runs the ecosystem simulations and Burn policies in
the browser. The root page starts a real environment immediately. Navigation
switches among the six ecosystem stages without leaving the site. Each stage
loads its curated best compatible checkpoint by default. The user can replace
the bunny checkpoint and, where applicable, the fox checkpoint with local
`.mpk` files.

The release is complete only when another tailnet device can open the URL,
watch every supported stage advance under inference, switch stages, upload a
checkpoint, and see the uploaded policy take control.

## Scope

The first release supports every value in `CurriculumStage::ALL`:

1. Forage
2. Survival
3. Shelter
4. Competition
5. Predator-prey
6. Obstacles

The `ecosystem-curriculum` target remains a native training orchestrator. The
site represents its six lessons as navigable environments instead of starting
training in WASM.

The architecture must allow later adapters for CartPole and the other trained
examples. Those adapters do not block this release.

The site does not train, save into `runs/`, invoke native commands, record
video, expose Bevy Remote Protocol, or require a backend service. Uploaded
checkpoint bytes remain in the browser tab.

## Current repository facts

- Bevy is pinned to `0.18`, Burn is pinned to `0.21`, and native inference uses
  `burn::backend::Flex`.
- The default `render` feature enables Linux `x11` and `wayland`. That feature
  graph cannot be used unchanged for `wasm32-unknown-unknown`.
- Burn autodiff, Tokio, Clap, native video work, filesystem checkpoint loading,
  and background trainer threads are mixed into the current example surface.
  The web target needs an inference-only dependency path.
- `RecurrentPpoPolicy::load` reconstructs an architecture and loads a
  `NamedMpkFileRecorder<FullPrecisionSettings>` record from a `Path`.
- Burn 0.21 supplies `NamedMpkBytesRecorder<FullPrecisionSettings>`. It uses the
  same named MessagePack encoding and can load fetched or uploaded bytes.
- Every ecosystem stage uses a fixed local observation width, global state
  width, action width, and maximum agent count in the current profile. Older
  checkpoints still differ in record shape or environment tuning.
- Predator-prey and obstacles require a synchronized bunny and fox pair.
- `runs/` is ignored and contains diagnostic, obsolete, and incompatible
  checkpoints. File modification time or a filename containing `best` is not a
  sufficient release-selection rule.
- The installed native Rust toolchain has no WASM standard library. The flake
  must provide a toolchain with `wasm32-unknown-unknown` rather than depend on a
  user-level `rustup target add`.
- The current tailnet host is `nixos.tail87afcc.ts.net` at `100.123.85.8`.

## Tooling decision

Use Trunk `0.21.14` for the site build and development server. Trunk invokes
Cargo and `wasm-bindgen`, copies declared assets, watches source and asset
changes, rebuilds without restarting the server, and reloads connected
browsers. Pin the Trunk version in Nix and in `Trunk.toml`.

Use direct `wasm-bindgen-test` only for focused Rust-to-browser tests. Do not use
`wasm-pack` as the site bundler. Its npm-package workflow does not replace the
asset graph, development server, or reload behavior needed here.

Use Bevy's WebGL2 renderer for the first release. Use Burn Flex for CPU
inference. WebGPU would add a second GPU workload beside Bevy. WASM threads
would require cross-origin isolation headers. Neither is needed for these small
policies. Use Tailscale Serve HTTPS for the final shared URL so later renderer
experiments do not require a serving migration.

Use one WASM binary and fragment routes such as
`#/ecosystem/predator-prey`. Fragment routing works on any static file server
without rewrite rules.

Primary sources:

- [Bevy 0.18 WASM build instructions](https://github.com/bevyengine/bevy/blob/v0.18.1/examples/README.md#wasm)
- [Bevy 0.18 release and feature collections](https://bevy.org/news/bevy-0-18/)
- [Burn 0.21 WASM guidance](https://burn.dev/burn-book/advanced/web-assembly.html)
- [Burn 0.21 Flex release rationale](https://burn.dev/blog/release-0.21.0/)
- [Burn named MessagePack bytes recorder](https://docs.rs/burn/0.21.0/burn/record/struct.NamedMpkBytesRecorder.html)
- [Trunk repository and supported workflow](https://github.com/trunk-rs/trunk)
- [Trunk configuration, watch, serve, and tool pinning](https://trunk-rs.github.io/trunk/guide/configuration/)
- [wasm-bindgen guide](https://rustwasm.github.io/docs/wasm-bindgen/)
- [W3C File API](https://www.w3.org/TR/FileAPI/)
- [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve)

Do not use `trunkrs.dev` as a source. That domain no longer serves the Trunk
project. The project now links to `trunk-rs.github.io/trunk`.

## Runtime architecture

The browser owns one `DemoApp` with these closed states:

- `EnvironmentId`: `Forage`, `Survival`, `Shelter`, `Competition`,
  `PredatorPrey`, or `Obstacles`.
- `PolicyRole`: `Bunny` or `Fox`.
- `PolicySource`: `Bundled` or `Uploaded`.
- `LoadState`: `Fetching`, `Validating`, `Ready`, or `Failed`.
- `PlaybackState`: `Running`, `Paused`, `EpisodePause`, or `ResetReady`.

Changing the route creates the selected `SimulationConfig`, fetches its
versioned checkpoint files, validates them, creates fresh recurrent memory for
every agent, and starts deterministic mean-action inference. A route change
must never retain an agent, recurrent memory, policy role, or environment
tuning from the previous stage.

The simulation advances at its existing fixed 0.1-second step. Rendering can
interpolate between fixed states. Inference runs once per living agent per
simulation step. Episode termination shows the existing one-second terminal
pause, resets the environment with the next deterministic seed, and resets all
recurrent memory.

The page shows the selected environment, policy source, checkpoint name and
hash prefix, load state, episode, simulation step, simulation speed, and any
load error. Controls provide environment navigation, pause, restart, speed,
reset-to-bundled, bunny upload, and fox upload for the two predator stages.

## Checkpoint release contract

Add a typed, versioned manifest under `web/checkpoints/manifest.json`. Each
environment entry contains:

- schema version;
- environment ID and title;
- checkpoint profile and experiment tuning;
- algorithm name and architecture dimensions;
- required policy roles;
- versioned checkpoint URL, byte length, and SHA-256 for each role;
- source run ID and training seed;
- summary artifact values used to select it;
- qualification state: `qualified` or `best-compatible-available`;
- the current source commit and Burn version.

Use content-addressed filenames such as
`survival-bunny-<sha256-prefix>.mpk`. Compile the small manifest into the WASM
binary and copy checkpoint files as static assets. Updating the manifest then
produces a new hashed WASM file, while checkpoint URLs change with their
content. This prevents a browser cache from pairing new metadata with old
weights.

Create a native export command that accepts an environment and run directory.
It validates the checkpoint through the existing native load path, runs a
fixed-observation inference probe, copies required `.mpk` and configuration
files, computes hashes, and writes one typed manifest entry. It must refuse:

- a stage mismatch;
- missing or mismatched bunny and fox sidecars;
- an unsupported checkpoint profile;
- a model that does not match the declared architecture;
- non-finite probe output;
- a run without the evidence needed for its declared qualification state.

Select defaults from current held-out evidence and current load compatibility.
Do not select by modification time. Re-evaluate the strongest current candidate
for each stage. If no candidate is qualified, ship the best compatible
candidate with the exact `best-compatible-available` label. If no compatible
candidate behaves usefully, train or fine-tune natively before exporting it.

The browser verifies bundled file length and SHA-256 before decoding. An upload
has an 8 MiB per-file cap, decodes with the selected environment's declared
architecture, and must produce finite probe output before activation. A raw
`.mpk` has no trustworthy stage metadata. The UI labels it with the selected
stage and `uploaded metadata unverified`. When a matching configuration
sidecar is supplied, the browser also validates its stage and tuning.

Single-policy environments accept one bunny `.mpk`. Predator-prey and
obstacles expose separate bunny and fox inputs. A new policy becomes active
only at an episode boundary so one trajectory never mixes recurrent states or
policy versions.

## Rust boundaries and file map

Preserve existing path-based load methods and public error shapes. Add a new
bytes-load API with its own typed error instead of inventing a filesystem path
for browser bytes.

Use this target layout:

```text
src/
  inference/
    mod.rs
    backend.rs
    checkpoint_bytes.rs
    recurrent_ppo.rs
  ecosystem/
    mod.rs
    domain.rs
    reward.rs
    rng.rs
    simulation.rs
    visual_snapshot.rs
examples/ecosystem/shared/
  demo.rs
  rendering.rs
  training.rs
  video.rs
web/
  Cargo.toml
  Trunk.toml
  index.html
  site.css
  checkpoints/
    manifest.json
    *.mpk
    *.config.json
  src/
    main.rs
    app.rs
    environment_id.rs
    manifest.rs
    policy_loader.rs
    route.rs
    viewer.rs
  tests/
    browser.spec.ts
```

Move the platform-neutral inference policy and ecosystem simulation into the
library. Keep the native trainer, CLI, worker threads, Inspector, video writer,
and filesystem orchestration in the examples. Re-export existing
`bevy_gym::training` policy types so current callers retain their paths.

Create `web` as a workspace package. Set `default-members = ["."]` so existing
root Cargo commands keep their current scope unless a workspace command is
explicit. The web package depends on `bevy-gym` with default features disabled
and only portable ecosystem inference enabled.

Split Cargo features by capability:

- portable ECS, simulation, Burn Flex inference, 2D rendering, UI, WebGL2;
- native training and autodiff;
- native window backends and Inspector;
- native video and process I/O;
- MuJoCo.

Do not enable `x11`, `wayland`, `multi_threaded`, Tokio, autodiff, MuJoCo,
Bevy Remote Protocol, or video dependencies in the WASM graph. Verify this with
`cargo tree --target wasm32-unknown-unknown` searches, not only a successful
compile.

## Nix and serving design

Extend `flake.nix` with a pinned Rust toolchain that includes
`wasm32-unknown-unknown`, Trunk `0.21.14`, the exact `wasm-bindgen-cli` version
used by `Cargo.lock`, Binaryen, and the browser-test dependencies.

Add these outputs:

- `packages.web-dist`: a pure release build whose output is the complete static
  `dist/` tree;
- `apps.web-serve`: `trunk serve --release`, bound to `127.0.0.1:8080`, with
  source and checkpoint watching plus browser reload enabled;
- `apps.web-check`: build the site, start an isolated server, and run browser
  smoke tests;
- `checks.wasm`: compile the exact inference-only WASM feature graph;
- `checks.web-checkpoint-parity`: run the native file-to-bytes recorder parity
  test;
- `checks.web-dist`: build the exact release package during `nix flake check`.

`nix run .#web-serve` is the user command. It keeps one server process alive
while Trunk rebuilds changed Rust, HTML, CSS, manifest, and checkpoint assets.
The same process must remain alive across a source change, and a connected
browser must reload after the successful rebuild.

Use `nix run .#web-serve-tailscale-https` for the verified loopback listener.
On this host, Tailscale Serve proxies HTTPS port 8443 to loopback port 8080
because Traefik owns port 443. Trunk's proxied reload socket uses `wss`. Do not
enable Tailscale Funnel or public internet access. Resolve the current MagicDNS
URL after setup instead of embedding the private tailnet suffix in application
code.

## Implementation sequence

### 1. Freeze acceptance and prove the toolchain

1. Record the dirty-worktree baseline and preserve unrelated changes.
2. Add a failing external test for bytes-based policy loading.
3. Add the `web` package skeleton and a failing WASM compile gate.
4. Add the pinned WASM toolchain and Trunk to the flake.
5. Build a blank Bevy canvas through Trunk before moving ecosystem code.

Exit gate: `nix build .#web-dist` produces an HTML, JavaScript, and WASM bundle that
opens in Chromium with no console error.

### 2. Separate portable inference

1. Extract the Flex inference backend and recurrent policy from native trainer
   concerns.
2. Add `load_named_mpk_bytes` with a typed bytes-load error.
3. Keep existing filesystem load and save behavior unchanged.
4. Prove one native file load and one bytes load return the same action and
   recurrent state for the same observation.
5. Compile the inference-only graph for WASM and audit forbidden native
   dependencies.

Exit gate: a current ecosystem checkpoint loads from bytes in native tests and
the same code compiles for `wasm32-unknown-unknown`.

### 3. Extract the portable ecosystem runtime

1. Move domain, reward, RNG, simulation, and visual snapshot code behind a
   narrow library facade.
2. Keep training, native watch, Inspector, and video modules native.
3. Update ecosystem examples to use the extracted runtime.
4. Run every existing focused ecosystem test after each coherent move.

Exit gate: native examples preserve their command behavior and the web package
can create and step every `CurriculumStage::ALL` environment.

### 4. Add one complete browser slice

1. Implement the manifest parser and bundled fetch path.
2. Export one current survival checkpoint.
3. Render survival with deterministic mean-action inference.
4. Add status, pause, reset, speed, and failure UI.
5. Add the survival route and browser motion probe.

Exit gate: the survival step counter and agent position both change under the
bundled checkpoint in a real browser.

### 5. Add navigation and all stages

1. Add the six fragment routes and environment selector.
2. Recreate environment, policies, and recurrent memory on every route change.
3. Export the best compatible defaults for all six stages.
4. Require synchronized bunny and fox policies for predator-prey and obstacles.
5. Preserve the current visual distinctions for food, water, shelter, species,
   solid obstacles, and thorns.

Exit gate: one browser session visits all six routes. Every route reaches
`Ready`, advances for at least five seconds, and shows the expected agent and
object types.

### 6. Add checkpoint upload

1. Read files through the browser File API without sending bytes off-device.
2. Enforce file count and size before decoding.
3. Validate record shape and finite probe output.
4. Swap only at an episode boundary and reset recurrent memory.
5. Show exact errors while retaining the last working policy.
6. Add reset-to-bundled behavior.

Exit gate: a known alternate checkpoint changes the displayed hash and policy
source, completes an episode, and can be reset to the bundled checkpoint. A
wrong-stage or malformed file is rejected without stopping the running world.

### 7. Close Nix, browser, and tailnet acceptance

1. Add the pure static package and persistent Trunk server app.
2. Run the browser suite against the release bundle.
3. Keep the server running through one watched asset change and verify reload.
4. Open the tailnet DNS URL from the browser and repeat the six-stage smoke.
5. Inspect representative survival and predator-prey screenshots.
6. Update README and HANDOFF only after code and tests are final.

Exit gate: the user opens the supplied tailnet URL and sees ecosystem inference
running without a local build command on the viewing device.

## Test and evidence matrix

Rust tests must cover:

- manifest schema version and every environment ID;
- every required and forbidden policy-role combination;
- missing, duplicate, oversized, hash-mismatched, malformed, and wrong-shape
  checkpoint input;
- path and bytes load parity;
- fresh recurrent memory on route, policy, and episode changes;
- stage configuration and checkpoint-sidecar agreement;
- deterministic seed reset and fixed-step progression;
- every `LoadState` and its displayed diagnostic;
- fallback to the last working policy after an upload failure.

Browser tests must cover:

- initial page load and bundled survival policy readiness;
- all six route changes;
- an increasing simulation step counter on each route;
- visible canvas motion rather than only timer progress;
- separate bunny and fox load state on predator-prey and obstacles;
- successful upload and reset-to-bundled;
- malformed upload rejection while playback continues;
- pause, restart, and speed controls;
- no uncaught page error, failed request, or console error;
- a screenshot for survival and predator-prey at a stable viewport.

Native-to-browser parity uses one exported checkpoint, fixed observation,
fixed recurrent memory, and deterministic mean action. Measure the initial
Flex difference before fixing a tolerance. Store the native expected vector and
the smallest tolerance justified by that measurement.

Record the release WASM byte size, compressed transfer size, checkpoint transfer
size, time to first rendered frame, checkpoint decode time, and steady-state
frame time. These are evidence, not success claims, until measured on the
release build.

## Required gates

Run the exact applicable commands after the final source edit:

```sh
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
personal-lints --repo . --dry-run
personal-lints --repo .
nix flake check
nix build .#web-dist
nix run .#web-check
```

Also inspect the WASM dependency graph for `x11`, `wayland`,
`multi_threaded`, `tokio`, `burn-autodiff`, `mujoco`, and native video/process
dependencies. Every match requires a concrete target-specific explanation or a
feature correction.

After the final build, start the loopback launcher, resolve the current
MagicDNS name, and verify both:

```text
http://127.0.0.1:8080/
https://nixos.tail87afcc.ts.net:8443/
```

The tailnet hostname and IP are machine state. Re-read them before final
handoff instead of assuming this research snapshot is still current.

## Risks and stop conditions

- If Avian's current parallel feature enters the WASM graph, split native and
  web dependency features. Do not enable WASM threads to preserve the existing
  feature graph.
- If the exact `wasm-bindgen-cli` in Nix differs from the crate schema version,
  align the lock and CLI explicitly. Do not let Trunk download an unpinned tool
  during a pure Nix build.
- If a current checkpoint loads but its sidecar describes another stage or
  tuning profile, reject it. A visually moving policy is not compatibility
  evidence.
- If no current compatible checkpoint exists for a stage, run native training
  or fine-tuning. Do not silently ship a random policy as `best`.
- If an uploaded record can only be identified by user-selected stage, label
  its metadata unverified. Named MessagePack weights do not prove environment
  provenance.
- If browser rendering and native watch diverge, keep one simulation and visual
  snapshot source. Do not maintain a second environment implementation for the
  site.
- If Tailscale Serve cannot proxy Trunk's reload socket, use the immutable
  release bundle for remote verification while keeping local watch mode. Do
  not replace the final HTTPS route with a public or all-interface server.

## Definition of done

- `nix build .#web-dist` produces a self-contained static site.
- `nix run .#web-serve` serves and rebuilds it without a server restart.
- The site is reachable through the current tailnet DNS name.
- All six ecosystem environments load their declared default checkpoints and
  visibly advance under Burn Flex inference.
- Predator-prey and obstacles load synchronized bunny and fox policies.
- Local checkpoint upload, validation, activation, failure recovery, and reset
  work in the browser.
- Browser automation passes against the release bundle with no console error.
- Native example behavior and public checkpoint paths remain compatible.
- The exact Rust, personal-lint, Nix, WASM graph, and browser gates pass.
- README and HANDOFF contain the final command, current URL, supported browser
  profile, checkpoint provenance, and verification boundary.
