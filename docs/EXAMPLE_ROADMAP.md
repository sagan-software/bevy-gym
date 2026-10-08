# Quality and examples roadmap

Owner request: 2026-10-08. This roadmap supersedes the delivery order in older
plans. Preserve their implementation evidence, but verify it against current code.
Execution state belongs in [EXAMPLE_STATUS.md](EXAMPLE_STATUS.md).

## Outcome

Make adding a trained bot to a Bevy game straightforward, with safe defaults,
strong types, and small tutorial examples. Provide a browser gallery modeled on
the [Bevy examples site](https://bevy.org/examples/), with working training and
inference for every delivered environment. Audit the entire library, trainers,
examples, browser packages, tests, assets, documentation, and CI.

The work includes 23 Gymnasium tasks, 17 Unity example families, 11 Godot video
examples, custom robot examples, and a separately verified AI Warehouse inventory.
Unity's “3D ball” and “3D balance ball” name the same family. Preserve its ordinary,
hard, and visual variants. Godot's “Robot F” maps to Robot FPS.

## Priority and publish checkpoints

Complete one tested achievement before expanding the next. Each checkpoint needs
its own descriptive commit, status update, and push to GitHub `main`. Preserve
unrelated staged changes. Fetch before pushing; integrate remote changes without
force. A passing local gate does not establish CI or deployment success.

### P0: audit and reference foundation

- Record existing behavior, dependencies, API risks, and example complexity.
- Remove imports used only to appease `unused_crate_dependencies`. Allow that
  lint for examples with a reason. Preserve dependencies actually needed by code.
- Separate tutorial entry points from training orchestration and benchmarking.
  Keep the complete concept readable top to bottom. Avoid both thousand-line
  examples and opaque one-line launchers that teach nothing.
- Inventory public constructors, accessors, conversions, mutable components,
  environment lifecycle, asynchronous policy responses, and trainer validation.
- Add ergonomic APIs alongside stable APIs. Do not silently change public types,
  errors, or serialized formats. Make any migration a separate reviewable choice.
- Publish the plan and initial findings, then the independently tested cleanup.

### P1: ARC Raiders-inspired robots

Research precedes robot implementation. Inspect official gameplay and development
footage of Wasps, Hornets, the jumping quadruped, and other robots. Download videos
with `yt-dlp`; preserve source URLs, timestamps, hashes, captions, and contact
sheets. Read Embark's accounts and relevant drone-control research. Separate
observed behavior, published methods, and project design choices.

1. Deliver a four-thruster hover environment with gravity, inertia, individual
   motor forces, torque, collision, bounded actions, and deterministic resets.
   Publish its tutorial, native/WASM tests, renderer, and baseline controller.
2. Train hover and upright recovery using batched simulation and a curriculum.
   Publish qualified checkpoints, learning curves, and browser training/inference.
3. Add an animated moving humanoid target, occlusion-aware perception, pursuit,
   target loss, remembered sightings, search, and reacquisition. Add hearing with
   explicit range, obstruction, and event lifetime. Prove hidden targets do not
   leak through actor observations. Distinguish semantic ray sensing from rendered
   camera vision; add an actual image-observation example for the latter.
4. Add independently destructible thrusters, impacts, and damaged flight training.
   Evaluate each single-thruster failure and selected multiple failures. Do not
   guarantee hover for mechanically uncontrollable damage states. Show recovery,
   degraded pursuit, landing, and failure outcomes honestly.
5. Deliver a joint-driven jumping quadruped with ground contact, balance,
   telegraphed jumps, target pursuit, attacks, and independently destructible legs.
   Train damaged locomotion and recovery. Rendering must follow physics rather
   than moving a decorative robot along a scripted path.
6. Add shared-policy groups, distinct robot policies, cooperative pursuit, and
   adversarial target learning. Explain curriculum, batch, multi-agent, and
   adversarial learning through separate runnable tutorials in the same arena.
7. Publish the robot gallery and beginner integration guide. Re-audit the public
   API against the examples before proceeding to Gymnasium.

Select one attractive Creative Commons four-thruster model after inspecting its
geometry, materials, license, and browser cost. Record model node-to-actuator
mapping. Reuse licensed humanoid animation where possible. Use original robot art
or licensed replacements, not extracted ARC Raiders assets. Gameplay footage is
reference evidence; asset redistribution requires its own permission.

### P2: Gymnasium

Finish and qualify each task in this order. Existing implementations are starting
points, not completion evidence.

- Classic Control: CartPole, MountainCar, continuous MountainCar, Pendulum, Acrobot.
- Toy Text: Cliff Walking, Frozen Lake, Taxi, Blackjack.
- Box2D family: Lunar Lander, Bipedal Walker, Car Racing.
- MuJoCo: Inverted Pendulum, Inverted Double Pendulum, Reacher, Pusher, Swimmer,
  Hopper, Half Cheetah, Walker 2D, Ant, Humanoid Stand Up, Humanoid.

Keep the existing Gymnasium source pin until a reviewed upgrade. Download all
23 official GIFs and inspect chronological contact sheets. Existing six-frame
sheets are useful inventory evidence; add denser samples for motion details.
Port source assets, geometry, camera, colors, and animation. Compare fixed states
and complete episode sequences against upstream.

For every task, specify observations, actions, reset distribution, dynamics,
reward, termination, truncation, options, and time limits. Compare against a
pinned upstream oracle. Same numeric seeds alone do not prove equivalent random
streams. Keep environment reward separate from training-only shaping.

The user permits Rapier2D or another suitable replacement for Box2D. A solver
replacement cannot promise identical trajectories: contact impulse differences
can change the first landing frame and terminal reward. Preserve task semantics
and set measured tolerances; retain an upstream-engine path when exact transition
parity is required. Label approximation claims explicitly. Prove a MuJoCo browser
engine path before promising faithful WASM MuJoCo environments.

Publish one complete environment per checkpoint. Keep unqualified examples out of
the default “working examples” catalog or show their exact incomplete status.

### P3: Unity ML-Agents

Use pinned scenes, agent scripts, training configuration, materials, meshes, and
official images under `Project/Assets/ML-Agents/Examples`. Include these families:

- Basic; 3D Balance Ball with hard and visual variants; GridWorld.
- Push Block; Wall Jump; Crawler; Worm; Food Collector; Hallway.
- Soccer Twos; Strikers versus Goalie; Walker; Pyramids.
- Match 3; Sorter; Cooperative Push Block; Dungeon Escape.

Start with Basic and Ball, then discrete navigation, manipulation, locomotion,
memory, and group tasks. Preserve task-specific action masks, branched actions,
mixed continuous/discrete actions, group rewards, recurrence, and termination.
Check historical revisions for examples absent from the current source tree.
Do not silently substitute Soccer Twos for Strikers versus Goalie.

Download official screenshots and available videos. Make video contact sheets and
compare camera, layout, geometry, materials, timing, and state changes. Record
behavioral differences caused by replacing Unity physics. Publish each family
after tutorial, conformance, learning, asset, and browser checks pass.

### P4: Godot RL Agents

Port only the eleven README examples with videos:

- 3D Car Parking; Item Sorting Cart; Hovercraft Racing; 3D Lander.
- MultiLevel Robot; Robot Volleyball; DownFall; MultiAgent Simple.
- Cross The Road; Score The Goal; Robot FPS.

Clone or initialize the pinned reference repository. Read each scene, agent,
reward function, sensor configuration, termination rule, and asset license.
Download all eleven README videos, retain timestamped contact sheets, and copy
permitted assets with attribution. Inspect per-example notices rather than assuming
the repository's MIT license covers every model. Publish one complete port at a time.

### P5: AI Warehouse

Verify the creator's repository and video inventory before defining ports.
The public `AIWarehouse/AIWarehouse` README currently says its code is private.
Do not invent an available source implementation. Record public videos and any
later releases; distinguish behavior reproductions from source-faithful ports.
Unavailable code or unclear asset rights block only the affected example.

## Definition of done for each example

- A focused tutorial names prerequisites and runs from a fresh clone. Comments
  explain the concept and decisions. No dummy dependency imports remain.
- A pinned source contract and licenses identify what was reproduced and changed.
- Native and WASM execute the same authoritative environment logic.
- Behavior tests cover every changed branch, invalid action, reset, time limit,
  observation boundary, and terminal path. Measure coverage and record exact gaps.
- Learned behavior beats declared baselines on held-out seeds. Separate training,
  checkpoint selection, and final evaluation seeds. Report multiple training
  seeds, failures, and damaged-agent outcomes. Scripted motion is not RL evidence.
- Browser training performs real optimizer updates; inference uses frozen weights.
  Both remain responsive. Pause, step, restart, speed, seed, policy upload/download,
  learning curves, return, and terminal status work without exposing debug clutter.
- Visual comparisons include reference frames, local frames, desktop and mobile
  inspection, sustained motion, occlusion, contacts, damage, and repeated resets.
- Release assets have provenance, hashes, notices, size, startup, and frame-time
  measurements. A fresh checkout needs no private checkpoint or `runs/` directory.
- Required gates pass after the final edit. CI and deployed URL checks have separate
  outcomes. Status records the commit, commands, results, artifacts, and next task.

## Required gates

Run the nearest focused regression first and preserve its genuine initial failure.
Follow the requested Rust implementation checkpoints, including final prose-only
documentation and runnable examples. Run these exact Rust gates through Nix:

```sh
nix develop --command cargo fmt --all -- --check
nix develop --command cargo test
nix develop --command cargo clippy --all-targets --all-features -- -D warnings
```

Use the installed `liamc-lints` workflow. Run changed-code coverage with
`cargo llvm-cov` when available, after confirming toolchain support. No repository
coverage command was found in the initial inspection; establish one before a
production change claims complete coverage. Compile-fail tests must prove APIs
that intentionally make misuse unavailable.

CI also requires fixture contracts, dependency tests, both browser packages,
browser Clippy for native and WASM, `nix run .#web-check`, and
`nix run .#gymnasium-check`. Run applicable Nix checks after Nix edits. Preserve
Chromium, Firefox, and WebKit coverage. Use the collaborative browser for visible
inspection; record unavailable visual checks without inventing results.

## Audit worklist

Review core state and lifecycle first, then plugin scheduling and multi-environment
isolation, environment math, trainers and replay, checkpoint loading, browser
workers and rendering, assets, tutorials, build reproducibility, and CI.
For each area, record inspected files, contract, concrete counterexample, severity,
smallest fix, passing boundaries, and missing evidence. Track pending areas in the
status document so “initial audit” cannot be confused with “full audit complete.”
