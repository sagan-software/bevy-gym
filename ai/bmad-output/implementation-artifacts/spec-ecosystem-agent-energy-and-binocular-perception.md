---
title: 'Ecosystem agent energy and binocular perception'
type: 'feature'
created: '2026-07-29'
status: 'done'
baseline_commit: 'aff3e82'
context:
  - '{project-root}/examples/ecosystem/PLAN.md'
  - '{project-root}/examples/ecosystem/EVIDENCE.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Ecosystem agents move too slowly, sense uniformly in every direction, and survive long enough without resources that training does not clearly teach efficient food and water seeking. The viewer also lets HUD pointer input move the camera and lacks drag panning and learning-rate graphs.

**Approach:** Introduce faster energy-aware locomotion, bounded overconsumption consequences, independently controllable binocular gaze, and need-weighted food and water rewards. Expose the meaningful experiment settings in the HUD, correct camera input ownership, and select defaults through measured headless trials before rendered verification.

## Boundaries & Constraints

**Always:** Rotation and eye movement consume only baseline needs; actual linear speed adds need drain without adding a reward term. Two forward-facing eye origins emit dense frontal and sparse peripheral rays, never rear vision, and gaze remains bounded relative to the head. Food and water reserves may exceed nominal fullness but stop at a hard capacity; fullness slowdown begins at a configurable threshold, and overfill above nominal fullness damages health. Food and absorbed-water rewards use the reserve before consumption: 125% at or below 10%, 100% through 50%, 90% through 75%, 75% below 90%, and zero from 90% upward. All simulation values shown as configurable must apply between PPO batches and be recorded in metrics. Existing deterministic seeding, fixed-shape tensors across curriculum stages, and timestep-weighted PPO updates remain intact.

**Ask First:** Adding a consumption action, weakening deterministic evaluation, or expanding beyond ecosystem examples and their reusable recurrent PPO boundary.

**Never:** Add direct movement, gaze, damage, or overconsumption rewards. Reward food or water at or above 90% reserve. Charge need drain for angular velocity or gaze. Let HUD scrolling or dragging zoom or pan the world. Claim the prior checkpoint remains compatible after observation or action dimensions change.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Linear movement | Full-speed translation | Faster travel and configurable extra hunger/thirst drain | Bounded finite tuning only |
| Stationary sensing | Zero translation with body turn or gaze | Baseline drain only; perception direction changes | Gaze clamps to its limit |
| Binocular vision | Configured active ray count | Rays divide across two eye origins with frontal density and no rear coverage | At least one ray per eye |
| High fullness | Reserve crosses slowdown threshold | Speed interpolates to the configured multiplier | Hard capacity prevents unbounded values |
| Overconsumption | Reserve exceeds nominal fullness | Health and hit points decrease without reward shaping | Death records overconsumption |
| HUD pointer input | Scroll or drag over egui | HUD alone consumes the input | Camera state remains unchanged |
| World pointer input | Left or middle drag outside egui | Camera pans proportionally to zoom | Zero mouse motion is inert |

</frozen-after-approval>

## Code Map

- `examples/ecosystem/shared/domain.rs` -- fixed observation/action contracts and validated live tuning.
- `examples/ecosystem/shared/simulation.rs` -- Avian locomotion, consumption, physiology, perception, and metrics.
- `examples/ecosystem/shared/training.rs` -- three-axis policy boundary, durable metrics, and learning evidence.
- `examples/ecosystem/shared/rendering.rs` -- eye/ray visualization, HUD controls, graphs, and camera input.
- `examples/ecosystem/shared/demo.rs` -- visual defaults matched to the selected headless profile.

## Tasks & Acceptance

**Execution:**
- [x] `domain.rs` -- add validated movement, fullness, and gaze settings; extend the action and tensor contracts.
- [x] `simulation.rs` -- implement faster locomotion, translation-only drain, bounded overfill, binocular gaze, and observable diagnostics through red-green slices.
- [x] `training.rs` and policy call sites -- migrate to the new action shape, persist all settings, and retain deterministic reward evaluation.
- [x] Need-weighted reward -- add configurable food and absorbed-water rewards with exact reserve-zone multipliers and zero reward from 90% fullness.
- [x] `rendering.rs` -- draw two eye origins and cones, add live controls and learning-velocity/optimizer-rate graphs, isolate egui pointer input, and add left/middle drag panning.
- [x] Headless experiments -- compare bounded physiology profiles, select a short default that improves disjoint-seed survival and resource use, then reproduce it with the no-argument training profile.
- [x] Rendered verification -- run the exact visual command, exercise HUD scrolling and drag panning, and capture baseline and learned screenshots.

**Acceptance Criteria:**
- Given equal initial physiology, when one agent translates and another only rotates or moves its gaze, then only translation adds need drain.
- Given maximum gaze commands, when observations are sampled, then both eye cones remain forward of the head and use separate origins.
- Given reserves at the slowdown and overfill boundaries, when the world steps, then configured speed reduction and damage occur at their exact thresholds.
- Given hunger or thirst at each zone boundary, when a compatible resource is consumed, then its base reward uses the exact pre-consumption multiplier.
- Given either reserve at or above 90%, when a compatible resource is consumed, then that resource adds zero reward.
- Given HUD hover, when the wheel or drag input occurs, then camera transform and scale do not change.
- Given world hover, when either supported mouse button drags, then the camera pans without changing simulation state.
- Given a completed training update, when the HUD redraws, then survival, survival change per update, and actor/critic optimizer rates have graphed points.
- Given the selected default profile, when trained headlessly and rendered, then durable evaluation data demonstrates improvement over step zero within the demo budget.

## Spec Change Log

- 2026-07-29: Human changed the reward constraint from survival-only to the
  exact hunger and thirst multiplier zones implemented above.

## Design Notes

Eye control uses one conjugate gaze axis for both eyes. Separate eye origins provide binocular parallax without adding two independently divergent eye controls. Reserved ray slots are ordered by perceptual importance so lowering the live count removes peripheral rays before dense frontal rays.

## Verification

**Commands:**
- `cargo test --no-default-features --example ecosystem-survival` -- headless mechanics and policy contracts pass.
- `cargo test --example ecosystem-survival --all-features` -- HUD and rendered contracts pass.
- `cargo test` -- reusable trainer regressions pass.
- `cargo clippy --all-targets --all-features -- -D warnings` -- no warning remains.
- `personal-lints --repo .` -- no candidate-local diagnostic remains; repository/toolchain boundaries are reported exactly.
- `cargo run --no-default-features --release --example ecosystem-survival -- train` -- selected defaults produce durable learning evidence.
- `cargo run --example ecosystem-survival` -- binocular rays, graphs, pointer isolation, and drag panning are inspected in the rendered demo.

## Suggested Review Order

**Agent contract**

- Start with the validated controls and fixed binocular policy shape.
  [`domain.rs:406`](../../../examples/ecosystem/shared/domain.rs#L406)

- Inspect paired frontal and peripheral ray ordering.
  [`domain.rs:111`](../../../examples/ecosystem/shared/domain.rs#L111)

**Simulation behavior**

- Translation speed, fullness slowdown, body turn, and gaze meet at action application.
  [`simulation.rs:947`](../../../examples/ecosystem/shared/simulation.rs#L947)

- Consumption applies need-weighted reward before physiology applies movement drain and overfill damage.
  [`simulation.rs:1123`](../../../examples/ecosystem/shared/simulation.rs#L1123)

**Training proof**

- Batch boundaries persist tuning and reset selection before comparing policies.
  [`training.rs:737`](../../../examples/ecosystem/shared/training.rs#L737)

- Final headless and disjoint results establish rapid resource learning.
  [`EVIDENCE.md:545`](../../../examples/ecosystem/EVIDENCE.md#L545)

**Interactive demo**

- The HUD owns live settings and applies them at explicit boundaries.
  [`rendering.rs:995`](../../../examples/ecosystem/shared/rendering.rs#L995)

- Survival velocity and bunny optimizer rates remain separate graph series.
  [`rendering.rs:1217`](../../../examples/ecosystem/shared/rendering.rs#L1217)

- Camera input filters HUD capture before wheel zoom or drag panning.
  [`rendering.rs:1604`](../../../examples/ecosystem/shared/rendering.rs#L1604)

**Verification support**

- The completed render shows binocular rays and the measured survival gain.
  [`ecosystem-training-complete.png`](screenshots/ecosystem-training-complete.png)
