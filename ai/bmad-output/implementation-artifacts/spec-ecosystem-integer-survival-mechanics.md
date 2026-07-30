---
title: 'Ecosystem rounded geometry and integer survival mechanics'
type: 'feature'
created: '2026-07-29'
status: 'done'
baseline_commit: 'b01f9b1691335681704b386bf84c25e88460ba47'
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Water interaction sensors are visible to perception rays, so rays cast from inside a well stop at the square sensor boundary. Square agents and world objects also create corner contacts, while the current float-based physiology and dense survival reward obscure the requested survival objective.

**Approach:** Make interaction sensors invisible to ray queries, use matching circular or oval visuals and colliders for movable and interactable objects, and replace continuous physiology with discrete points and timed damage. End episodes after a configurable number of seconds and emit the health-weighted survival reward only when the episode ends, while retaining need-weighted food and water bonuses.

## Boundaries & Constraints

**Always:** Keep the actor and critic tensor widths stable. Keep perception egocentric and derived from the same samples used by the HUD rays. Use a 5 HP, 5 satiation, and 5 hydration default profile. Reduce each need by one point every five simulated seconds. While a need is zero, deal one HP every three starvation seconds and every two dehydration seconds. Add one satiation point per food event and one hydration point per completed drinking event. Default the episode to 20 simulated seconds. At termination or truncation, add `survived_seconds * remaining_hp / maximum_hp` to resource bonuses earned during that transition. Keep numeric conversions at observation, reward, metric, and rendering boundaries.

**Ask First:** Changes to the stable observation width, action width, checkpoint tensor layout, or non-ecosystem environments.

**Never:** Let the well drinking sensor become a semantic ray hit. Restore square colliders or visuals for agents, food, wells, trees, rocks, or thorns. Claim new learning evidence from an old checkpoint or a pre-change run.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Water perception | Agent is inside the circular drinking sensor | Rays ignore the sensor and hit only visible semantic bodies or extend to full range | A missing semantic component is a miss, not a short ray |
| Need timing | Satiation or hydration is positive | Exactly one point is removed after each configured five-second interval | Counters saturate and points never underflow |
| Deprivation | A need remains at zero | HP falls by one at its starvation or dehydration interval | Simultaneous due damage is cumulative and HP saturates at zero |
| Resource use | Agent consumes compatible food or completes drinking | The matching need gains one point up to five and earns its pre-consumption need-weighted bonus | Full needs remain capped and earn zero bonus |
| Horizon | Agent reaches 20 seconds with 90 percent HP | Final transition includes 18 survival points plus any resource bonus | Later steps return `EpisodeFinished` |
| Early death | Agent dies before the horizon | Final transition uses actual survived seconds and remaining HP fraction | Death status and cause remain explicit |

</frozen-after-approval>

## Code Map

- `examples/ecosystem/shared/domain.rs` -- discrete tuning profile, simulation contract, snapshots, and stable tensor metadata.
- `examples/ecosystem/shared/simulation.rs` -- Avian shapes, ray queries, contacts, physiology clocks, observations, and rewards.
- `examples/ecosystem/shared/rendering.rs` -- oval and circular meshes plus Inspector-egui controls and labels.
- `examples/ecosystem/shared/demo.rs` -- visual trainer default horizon and worker tuning.
- `examples/ecosystem/shared/training.rs` -- training defaults, persisted tuning, metrics, and checkpoint compatibility.
- `examples/ecosystem/shared/video.rs` -- recorded-playback horizon wiring.
- `examples/ecosystem/{README.md,EVIDENCE.md}` -- commands, mechanics, and fresh verification evidence.

## Tasks & Acceptance

**Execution:**
- [x] `examples/ecosystem/shared/simulation.rs` -- add failing regression tests, then implement sensor filtering, rounded colliders, discrete physiology, and terminal reward.
- [x] `examples/ecosystem/shared/domain.rs` -- replace float reserve and step-limit tuning with bounded integer point and second settings while preserving tensor shapes.
- [x] `examples/ecosystem/shared/rendering.rs` -- render matching circles and ovals and expose second- and point-based live controls.
- [x] `examples/ecosystem/shared/{demo.rs,training.rs,video.rs}` -- propagate the 20-second default and persist the new tuning schema.
- [x] `examples/ecosystem/{README.md,EVIDENCE.md}` -- describe the current behavior and record fresh headless and visual proof.

**Acceptance Criteria:**
- Given an agent inside the well sensor, when perception updates, then no ray terminates at the drinking sensor boundary.
- Given contact with any rounded scene object, when physics resolves movement, then no square corner can trap the oval agent.
- Given a configured 10-HP maximum, when a 20-second episode ends with 9 HP, then the survival component is exactly 18.
- Given each physiology interval boundary, when one simulation step crosses it, then the requested integer state transition occurs once.
- Given rendered and headless runs, when the example trains and plays, then both use the same mechanics and fresh metrics.

## Spec Change Log

- 2026-07-29 review clarification: The 90 percent reward example uses the configurable 10-HP profile because the 5-HP default has 20 percent increments. The 5-point need-loss interval is the idle clock; translation adds the previously requested 25 percent movement cost. KEEP the general health-weighted formula, integer default maxima, and translation-only movement cost.

## Design Notes

Map boundaries remain rectangular because they define the square playable area. The rounded-shape rule applies to agents and discrete world objects. Physics can retain fractional time and well capacity internally, but agent physiology remains integer state. The five-second need interval is the idle rate. Translation advances that clock by the configured movement-cost percentage, while rotation and gaze do not. Observation and reward APIs remain floating-point because the policy contract requires normalized tensors and scalar rewards.

## Verification

**Commands:**
- `cargo test --example ecosystem-survival` -- ecosystem mechanics and regression tests pass.
- `cargo fmt --all -- --check` -- all Rust formatting passes.
- `cargo test` -- the full test suite passes.
- `cargo clippy --all-targets --all-features -- -D warnings` -- strict Clippy passes.
- `personal-lints --repo .` -- candidate-local personal lint findings are clear.
- `cargo run --no-default-features --release --example ecosystem-survival -- train` -- fresh headless training produces improving evaluation data.
- `cargo run --release --example ecosystem-survival` -- rendered mode shows full-range misses, rounded shapes, correct HUD values, and successful resource contacts.

## Suggested Review Order

**Simulation contract**

- Start with the fixed-step ordering that coordinates motion, contacts, physiology, perception, and reward.
  [`simulation.rs:450`](../../../examples/ecosystem/shared/simulation.rs#L450)

- Inspect discrete physiology clocks and explicit combined-damage attribution.
  [`simulation.rs:1267`](../../../examples/ecosystem/shared/simulation.rs#L1267)

- Verify the terminal survival formula remains isolated and health-weighted.
  [`simulation.rs:1675`](../../../examples/ecosystem/shared/simulation.rs#L1675)

**Configuration and playback integrity**

- Review bounded integer tuning without changing policy tensor widths.
  [`domain.rs:407`](../../../examples/ecosystem/shared/domain.rs#L407)

- Check fail-closed checkpoint saves and immutable stage-aware tuning sidecars.
  [`training.rs:1316`](../../../examples/ecosystem/shared/training.rs#L1316)

- Check exact stage, predecessor, profile, and pair validation at load boundaries.
  [`training.rs:2024`](../../../examples/ecosystem/shared/training.rs#L2024)

- Confirm video segments share one profile and record effective tuning.
  [`video.rs:330`](../../../examples/ecosystem/shared/video.rs#L330)

**Visual behavior and evidence**

- Review checkpoint playback setup and stage-aware policy loading.
  [`rendering.rs:309`](../../../examples/ecosystem/shared/rendering.rs#L309)

- Confirm rays ignore the circular drinking sensor at the public simulation seam.
  [`simulation.rs:2048`](../../../examples/ecosystem/shared/simulation.rs#L2048)

- Read the fresh learning curve, held-out evaluation, and screenshot evidence.
  [`EVIDENCE.md:1`](../../../examples/ecosystem/EVIDENCE.md#L1)
