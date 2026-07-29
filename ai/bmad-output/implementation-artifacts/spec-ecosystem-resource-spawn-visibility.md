---
title: 'Ecosystem resource spawn visibility'
type: 'bugfix'
created: '2026-07-29'
status: 'done'
baseline_commit: '2fffac738b8e87153606d9bacba79a2769c328b9'
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The survival lesson spawns its first food directly between the bunny and the well. The bunny cannot initially perceive both choices, and the direct route teaches it to drive into the solid well repeatedly.

**Approach:** Spawn food and water at separate visible bearings in front of the bunny. Keep both resources outside initial interaction range and preserve deterministic route rotation between episodes.

## Boundaries & Constraints

**Always:** The initial local observation contains both food and water. Their ray hits use separate bearings. Both resources begin beyond their interaction ranges. The complete layout rotates with the seeded route angle.

**Ask First:** Any change to collision behavior, reward calculation, observation shape, or policy-network architecture.

**Never:** Remove the solid well collider, add hidden steering, place resources behind the bunny, or specialize the layout to one world-space direction.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Survival reset | Valid survival config and seed | Food and water are visible at separate forward bearings and outside contact range | N/A |
| Rotated reset | Different valid seeds | The layout rotates while preserving relative visibility and clearance | N/A |

</frozen-after-approval>

## Code Map

- `examples/ecosystem/shared/simulation.rs` -- survival spawn layout, semantic observations, and regression tests.

## Tasks & Acceptance

**Execution:**
- [x] `examples/ecosystem/shared/simulation.rs` -- add a failing reset regression and separate the initial resource bearings.

**Acceptance Criteria:**
- Given a survival reset, when the first observation is sampled, then it contains food and water hits on separate ray bearings.
- Given the initial world state, when resource clearances are measured, then neither food nor water overlaps the bunny or starts active.
- Given several seeds, when the survival lesson resets, then the same relative guarantees hold after seeded rotation.

## Spec Change Log

## Verification

**Commands:**
- `cargo test --example ecosystem-survival survival_reset_faces_visible_food_and_water` -- expected: focused regression passes.
- `cargo fmt --all -- --check` -- expected: formatting passes.
- `cargo test` -- expected: all tests pass.
- `cargo clippy --all-targets --all-features -- -D warnings` -- expected: no warnings.
- `personal-lints --repo .` -- expected: no candidate-local diagnostics.

## Suggested Review Order

**Spawn layout**

- Start with the deterministic split that puts both resources in the binocular field.
  [`simulation.rs:696`](../../../examples/ecosystem/shared/simulation.rs#L696)

- Review the distance and lateral offset that define the initial choice geometry.
  [`simulation.rs:43`](../../../examples/ecosystem/shared/simulation.rs#L43)

**Regression evidence**

- Verify opposite observation rays and empty Avian contact sets across rotated seeds.
  [`simulation.rs:2089`](../../../examples/ecosystem/shared/simulation.rs#L2089)

- Confirm a steering agent can still reach and consume the relocated food.
  [`simulation.rs:2277`](../../../examples/ecosystem/shared/simulation.rs#L2277)
