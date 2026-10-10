---
version: 1
slug: "examples-robots-standing-viewer-rs"
primary_target: "examples/robots/standing_viewer.rs"
related_targets: ["examples/robots/world_scene/viewer.rs","scripts/build_drone_skills.sh"]
---

# Shared-world inspection

Extend the existing standing inspection scene. Preserve its visual system and default standing behaviour.
The user watches frozen RL inference and checks all six physical robots.
Competition qualification remains unfinished.

## Direction contract

THESIS: Show three drones and three droids in the common physical world, with their policy identities and playback state.

OWN-WORLD: Retain Mona Sans, the light grey sky, muted ground, licensed models and charcoal controls from standing inspection.

STORY: Run the clip, inspect any robot, pause, step, reset and change speed. Free camera input affects presentation only.

FIRST VIEWPORT: The six robots occupy the scene. Playback state stays at the upper left.
Controls and both checkpoint identities occupy the lower left and wrap at narrow widths.
The qualification label remains visible. The guide and asset credits remain linked.

FORM: A precise extension of standing inspection. No concept roll applies; the incumbent composition is the authority.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

The existing implementation supplies visual authority; DESIGN.md is absent.
Preserve the incumbent system. This extension creates no raster assets or replacement identity.

## Invariants

- Select `standing` or `shared-world` through a closed mode before loading policies.
- Preserve standing as the default. Reject unsupported explicit modes before inference.
- Keep the same standing executable; use native `--scene shared-world` and a validated browser attribute.
- Load the recorded hover and standing checkpoints through the tested shared session.
- Project snapshots onto all six licensed models. Do not play locomotion clips or move authoritative bodies through transforms.
- Capture thirteen unique body bones for each droid within its own scene root.
- Preserve original drone model alignment and named rotor pivots.
- Block playback until every model is ready. Show model or policy failure without a fallback.
- Reuse fixed 20 ms physics. Speed changes the number of policy steps, not actuator values or timestep.
- Follow any of six agents, show all six, or use a free camera. Camera input cannot enter actor encoders.
- Compare playback and reset with direct frozen sessions; test mode rejection, readiness and camera isolation.
- Run exact native, WASM, strict and personal lint gates. Record measured coverage gaps.
- Inspect rendered desktop and narrow views. Save complete clips, screenshots and console evidence through T3 preview.
- Use one bounded visual fix batch, the installed skill's finish reviewer and documenter.
