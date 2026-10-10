---
version: alpha
name: Bevy Gym Browser Skills
description: Visual rules for Bevy Gym's browser robot scenes.
colors:
  primary: "#B8E1E6"
  charcoal: "#171A1C"
  shell-text: "#EDF0F1"
  control-text: "#FFFFFF"
  world-sky: "#B8C4C9"
  world-ground: "#8C998F"
  reset-marker: "#D6D6C4"
  control: "#383D40"
typography:
  scene-title:
    fontFamily: "Mona, sans-serif"
    fontSize: "24px"
  status:
    fontFamily: "Mona, sans-serif"
    fontSize: "15px"
  control-label:
    fontFamily: "Mona, sans-serif"
    fontSize: "15px"
  footer-link:
    fontFamily: "Mona, sans-serif"
    fontSize: "14px"
  robot-label:
    fontFamily: "Mona, sans-serif"
    fontSize: "13px"
  checkpoint-id:
    fontFamily: "Mona, sans-serif"
    fontSize: "13px"
rounded:
  square: "0px"
spacing:
  scene-inset: "16px"
  panel-inset: "12px"
  header-gap: "6px"
  control-gap: "8px"
  control-panel-gap: "10px"
  button-padding-x: "12px"
  button-padding-y: "10px"
  footer-padding: "18px"
  footer-link-row-gap: "12px"
  footer-link-column-gap: "24px"
  footer-mobile-row-gap: "10px"
  footer-mobile-column-gap: "20px"
components:
  footer-links:
    textColor: "{colors.primary}"
    typography: "{typography.footer-link}"
    padding: "{spacing.footer-padding}"
  footer-links-hover:
    textColor: "{colors.control-text}"
---

# Design System: Bevy Gym Browser Skills

## Overview

The shared-world browser skill shows three drones and three droids in one Bevy canvas. Playback state and controls sit over the scene. The guide and model credits appear below it.

The standing and shared-world scenes use the same Mona Sans asset, sky, ground, reset markers, and charcoal panels. The shared-world viewer extends the standing inspection layout to six robots and two checkpoint identities.

Key characteristics:

- The scene stays visible behind the status and control panels.
- Mona Sans appears in the browser shell and Bevy UI.
- The scene uses a light grey sky, muted ground, and the licensed robot models.

## Colors

The browser shell uses pale blue for links and focus. The physical scene uses a light grey sky, muted ground, and pale reset markers; charcoal panels separate text and controls from the scene.

### Primary

- `primary` colors the training guide and asset links, text selection, and visible focus in `robot-web/skills/styles.css`.

### Neutral

- `charcoal` is the browser background and the status and control panel fill in `robot-web/skills/styles.css`, `examples/robots/standing_scene/view.rs`, and `examples/robots/world_scene/view.rs`.
- `shell-text` colors browser-shell text in `robot-web/skills/styles.css`.
- `control-text` is Bevy 0.18.1's default `TextColor`; `examples/robots/world_scene/view.rs` does not override control text color.
- `world-sky`, `world-ground`, and `reset-marker` use the values shared by `examples/robots/standing_scene/view.rs` and `examples/robots/world_scene/view.rs`.
- `charcoal` colors robot labels, while `control` colors playback buttons in `examples/robots/world_scene/view.rs`.

## Typography

The browser stylesheet names `assets/fonts/MonaSans-VariableFont.ttf` `Mona`; Bevy loads the same file for its overlay text. The source sets sizes by role but declares no custom font weight or line height.

### Hierarchy

- `scene-title` labels the inspection trial in `examples/robots/world_scene/view.rs` and `examples/robots/standing_scene/view.rs`.
- `status` and `control-label` cover the readout and playback buttons in those scene files.
- `footer-link` is used below the canvas in `robot-web/skills/styles.css`.
- `robot-label` and `checkpoint-id` identify each robot and the two frozen policies in `examples/robots/world_scene/view.rs`.

## Layout

The browser page uses a full viewport-height column. The scene canvas grows above a footer that holds the guide and asset links. At widths of 600 pixels or less, the scene keeps a taller minimum height and the footer link gaps tighten; the links wrap.

The Bevy overlay fills the canvas with a 16-pixel inset. The status panel anchors at the upper left. The control panel anchors at the lower left and wraps its buttons. Its maximum width is 600 pixels in the shared-world view; the standing view keeps its 520-pixel maximum. Both panels use a 12-pixel inset. These values come from `robot-web/skills/styles.css`, `examples/robots/standing_scene/view.rs`, and `examples/robots/world_scene/view.rs`.

## Elevation & Depth

The robot scene gets depth from its directional light, cast shadows, and model materials. The status and control panels use solid fills. `robot-web/skills/styles.css` declares no box shadows.

## Shapes

The Bevy panels and buttons use square corners through Bevy's zero-radius `BorderRadius::DEFAULT`. The browser footer uses underlined links with a three-pixel underline offset. Its visible-focus outline is two pixels wide with an inset three-pixel offset, as declared in `robot-web/skills/styles.css`.

## Components

### Status panel

The upper panel groups a 24-pixel trial title and a 15-pixel status readout. In the shared-world view, the readout retains playback state and the visible `Unqualified for competition` label. The panel uses a 12-pixel inset and a six-pixel gap, as defined in `examples/robots/world_scene/view.rs`.

### Playback controls

The lower panel groups playback controls, view selection, and the two checkpoint identities. Its buttons have a 44-pixel minimum height and 12-by-10-pixel padding. Button rows wrap with an eight-pixel gap. The panel's maximum width expands for the shared-world labels. `examples/robots/standing_scene/view.rs` and `examples/robots/world_scene/view.rs` define the two widths.

### Robot labels

Each robot label uses centered 13-pixel Mona Sans text over a 90-pixel node. `examples/robots/world_scene/view.rs` positions each label from its robot's projected body position.

### Browser footer links

The footer displays the training guide and asset credits as underlined links. The shared-world page lists separate drone and droid credits. Links turn white on hover and keep the visible-focus outline. `scripts/build_drone_skills.sh` supplies the shared-world guide and credit links; `robot-web/skills/styles.css` supplies their styles.

## Do's and Don'ts

### Do

- Do reuse the bundled Mona Sans face in the browser shell and Bevy overlays.
- Do keep the guide and asset credits below the canvas, with underlines and visible focus.

### Don't

- Don't replace the licensed drone or mannequin assets when extending these browser scenes.
