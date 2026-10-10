# Shared-world viewer documentation handoff

The shipped `impeccable_documenter` definition was unavailable. This pass used `/home/sagan/.agents/skills/impeccable/reference/degraded/documenter.md` and `reference/document.md`.

Files written: `DESIGN.md`, `.impeccable/design.json`, and this record. `PRODUCT.md` is absent. This pass changed no source, tests, captures, or raster assets.

Palette: browser shell uses `#B8E1E6`, `#171A1C`, and `#EDF0F1`; scene tokens are `#B8C4C9`, `#8C998F`, `#D6D6C4`, `#171A1C`, and `#383D40`.
Type ramp: Mona Sans; title 24 px; status and controls 15 px; robot and checkpoint labels 13 px; footer links 14 px. The sources declare no custom font weight or line height.
Named rules: none recorded.
Layout and shape: full-height canvas with a 16 px overlay inset; panels use 12 px insets; shared-world control panel max width is 600 px; Bevy panels and buttons have square corners.
Components: buttons use a 44 px minimum height and 12-by-10 px padding; robot labels use a 90 px node; footer padding is 18 px with 12-by-24 px gaps, tightening to 10-by-20 px at 600 px; focus outline is 2 px with a -3 px offset.

Source values were checked in `robot-web/skills/styles.css`, `examples/robots/standing_scene/view.rs`, `examples/robots/world_scene/view.rs`, `examples/robots/world_scene/viewer.rs`, `scripts/build_drone_skills.sh`, `Cargo.lock`, and the font/model license files. The footer links and their responsive styles are in `robot-web/skills/styles.css`; scene colors and overlay sizes are in the two `view.rs` files. Bevy 0.18.1 supplies the white default control text. The license files are `assets/fonts/OFL.txt`, `assets/robots/README.md`, and `assets/robots/survival/README.md`.

Finish review used a generated fallback and returned `ship`. It later reported clipping from a scaled mobile preview. The reviewer rechecked raw 566×1224 frames at 14 s and 16 s, then retracted that finding. Footer labels and checkpoint identities remained readable; no layout change was indicated.

The original control recordings predate the startup-error fix; presentation source is unchanged. Later release captures verify six-policy playback at desktop and narrow widths. Invalid modes and missing canvases show errors before any model request. Native window interaction remains unverified. The evidence record retains runtime hashes, media provenance and the preview disconnection after release checks.

Format checks against the official alpha spec passed: frontmatter YAML and sidecar JSON parse, token references resolve, each color has an eight-step ramp, component references resolve, and canonical section order matches. The documentation does not claim competition qualification.
