# Drone survival upgrade

The playable scene will become a survival game with a licensed animated humanoid,
a matching pistol, and industrial cover. The existing drone remains the opponent.
The full examples roadmap remains active.

## Asset direction

The user selected the mannequin with neutral armor and dark joints.
Use Quaternius's Universal Animation Library Standard v3.0.
It includes a skinned humanoid and authored idle, jog, sprint, pistol aim, shoot,
reload, hit, and death clips. Use its in-place export so physics owns movement.
Recolor its armor and joints to match the drone and industrial props.

The alternative is Quaternius's Animated Robot, also distributed as RobotExpressive
by three.js. It includes running and death but lacks authored pistol clips.
The user reviewed both references and selected the mannequin.

Use the CC0 Sci-Fi Essentials Standard pistol and selected crates or barrels.
Inspect the matching Modular Sci-Fi MegaKit for building pieces with usable entrances.
Keep original licenses, source URLs, download hashes, and any asset transformations.
Only redistribute assets whose individual license and provenance are verified.

Sources:

- [Universal Animation Library](https://quaternius.itch.io/universal-animation-library)
- [Sci-Fi Essentials](https://quaternius.com/packs/scifiessentialskit.html)
- [Animated Robot](https://quaternius.com/packs/animatedrobot.html)
- [RobotExpressive license](https://github.com/mrdoob/three.js/blob/dev/examples/models/gltf/RobotExpressive/README.md)

## Acceptance checklist

- Replace the primitive humanoid and pistol with licensed models.
- Blend idle and locomotion according to measured movement, including blocked movement.
- Attach the pistol to the hand and align firing with the authoritative aim direction.
- Show authored shooting and death motion without moving the physics capsule through animation.
- Keep pickup, ammunition, rotor damage, explosions, and reset functional.
- Add a survival clock that stops on death and resets for a new run.
- Preserve a clear route into cover and a route for the drone to investigate.
- Give every gameplay-relevant solid matching collision and sight occlusion.
- Keep decoration from promising cover that the sensor ignores.
- Keep render-only asset loading out of the minimal sensing and training guides.
- Verify native tests, browser execution, strict Clippy, personal lint, and asset licensing checks.
- Inspect desktop and narrow views together, then confirm any observed fixes once.
- Show screenshots and recordings of model, animation, and gameplay progress.
- Push tested model/animation and survival/environment checkpoints to main.

## Invariants

Physics owns position and collision. Animation only projects measured motion and combat state.
The navigator receives filtered sightings and known static geometry, never hidden player coordinates.
Survival is either running with elapsed simulation time or ended with its final elapsed time.
Reset clears the run, damage, weapon ownership, perception, and death state.
A failed asset load must produce a visible diagnostic instead of an invisible player.

## Status

The mannequin, pistol, and crates render in the browser. The user selected the
neutral armor and dark joints. Authored locomotion, upper-body aim, recoil, and
death clips project the existing game state. Asset loading pauses simulation.
The survival clock stops on robot death, drone destruction, or a flight error.
Reset starts a new run.

The current run has one drone; defeating it ends the run. All 86 scene tests and
20 navigation tests pass. Four asset contract tests pass. A headless integration
test loads the real mannequin, finds its hand and animation graph, and verifies
that a missing scene freezes simulation and displays the loading error.

The optimized browser build includes the final pistol roll and ground-placement
fixes. Both fixes have failing regressions followed by passing tests. Desktop and
narrow inspection confirms model loading, controls, shooting, and layout.
Later source edits added tests only; production code remains the rendered version.

The native release gates and standard browser suite pass. Branch coverage is
still running in `bevy-gym-survival-branch-coverage-20261009.service` with
`RUSTC_BOOTSTRAP=1`. Logs are under `/home/sagan/.cache/bevy-gym-quality-validation`.
The earlier coverage run lacked that setting and was stopped; do not use its output.
The [status document](EXAMPLE_STATUS.md) and [qualification record](progress/drone-survival-model.json)
track the checkpoint and remaining verification.

Remaining game work includes directional locomotion, current-frame hand attachment,
building detail, multi-wave survival, and a learned humanoid opponent.
The current navigator uses programmed search over learned motor control.
Downloaded references and original archives remain under `runs/quality-research/survival-assets`.

The source root-motion jog advances 5 metres in 28/30 seconds. Its authored speed
is about 5.357 m/s; the arena moves at 4 m/s. The next locomotion pass should derive
playback speed from measured displacement and that authored speed. The source is
`UAL1_Standard_RM.glb`, `Jog_Fwd_Loop`, root translation from zero to Z=5.
The shipped model uses the in-place clip; physics retains movement authority.
