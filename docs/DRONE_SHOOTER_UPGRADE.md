# Drone shooter upgrade

The pursuit example is the current priority. The user rejected its fixed camera,
weapon jitter, forward-only strafing, and weak hit feedback. Preserve the neutral
mannequin and dark joints. Build a hiding-and-shooting game with two drones and
two droids, including one player-controlled droid and one AI teammate.

## Delivery order and acceptance

1. Record the current strafing and aiming, extract a contact sheet, and inspect it.
2. Add an orbiting shoulder camera, camera-relative movement, right-button ADS,
   centre reticle, collision clearance, and browser pointer capture with Escape release.
3. Attach the pistol to the current animated hand and match locomotion to movement.
4. Replace hitscan with swept projectiles. Apply physical linear and angular impulses
   at the hit point. Smoke the specific rotor after its first hit; destroy it after two.
5. Add grounded muzzle flash, muzzle smoke, visible projectile tracers, and recoil.
6. Integrate sagan-software/bevy-ragdoll for physical death and animation handoff.
7. Add two independently simulated drones and two droids with team-aware targeting.
   The teammate must use observations rather than hidden enemy coordinates.
8. Add green idle, yellow suspicion, red pursuit, and orange searching lights.
   Play distinct transition sounds, gunfire, footsteps, and drone audio.
9. Integrate the compatible bevy-raytraced-audio adapter against the same cover geometry.
10. Validate each checkpoint in the browser, show video evidence, and push to main.

Each checkpoint requires focused failing regressions, passing native tests, formatting,
strict native and WASM Clippy, personal lint, applicable coverage, and browser inspection.
Run tutorial examples and documentation tests after the final prose edit.
Do not call the complete shooter finished while any numbered behavior remains absent.
The original full examples roadmap remains pending behind this work.

## Invariants

Camera state is presentation state. It cannot move the physics capsule through cover.
Aiming chooses a world point; projectiles originate at the weapon and collide along each step.
An impact applies damage and impulse once. A destroyed rotor cannot receive another hit.

Smoke follows its owning rotor. Reset removes projectiles, smoke, impulses, and ragdolls.
Actor identity and team determine ownership; rendering must not assume one global hand.
A droid is animated while alive and physically simulated after death.

A drone's light and transition sound derive from its perception state.
No extracted ARC Raiders audio or models are redistributed.

## Reference and compatibility evidence

The baseline recording and contact sheet are `progress/shooter-strafe-before.mp4`
and `progress/shooter-strafe-before.jpg`. The pistol reads stale `GlobalTransform`
before animation and transform propagation. All sideways movement uses a forward jog.

The official [gameplay reveal](https://www.youtube.com/watch?v=OpCooWm-PDs)
provides the shoulder framing and aiming reference. A 00:30–01:15 clip is stored
under `runs/quality-research/shooter`. The previously downloaded
[Building ARC Machines documentary](https://www.youtube.com/watch?v=DRlhpzc7ImA)
and its aerial and damage contact sheets remain under `runs/quality-research`.

The camera uses [bevy_panorbit_camera 0.34.0](https://github.com/Plonq/bevy_panorbit_camera)
with Bevy 0.18. Its orbit solver runs behind shooter input and the arena's collision
query. The alternative bevy_third_person_camera 0.4.0 includes unconditional Avian
and unified-input dependencies, which this scene does not need.

[bevy-ragdoll](https://github.com/sagan-software/bevy-ragdoll) currently pins
Bevy 0.19.1 and bevy_rapier3d 0.36.0. This project uses Bevy 0.18.1.
Resolve that version boundary before integrating its runtime; adding two Bevy
versions would not make their components interoperable.
The raytraced-audio repository has a Bevy 0.18 compatibility adapter to inspect.

## Status

The camera checkpoint adds mouse orbit, camera-relative WASD, right-button ADS,
a centre reticle, wall clearance, and an equipped pistol on start and reset.
Hip view uses a 3.2-metre boom and 60-degree vertical field of view; ADS uses
1.5 metres and 40 degrees. Escape releases pointer capture. The embedded preview
rejects pointer lock, so right-button drag remains available there.

Weapon projection now reads current animated local transforms after animation.
A regression proves stale global transforms cannot shift the held pistol.
Locomotion playback follows measured travel speed; pelvis turns and reverse playback
support directional travel. The [ADS recording](progress/shooter-ads-strafe.mp4) and
[contact sheet](progress/shooter-ads-strafe.jpg) still show abrupt direction changes.
Authored directional clips or improved blending remain required; animation quality
is not accepted as complete.

The footage includes synthetic keyboard and pointer
input delivered through the browser. Full pointer capture was unavailable there.

The checkpoint passes 97 scene tests and 20 navigation tests. Its selected camera,
locomotion, and weapon files have full native line and branch coverage, including
their test code. This is not a whole-change coverage percentage. Remaining coverage
boundaries are listed in [the record](progress/shooter-camera-coverage.json).
Physical impacts, player projectiles, rotor smoke, ragdolls, two-versus-two agents,
state lights, and raytraced audio remain separate unfinished checkpoints.

The optimized runtime is `a7e93bc80af7fdb86834af24d4a26ff28d853d172259d4841af8cc6262ce862b`.
Its [final recording](progress/shooter-camera-final.mp4) and
[contact sheet](progress/shooter-camera-final.jpg) show ADS, orbit, strafing, and shooting.
