# Playable drone pursuit

User expansion: 2026-10-09. This is part of the first robot milestone, before
Gymnasium ports. Keep the small flight guides available alongside this game.

## Requested scene

Build a gray blockout arena with colored landmarks, a house with windows,
a traversable pipe, corners, and cover. An animated humanoid robot can hide,
find and loot a pistol, aim, and shoot. Support both a trained humanoid policy
and a player controlling the same character through the same actions.

The drone must find, follow, lose, search for, and reacquire that character.
It must investigate windows and pipe openings using its available observations.
Hidden target positions must not enter the actor's observations. Last sightings
and audible events need finite lifetimes and explicit uncertainty.

Give the drone a visible weapon with an audible and visible spin-up before firing.
Give its body health and each rotor a separate weak point. Initially, two pistol
hits destroy one rotor; this is a local balance choice, not a claim about ARC.
A destroyed rotor pops with a flash, sparks, and smoke. The corresponding motor
loses force and reaction torque.

A fatal crash or destroyed body bursts into
simulated debris. Reset removes debris and effects and restores the full scene.

The humanoid must learn or let a player choose to seek cover, loot, aim, and fire.
The drone must learn flight, pursuit, search, and damage adaptation. Scripted
controllers may establish baselines, but must be labeled as baselines.

## Reference and evidence

The Wasp is the closest initial gameplay reference: the community
[Wasp entry](https://arcraiders.wiki/wiki/Wasp) describes projectile volleys and
thruster weak points. The [Hornet entry](https://arcraiders.wiki/wiki/Hornet)
describes a stun-round support role and rear-thruster weak points. These are
secondary descriptions, not evidence of Embark's implementation or exact timings.

A denser contact sheet now samples 02:34–02:47 of the official documentary at
four frames per second. It shows bright bursts, falling parts, and lingering smoke.
It does not establish wind-up time, hit counts, or which component was destroyed.
The hashes and interval are in `progress/drone-pursuit-reference.json`.

Use the existing official Building ARC Machines footage and its recorded sources
in [EXAMPLE_RESEARCH.md](EXAMPLE_RESEARCH.md) for visual investigation.

A search also located an apparent official development-video post at
<https://x.com/ARCRaidersGame/status/2090106639193985395>. The browser fetch failed;
its contents have not been verified. `yt-dlp` also reported no video in that post.
Retrieve an accessible official copy before using it as evidence. Record attack warning, firing, rotor loss, smoke, and crash sequences
with timestamps and contact sheets. Do not infer an RL architecture from footage.

Use licensed or original assets. The already inspected CC0 animated humanoid is
a candidate; a robotic mannequin is also acceptable. Do not copy Unreal mannequin
assets without checking their applicable license and permitted distribution.

## Delivery checkpoints

1. Finish validated local checkpoint playback and record the current learned flight.
2. Add rotor flashes, smoke, and collision-driven crash debris with bounded lifetimes.
3. Add the arena and animated humanoid with player movement, camera, and reset.
4. Add typed health, rotor hitboxes, pistol pickup, ammunition, aim, and firing.
5. Add occlusion-aware sensors, remembered sightings, sound events, and weapon spin-up.
6. Train pursuit and search against a labeled baseline target, then varied opponents.
7. Train the humanoid's cover, loot, and shooting behavior through the player action seam.
8. Add adversarial training and publish held-out evaluation plus browser play.

Publish each tested checkpoint to main with screenshots and a short recording.
Keep deterministic environment state separate from presentation particles.
Bound debris counts and simulate collisions; do not silently change the qualified
hover environment's mass, thrust, reward, or terminal rules for visual effects.

## Acceptance checks

- A rotor emits one destruction event only; later hits cannot repeat its explosion.
- Body death and fatal crash emit one death event; live control stops after death.
- Smoke follows the damaged rotor; detached debris follows its own physics.
- Reset clears effects, damage, ammunition changes, pickups, memory, and terminal state.
- Walls block movement, shots, and sight; window and pipe openings permit valid rays.
- A hidden target cannot affect actor input without an allowed sight or sound event.
- Spin-up completes before a shot; losing a valid firing opportunity cancels or
  interrupts firing according to the documented weapon state machine.
- Unarmed actors cannot fire; pickup requires proximity and an available pistol.
- Player and learned humanoid actions use the same validation and simulation path.
- Separate evaluation measures survival, pursuit, reacquisition, rotor damage,
  kills, and resource use on withheld layouts, seeds, and opponents.
- Browser playback, player input, training, and reset work at desktop and narrow sizes.

Current status: checkpoint loading and destruction effects are implemented. Rotor
failure emits a burst and smoke; task termination hides the body and spawns eight
colliding debris proxies. The arena, humanoid gameplay, weapons, and pursuit remain
pending. These effects do not demonstrate learned damage recovery.
