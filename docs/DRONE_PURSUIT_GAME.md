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
colliding debris proxies. The arena now has a player-controlled robot. Weapons, drone pursuit, and
humanoid training remain pending. These effects do not demonstrate learned damage recovery.

## Explore the arena

Run the player-controlled scene:

```sh
nix develop --command cargo run --features robots --example drone-pursuit
```

Hold `W`, `A`, `S`, and `D` to move. Press `R` to reset. The on-screen movement
buttons also accept held input. The camera follows the character and moves closer
when a wall obstructs its view. This checkpoint has no drone opponent or weapons.

Run the short headless movement guide:

```sh
nix develop --command cargo run --features robots --example pursuit-walk
```

Build and serve the browser scene:

```sh
nix develop --command scripts/build_pursuit_viewer.sh
python3 -m http.server 8000 --directory robot-web/pursuit-dist
```

Open `http://localhost:8000`. Published builds also link this scene from the
examples gallery. Use `--release` to produce `robot-web/pursuit-dist-release`.

## Arena movement contract

The example's private `Arena` helper owns a 26-metre blockout and capsule movement.
One closed `Movement` action advances 20 milliseconds. Walking speed is limited
to 4 metres per second; diagonal input does not increase that limit. Opposite
buttons cancel. The actor receives no raw solver handle.

The capsule is 1.8 metres tall and 0.3 metres in radius. Gravity is 9.81 metres
per second squared. It can step over obstacles up to 0.25 metres high, including
the pipe lip. The one-metre window sill blocks walking. Reset restores the capsule
centre, falling velocity, facing direction, and walking phase.

Rapier 0.36 supplies the
[kinematic character controller](https://rapier.rs/docs/user_guides/rust/character_controller_setup/).
The controller resolves translation against static colliders. It does not train
humanoid balance or physical leg motion. The robot's original geometric model
uses articulated shoulder and hip animation driven by measured displacement.

One list of 31 oriented boxes supplies both meshes and colliders. It includes
the floor, perimeter, windowed house, roof, four cover blocks, and twelve pipe
segments. House bounds are X=-8..-2 and Z=-6..0 metres. Its front window spans
X=-6..-4 and Y=1..2.5 metres; its east door spans Z=-4..-2 and Y=0..2.4 metres.
The pipe runs along Z=-2..4 metres, centred at X=5 and Y=1.7 metres.

The same static geometry answers nearest-obstruction queries for the camera.
Invalid query coordinates fail closed at zero distance; zero-length segments
have no obstruction. This is the geometry basis for later sight and shot queries.
No perception model or hidden-target observation has been added yet.

Tests cover all sixteen button combinations, fixed movement, walls, door and pipe
traversal, window and roof obstruction, gravity, reset, camera obstruction, walking
animation, held buttons, and a reset press released before the next frame.

The [browser recording](progress/pursuit-arena.mp4) shows pipe traversal, reset,
and house entry. [The evidence record](progress/pursuit-arena.json) identifies the
verified bundle, source hashes, and remaining verification gaps.

## Combat rules

The private combat helpers now model damage and pistol ownership. They are not
connected to the visible arena yet. Run their short headless guide:

```sh
nix develop --command cargo run --features robots --example pursuit-combat
```

A rotor advances from intact to damaged to destroyed. Its second hit returns
`Damage::RotorDestroyed` with the existing `DroneMotor` identity. Further hits
return `Damage::Ignored`. Rotor damage does not decrease body health.

Six body
hits or a fatal crash return `Damage::Destroyed` once. All later damage is ignored.
These hit counts are local balance choices, not measurements of ARC Raiders.

The pistol starts unowned at X=-1, Y=0.92, Z=8 metres. Pickup accepts a character
centre within 1.5 metres, including equality. Non-finite coordinates or distance
return `PickupError::InvalidPosition` before other checks. An owned pistol returns
`AlreadyOwned`; an unowned pistol beyond the radius returns `TooFar`.

Pickup grants twelve rounds. `fire` checks ownership, remaining ammunition, and
cooldown in that order. Rejections return `Unarmed`, `Empty`, or `CoolingDown`
without spending a round. Each accepted shot spends one round and starts a
250-millisecond cooldown.

`advance` consumes simulation time and saturates at zero.
There is no reload yet. Replacing each helper with `Default` restores its initial state.

The guide now traces aimed shots through the arena. Live motor failure, effects,
and player controls still need to connect to these rules. The library's qualified
hover environment is unchanged.
[Coverage evidence](progress/pursuit-combat-coverage.json) records the tested
health and pistol branches; it does not claim a playable combat scene.

## Aimed shots

`Aim::try_from` accepts an origin within ±1,000 metres on each axis and a finite,
nonzero direction. `Target::try_from` accepts the same position bounds and a finite,
nonzero quaternion. Bounds include both endpoints. Each constructor validates
position first. Scaling before normalization supports finite subnormal and large
magnitudes without overflowing or underflowing their squared lengths.

Invalid aim origins return `InvalidAim::Origin`. Invalid directions retain Bevy's
`InvalidDirectionError` inside `InvalidAim::Direction`. Invalid target positions
and rotations return `InvalidTarget::Position` and `InvalidTarget::Rotation`.
The stored direction is `Dir3`; the stored quaternion is normalized.

A shot travels at most 30 metres, including an impact at exactly that distance.
It spends one pistol round before querying geometry. Unarmed, empty, or cooling
pistols return their existing error without changing target health. Misses and
wall impacts spend a round. A miss returns the world point exactly 30 metres
along the aim direction.

The nearest live part receives damage. Static geometry wins equal-distance ties,
including a muzzle inside both a wall and the body. Body/rotor ties prefer the
body; rotor ties follow `DroneMotor::ALL`. Destroyed rotor hitboxes disappear, and
body death removes all target hitboxes. The same arena geometry blocks movement,
shots, and camera sight lines; windows and pipe openings remain open.

The central shot box has half extents X=0.17, Y=0.104, Z=0.18 metres. Each rotor
uses a sphere of radius 0.12 metres at its existing motor centre. Front-left is
(-0.2505, 0.0875, -0.2606) metres; front-right has positive X. Rear-right has
positive X and Z; rear-left has negative X and positive Z. These are shot proxies;
the qualified flight mass and inertia collider are unchanged.

Rays transform into body coordinates before querying the box and four spheres.
[Parry 0.31.1's RayCast](https://docs.rs/parry3d/0.31.1/parry3d/query/trait.RayCast.html),
re-exported by Rapier 0.36, supplies these intersections. Unit ray direction makes
the returned impact parameter a distance in metres. Solid queries report zero
when the muzzle starts inside a hitbox. One arena query resolves static occlusion.

Thirteen native cases include two existing arena checks. Eleven external shot
cases also pass in Chrome/WASM. They cover rotated weak points, nearer-part
shielding, wall/window/pipe geometry, range, malformed inputs, extreme finite
magnitudes, death, and rejected firing. [Coverage evidence](progress/pursuit-shots-coverage.json)
records every measured shot-helper line and both outcomes of thirteen conditions.
Visible aiming and combat integration remain pending.
