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
colliding debris proxies in the hover viewer. The arena now has a player-controlled
robot, a lootable pistol, and a flying drone driven by the bundled healthy hover
policy. Rotor damage disables motor forces; flight termination triggers destruction.
Pursuit, humanoid training, and learned damage recovery remain pending.

## Explore the arena

Run the player-controlled scene:

```sh
nix develop --command cargo run --features robots --example drone-pursuit
```

Hold `W`, `A`, `S`, and `D` to move. Press `R` to reset. The on-screen movement
buttons also accept held input. The camera follows the character and moves closer
when a wall obstructs its view.

Press `E` near the pistol to collect it. Point at
the drone and click to fire; holding the button repeats at the weapon cooldown.
`Space` and the Fire button use the last captured aim. The drone uses its learned
hover controller while the player moves and shoots.

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
The geometric sight model below queries this layout from the rendered drone's
body-mounted camera. A pursuit actor remains unfinished.

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

The guide traces aimed shots through the arena. The player-controlled scene uses
the same health and pistol rules. Combat drives destruction effects and disables
destroyed motors before the next flight action. The library's qualified default
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
The visible scene now uses these shot helpers.

## Player combat checkpoint

In checkpoint `5164de6`, the licensed drone stayed at (0, 2, 1) metres. Pickup hides
the ground pistol and shows it in the robot's right hand. Aim rotates the body and
shoulder; movement remains independent.

A damaged rotor keeps a visible orange
marker. Its second hit hides that rotor. Six body hits hide the entire target.
Reset restores the pistol, ammunition, target, and all rotor meshes.

The camera ray chooses the nearest visible aim point. The actual shot starts
0.4 metres above the character centre, inside its collision capsule. A displayed
barrel penetrating a wall therefore cannot bypass shot occlusion. The camera
cannot directly supply a shot origin. `Action::Fire` accepts only a `Dir3`.

Pickup and fire enter one fixed simulation action queue. Pickup takes precedence
over a simultaneous shot and survives render frames until the simulation consumes
it. Clicking movement controls does not shoot into the world. Cooling-down attempts
retain the preceding feedback; other rejected actions display their reason.

Accepted shots show one 120-millisecond trace and impact pulse. New shots replace
that bounded trace. Combat now shares the hover viewer's explosion, smoke, and
debris renderer. Flight and motor failure still need to connect to combat.

Nineteen viewer tests cover input, pickup, aim, hand alignment, feedback, visibility,
and reset. The fourteen native shot tests include two arena tests; twelve external
shot cases pass in Chrome/WASM. [Coverage](progress/pursuit-visible-combat-coverage.json)
records source hashes and the uninstrumented native startup lines.

[The browser recording](progress/pursuit-visible-combat.mp4) demonstrates pickup,
body and rotor hits, destruction, and reset. Desktop and 390-pixel layouts were
inspected. [The evidence record](progress/pursuit-visible-combat.json) retains
artifact hashes and the remaining verification gaps.

## Combat destruction

The second rotor hit emits one flash, twelve sparks, and four seconds of smoke.
Flashes use a soft radial texture and face the camera; sparks remain small spheres.
The smoke follows the named rotor after its mesh disappears. Body death emits a
larger flash, twelve sparks, two seconds of smoke, and eight colliding fragments.
These fragments are box proxies, not fractures of the licensed drone mesh.

The fragments collide with the arena's floor, house, pipe, and cover. Their private
Rapier world is built from the same immutable blocks used by the arena. They inherit
the moving body's last authoritative linear velocity, then receive the existing
visual burst velocities. Wreckage expires after five seconds.

Combat emits typed `Reset`, `Rotor(DroneMotor)`, and `Destroyed` events. The private
queue retains at most one reset, four rotor events, and one body event. Presentation
consumes these events without changing health. Reset replaces pending events and
clears active particles, smoke emitters, and debris before any new destruction.

Particle assets are shared across both viewers and reused across resets. Presentation
time advances independently of the combat cooldown, with each frame capped at
100 milliseconds. Debris uses fixed 16,667-microsecond steps. These effects do not
change the qualified hover environment, rewards, motor forces, or observations.

Checkpoint `2f80f89` passed 26 pursuit-viewer tests and 50 hover-viewer tests. They cover
one-shot events, same-frame reset and damage, effect expiry, stable material counts,
and fragments bouncing off the house wall. [Coverage](progress/pursuit-destruction-coverage.json)
records the native startup gap and the defensive missing-fragment-body branch.

[The stationary combat-destruction recording](progress/pursuit-destruction.mp4) shows
rotor smoke, a body explosion, moving debris, and reset in the optimized browser
build. Desktop and narrow views pass. [The visual record](progress/pursuit-destruction.json)
retains artifact hashes and distinguishes this stationary target from live flight.

## Live flight and combat

The private `Flight` controller builds `DroneHover` from the arena's immutable
boxes, omitting the floor that the task already owns. It loads `recovery.mpk`
with the existing twelve-feature architecture. Reset uses seed 42, retains loaded
weights, clears recurrent memory, and repairs damage. Loading or inference failure
stops flight and displays a diagnostic; reset retries a failed load.

Every fixed action applies destroyed rotor states before requesting the next
validated motor command. Failed motors contribute neither force nor reaction
torque. Flight observations update shot geometry without repairing health.
The target validates and normalizes each pose; quaternion components can differ
from the solver by rounding. Position remains unchanged by this projection.

A collision or flight-region exit destroys the body once and stops subsequent
flight actions. Body damage also stops flight before another policy action.
The character can still move after drone death. Debris inherits the last measured
world velocity. Reset clears combat and presentation effects together with flight.

Thirty-four pursuit-viewer tests and all 50 hover-viewer tests pass. Fifteen native
shot tests include the two shared arena cases; thirteen shot cases pass in Chrome.
The tests cover a minute of healthy hovering, physical failure after rotor hits,
absorbing body death, deterministic reset, invalid pose rejection, and failed
checkpoint or inference handling. This does not qualify pursuit or damage recovery.

[Coverage](progress/pursuit-flight-coverage.json) records the tested branches and
source hashes. The guard for a non-finite solver pose remains unhit; direct pose
rejection and controller failure have separate tests. The existing defensive
missing-fragment-body branch also remains unhit. Native window execution and
mobile touch input remain unverified.

[The live-flight recording](progress/pursuit-flight.mp4) shows rotor destruction,
loss of control, and crash debris in the optimized browser build. Desktop and
390-pixel views were inspected. Narrow reset restores the drone and pistol.
[The visual record](progress/pursuit-flight.json) records the runtime, hashes,
recording excerpt, and verification limits.

## Sight and finite memory

Run the short sensor guide:

```sh
nix develop --command cargo run --features robots --example pursuit-sight
```

The guide sees a robot through the house window, loses it behind the wall, and
forgets its last position. `Sight` returns one of three `Contact` variants:
`Unknown`, `Visible(point)`, or `Remembered { point, age }`. A remembered point
is an earlier measurement, not the robot's current position. Hidden movement
cannot update that point or supply target velocity. Reset calls `forget`.

The camera uses a 90-degree full cone and an inclusive 20-metre range. Its direction
is a `Dir3`. Observer and character centres must be finite and within ±1,000 metres
on every axis. Invalid centres produce no sighting and age existing memory.

The sensor tests head, chest, and hip proxies in that order, at vertical offsets
of +0.6, +0.2, and −0.3 metres from the character centre. It returns the first
exposed point. These samples approximate body visibility; they are not rendered
pixels or a full silhouette. A coincident sample has no viewing direction.

Occlusion uses the arena's existing solid-ray query and immutable geometry.
[Rapier 0.36 documents bounded rays and solid origins](https://rapier.rs/docs/user_guides/rust/scene_queries/).
Each sight ray extends one millimetre beyond its sample to close floating-point
gaps at wall faces. That margin can hide a sample less than one millimetre in
front of a wall. It does not change the range or cone test on the sampled point.
Windows, the doorway, and pipe openings permit clear rays; solid surfaces block them.

Each successful sample replaces the measured point and resets its age to zero.
Each unsuccessful sample adds elapsed simulation time with saturating arithmetic.
At an age of three seconds or more, the observation becomes `Unknown` and retains
no position. Reacquisition starts a fresh sighting. Range, cone, body proxies,
occlusion margin, and expiry are local game choices, not claims about ARC Raiders.

Seven sensor cases pass natively and in Chrome/WASM. Native execution also runs
two shared arena cases. Tests cover range and cone boundaries, exposed body points,
wall-face rounding, invalid centres, hidden movement, exact expiry, saturation,
reacquisition, and reset. [Coverage](progress/pursuit-sight-coverage.json) records
checkpoint `7f5fd6a`: all 48 instrumented sensor lines, both outcomes of all ten
instrumented sensor branches, and all 17 instrumented guide lines.

The model and guide are implemented. The rendered integration is described below.
Sensor-only pursuit actor inputs and return fire remain pending. Hearing is described below.

## Body-mounted sight in the arena

The playable drone now samples sight after flight and character movement on each
20-millisecond action. Its camera centre is at body-local (0, 0, −0.22) metres,
facing local negative Z. The physical body orientation determines camera direction;
the sensor never turns toward a hidden character position. The existing healthy
hover policy receives the same twelve flight features as before.

`Game` owns one `Sight`. Its read-only `contact` accessor returns the latest
filtered observation without querying world state again. Body damage clears
memory immediately on death. Flight termination and controller failure clear
memory before the next observation. Reset also discards every prior sighting.

The HUD reports `Drone sight: visible`, `last seen`, `none`, or `offline`.
An original camera-lens mesh follows the same eye pose. Its red, amber, cyan,
and dark materials match those states; text supplies the same information without
requiring colour recognition. Reset and observation changes reuse existing assets.
The lens is too small to assess its colour in the wide browser captures.

Thirty-nine pursuit-viewer tests and all 50 hover-viewer tests pass. Nine native
sensor cases include the shared arena tests; seven sensor cases pass in Chrome.
The integration tests cover actual character movement into view, camera pose,
immediate body death, physical death, controller failure, and reset. Projection
tests check every label, material choice, visibility state, and stable asset counts.
[Coverage](progress/pursuit-perception-coverage.json) records the native-startup gap,
the existing invalid-solver-pose guard, and unhit test-only fallback paths.

[The browser recording](progress/pursuit-perception.mp4) shows the robot entering
view, walking into the house, and losing its remembered sighting. Desktop and
390-pixel views were inspected. Narrow reset and pistol pickup work with keyboard
input; actual mobile touch input remains unverified. [The visual record](progress/pursuit-perception.json)
retains source and media hashes and distinguishes observed behavior from test evidence.

The drone still hovers. It does not choose where to look, pursue the robot, or fire.
Hearing is connected below. Telegraphed return fire comes next, before pursuit training.

## Finite hearing events

The playable arena emits footsteps after each 0.8 metres of grounded travel and
one gunshot per accepted pistol round. Idle, blocked, and airborne movement stays
silent. Unarmed attempts, cooldown rejections, and an empty magazine cannot refresh
hearing. Reset, body death, physical death, and controller failure clear both senses.

The local hearing model retains only an event class, an eight-way world-relative
bearing, and elapsed simulation time. North is negative Z; east is positive X.
Coincident horizontal positions have an unresolved bearing. Cardinal sectors own
their 22.5-degree boundary ties. No emitter or listener coordinate is retained.

Footsteps reach eight metres and gunshots reach 24 metres, including the boundary.
An obstruction halves range once; windows and pipe openings use the shared geometry.

A new audible event replaces the previous cue. Invalid or inaudible events preserve
its age. Cues expire at two seconds; age addition saturates. Query positions must
be finite and within ±1,000 metres on every axis. These ranges and timing are game
choices, not measurements of ARC Raiders. The model does not simulate diffraction,
reflections, material acoustics, or masking.

Epic's [AI Perception documentation](https://dev.epicgames.com/documentation/unreal-engine/ai-perception-in-unreal-engine?lang=en-US)
and [Report Noise Event](https://dev.epicgames.com/documentation/unreal-engine/BlueprintAPI/AI/Perception/ReportNoiseEvent?lang=en-US)
describe explicit hearing events, range, and finite stimulus age. They informed this
design; they do not establish how Embark implements enemy hearing. Audible sound
playback remains pending.

Run the minimal guide:

```bash
nix develop --command cargo run --no-default-features --features robots --example pursuit-hearing
```

The guide contrasts an inaudible footstep with a gunshot through the same wall,
reads the coarse direction, then expires it. The HUD shows cues such as
`Drone sight: none · shot south`; sight and hearing expire independently.
HUD text updates only when its displayed state changes.

All 45 pursuit-viewer tests and 50 hover-viewer tests pass. Eight native hearing
cases include two shared arena tests; six hearing cases pass in Chrome/WASM.
Tests cover clear and obstructed range endpoints, every bearing, sector ties,
invalid input, silence, replacement, expiry, reset, accepted and rejected shots,
blocked movement, falling, landing, and offline UI. Root tests, strict native/WASM
Clippy, and the changed-line personal Rust gates pass. The full personal Rust lint
backlog remains; the Nix gate still reports existing source-root warnings at
`flake.nix:119` and `flake.nix:199`.

[Coverage](progress/pursuit-hearing-coverage.json) records hits on every changed
instrumented line and both outcomes of every added production branch. The runnable
guide is included. Native window startup and the existing invalid-solver-pose
branch remain outside this coverage.

[The browser recording](progress/pursuit-hearing.mp4) shows a gunshot cue, expiry,
footsteps, and expiry after stopping. Desktop and 390-pixel views were inspected;
hearing text and controls fit. Actual mobile touch input remains unverified.
[The visual record](progress/pursuit-hearing.json) retains source and media hashes.
The healthy hover policy receives no new inputs and does not pursue these cues.
