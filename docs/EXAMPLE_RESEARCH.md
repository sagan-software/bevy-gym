# Example reference research

Inspected 2026-10-08. Research supports the [roadmap](EXAMPLE_ROADMAP.md);
it does not establish that any new port works.

## ARC Raiders and Embark

[Transforming animation with machine learning](https://medium.com/embarkstudios/transforming-animation-with-machine-learning-27ac694590c),
Tom Solberg, Embark, 2021-02-26, describes reward-trained physical locomotion,
recovery from impacts, traversal, and continued movement after limb loss. Designers
provide goals and movement constraints. The article does not publish a complete
training recipe or reusable robot models.

[Learning to Move](https://schedule.gdconf.com/session/learning-to-move-physics-based-enemy-locomotion-in-arc-raiders/917319)
is Martin Singh-Blom's GDC 2026 talk. The agenda verifies the session; the full
recording and slides still need inspection.

In a [direct interview](https://80.lv/articles/inside-the-magic-of-machine-learning-that-powers-enemy-ai-in-arc-raiders),
Singh-Blom describes physical simulation, utility AI, randomized training setups,
and adversarial motion priors. He says the game does not train online and avoids
server-rendered camera observations. Earlier height-grid perception failed indoors.
Our camera-vision and hearing lessons are requested project features; these sources
do not establish their exact implementation in ARC Raiders.

The official [Building ARC Machines documentary](https://www.youtube.com/watch?v=DRlhpzc7ImA)
is published by the ARC Raiders channel, dated 2025-12-05, and lasts 22:02.
Downloaded the public 720p HLS video and English captions with `yt-dlp` 2026.08.19.
The initial direct media format returned HTTP 403; the public HLS format succeeded.
The [earlier Embark Vimeo demonstration](https://vimeo.com/345027169) requires login
through the current downloader and remains unavailable locally.

Inspected 20-frame documentary sheets for these intervals:

- 02:20–03:20: aerial gameplay shows small drones above the player, red targeting
  beams, firing, and cover beside a large wall. Review tighter shots before fixing
  the model silhouette or measuring movement speeds.
- 05:48–06:28: flight, impacts, firing, and falling debris appear between interviews.
  The frames support physical reactions but do not prove the control algorithm.
- 12:23–14:03: editor footage shows articulated multi-legged prototypes and
  multiple robots in a training scene. Leg poses change with body position.
- 15:56–18:16: mixed interviews, gameplay, and an editor terrain view. These
  frames do not expose a reproducible training configuration or measured results.

Local files are `runs/quality-research/arc-documentary.mp4`,
`arc-documentary.en.vtt`, and `arc-media-manifest.json`. The manifest records the
video hash and intervals. These overview sheets alone do not establish movement timing.

## Drone and locomotion methods

[Flightmare](https://uzh-rpg.github.io/flightmare/) separates dynamics and rendering
and supports batched quadrotor learning. Its
[dynamics implementation](https://github.com/uzh-rpg/flightmare/blob/master/flightlib/src/dynamics/quadrotor_dynamics.cpp)
is a reference for forces, moments, inertia, and motor response. Pin a revision
before adopting equations or tests. This is research input, not a WASM dependency.

[gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) provides
single-agent and multi-agent examples. Its `BaseAviary` distinguishes physics and
control rates and validates their ratio. Compare hover and motor-force oracles
against a pinned revision before implementing the robot environment.

[QuadSwarm](https://github.com/Zhehui-Huang/quad-swarm-rl) is a reference for batched
multi-drone tasks. Inspect collision handling, observation ownership, and evaluation
before adopting any algorithm or metric.

[AMP](https://arxiv.org/abs/2104.02180v2) uses motion clips to train a discriminator
that supplies a style reward alongside task rewards. This is distinct from an
adversarial target agent. A jumping-robot lesson can explain both separately.
Use licensed authored motion clips if implementing AMP; do not infer joint targets
from a gameplay camera without recording that approximation.

[Fault-tolerant quadrotor control](https://arxiv.org/abs/2503.02649v1) combines
reinforcement learning, behavior cloning, and fault supervision. It reports
single-rotor failure experiments. Its abstract is inspected; equations and
experimental limits still need review. It does not prove arbitrary damage is
recoverable for our robot geometry.

The provisional approach is a shared native/WASM rigid-body simulation, bounded
per-thruster actions, separate physics and policy ticks, and a curriculum from hover
to pursuit and damage. Select the physics library only after checking current Bevy
compatibility and browser support. Keep a simple baseline controller for diagnosis;
learned-policy claims require trained weights and held-out results.

## Additional locomotion reference

[ARC-RL](https://github.com/CarloRomeo427/ARC_RL/tree/abca8c8377f25379b87c9705614911d6824af9c9)
is an independent project with a twelve-joint quadruped model named Leaper.
At this pinned revision, its MJCF uses primitive geometry and its environment
rewards forward-speed tracking, health, and gait compliance. It is a locomotion
reference; it does not implement our requested jump attacks, limb destruction,
perception, or pursuit experience.

Its README calls the license MIT, but the actual LICENSE adds an author-
acknowledgment condition and contains a placeholder surname. Preserve the actual
license text and resolve attribution before copying code or models. No code or
models from this project have been added to distributable assets.

[Embark's Unreal Engine interview](https://www.unrealengine.com/developer-interviews/embark-studios-build-the-award-winning-arc-raiders-with-unreal-engine)
describes physics, learned locomotion, and procedural animation together. It also
describes player-facing acoustic simulation. It does not specify an enemy hearing
sensor or publish a reproducible drone controller.

[The motor-damage contract](DRONE_DAMAGE.md) separates force failure, physical
thruster separation, and learned recovery. It records the current model's static
lift and yaw constraints before changing the public API.

## Model candidates

[Alexandre.ltrgn's Drone](https://sketchfab.com/3d-models/drone-688425e1cddb4a288a12b00246a76d43)
is listed as CC Attribution, 37.2k triangles. Download access, separate thruster
nodes, texture cost, and appearance still need inspection.

An [Innerscene quadcopter](https://www.innerscene.com/tools/library/3d-parts/quadcopter-with-flight-electronics-a25968c1)
search result advertises CC0 and four rotors, but direct retrieval returned 403.
It is not an approved asset. Neither candidate is selected yet. Do not substitute
unverified license claims for a downloaded asset and its notice.

## Gymnasium

Existing source pin: `7a1191388aa4aa973d3a5e4b039899cd99cc991f`.
[Official catalogs](https://gymnasium.farama.org/environments/classic_control/)
link the Classic Control, Toy Text, Box2D, and MuJoCo families.
`docs/visual-comparisons/references/manifest.json` contains 23 official GIF URLs,
SHA-256 hashes, durations, and six-frame contact sheets. Preserve this provenance
and existing fixed-state renderer fixtures. Inspect denser samples when matching
motion. Existing approximation folders must not be reported as conformance proof.

## Unity ML-Agents

Source revision: `7a03145ae48ad354821bd89e0243d99332149ace`.
The [official inventory](https://github.com/Unity-Technologies/ml-agents/blob/7a03145ae48ad354821bd89e0243d99332149ace/docs/Learning-Environment-Examples.md)
contains 17 families. Downloaded all 17 linked images. Scenes and scripts are under
`Project/Assets/ML-Agents/Examples`, and training configuration is under `config`.
The root `LICENSE.md` is Apache-2.0. Check asset-specific notices during import.
The documentation still lists Strikers versus Goalie; the current example tree
has no directory with that name. Resolve it through history before implementing.

Inspected all 17 official images. They share a blue-gray background, dark tiled
floors, soft shadows, and bright agent colors. Preserve each scene's source camera
and scale rather than applying one generic scene layout.

- Basic: a long platform, a blue cube agent with a yellow headband, and two green
  spheres of different sizes.
- 3D Balance Ball: repeated blue headband cubes, tilted top surfaces, and small
  white-gray balls.
- GridWorld: a square walled arena, hollow blue agent, red and green markers, and
  a separate top-down agent-view inset.
- Push Block: a walled rectangle, one white block, and a green goal strip.
- Wall Jump: a translucent blue wall, a white block, an airborne cube agent,
  and a green goal patch beyond the wall.
- Crawler: a blue cylindrical body, four jointed blue-white legs, a blue ground
  ring, a green direction arrow, and a green target cube.
- Worm: a segmented blue body with white joints, a headband, eyes, and a green
  target cube.
- Food Collector: multiple small agents, scattered red and green spheres, a
  purple beam, and repeated bounded arenas.
- Hallway: a long walled corridor, an early symbol panel, and two marked goal pads.
- Soccer Twos: two blue and two purple cube agents, a soccer ball, white pitch
  lines, nets, colored goal areas, and rounded end walls.
- Strikers versus Goalie: the same pitch style with two blue attackers and one
  purple defender. This distinct team arrangement needs its own behavior source.
- Walker: a jointed humanoid with white upper body, blue hips and thighs, a
  headband, a blue ground ring, and a green direction arrow and target.
- Pyramids: separated cross-shaped walls, stacks of cubes, a green button,
  and a highlighted stack supporting a green cube.
- Match 3: a yellow board with a dark frame and colored face tiles, stars,
  and triangles.
- Sorter: a circular arena with numbered white and green blocks along its edge.
- Cooperative Push Block: three colored agents, differently sized numbered
  blocks, and a green goal strip.
- Dungeon Escape: blue sword-carrying agents, a green key-carrying enemy,
  columns, a rear door, and a purple portal.

## Godot RL Agents

Source revision: `d65963648439167f4902043376321c15d3df0e3a`.
The [README](https://github.com/edbeeching/Godot_RL_agents_examples/blob/d65963648439167f4902043376321c15d3df0e3a/README.md)
contains exactly eleven video examples requested by the user. All eleven videos
downloaded successfully and have 20-frame chronological contact sheets. The root
license is MIT; per-example asset notices also exist. The codebase contains other
examples, which are outside this request.

Paths under `examples/`: `3DCarParking`, `ItemSortingCart`, `HovercraftRacing`,
`3DLander`, `MultiLevelRobot`, `RobotVolleyball`, `DownFall`, `MultiAgentSimple`,
`CrossTheRoad`, `ScoreTheGoal`, and `RobotFPS`. CrossTheRoad contains Starter and
Completed versions; use Completed as the behavior reference.

`runs/quality-research/media-manifest.json` records URLs, hashes, byte sizes,
durations, and sheet paths. All eleven sheets are inspected; scene-level contracts
remain pending. Visible reference details are:

- 3D Car Parking: white bay markings, parked colored cars, orange perimeter posts,
  a green goal arrow, and a central grass island with trees.
- Item Sorting Cart: a yellow cart on a horizontal strip, cyan and orange end
  containers, grass, trees, and repeated arenas behind the active cart.
- Hovercraft Racing: a blue-purple craft with side fan pods, dashed white road
  stripes, black-yellow barriers, yellow gates, and a chase camera.
- 3D Lander: gray rocky terrain, a small upright craft, a translucent green region,
  and a black sky.
- MultiLevel Robot: separated gray tiled platforms, an orange robot, green goal
  volumes, yellow objects, and later red agents.
- Robot Volleyball: two orange wheel robots, a yellow ball, a yellow court with a
  center net, grass, trees, and scores in the top corners.
- DownFall: colored rounded humanoids, platforms with gaps, spiked rollers,
  striped obstacles, and yellow goal arches.
- MultiAgent Simple: an orange robot, green pads, a yellow slatted plank over a
  brown void, and repeated arenas. Read the scene before inferring cooperation.
- Cross The Road: orange tiles, a gray traffic lane with white cars, green trees,
  a small green robot, and an elevated camera.
- Score The Goal: a gray fenced arena, three colored goals and matching balls,
  a green robot with a white selection ring, and surrounding grass and trees.
- Robot FPS: first-person and overhead views, colored wall blocks, wooden crates,
  a central hut with a pink roof, colored robots, and purple projectiles.

## AI Warehouse

The [creator's public README](https://github.com/AIWarehouse/AIWarehouse) says the
code is private. GitHub's public repository inventory returned only that profile
repository. Source-faithful ports cannot yet be specified. Verify future releases
and retain a separate list of video-inspired reproductions with explicit limits.

## Gallery direction

The [Bevy gallery](https://bevy.org/examples/) is the requested visual reference.
Use categorized thumbnail cards, direct example links, and source access. Keep
training/inference and necessary controls on the example page. Reuse existing
browser worker functionality. Preserve exact task names and qualification status.
Inspect desktop and mobile rendering before publishing. The T3 preview host was
unavailable during this initial research pass, so no live gallery review is claimed.

## Additional inspected gameplay

Downloaded [I am Leaper](https://www.youtube.com/watch?v=wrBP07lVO5c) and inspected
a five-second overview plus 24-frame sheets for 00:50–01:05 and 01:20–01:35. The
robot spreads its long articulated legs during airborne approaches, closes distance
across height changes, and lands near players and cover. Shots cut between scenes,
so these sheets cannot establish a single continuous jump trajectory. A continuous
clip is required before measuring launch speed, recovery time, or contact order.

A closer inspection sampled 00:48–01:00 and 01:20–01:32 at four frames per second.
The continuous 00:55–00:57.6 rooftop approach shows leg spread and overhead motion.
Its twenty-frame contact sheet samples eight frames per second. Neither takeoff
nor landing is fully visible, so this excerpt cannot establish jump distance or
full airborne duration. The clip and contact-sheet hashes are in the media inventory.

A 24-frame documentary sheet for 02:30–02:45 shows a four-lobed aerial silhouette,
bright thruster or attack effects, downward pursuit near a wall, impact, and debris.
Use those visible events to define comparison shots. Do not infer hearing, online
learning, or a particular controller from them.

[The committed media inventory](reference-inventory.json) retains source URLs,
content hashes, durations, and source revisions independently of local downloads.
Raw gameplay videos remain local reference material.

[NateGazzard's four-rotor drone](https://poly.pizza/m/DNbUoMtG3H) is now the
viewer model. Its [asset record](../assets/robots/README.md) retains the license,
geometry inspection, and actuator alignment. [Silly Fear's drone](https://poly.pizza/m/3Ae_y67lzvd)
remains an uninspected alternative.

## Animated target asset

[Quaternius's Adventurer](https://poly.pizza/m/5EGWBMpuXq) is a CC0 character from
the [Ultimate Modular Men pack](https://quaternius.com/packs/ultimatemodularcharacters.html).
The pack's Google Drive individual-file download returned a quota-exceeded HTML
page. The creator's Poly Pizza listing supplied a valid 1,944,116-byte GLB instead.
It contains five meshes, five skins, and 24 named animation clips, including walk,
run, hit reactions, and death. The media inventory records its hash and all clip names.

The [inspection screenshot](progress/adventurer-target-inspection.png) and
[walking clip](progress/adventurer-target-inspection.mp4) show the actual downloaded
model in a browser asset viewer. Bevy animation playback, root motion, collision,
target behavior, and perception integration remain unverified.
