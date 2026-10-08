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
- 15:56–18:16: training discussion and examples; contact sheet generated, detailed
  behavioral notes pending.

Local files are `runs/quality-research/arc-documentary.mp4`,
`arc-documentary.en.vtt`, and `arc-media-manifest.json`. The manifest records the
video hash and intervals. Dense jump and damaged-flight sequences remain required;
these overview sheets alone do not establish their timing.

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
durations, and sheet paths. Visual inspection and scene-level contracts are pending.

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
