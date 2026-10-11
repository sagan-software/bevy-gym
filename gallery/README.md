# Browser examples index

The deployment copies this directory to `/bevy-gym/examples/`. The robot cards
open independent frozen-policy viewers. Drones versus droids opens the shared
six-agent physical scene. Combat rules, competitive training and qualification
remain unfinished. The standing viewer is an unqualified PPO trial. Existing `/robots/pursuit/` links
open the same six-policy scene; `/robots/hover/` opens the frozen PPO hover lesson.

The droid curriculum lists standing, recovery, walking and running, low cover,
jumping, aiming and firing, and evasion. Only standing currently has a scene.
The [curriculum guide](../docs/ROBOT_SKILL_CURRICULUM.md) records prerequisites,
training commands and promotion rules. Failed training never promotes a lesson.

Each card uses a fresh ten-second silent browser recording and a poster extracted
from its first frame. [The recording manifest](recordings.json) retains source
recording identities, source and output hashes, runtime builds, trim offsets,
policy evidence and mode. Every MP4 contains 240 frames at 24 fps, at 960 × 600.
All ten files passed complete decoding and duration checks.

Robot recordings came from local release builds. The world recording includes
the current Drones versus droids name. Classic Control recordings came from the public browser routes.
CartPole, both MountainCar tasks and Acrobot use bundled RL policies. Pendulum
shows PPO training from seed 42 because its bundled checkpoint is unavailable.

Before training Pendulum, select Train from scratch. The default inference
route currently reports the missing checkpoint; it does not substitute actions.
Recordings establish visible execution, not policy qualification.

Previews load and play only while visible. Hidden tabs and off-screen previews
pause. Reduced motion starts with posters. The preview button permits explicit
playback and pause. When autoplay fails, the button offers an explicit retry.

Eight focused tests cover every production line, branch and function in the two
preview modules. The full robot browser JavaScript suite passes 39 tests.
The two gallery contract tests check routes, playback attributes, unique media
and hashes. The [name and route record](../docs/progress/world-scene/name-and-gallery.json)
retains current source hashes, runtime identity and fresh native coverage.

The overlay title has three native hits. Window creation at `viewer.rs:26` has
zero native hits because these tests do not open a window. The release browser
created that window and rendered all six models. Complete native coverage is
not claimed.

Desktop and 390-pixel layouts were inspected through the collaborative browser.
The curriculum and Classic Control sections also passed narrow inspection.
Playback, pause and off-screen loading worked. Physical mobile devices remain
unverified. The design detector reports existing Mona Sans and robot-design
palette differences; this refinement preserves the gallery’s existing design.

The drone model is NateGazzard’s [CC BY 3.0 asset](../assets/robots/README.md).
The droid model is Quaternius’s [CC0 mannequin](../assets/robots/survival/README.md).
The gallery retains both credits and uses the repository’s
[Mona Sans font](../assets/fonts/README.md).
