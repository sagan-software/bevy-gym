# Body-mounted collision images

Inspect the camera without advancing any agent:

```sh
nix develop --command cargo run --no-default-features --features robots --example robot-battle
```

The guide prints the initial drone health and `17 x 13 RGB camera image`.
It supplies no autonomous actions and proves no learned combat.

`RobotBattle::image(owner)` and a running `RobotMatch::image(owner)` return a
`RobotImage`. The image retains its opaque physical frame and camera owner.
Its borrowed pixel buffer contains 13 rows, 17 columns and three RGB bytes per
pixel. Rows run top to bottom; columns run left to right. The buffer uses 663 bytes.

The local profile uses a 90-degree horizontal field of view, square pixels and
an inclusive 20 m distance along each normalized ray. The camera faces the owner's
local negative Z axis. Its origin is body-local `(0, 0, -0.317)` m for drones and
head-local `(0, 0, -0.1)` m for droids. Physical body rotation determines the view.
Spectator controls and enemy positions cannot select its direction.

Pixel centres sample the local plane at `z = -1` m. For zero-based column `c` and
row `r`, `x = (2c - 16) / 17` m and `y = (12 - 2r) / 17` m. The plane spans
2 m horizontally and `26/17` m vertically. Body rotation transforms the direction
before normalization. The nearest solid blocks every farther solid; all owner
colliders are excluded. Reset poses and equal-distance ordering follow
[the collision-query contract](ROBOT_RAY_QUERY.md).

Flat RGB materials encode empty space as `[16, 24, 40]`, environment solids as
`[96, 104, 112]`, drones as `[32, 176, 208]`, and droids as `[224, 160, 48]`.
Pixels contain no slot, segment identity, hidden position or depth value.
This renders collision proxies, not the spectator meshes, lighting or weapon effects.
It does not implement hearing or a trained perception policy.

A running capture preserves physics, health, weapon state and the common frame.
Finished matches return `Finished(result)`; failed matches return `Failed` with
the original cause. Reset restores image capability. Invalid physical state fails
before image output. Capture never chooses movement, aim, trigger or alert state.

The projection is a local sensor profile. Bevy 0.18.1 defines its perspective
`fov` vertically and its aspect ratio as width divided by height in the
[official projection documentation](https://docs.rs/bevy/0.18.1/bevy/camera/struct.PerspectiveProjection.html).
An equivalent frustum would use aspect ratio `17/13` and vertical FOV
`2 * atan(13/17)` radians. This sensor limits Euclidean ray distance; Bevy's far
plane limits view-axis distance. Those limits are distinct.

Native and browser checks cover literal pixels, all six owners, body rotation,
20 m range, nonfinite-state rejection, terminal rejection and reset. Moving an
enemy behind solid cover changes no image byte. These checks do not establish a
trained fighting policy or full mesh-image parity. Standing, walking, rollout
collection and competitive fighting remain unfinished.

Fresh coverage hits 94 of 99 handwritten image regions. All four pixel materials, both camera mounts and
all three match states execute. Five defensive error regions remain uncovered:
`image.rs:71:95-96`, `72:25-87`, `73:21-22`, `73:23-24` and `76:49-50`.
They handle ray construction or collision-query failure after the immutable world
snapshot has passed validation. Nonfinite source-state rejection is covered.
[The validation record](progress/robot-image-validation.json) retains the exact
executables, command scopes and uncovered coordinates.
