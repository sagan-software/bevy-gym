# Articulated droid standing

Status: physical environment implemented; RL training, scene and qualification pending.
No standing checkpoint exists. This reset-only command prints physical segment centres:

```sh
nix develop --command cargo run --no-default-features --features robots \
  --example droid-standing-contract
```

The command never steps an autonomous agent. Deterministic torque fixtures exist only
in tests. A future trainer and frozen-policy scene must use the same `DroidStanding`
implementation. No standing competence or rendered mannequin alignment is claimed.

## Physical model

`DroidStanding` implements `Env` with `DroidAction`, `DroidObservation` and unit info.
The private Rapier 0.36 world contains thirteen dynamic segments, twelve joints and
a fixed floor. Total mass is 70 kg.

Rapier derives each segment's inertia from its
box collider and mass. Every action lasts 20 ms, with four 5 ms physics substeps.
Gravity is 9.81 m/s² downward. Bodies cannot sleep and use continuous collision detection.

The [joint guide](https://rapier.rs/docs/user_guides/rust/joints/) controls Rapier's
joint semantics.
The [force guide](https://rapier.rs/docs/user_guides/rust/rigid_body_forces_and_impulses/)
controls persistent torque application. These APIs provide mechanics; the local
robot dimensions, masses, torque bounds and limits are application choices.
This simplified robot is not a validated human biomechanics model.

Each collider has friction 0.8. Adjacent linked segments cannot collide with each
other; nonadjacent self-contact remains enabled. The floor's upper face is y=0,
with half-width 10 m. Contacts and joint constraints never choose policy actions.
There are no posture motors, damping controllers, animation clips or root movement.

The [collision profile](../src/robots/droid/geometry.rs) records every box centre,
half extent, mass, anchor and angular limit. Coordinates are metres in a common
bind frame: +X right, +Y up, -Z forward. All segment axes align in the bind pose.
A unit quaternion rotates each segment's local axes into world coordinates.

Masses are pelvis 12 kg, torso 24 kg, head 5 kg, each upper arm 2 kg, each forearm
including hand 1.5 kg, each thigh 7 kg, each calf 3.5 kg and each foot 0.5 kg.

## Action contract

`DroidAction::try_from([f32; 26])` rejects the entire action when any fraction is
non-finite or outside inclusive [-1, 1]. It returns `InvalidDroidAction` before
physics can receive the action. Valid fractions, including signed zero, are preserved.
`DroidActuator::ALL` fixes this order:

1. Spine X, Y, Z.
2. Neck X, Y, Z.
3. Left shoulder X, Y, Z; left elbow Y.
4. Right shoulder X, Y, Z; right elbow Y.
5. Left hip X, Y, Z; left knee X; left ankle X, Z.
6. Right hip X, Y, Z; right knee X; right ankle X, Z.

At each substep, the environment clears previous user torques. For each actuator,
torque equals its dimensionless fraction times its Nm bound times its current
parent-local unit axis rotated into world coordinates. The child receives that
world torque and the parent receives its opposite. Zero requests no actuator torque;
passive constraints and contacts still act. Axis directions follow the right-hand rule.

Maximum absolute torque is 80 Nm per spine axis, 8 Nm per neck axis, 30 Nm per
shoulder axis, 20 Nm per elbow, 120 Nm per hip axis, 100 Nm per knee and 60 Nm per
ankle axis. Limits below use Rapier angular solver coordinates in radians, not
an independently defined Euler-angle decomposition. Translation is locked at every joint.

- Spine X/Y/Z: [-0.5, 0.5]. Neck X/Y/Z: [-0.4, 0.4].
- Shoulders X/Y/Z: [-1.8, 1.8].
- Left elbow Y: [-2.4, 0.1]. Right elbow Y: [-0.1, 2.4]. Other elbow axes are locked.
- Hips X: [-1.5, 1.5], Y: [-0.7, 0.7], Z: [-0.6, 0.6].
- Knees X: [-2.4, 0.12]. Other knee axes are locked.
- Ankles X: [-0.65, 0.65], Z: [-0.4, 0.4]. Ankle Y is locked.

## Observation and lifecycle

`DroidObservation::body` returns a read-only `DroidBodyState`. Each state contains
world position in metres, a segment-to-world unit quaternion, world linear velocity
in m/s, world angular velocity in rad/s and an active-floor-contact flag.
Joint-relative rotations and velocities can be derived from linked segment states.
The policy tensor encoding remains unfinished; this is the physical observation API.

`DroidBody::ALL` orders pelvis, torso, head, left upper arm, left forearm, right upper
arm, right forearm, left thigh, left calf, left foot, right thigh, right calf, right foot.
Callers cannot access the physics world or replace its body transforms through this API.

Reset samples one common yaw in ±π radians and pitch/roll in ±0.01 radians, then
lifts the bind pose by 0.03 m. All velocities start at zero. The common transform
preserves coincident joint anchors. `reset(Some(seed))` restarts the reset stream;
`reset(None)` continues it. Reset rebuilds all bodies, contacts and solver state.

A substep terminates when the pelvis falls below 0.5 m, a non-foot segment contacts
the floor, or any segment leaves x/z inclusive [-5, 5] m or y inclusive [0, 3] m.
Non-finite position, orientation or velocity also terminates. The first terminal
substep freezes the observation; later actions return that observation and zero reward.
The environment has no implicit horizon. A trainer must impose its declared time limit.

Continuing reward is `upright * height / (1 + drift + speed)`:

- `upright` is `(1 + torso_up.y) / 2`, clamped to [0, 1].
- `height` is pelvis height in metres divided by 0.99 m, clamped to [0, 1].
- `drift` is squared horizontal distance from the origin in m² divided by 1 m².
- `speed` is squared pelvis speed in m²/s² divided by 1 m²/s².

Termination earns zero. Reward shaping is not a qualification criterion. Standing
promotion thresholds, held-out seeds and perturbation cases must be frozen before
training. Checkpoint identity, training provenance and independent evaluation remain absent.

## Mannequin provenance

Bind anchors come from the neutral-armour mannequin with dark joints in
`assets/robots/survival/mannequin.glb`. Quaternius released Universal Animation Library
Standard v3.0 under CC0. The [manifest](../assets/robots/survival/manifest.json) preserves
source URLs, licence, original archive hashes and recolouring details.
The current model SHA-256 is
`a0b15a89d3b21bf20ac48f8ac32fa16f20a9f32b170dfacd01a5375d7190d97f`.

The profile rotates the model's +Z-forward bind coordinates by π about Y. Spine uses
`spine_01`; neck uses `neck_01`; shoulders use `upperarm_l/r`; elbows use `lowerarm_l/r`;
hips use `thigh_l/r`; knees use `calf_l/r`; ankles use `foot_l/r`. Collision boxes
approximate the armoured segments. Bone-to-physics rendering and its visual validation
remain unfinished. No animation clip is loaded or played by this environment.

## Validation and remaining work

The [evidence record](progress/droid-standing-physics.json) preserves source hashes,
red-test output, native/browser results, coverage, lint gaps and the concurrent travel
run snapshot. Four public droid tests and seven internal physics tests pass. The
browser suite passes 180 tests, including all four public droid contract tests.

Native coverage hits all 439 instrumented droid-module lines, including tests, and
all 30 production branch outcomes. Constant profile data and enum declarations have
no instrumented line counts. Failing assertion branches in tests remain unexecuted.
Strict repository Clippy passes; strict personal Clippy retains unchanged repository
findings. Changed-line personal Rust lint is clean. Two unchanged Nix source-filter
findings remain at `flake.nix:119` and `flake.nix:200`.

Actor encoding, training and inference commands, frozen qualification thresholds,
checkpoint evidence, physical mannequin rendering and browser recordings remain
unfinished. The reset-only guide is not the independently runnable trained standing
lesson required by the curriculum. No existing pursuit controller was extended.
