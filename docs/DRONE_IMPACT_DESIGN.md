# Drone impact boundary

An episode is Flying or Ended. Only Flying accepts an impulse. Reset restores the
seeded initial motion. Applying an impulse does not advance time or consume randomness.

`DroneImpulse` stores a validated body-local contact point in metres and world-space
momentum in newton-seconds. The lesson accepts contact components within ±1 metre
and momentum magnitude at most 10 N·s. Both vectors must be finite; zero is valid.
These are local flight-lesson limits, not a physics standard. Validation checks the
contact point before momentum. The world contact point is derived from the current
body transform when the impulse is applied.

Malformed input returns `InvalidDroneImpulse`. An ended episode returns
`DroneImpulseRejected`, preserving all solver state. The existing motor-failure
error and its diagnostic message remain unchanged.

Rapier 0.36 applies a world impulse at a world point and derives angular impulse
from `(point - centre_of_mass).cross(momentum)`. The existing private rigid body
remains authoritative. No raw solver accessor is added. Applying a hit takes
constant time and allocates no input-sized storage.

Controlling source: Rapier 0.36 `RigidBody::apply_impulse_at_point`,
<https://docs.rs/rapier3d/0.36.0/rapier3d/dynamics/struct.RigidBody.html#method.apply_impulse_at_point>.
The installed source was inspected because the web documentation fetch failed.

Acceptance covers finite bounds on each axis, both signs, zero, diagonal magnitude,
nonfinite values, validation order, centre and offset hits, rotated bodies, no time
advance, reset, terminal rejection, error messages, and inaccessible raw fields.
