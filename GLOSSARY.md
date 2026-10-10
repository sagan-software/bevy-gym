# Robot examples

These terms describe flight control and the shared physical world.

## Language

Flight goal: A requested world position and horizontal heading for the drone.
Avoid: destination pose.

Flight pilot: The learned controller that converts a flight observation and flight
goal into four motor commands.
Avoid: navigator, pursuit agent.

Navigator: The controller that selects flight goals from observations of the arena.
Avoid: flight pilot.

## Shared world

Robot world: The single physical solver containing all six robot bodies and
their constraints. `RobotWorld` owns it.
Avoid: match.

Match: A competitive episode that applies sensors, weapons, health, rewards and
termination rules to the robot world. This layer remains unfinished.
Avoid: robot world.

Robot snapshot: Read-only physical states captured at one common decision boundary.
`RobotSnapshot` contains all six states; actor observations require sensor isolation.

Robot frame: The opaque world/reset scope and boundary ordinal that binds a
complete actuator request to its source snapshot. `RobotFrame` is its API type.

Robot slot: One of three stable positions within a team, independent of physical
body handles. `RobotSlot::ALL` defines array order.

Robot identity: A robot's team and slot together. `RobotId` derives its team
from its `Drone` or `Droid` variant.
