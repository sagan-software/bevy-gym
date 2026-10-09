# Drone examples

These terms describe flight control in the drone examples.

## Language

Flight goal: A requested world position and horizontal heading for the drone.
Avoid: destination pose.

Flight pilot: The learned controller that converts a flight observation and flight
goal into four motor commands.
Avoid: navigator, pursuit agent.

Navigator: The controller that selects flight goals from observations of the arena.
Avoid: flight pilot.
