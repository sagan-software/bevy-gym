# Shared weapon timing

`RobotWeapon` applies the drone and droid timing rules to the current reinforcement-learning actor's `CombatSignal`. Every requested alert category supplies a trigger request; every released alert category leaves the trigger released. Sensors and target state are absent from this API. Policy inference still has to choose aim and movement.

Construct one weapon per living agent with `RobotWeapon::from(RobotTeam::Drones)` or `RobotWeapon::from(RobotTeam::Droids)`. Call `advance` once per 20 ms environment action, after collecting every living agent's actions from the same snapshot. `phase` reads timing; `disable` permanently blocks that weapon. Episode reset constructs fresh weapons. No mutable phase or raw timing backend is exposed.

A drone's first requested step starts an 800 ms charge. Forty further requested steps complete that charge and discharge the first round. Two later rounds each require a 120 ms delay and a current trigger request.

Releasing during charge discards progress. Releasing during a volley cancels its later rounds and starts 1,200 ms recovery. Completed volleys pay the same recovery. Finishing recovery cannot discharge on that step; another decision boundary is required.

A ready droid discharges one requested pistol round immediately. Its magazine starts with twelve rounds and never reloads automatically. Each discharge starts a 250 ms cooldown. With 20 ms action boundaries, the remaining time is 10 ms after twelve cooling steps; the thirteenth cooling step finishes recovery at 260 ms. That step cannot discharge. The final round still pays cooldown before the empty state becomes visible.

`RobotWeaponOutcome::Fired` is a mechanical discharge event. The [shared combat mechanics](ROBOT_BATTLE.md) now connect muzzle poses, body collisions, health and actuator shutdown. Rotor impacts, fatal-contact rules, match termination, competitive rollouts and trained combat policies remain unfinished. This module does not establish working fighting or qualify an agent. The runnable `robot-weapon-timing` example prints initial phases without supplying autonomous trigger requests.

Nine public tests pass on native and WASM. Two private tests verify positive time remainders and descending magazine counts. Native format, strict Clippy, full tests and personal Rust lint pass. Measured production coverage hits all 151 regions and both outcomes of all six conditional branches. Nix lint retains ten findings matching the preserved baseline. The [validation record](progress/robot-weapon-validation.json) identifies the source and commands.
