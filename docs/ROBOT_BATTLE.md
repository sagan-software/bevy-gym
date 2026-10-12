# Shared combat mechanics

Inspect the initial combat state:

```sh
nix develop --command cargo run --no-default-features --features robots --example robot-battle
```

`RobotBattle` owns one shared `RobotWorld`, six health states, six weapons and bounded projectile storage. `snapshot` returns physical measurements, health and weapon phases. `advance` accepts a complete `RobotActions` frame plus three drone and three droid `CombatSignal` values. Policies must supply those requests; the environment cannot select movement, aim, alert state or triggers. The example prints initial health without supplying autonomous actions.

Source-frame, clock and physical-state validation precede weapon updates. Foreign or stale actions stop the world with the original `RobotWorldFailure`; later requests remain stopped until reset. Rejection consumes no ammunition, starts no charge and creates no projectile. `reset` rebuilds physics, health, weapons and projectiles with a fresh frame scope. Callers cannot access the mutable physics backend.

All living weapon outcomes are collected from the same boundary before damage changes health. Droid discharges trace the actual right-forearm muzzle before physical advancement. Drone discharges retain their launch direction and owner. Each drone round sweeps 0.36 m per 20 ms boundary, corresponding to 18 m/s, against the resulting physical collider poses. An unobstructed round expires after 100 segments, or two seconds.

Each round hits the nearest solid once and then disappears. The owner is excluded; other robots, including teammates, can receive body damage. Floor hits consume rounds without reducing robot health. Existing six-hit drone and three-hit droid body limits apply. Destroyed bodies remain in the solver, receive passive physics and cannot exert motor or joint force. Destroyed weapons reject later triggers; already launched rounds continue.

Drone observations report the same motor failures that constrain physical thrust. A failed rotor contributes the same force as a valid zero request for that rotor. Destroyed droids receive zero torque through `DroidAction::default`; every default actuator fraction is zero. This is mechanical shutdown, not an autonomous controller.

This integration does not implement rotor collision surfaces, weapon mass or recoil, fatal-contact rules, match termination, combat sensors, competitive rollout collection or trained competitive policies. Projectile collision uses endpoint-frame poses at 20 ms boundaries; it does not provide continuous collision against moving targets between those boundaries. The published shared-world diagnostic still uses `RobotWorld`. Qualified standing, walking and fighting remain unfinished.

Six public cases pass on native and WASM. Private tests cover physical pistol impact, prior-force shutdown, rotor thrust suppression, absorbing body damage, weapon-state preservation and projectile lifetime. Native format, strict Clippy, full tests and personal Rust lint pass. Coverage hits 373 of 390 production regions and both outcomes of all 12 conditional branches. The 17 unhit regions cover geometry-error handling and error propagation; the [validation record](progress/robot-battle-validation.json) lists their exact locations. Nix lint retains ten findings matching the preserved baseline.
