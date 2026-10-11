# Shared damage rules

`RobotHealth` records body life and rotor integrity for one team. Construct fresh health with `RobotHealth::from(RobotTeam::Drones)` or `RobotHealth::from(RobotTeam::Droids)`. Team identity remains readable after death. Health transitions constrain mechanics; they do not choose movement, aim or trigger requests.

A drone's body needs six hits to be destroyed. A droid's body needs three hits. `hit_body` returns `RobotDamage::Hit` for earlier hits and `RobotDamage::Destroyed` once for the final hit. Further body hits return `RobotDamage::Ignored`.

Each drone rotor needs two hits. Its first hit leaves the motor working; its second hit returns `RobotDamage::RotorDestroyed(motor)` and makes `motor_state` report `DroneMotorState::Failed`. Destroyed rotors ignore further hits without reducing body life. All four failed rotors can coexist with a living body; physical crash handling remains the environment's responsibility.

`destroy` reports a fatal transition once. A destroyed drone retains its last rotor observations, but every motor reports failure. Rotor hits on a dead drone leave those observations unchanged. Both living and destroyed droids reject rotor inspection and rotor hits with `RobotRotorHitRejected` before lifecycle denial; rejection changes no health. Episode reset constructs fresh health rather than reviving the old value.

`body_hits_remaining` returns a read-only `RobotHitCount` bounded from zero through six. Living team-specific states cannot retain zero hits or give a droid more than three hits. Raw integer construction and mutable state access are unavailable. Inspection copies contain no mutable alias to the original health.

Run the configuration example:

```sh
nix develop --command cargo run --no-default-features --features robots --example robot-health
```

The example prints initial hit limits without supplying autonomous actions. Shared-world projectile collisions, motor force suppression, droid torque shutdown and competitive rollout integration remain unfinished. These mechanics tests do not establish working fighting or qualify a policy.

Seven public tests pass on native and WASM. Two private tests cover physical rotor storage order and default rotor integrity. Native format, strict Clippy, full tests and personal Rust lint pass. Coverage hits all 154 production regions; this module has no instrumented conditional branches. Nix lint retains ten findings matching the preserved baseline. The [validation record](progress/robot-health-validation.json) identifies the source and commands.
