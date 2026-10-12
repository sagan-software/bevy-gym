# Physical muzzle geometry

Run the geometry example:

```sh
nix develop --command cargo run --no-default-features --features robots --example robot-muzzle
```

`RobotMuzzle::try_from((&snapshot, owner))` derives a weapon mount from an authoritative `RobotSnapshot`. It retains the owner's identity and source frame. It cannot select a target, choose aim or request a shot. Its origin, normalized direction and range are read-only.

The drone mount lies 0.317 m along local -Z from the body centre: the existing 0.297 m collision half-extent plus 0.02 m clearance. Its range is 36 m, corresponding to the existing projectile speed of 18 m/s over two seconds. The droid mount lies 0.197 m along local +X from the right forearm centre: the existing 0.177 m half-extent plus 0.02 m clearance. Its pistol range is 30 m. Body orientation transforms each mount into world coordinates; arm articulation determines droid aim.

`RobotWorld::cast_muzzle` validates physical state before comparing source frames. It excludes every collider belonging to the owner and returns the nearest solid under the existing ray-query ordering. Stale, reset and foreign frames return `RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)` without changing the world. A stopped world returns its original stopped-world cause first. Moving the world does not invalidate its frame.

These queries do not consume ammunition, advance weapon timing, move projectiles, apply recoil or damage bodies. Shared combat integration and qualified standing, walking and fighting policies remain unfinished. The example prints geometry without supplying autonomous actions.

Four public cases pass on native and WASM. Two private tests cover a rotated mount and invalid internal pose rejection. Native format, strict Clippy, full tests and personal Rust lint pass. Coverage hits all 65 production regions and both frame-validation outcomes. Nix lint retains ten findings matching the preserved baseline. The [validation record](progress/robot-muzzle-validation.json) identifies the checks.
