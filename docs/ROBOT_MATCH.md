# Match lifecycle

Inspect initial health without supplying autonomous actions:

```sh
nix develop --command cargo run --no-default-features --features robots --example robot-battle
```

`RobotMatch::try_from(Duration)` owns the [shared combat mechanics](ROBOT_BATTLE.md). Duration must contain 1 through `u32::MAX` complete 20 ms frames. Zero, partial frames and larger durations return `RobotMatchError::InvalidDuration` before allocating combat bodies. The example chooses two minutes; the API has no implicit duration.

`status` reports exactly one of `Running`, `Finished(result)` or `Failed(error)`. Running accepts complete policy motor, torque and trigger requests through `advance`. Finished retains a read-only final snapshot. Failed retains the original `RobotWorldError`. Neither terminal state retains a mutable combat backend. Public callers cannot access or extract the owned backend.

Elimination requires an entire team to die and every launched round to settle. One death leaves a match running while both teams have living robots. Results distinguish `DronesWin`, `DroidsWin`, `MutualElimination` and `Timeout`. Settled elimination takes precedence over timeout at the same boundary. Timeout retires remaining projectiles with the combat backend.

Terminal advancement returns `Finished(result)` or `Failed(error)` before consuming actions, triggers or physical time. `snapshot` remains readable after a finish and returns the preserved diagnostic after failure. `reset` recreates six bodies, health, weapons, projectiles and frame scope, then restores the configured duration. This mechanical layer owns no policy memory; the controller layer must reset its recurrent memories separately.

Thirteen combat cases pass on native and WASM, including five match cases. They cover exact duration boundaries, timeout, sticky failure, reset and physical elimination. Private cases cover all elimination outcomes, pending-round exclusion, the exact maximum duration and diagnostic sources. Native format, strict Clippy, full tests and personal Rust lint pass. Fresh coverage hits all 134 reported production regions in the match lifecycle scope and both outcomes of three conditional branches. The [validation record](progress/robot-match-validation.json) records source and executable hashes.

These tests use isolated mechanics fixtures. They do not qualify standing, walking or fighting. The published diagnostic still uses `RobotWorld`; the competitive viewer, sensors, rollout collection and trained competitive policies remain unfinished. No new screenshot or recording accompanies this mechanics change.
