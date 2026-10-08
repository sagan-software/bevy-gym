# PettingZoo Classic examples

This folder ports the [PettingZoo Classic environments](https://github.com/Farama-Foundation/PettingZoo/tree/38e73889c04cedf7b92eb65d74bb6484f69c8c33/pettingzoo/classic) from the pinned PettingZoo submodule to Bevy Gym. PettingZoo uses the [Apache License 2.0](https://github.com/Farama-Foundation/PettingZoo/blob/38e73889c04cedf7b92eb65d74bb6484f69c8c33/LICENSE).

Each Rust file starts with the complete upstream Python module docstring. The examples include source-compatible rules, observations, action masks, rewards, seeded resets, focused tests, checkpoint workflows, and Bevy renderers.

The [implementation ledger](IMPLEMENTATION.md) records the exact comparison protocol, official targets, honest results, source files, and test coverage. Fixed game lines and fixed card orders exist only as deterministic rules tests.

## Running an example

Every versioned target supports `train`, `eval`, `watch`, `gif`, and `video`.

```sh
cargo run --example rps-v2 -- train
cargo run --example connect-four-v3 -- train --steps 20480 --seed 0
cargo run --example chess-v6 -- eval --checkpoint runs/chess-v6-dqn/<run-id>/best.mpk
cargo run --example go-v5 -- watch --checkpoint runs/go-v5-dqn/<run-id>/best.mpk
```

Use `cargo run --example <name> -- --help` for all options.

## Environments

- `rps-v2`
- `tictactoe-v3`
- `connect-four-v3`
- `chess-v6`
- `texas-holdem-no-limit-v6`
- `texas-holdem-v4`
- `leduc-holdem-v4`
- `go-v5`
- `hanabi-v5`

Runtime images and fonts live under `assets/` with their upstream paths preserved.
