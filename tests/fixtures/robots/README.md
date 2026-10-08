# Drone native reference

`drone-native.jsonl` records the public drone environment using Rust 1.97.1 on
`x86_64-unknown-linux-gnu`. Its first 1,420 calm results still match `bfe088d`.
Another 451 results cover disturbed starts. Each JSON line contains one complete
result. Integer fields preserve float bits; they are not rounded measurements.

The ten motor sequences and both reset modes live in
[`tests/drone_parity.rs`](../../drone_parity.rs). Ordinary tests compare the fixture
without writing it. Only the ignored `regenerate_native_fixture` test writes it.
See [the comparison guide](../../../docs/DRONE_PARITY.md) for commands, provenance,
counts, and verification limits.
