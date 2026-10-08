# Drone platform comparison

Status: native and interactive browser comparison passed, 2026-10-08.

This checkpoint compares the same public `DroneHover` API in native Rust and
browser WebAssembly. It changes no production API. A native-generated fixture
tests platform equivalence; the gravity and torque tests remain the independent
physical checks.

## Contract and method

Use `wasm-bindgen-test` 0.3.71, which matches the locked `wasm-bindgen` 0.2.121.
The [pinned package manifest](https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.121/crates/test/Cargo.toml)
defines that pairing. The [browser testing guide](https://wasm-bindgen.github.io/wasm-bindgen/wasm-bindgen-test/browsers.html)
documents `run_in_browser` and interactive `NO_HEADLESS=1` execution.
The [pinned runner](https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.121/crates/cli/src/wasm_bindgen_test_runner.rs)
accepts `WASM_BINDGEN_TEST_ADDRESS` for the local test server.

- Keep actions validated through `DroneAction::try_from`.
- Record all 13 observation scalars as their `f32` bit patterns.
- Record each reward as its `f64` bit pattern and each status as a closed variant.
- Compare every reset and action result, including terminal repetition and reset.
- Exercise hover, power-off, climb, roll, pitch, yaw, and changing motor commands.
- Include disturbed hover, disturbed climb, and disturbed changing commands.
- Include seeds zero, 42, and `u64::MAX`; include seeded and continuing resets.
- Apply a 500-action `TimeLimit` and distinguish termination from truncation.
- Keep the fixture generator native-only and ignored during ordinary tests.
- Treat a mismatch as a failed gate. Do not replace exact comparison with a tolerance
  unless a researched limitation requires a documented contract change.

Fixture data is bounded test evidence. It does not define a public interchange
format or prove equivalence for all possible actions and platforms.

## Acceptance checklist

- [x] Run the existing eight public drone contract tests in the browser.
- [x] Compare every recorded scalar, reward, status, and reset in both targets.
- [x] Show the browser test result and retain its screenshot.
- [x] Keep native generation explicit; reject an empty or shortened fixture.
- [x] Run root format, tests, strict all-target/all-feature Clippy, and personal lints.
- [x] Run WASM test compilation and browser execution after the final source edit.
- [x] Record fixture provenance, commands, exact counts, and verification limits.
- [x] Finish with prose-only documentation, the guide, and documentation tests.

## Results and reproduction

The [fixture](../tests/fixtures/robots/drone-native.jsonl) contains 1,871 results:
24,323 observation floats, 1,841 rewards, and 30 resets. Every float bit and outcome
matched in native x86-64 Linux and the T3 Chromium browser. Both used Rust 1.97.1
test-profile builds. All eight hover and four recovery contract tests also passed
in both targets. The original 1,420 calm results remain byte-for-byte unchanged.
The [evidence record](progress/drone-parity.json) retains the fixture hash, per-case
counts, source hashes, tool versions, and coverage gaps.

Run the native tests:

```sh
nix develop --command cargo test --locked --no-default-features --features robots --lib --test drone_hover --test drone_recovery --test drone_parity
```

CI uses the pinned Nix Chromium and driver:

```sh
nix run .#drone-browser-check
```

That wrapper passed in [CI](https://github.com/sagan-software/bevy-gym/actions/runs/37848277209).
Local browser evidence comes from the interactive runner:

```sh
NO_HEADLESS=1 WASM_BINDGEN_TEST_ADDRESS=0.0.0.0:8770 \
  CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  nix develop --command cargo test --locked --no-default-features \
  --features robots,browser --target wasm32-unknown-unknown --test drone_parity
```

Open the printed port on the development host. Replace `drone_parity` with
`drone_hover` or `drone_recovery` to run their contract tests. The interactive runner stays open
after the page reports its result. Stop the runner after inspection.

When an intentional physics change requires a new reference, regenerate it explicitly:

```sh
nix develop --command cargo test --locked --no-default-features --features robots \
  --test drone_parity regenerate_native_fixture -- --ignored
```

Review the changed results before accepting the fixture. Generation does not establish
physical correctness. Do not regenerate it to conceal a browser mismatch.

Root tests, strict Clippy, WASM test Clippy, Nix lints, and workflow lint passed.
Personal Rust lints reported no changed-line diagnostics. The full-package run still
has 51 distinct Clippy errors and two Dylint warnings in unchanged files.
The comparison has full measured native branch coverage; its ignored generator's ten
lines were executed separately and remain unhit in the coverage run. Release-profile
bit equality, other architectures, native GUI interaction, and mobile touch are unverified.
