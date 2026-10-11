# Joint recurrent combat policy

The joint PPO API learns continuous commands and categorical trigger-request and
alert-state choices. Physical combat training remains unfinished.

## Quick start

Run the diagnostic from the repository root:

```sh
nix develop --command cargo run --no-default-features --example recurrent_combat
```

It performs 200 PPO updates over 6,400 sampled transitions. Reward grades the
policy's categorical choice; it never supplies that choice. This bandit does not
qualify locomotion, perception or fighting.

## Distribution and checkpoint contract

`RecurrentCombatAgent` shares one recurrent representation between a bounded
Gaussian head and an eight-column categorical head. `CombatSignal::ALL` fixes the
column order: idle, suspicious, pursuit and search, each with released then requested
trigger. Sampling draws the continuous commands before the categorical choice.
Deterministic inference uses continuous means and the categorical mode; ties select
the first column.

`RecurrentCombatAction::log_probability` sums the corrected bounded Gaussian log
density and categorical log probability. PPO forms one ratio and clips that joint
ratio once. It preserves each lane's initial recurrent memory and excludes padding
before probability arithmetic. Actor calls accept observations and their own memory.
Global state enters the separate critic call.

The entropy term combines a fresh Monte Carlo estimate of bounded continuous entropy
with exact categorical Shannon entropy. The continuous estimate uses the configured
Jacobian floor. `RecurrentCombatUpdate` reports this joint estimate; the existing
Gaussian learner and its metrics retain their previous contract.

Frozen byte loading validates configuration and every tensor before activation.
The joint record differs from the Gaussian-only record, which the joint loader
rejects. Callers must retain observation widths, action bounds, exploration bounds,
Jacobian floor and column order with the weights. Bytes alone do not establish a
training history or qualification.

Each minibatch checks both objective values before applying gradients. An error in
a later minibatch can retain earlier completed updates. Stored exploration scales
are projected after Adam while retaining parameter identity and optimizer state.

The [official MAPPO mixed-action implementation](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/algorithms/utils/act.py)
provides the joint-probability reference. This API uses pinned Burn 0.21.0
[`log_softmax`](https://docs.rs/burn/0.21.0/burn/tensor/activation/fn.log_softmax.html).
Competitive opponent selection and the physical team collector remain separate work.

## Verification

The [verification record](progress/recurrent-combat-policy-20261011/summary.json)
retains the failing public test, native and WASM gates, personal lints and fresh
instrumented-object coverage. Tests prove actual categorical learning, trained-byte
reload, complete decision identity, unequal lanes, padding exclusion, validation
order and non-finite failure handling. All eight sides of four recorded branches
were exercised. The record names every uncovered defensive region and its reason.

The next physical work is stable droid standing, commanded walking, combat rules,
sensor isolation and batched two-team collection. No fighting policy is qualified.
