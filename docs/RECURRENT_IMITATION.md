# Learn from an observation sequence

Run the [remembered-cue guide](../examples/robots/remember.rs):

```sh
cargo run --no-default-features --features browser-training --example remember-cue
```

The first observation supplies a direction. Three blank observations follow.
The trained actor chooses approximately -0.8 after the negative cue and +0.8 after
the positive cue. Its current observation is identical in both cases.
The difference comes from its recurrent memory.

`RecurrentPpoAgent::behavior_clone_sequence` trains one contiguous demonstration.
It reuses `RecurrentBehaviorSample` and `RecurrentMemory`; no new public type is
required. Gradients pass through earlier observations within the sequence.
The supplied initial memory is borrowed and detached from earlier computation.

Keep samples in temporal order within one episode. At inference, pass each
returned `next_memory` to the next decision. Reset memory when a new episode
starts. Activation storage grows with sequence length, so callers control the
length of each training chunk.

The three cloning methods have different memory behavior:

- `behavior_clone` starts every sample with zero memory.
- `behavior_clone_with_memory` fits independent samples with explicit initial states.
- `behavior_clone_sequence` carries memory and gradients through one ordered sequence.

All three update only the actor. Sequence cloning returns mean squared error
over timesteps and normalized action dimensions. Each action interval maps to
[-1, 1], so different physical action ranges do not change their relative scale.
Sample validation runs before initial-memory validation. Invalid input leaves
both actor parameters and the next optimizer update unchanged.

The guide enables `browser-training`, which also makes the training API available
in WASM. Six integration tests pass natively and in Chrome/WASM. The standard browser check
now runs them automatically. They cover
remembered cues, initial memory, action scaling, validation order, inclusive bounds,
and unchanged state after rejection. A private encoder test checks owned and
borrowed row order. The [verification record](progress/recurrent-sequence-validation.json)
and [coverage record](progress/recurrent-sequence-coverage.json) retain the evidence.

This is a training primitive for the [drone search curriculum](DRONE_SEARCH.md).
It does not establish a qualified drone-search policy or add browser game controls.
