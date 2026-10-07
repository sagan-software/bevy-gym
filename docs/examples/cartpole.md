# CartPole DQN trainer and visualizer

Self-contained CartPole-v1-style example using `bevy-gym`, Burn, and an optional Bevy visualizer.
The trainer writes run metadata, metrics, eval records, and Burn `best.mpk` checkpoints under
`runs/cartpole-dqn/<run-id>/`.

## Usage

Train a policy from a random initial network with DQN replay updates:

```sh
cargo run --example cartpole --release -- train
```

Evaluate a saved policy:

```sh
cargo run --example cartpole --release -- eval --checkpoint runs/cartpole-dqn/<run-id>/best.mpk
```

Create a 30-second training timelapse video from periodic eval checkpoints:

```sh
cargo run --example cartpole --release -- video \
  --checkpoint runs/cartpole-dqn/<run-id> \
  --output runs/cartpole-dqn/<run-id>/cartpole-training-timelapse.mp4
```

The video renderer uses Gymnasium's CartPole cadence: 50 FPS, matching the environment's 0.02 second
state update interval. It shows the best checkpoint for five seconds, then the first, 33 percent,
and 66 percent checkpoints for five seconds each. It finishes with the best checkpoint for ten
seconds.

Run the realtime visualizer with Bevy Remote Protocol and BRP extras enabled:

```sh
cargo run --example cartpole --features bevy_remote --release -- watch \
  --checkpoint runs/cartpole-dqn/<run-id>/best.mpk
```

With the lighter `render` feature, a no-subcommand run opens the latest saved checkpoint directly:

```sh
cargo run --example cartpole --features render --release
```

Use `train-watch` explicitly when you want one command to train a fresh policy and then open the
visualizer.

Capture a visual verification image from the app itself:

```sh
cargo run --example cartpole --features bevy_remote --release -- watch \
  --checkpoint runs/cartpole-dqn/<run-id>/best.mpk \
  --screenshot runs/cartpole-dqn/<run-id>/cartpole.png
```

`bevy_remote` enables BRP on port `15702` by default and includes `bevy_brp_extras`, so MCP/BRP
tools can inspect the scene and request screenshots. Override the port with `BRP_EXTRAS_PORT`.

## What it demonstrates

- `Env` implemented directly for a local `CartPole` type
- Burn `Flex` inference and `Autodiff<Flex>` training with a small DQN policy network
- A step-0 `initial_mean_reward` eval, followed by replay-buffer DQN updates and periodic evals
- `NamedMpkFileRecorder` checkpoints saved as `best.mpk` and `checkpoints/latest.mpk`
- Periodic `checkpoints/step-*.mpk` snapshots for training timelapse videos
- `BevyGymPlugin` stepping the environment at 50 Hz in the visualizer
- `ActionRequest<CartPole>` and `ActionResponse<CartPole>` as the model-policy boundary
- Gymnasium-style 2D visuals: white background, black track/cart/pole, hinge, wheels, and boundary markers
- BRP/MCP inspection of named entities such as `CartPole Cart`, `CartPole Pole`, and `CartPole Hinge`

## Notes

Default training does not use heuristic imitation. `warmup_batches` defaults to `0`, so the saved
`best.mpk` should be justified by eval improvement from the initial random-policy score. Use
`--smoke` for a fast compile/runtime check. `--warmup-batches N` is available only as an explicit
experiment when comparing heuristic behavior cloning against DQN-from-scratch runs.
