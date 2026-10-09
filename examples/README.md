# Examples

## Remember an earlier observation

The [remembered-cue guide](../docs/RECURRENT_IMITATION.md) teaches a recurrent actor
to retain a direction after that direction disappears from its observations.
The example contains only the demonstration, training loop, and inference calls.

## Ecosystem curriculum

The [ecosystem suite](ecosystem/README.md) is an incremental recurrent PPO
curriculum using Bevy, Burn, and Avian2D. Its four stages cover solo survival,
multi-agent competition, predator-prey learning, and obstacle avoidance.

Start the survival example with `cargo run --features ecosystem-inference --example ecosystem-survival`. It
opens the visual environment, trains in the background, and displays live
reward and learning curves in an Inspector-egui HUD.

Validated training videos will be embedded here after each stage passes its
declared held-out gate. The required clips are not represented by placeholders:
every embedded video must come from the checkpoint and manifest documented in
[ecosystem/EVIDENCE.md](ecosystem/EVIDENCE.md).
