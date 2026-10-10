//! Inspect frozen PPO torque control of the physical droid, including its failed balance.
#[expect(
    dead_code,
    reason = "The scene loads frozen weights; trainer commands are separate examples."
)]
#[path = "learning/inference.rs"]
mod learning;
#[expect(
    dead_code,
    reason = "Frozen playback shares the standing model and encoding with native training."
)]
mod standing;
mod standing_scene;

/// Render physical observations; authored animations never choose living motion.
fn main() {
    standing_scene::run();
}
