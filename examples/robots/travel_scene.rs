//! Inspect frozen reinforcement-trained arrival through the original travel physics.
#[expect(
    dead_code,
    reason = "Each standalone binary uses one entry point from the shared scene module."
)]
mod skill_scene;

/// Start the same sampled travel task used by PPO collection and evaluation.
fn main() {
    skill_scene::run_travel();
}
