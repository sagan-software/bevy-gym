//! Watch the frozen RL curriculum policy in the standalone disturbed recovery lesson.
#[expect(
    dead_code,
    reason = "Each standalone binary uses one entry point from the shared scene module."
)]
mod skill_scene;

/// Select the same environment factory used by training and evaluation.
fn main() {
    skill_scene::run(
        "Disturbed recovery",
        bevy_gym::robots::DroneHover::disturbed,
    );
}
