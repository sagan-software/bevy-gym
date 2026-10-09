//! Watch the frozen RL curriculum policy in the standalone disturbed recovery lesson.
mod skill_scene;

/// Select the same environment factory used by training and evaluation.
fn main() {
    skill_scene::run(
        "Disturbed recovery",
        bevy_gym::robots::DroneHover::disturbed,
    );
}
