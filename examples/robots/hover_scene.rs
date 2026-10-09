//! Watch the frozen RL curriculum policy in the standalone calm hover lesson.
mod skill_scene;

/// Select the same environment factory used by training and evaluation.
fn main() {
    skill_scene::run("Hover", bevy_gym::robots::DroneHover::default);
}
