//! Run calm hover with the frozen reinforcement-trained curriculum checkpoint.

mod skill_inference;

/// Infer every motor command through the shared curriculum episode implementation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    skill_inference::run(bevy_gym::robots::DroneHover::default)
}
