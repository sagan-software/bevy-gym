//! Learn to retain a cue after it disappears from the observation.
//!
//! Run with `cargo run --no-default-features --features browser-training --example remember-cue`.

use bevy_gym::training::{
    RecurrentBehaviorSample, RecurrentPpoAgent, RecurrentPpoConfig, SeedConfig,
};

/// Train two demonstrations, then act from memory while the current input is blank.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 8,
        critic_hidden_sizes: vec![4],
        actor_learning_rate: 0.02,
        ..RecurrentPpoConfig::default()
    };
    let mut learner =
        RecurrentPpoAgent::new(1, 1, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(73))?;
    let initial = learner.policy().initial_memory();

    // The first frame supplies a direction. Later frames contain no direction.
    let demonstrations = [-1.0, 1.0].map(|cue| {
        (0..4)
            .map(|step| RecurrentBehaviorSample {
                observation: vec![if step == 0 { cue } else { 0.0 }],
                action: vec![if step == 0 { 0.0 } else { cue * 0.8 }],
            })
            .collect::<Vec<_>>()
    });
    for _ in 0..250 {
        for demonstration in &demonstrations {
            learner.behavior_clone_sequence(demonstration, &initial)?;
        }
    }

    // Carry the returned memory within an episode; reset it for a new episode.
    let policy = learner.policy();
    for cue in [-1.0, 1.0] {
        let first = policy.mean_action(&[cue], &initial)?;
        let later = policy.mean_action(&[0.0], &first.next_memory)?;
        let action = later.action;
        println!("Earlier cue: {cue}; action with a blank observation: {action:?}");
    }
    Ok(())
}
