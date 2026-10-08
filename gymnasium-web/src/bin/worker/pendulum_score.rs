//! Bounded frozen-policy scoring retains the final state before any reset.
use bevy_gym::environments::{Pendulum, PendulumAction};
use bevy_gym::training::{RecurrentPpoConfig, RecurrentPpoPolicy};
use bevy_gym::Env;

/// Evaluate exactly 200 mean actions with constant environment and recurrent state.
pub(super) fn evaluate(bytes: Vec<u8>, seed: u64) -> Result<serde_json::Value, String> {
    // Validate the frozen architecture before creating an independent reset stream.
    let config = RecurrentPpoConfig {
        actor_hidden_size: 32,
        critic_hidden_sizes: vec![64, 32],
        ..RecurrentPpoConfig::default()
    };
    let policy = RecurrentPpoPolicy::load_bytes(bytes, 3, 3, 1, &[-2.0], &[2.0], &config)
        .map_err(|error| error.to_string())?;
    let mut environment = Pendulum::default();
    let mut observation = environment.reset(Some(seed)).observation;
    let mut memory = policy.initial_memory();
    let mut reward = 0.0;
    let mut upright_steps = 0_u16;
    for index in 0..200 {
        // Normalize angular velocity by 8 rad/s only at the policy input boundary.
        let encoded = [observation[0], observation[1], observation[2] / 8.0];
        let sampled = policy
            .mean_action(&encoded, &memory)
            .map_err(|error| error.to_string())?;
        memory = sampled.next_memory;
        let torque = sampled
            .action
            .first()
            .copied()
            .ok_or("Pendulum policy omitted torque")?;
        let action = PendulumAction::try_from(torque).map_err(|error| error.to_string())?;
        let result = environment.step(action);
        observation = result.observation;
        reward += result.reward;
        // The time-limit transition contributes its state before the environment resets.
        let [angle, velocity] = environment.state();
        upright_steps += u16::from(upright_sample(index, angle, velocity));
    }
    Ok(serde_json::json!({"reward":reward,"upright_steps":upright_steps}))
}

/// Count indices 50 through 199 with inclusive 15-degree and 1-rad/s limits.
fn upright_sample(index: usize, angle: f64, velocity: f64) -> bool {
    (50..200).contains(&index)
        && angle.cos() >= (std::f64::consts::PI / 12.0).cos()
        && velocity.abs() <= 1.0
}

#[cfg(test)]
mod tests {
    use super::upright_sample;

    #[test]
    fn dwell_window_includes_the_final_transition_and_excludes_both_neighbors() {
        for index in [0, 49, 200, usize::MAX] {
            assert!(!upright_sample(index, 0.0, 0.0));
        }
        for index in [50, 199] {
            assert!(upright_sample(index, 0.0, 0.0));
        }
        assert_eq!(
            (0..200)
                .filter(|index| upright_sample(*index, 0.0, 0.0))
                .count(),
            150
        );
    }

    #[test]
    fn upright_limits_are_inclusive_and_full_turns_preserve_angle() {
        let limit = std::f64::consts::PI / 12.0;
        for angle in [0.0, limit, -limit, std::f64::consts::TAU] {
            for velocity in [-1.0, 0.0, 1.0] {
                assert!(upright_sample(199, angle, velocity));
            }
        }
        for angle in [limit + 1e-9, -limit - 1e-9, std::f64::consts::PI] {
            assert!(!upright_sample(50, angle, 0.0));
        }
        for velocity in [-1.0 - 1e-9, 1.0 + 1e-9] {
            assert!(!upright_sample(50, 0.0, velocity));
        }
    }
}
