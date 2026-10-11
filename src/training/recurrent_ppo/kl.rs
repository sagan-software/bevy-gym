//! Read-only conditional Gaussian divergence for evaluating actual policy updates.

use super::{
    validate_memory, validate_vector, RecurrentMemory, RecurrentPpoError, RecurrentPpoPolicy,
};

impl RecurrentPpoPolicy {
    /// Measure conditional `KL(previous || self)` in nats before tanh and action scaling.
    /// Both actors receive the same local observation and their separately supplied memories.
    /// This performs two actor forwards and changes neither policy nor caller memory.
    /// Critic state, action bounds and sampled actions do not enter this latent Gaussian measurement.
    /// For a trajectory, advance each policy's memory separately and average over actual observations.
    /// The PPO update's existing KL metric is measured before its optimizer step; this method
    /// permits a separate measurement afterward. The policy-update direction follows
    /// <https://arxiv.org/pdf/1707.06347>, equation4, August2017.
    ///
    /// # Errors
    /// Checks matching observation and action widths, observation values, previous memory,
    /// then current memory before either actor forward. Retains inference errors and rejects
    /// nonfinite Gaussian parameters or a divergence outside finite f64 precision.
    pub fn pre_tanh_kl_from(
        &self,
        previous: &Self,
        observation: &[f32],
        previous_memory: &RecurrentMemory,
        memory: &RecurrentMemory,
    ) -> Result<f64, RecurrentPpoError> {
        // Profile mismatches cannot reach observation conversion or either actor.
        for (field, expected, actual) in [
            (
                "KL observation",
                previous.observation_dim,
                self.observation_dim,
            ),
            (
                "KL action",
                previous.action_low.len(),
                self.action_low.len(),
            ),
        ] {
            if expected != actual {
                return Err(RecurrentPpoError::DimensionMismatch {
                    field,
                    expected,
                    actual,
                });
            }
        }
        validate_vector(observation, self.observation_dim, "observation")?;
        validate_memory(previous_memory, previous.hidden_size)?;
        validate_memory(memory, self.hidden_size)?;
        let (old_mean, old_log_std, _) = previous.distribution(observation, previous_memory)?;
        let (mean, log_std, _) = self.distribution(observation, memory)?;
        diagonal_kl(&old_mean, &old_log_std, &mean, &log_std)
    }
}

/// Sum independent dimensionless Gaussian divergences using f64 arithmetic.
/// Width and finite-value checks precede arithmetic; the input vectors remain borrowed.
fn diagonal_kl(
    previous_mean: &[f32],
    previous_log_std: &[f32],
    mean: &[f32],
    log_std: &[f32],
) -> Result<f64, RecurrentPpoError> {
    let width = previous_mean.len();
    for (values, field) in [
        (previous_mean, "previous Gaussian mean"),
        (previous_log_std, "previous Gaussian log std"),
        (mean, "Gaussian mean"),
        (log_std, "Gaussian log std"),
    ] {
        validate_vector(values, width, field)?;
    }
    let mut total = 0.0;
    for (dimension, (((old_mean, old_scale), mean), scale)) in previous_mean
        .iter()
        .zip(previous_log_std)
        .zip(mean)
        .zip(log_std)
        .enumerate()
    {
        // Log scales and mean differences are dimensionless pre-tanh values.
        let log_ratio = f64::from(*old_scale) - f64::from(*scale);
        let difference = f64::from(*old_mean) - f64::from(*mean);
        let normalized_difference = if difference == 0.0 {
            0.0
        } else {
            (difference.abs().ln() - f64::from(*scale)).exp()
        };
        let value =
            (0.5 * normalized_difference).mul_add(normalized_difference, variance_kl(log_ratio));
        total += value;
        if !total.is_finite() {
            return Err(RecurrentPpoError::NonFiniteValue {
                field: "pre-tanh KL",
                dimension,
            });
        }
    }
    Ok(total)
}

/// Evaluate `(exp(2*r)-1)/2-r` without losing tiny represented scale differences.
/// All operands and coefficients are dimensionless. Below |r|=0.001, terms through
/// r^7 leave a truncation error below f64 rounding at that threshold.
fn variance_kl(r: f64) -> f64 {
    if r.abs() < 0.001 {
        let polynomial = r.mul_add(4.0 / 315.0, 2.0 / 45.0);
        let polynomial = r.mul_add(polynomial, 2.0 / 15.0);
        let polynomial = r.mul_add(polynomial, 1.0 / 3.0);
        let polynomial = r.mul_add(polynomial, 2.0 / 3.0);
        let polynomial = r.mul_add(polynomial, 1.0);
        r * r * polynomial
    } else if r > 350.0 {
        // Divide inside the exponent so an intermediate exp(2*r) cannot overflow a finite result.
        r.mul_add(2.0, -std::f64::consts::LN_2).exp() - r - 0.5
    } else {
        (2.0 * r).exp_m1().mul_add(0.5, -r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small differences, range edges and large finite results retain their Gaussian meaning.
    #[test]
    fn gaussian_arithmetic_covers_small_and_large_scale_changes() {
        for tiny in [f32::from_bits(1), -f32::from_bits(1)] {
            let r = f64::from(tiny);
            let value = diagonal_kl(&[0.0], &[tiny], &[0.0], &[0.0]).unwrap();
            assert_eq!(value.to_bits(), (r * r).to_bits());
        }
        for r in [-350.0_f64, -0.001, 0.001, 350.0] {
            let expected = (2.0 * r).exp_m1().mul_add(0.5, -r);
            let actual = variance_kl(r);
            assert!((actual - expected).abs() <= expected.abs() * 1e-12);
        }
        for r in [-0.000_999, 0.000_999] {
            let expected = (2.0_f64 * r).exp_m1().mul_add(0.5, -r);
            assert!((variance_kl(r) - expected).abs() < 1e-18);
        }
        let large = diagonal_kl(&[0.0], &[355.0], &[0.0], &[0.0]).unwrap();
        assert!(large.is_finite() && large > 1e308);
        let large_mean = diagonal_kl(&[1.0], &[-355.0], &[0.0], &[-355.0]).unwrap();
        assert!(large_mean.is_finite() && large_mean > 1e308);
        assert_eq!(
            diagonal_kl(&[], &[], &[], &[]).unwrap().to_bits(),
            0_f64.to_bits()
        );
        assert!(diagonal_kl(&[f32::MAX], &[0.0], &[-f32::MAX], &[0.0])
            .unwrap()
            .is_finite());
    }

    /// Width and finite-value rejection precede arithmetic; overflow names its exact dimension.
    #[test]
    fn malformed_gaussians_and_unrepresentable_divergence_fail() {
        for (old_scale, mean, scale, field) in [
            (&[][..], &[0.0][..], &[0.0][..], "previous Gaussian log std"),
            (&[0.0][..], &[][..], &[0.0][..], "Gaussian mean"),
            (&[0.0][..], &[0.0][..], &[][..], "Gaussian log std"),
        ] {
            assert!(matches!(diagonal_kl(&[0.0], old_scale, mean, scale),
                Err(RecurrentPpoError::DimensionMismatch { field: actual, expected: 1, actual: 0 }) if actual == field));
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for (old_mean, old_scale, mean, scale, field) in [
                (invalid, 0.0, 0.0, 0.0, "previous Gaussian mean"),
                (0.0, invalid, 0.0, 0.0, "previous Gaussian log std"),
                (0.0, 0.0, invalid, 0.0, "Gaussian mean"),
                (0.0, 0.0, 0.0, invalid, "Gaussian log std"),
            ] {
                assert!(
                    matches!(diagonal_kl(&[old_mean], &[old_scale], &[mean], &[scale]),
                    Err(RecurrentPpoError::NonFiniteValue { field: actual, dimension: 0 }) if actual == field)
                );
            }
        }
        for result in [
            diagonal_kl(&[0.0], &[356.0], &[0.0], &[0.0]),
            diagonal_kl(&[1.0], &[-356.0], &[0.0], &[-356.0]),
        ] {
            assert!(matches!(
                result,
                Err(RecurrentPpoError::NonFiniteValue {
                    field: "pre-tanh KL",
                    dimension: 0
                })
            ));
        }
        assert!(matches!(
            diagonal_kl(&[1.0; 2], &[-355.0; 2], &[0.0; 2], &[-355.0; 2]),
            Err(RecurrentPpoError::NonFiniteValue {
                field: "pre-tanh KL",
                dimension: 1
            })
        ));
    }

    /// Invalid caller memory cannot reach either actor, even when the other memory is valid.
    #[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
    #[test]
    fn invalid_memory_prevents_all_actor_forwards() {
        use crate::training::{RecurrentPpoAgent, RecurrentPpoConfig, SeedConfig};
        let policy = RecurrentPpoAgent::new(
            1,
            1,
            1,
            &[-1.0],
            &[1.0],
            RecurrentPpoConfig::default(),
            SeedConfig::from_root(73),
        )
        .unwrap()
        .policy();
        let memory = policy.initial_memory();
        let mut invalid = memory.clone();
        invalid.hidden.clear();
        super::super::ACTOR_FORWARD_CALLS.with(|calls| calls.set(0));
        policy
            .pre_tanh_kl_from(&policy, &[0.25], &memory, &invalid)
            .unwrap_err();
        policy
            .pre_tanh_kl_from(&policy, &[0.25], &invalid, &memory)
            .unwrap_err();
        assert_eq!(
            super::super::ACTOR_FORWARD_CALLS.with(std::cell::Cell::get),
            0
        );
    }
}
