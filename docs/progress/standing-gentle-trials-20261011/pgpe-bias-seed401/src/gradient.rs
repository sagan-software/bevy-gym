//! Fixed-variance symmetric PGPE mean updates; no environment or action generation.
/// Four-seed episode-return averages for opposite perturbations of the same actor biases.
#[derive(Clone, Copy)]
pub(crate) struct Pair {
    /// Dimensionless pre-tanh bias offsets, held throughout each episode.
    pub(crate) noise: [f64; 26],
    /// Discounted dense return for mean plus noise.
    pub(crate) positive: f64,
    /// Discounted dense return for mean minus noise.
    pub(crate) negative: f64,
}

/// Apply the symmetric PGPE mean update from Eq.15 of Sehnke et al. (2010).
/// <https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Publications/Neural-Networks-2010-Sehnke.pdf>.
/// Local profile: only26 actor biases vary; covariance remains fixed; pairs average4 shared seeds.
/// Dense reward <=1 and gamma=0.995f32 imply discounted return <201; that bound normalizes returns.
/// Mean and perturbations are dimensionless pre-tanh coefficients; rate is dimensionless.
/// Validate every pair before computing a candidate; failures never mutate the supplied mean.
pub(crate) fn update_mean(
    mean: &[f64; 26],
    pairs: &[Pair],
    rate: f64,
) -> Result<[f64; 26], &'static str> {
    if pairs.is_empty() {
        return Err("PGPE requires at least one symmetric pair");
    }
    if !rate.is_finite() || rate <= 0.0 {
        return Err("PGPE rate must be finite and positive");
    }
    if mean
        .iter()
        .any(|value| !value.is_finite() || !(*value as f32).is_finite())
    {
        return Err("invalid PGPE mean");
    }
    for pair in pairs {
        if !pair.positive.is_finite()
            || !pair.negative.is_finite()
            || pair.positive > 201.0
            || pair.negative > 201.0
        {
            return Err("PGPE return violates its finite upper bound");
        }
        if pair
            .noise
            .iter()
            .any(|value| !value.is_finite() || !(*value as f32).is_finite())
        {
            return Err("invalid PGPE perturbation");
        }
    }
    let mut gradient = [0.0; 26];
    for pair in pairs {
        let positive_gap = 201.0 - pair.positive;
        let negative_gap = 201.0 - pair.negative;
        let scale = positive_gap.max(negative_gap);
        // Equal perfect returns produce no gradient; scaling gaps avoids overflow in their sum.
        let coefficient = if scale == 0.0 {
            0.0
        } else {
            let positive = positive_gap / scale;
            let negative = negative_gap / scale;
            (negative - positive) / (negative + positive)
        };
        for (sum, noise) in gradient.iter_mut().zip(pair.noise) {
            *sum += noise * coefficient / (pairs.len() as f64);
        }
    }
    let mut candidate = *mean;
    for (value, gradient) in candidate.iter_mut().zip(gradient) {
        // Preserve signed-zero parameters when the measured gradient is exactly zero.
        if gradient != 0.0 {
            *value += rate * gradient;
        }
        if !value.is_finite() || !(*value as f32).is_finite() {
            return Err("PGPE update exceeds finite actor precision");
        }
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Batch averaging and zero gradients retain their distinct meanings.
    #[test]
    fn averaging_and_signed_zero_are_preserved() {
        let pairs = [
            Pair {
                noise: [1.0; 26],
                positive: 201.0,
                negative: 0.0,
            },
            Pair {
                noise: [3.0; 26],
                positive: 201.0,
                negative: 0.0,
            },
        ];
        assert_eq!(update_mean(&[0.0; 26], &pairs, 0.2).unwrap(), [0.4; 26]);
        let zero = update_mean(
            &[-0.0; 26],
            &[Pair {
                noise: [1.0; 26],
                positive: 0.0,
                negative: 0.0,
            }],
            0.2,
        )
        .unwrap();
        assert!(zero
            .into_iter()
            .all(|value| value.to_bits() == (-0.0_f64).to_bits()));
    }
    /// Scaled reward gaps handle extreme finite returns without summation overflow.
    #[test]
    fn finite_extremes_and_update_overflow_are_distinct() {
        let pair = Pair {
            noise: [1.0; 26],
            positive: -f64::MAX,
            negative: 201.0,
        };
        assert_eq!(update_mean(&[0.0; 26], &[pair], 0.2).unwrap(), [-0.2; 26]);
        let pair = Pair {
            noise: [f64::from(f32::MAX); 26],
            positive: 201.0,
            negative: 0.0,
        };
        assert!(update_mean(&[f64::from(f32::MAX); 26], &[pair], 0.2).is_err());
        for rate in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(update_mean(&[0.0; 26], &[pair], rate).is_err());
        }
    }
    /// Opposite return order reverses the learned mean direction.
    #[test]
    fn symmetric_rewards_move_only_the_measured_parameter_direction() {
        let mut noise = [0.0; 26];
        noise[0] = 1.0;
        noise[1] = -1.0;
        let pair = Pair {
            noise,
            positive: 201.0,
            negative: -1.0,
        };
        let result = update_mean(&[0.0; 26], &[pair], 0.2).unwrap();
        assert_eq!(result[0], 0.2);
        assert_eq!(result[1], -0.2);
        assert!(result[2..].iter().all(|value| *value == 0.0));
        let reverse = Pair {
            noise,
            positive: -1.0,
            negative: 201.0,
        };
        let result = update_mean(&[0.0; 26], &[reverse], 0.2).unwrap();
        assert_eq!(result[0], -0.2);
        assert_eq!(result[1], 0.2);
    }
    /// Equal returns and the perfect-return boundary produce no parameter change.
    #[test]
    fn equal_and_perfect_returns_preserve_every_mean() {
        for score in [0.0, 201.0] {
            let result = update_mean(
                &[0.25; 26],
                &[Pair {
                    noise: [1.0; 26],
                    positive: score,
                    negative: score,
                }],
                0.2,
            )
            .unwrap();
            assert_eq!(result, [0.25; 26]);
        }
    }
    /// Invalid pair values fail before producing a mutable parameter update.
    #[test]
    fn invalid_inputs_and_overflow_fail_without_mutating_the_mean() {
        let mean = [0.25; 26];
        assert!(update_mean(&mean, &[], 0.2).is_err());
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 201.001] {
            assert!(update_mean(
                &mean,
                &[Pair {
                    noise: [1.0; 26],
                    positive: invalid,
                    negative: 0.0
                }],
                0.2
            )
            .is_err());
            assert!(update_mean(
                &mean,
                &[Pair {
                    noise: [1.0; 26],
                    positive: 0.0,
                    negative: invalid
                }],
                0.2
            )
            .is_err());
        }
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(update_mean(
                &mean,
                &[Pair {
                    noise: [invalid; 26],
                    positive: 0.0,
                    negative: 0.0
                }],
                0.2
            )
            .is_err());
            assert!(update_mean(
                &[invalid; 26],
                &[Pair {
                    noise: [0.0; 26],
                    positive: 0.0,
                    negative: 0.0
                }],
                0.2
            )
            .is_err());
        }
        assert_eq!(mean, [0.25; 26]);
    }
}
