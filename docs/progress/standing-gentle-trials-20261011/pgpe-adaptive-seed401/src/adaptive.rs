//! Learned PGPE exploration scales; reward signals do not select actuator actions.
use crate::gradient::Pair;

/// Apply the variance part of Equation 15 from the PGPE primary paper.
/// <https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Publications/Neural-Networks-2010-Sehnke.pdf>.
/// Local profile averages the pair batch and clamps scales between exp(-9f32) and calibration.
/// Sigma and noise are dimensionless pre-tanh coefficients; the rate is dimensionless.
/// The baseline must be finite and below the known discounted-return bound of201.
/// Validate all inputs before deriving a candidate; the supplied scales remain unchanged.
pub(crate) fn update_sigma(
    sigma: &[f64; 26],
    ceilings: &[f64; 26],
    pairs: &[Pair],
    baseline: f64,
    rate: f64,
) -> Result<[f64; 26], &'static str> {
    if pairs.is_empty() {
        return Err("PGPE variance requires at least one pair");
    }
    if !baseline.is_finite() || baseline >= 201.0 {
        return Err("invalid PGPE variance baseline");
    }
    if !rate.is_finite() || rate <= 0.0 {
        return Err("invalid PGPE variance rate");
    }
    let floor = f64::from((-9.0_f32).exp());
    for (&scale, &ceiling) in sigma.iter().zip(ceilings) {
        if !scale.is_finite()
            || scale < floor
            || !ceiling.is_finite()
            || ceiling > 1.0
            || scale > ceiling
        {
            return Err("invalid PGPE variance limits");
        }
    }
    for pair in pairs {
        if !pair.positive.is_finite()
            || !pair.negative.is_finite()
            || pair.positive > 201.0
            || pair.negative > 201.0
        {
            return Err("invalid PGPE variance return");
        }
        if pair
            .noise
            .iter()
            .any(|v| !v.is_finite() || !(*v as f32).is_finite())
        {
            return Err("invalid PGPE variance noise");
        }
    }
    let mut gradient = [0.0; 26];
    for pair in pairs {
        // Half each return before addition; finite reward pairs cannot overflow their average.
        let advantage = (pair.positive * 0.5 + pair.negative * 0.5 - baseline) / (201.0 - baseline);
        for ((sum, &noise), &scale) in gradient.iter_mut().zip(&pair.noise).zip(sigma) {
            *sum += advantage * (noise * noise - scale * scale) / scale / (pairs.len() as f64);
        }
    }
    let mut candidate = *sigma;
    for ((value, gradient), &ceiling) in candidate.iter_mut().zip(gradient).zip(ceilings) {
        // Reject arithmetic failure before applying the local exploration bounds.
        let proposed = *value + rate * gradient;
        if !proposed.is_finite() {
            return Err("PGPE variance update is nonfinite");
        }
        *value = proposed.clamp(floor, ceiling);
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::Pair;
    /// Better wide samples increase exploration; worse wide samples reduce it.
    #[test]
    fn paired_returns_and_noise_control_variance_direction() {
        let pair = Pair {
            noise: [0.2; 26],
            positive: 101.0,
            negative: 101.0,
        };
        let raised = update_sigma(&[0.1; 26], &[0.4; 26], &[pair], 1.0, 0.1).unwrap();
        assert!(raised.iter().all(|v| (*v - 0.115).abs() < 1e-14));
        let pair = Pair {
            positive: -99.0,
            negative: -99.0,
            ..pair
        };
        let lowered = update_sigma(&[0.1; 26], &[0.4; 26], &[pair], 1.0, 0.1).unwrap();
        assert!(lowered.iter().all(|v| (*v - 0.085).abs() < 1e-14));
    }
    /// Local variance limits constrain an update without choosing actions.
    #[test]
    fn variance_limits_and_zero_updates_are_explicit() {
        let pair = Pair {
            noise: [1.0; 26],
            positive: 201.0,
            negative: 201.0,
        };
        assert_eq!(
            update_sigma(&[0.1; 26], &[0.2; 26], &[pair], 1.0, 1.0).unwrap(),
            [0.2; 26]
        );
        let pair = Pair {
            positive: -1000.0,
            negative: -1000.0,
            ..pair
        };
        assert_eq!(
            update_sigma(&[0.1; 26], &[0.2; 26], &[pair], 1.0, 1.0).unwrap(),
            [(-9.0_f32).exp() as f64; 26]
        );
        let pair = Pair {
            positive: 1.0,
            negative: 1.0,
            ..pair
        };
        assert_eq!(
            update_sigma(&[0.1; 26], &[0.2; 26], &[pair], 1.0, 0.1).unwrap(),
            [0.1; 26]
        );
    }
    /// Invalid normalization, scale, return and rate values fail before mutation.
    #[test]
    fn invalid_variance_inputs_are_rejected() {
        let sigma = [0.1; 26];
        let ceilings = [0.2; 26];
        let pair = Pair {
            noise: [0.2; 26],
            positive: 1.0,
            negative: 1.0,
        };
        assert!(update_sigma(&sigma, &ceilings, &[], 0.0, 0.1).is_err());
        for baseline in [201.0, 202.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(update_sigma(&sigma, &ceilings, &[pair], baseline, 0.1).is_err());
        }
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(update_sigma(&sigma, &ceilings, &[pair], 0.0, rate).is_err());
        }
        for value in [
            0.0,
            -1.0,
            1e-8,
            0.3,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let mut s = sigma;
            s[25] = value;
            assert!(update_sigma(&s, &ceilings, &[pair], 0.0, 0.1).is_err());
        }
        for value in [
            0.0,
            -1.0,
            0.01,
            1.01,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let mut s = ceilings;
            s[25] = value;
            assert!(update_sigma(&sigma, &s, &[pair], 0.0, 0.1).is_err());
        }
        for value in [202.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(update_sigma(
                &sigma,
                &ceilings,
                &[Pair {
                    positive: value,
                    ..pair
                }],
                0.0,
                0.1
            )
            .is_err());
            assert!(update_sigma(
                &sigma,
                &ceilings,
                &[Pair {
                    negative: value,
                    ..pair
                }],
                0.0,
                0.1
            )
            .is_err());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            assert!(update_sigma(
                &sigma,
                &ceilings,
                &[Pair {
                    noise: [value; 26],
                    ..pair
                }],
                0.0,
                0.1
            )
            .is_err());
        }
    }
    /// Arithmetic failure is rejected before clamping; matched noise gives no update.
    #[test]
    fn overflow_and_exact_noise_variance_have_distinct_outcomes() {
        let pair = Pair {
            noise: [1.0; 26],
            positive: 201.0,
            negative: 201.0,
        };
        assert_eq!(
            update_sigma(&[0.1; 26], &[0.2; 26], &[pair], 1.0, f64::MAX),
            Err("PGPE variance update is nonfinite")
        );
        let pair = Pair {
            noise: [0.1; 26],
            ..pair
        };
        assert_eq!(
            update_sigma(&[0.1; 26], &[0.2; 26], &[pair], 1.0, 0.1).unwrap(),
            [0.1; 26]
        );
    }
}
