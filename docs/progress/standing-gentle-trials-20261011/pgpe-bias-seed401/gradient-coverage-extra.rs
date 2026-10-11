//! Additional coverage boundary; frozen trainer source stays unchanged.
#[path = "src/gradient.rs"]
mod gradient;
/// A finite positive rate can overflow the candidate and must be rejected.
#[test]
fn finite_rate_overflow_rejects_infinite_candidate() {
    let pair = gradient::Pair { noise: [2.0; 26], positive: 201.0, negative: 0.0 };
    assert_eq!(gradient::update_mean(&[0.0; 26], &[pair], f64::MAX), Err("PGPE update exceeds finite actor precision"));
}
