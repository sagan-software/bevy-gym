//! Small deterministic random stream for environment reset sampling.

/// `SplitMix64` stream with scalar sampling helpers.
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    /// Current generator state.
    state: u64,
}

impl SplitMix64 {
    /// Construct a stream from one root seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate a uniform scalar in `[0, 1)`.
    pub fn unit_f64(&mut self) -> f64 {
        // Preserve deterministic state ordering across this implementation stage.
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Generate a uniform scalar in `[0, 1)`.
    pub fn unit_f32(&mut self) -> f32 {
        self.unit_f64() as f32
    }

    /// Generate one `f64` scalar in `[low, high)`.
    pub fn f64_between(&mut self, low: f64, high: f64) -> f64 {
        self.unit_f64().mul_add(high - low, low)
    }

    /// Generate one `f32` scalar in `[low, high)`.
    pub fn f32_between(&mut self, low: f32, high: f32) -> f32 {
        self.unit_f32().mul_add(high - low, low)
    }

    /// Generate one standard-normal scalar with the Box-Muller transform.
    pub fn standard_normal_f32(&mut self) -> f32 {
        let radius = (-2.0 * self.unit_f32().max(f32::MIN_POSITIVE).ln()).sqrt();
        let angle = std::f32::consts::TAU * self.unit_f32();
        radius * angle.cos()
    }

    /// Generate one `f64` standard-normal scalar with the Box-Muller transform.
    pub fn standard_normal_f64(&mut self) -> f64 {
        let radius = (-2.0 * self.unit_f64().max(f64::MIN_POSITIVE).ln()).sqrt();
        let angle = std::f64::consts::TAU * self.unit_f64();
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_seeds_produce_equal_streams() {
        let mut left = SplitMix64::new(42);
        let mut right = SplitMix64::new(42);

        for _ in 0..16 {
            assert_eq!(left.unit_f64().to_bits(), right.unit_f64().to_bits());
        }
    }

    #[test]
    fn normal_samples_are_finite() {
        let mut stream = SplitMix64::new(907);

        for _ in 0..1_000 {
            assert!(stream.standard_normal_f32().is_finite());
            assert!(stream.standard_normal_f64().is_finite());
        }
    }
}
