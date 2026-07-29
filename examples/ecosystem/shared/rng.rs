//! Small deterministic random stream used by environment generation.

/// `SplitMix64` stream with explicit seed ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SplitMix64 {
    /// Current generator state.
    state: u64,
}

impl SplitMix64 {
    /// Construct a stream from one derived environment seed.
    pub(super) const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Produce one uniformly scrambled integer.
    pub(super) const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Produce a finite scalar in `[0, 1)`.
    pub(super) fn unit_f32(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as u32;
        mantissa as f32 / 16_777_216.0
    }

    /// Produce a scalar in `[low, high)`.
    pub(super) fn f32_between(&mut self, low: f32, high: f32) -> f32 {
        self.unit_f32().mul_add(high - low, low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equal seeds must reproduce map and action randomness.
    #[test]
    fn equal_seeds_reproduce_the_stream() {
        let mut first = SplitMix64::new(42);
        let mut second = SplitMix64::new(42);
        let first_values = [first.next_u64(), first.next_u64(), first.next_u64()];
        let second_values = [second.next_u64(), second.next_u64(), second.next_u64()];
        assert_eq!(first_values, second_values);
    }
}
