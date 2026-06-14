//! Deterministic seed stream configuration.

/// Seed streams used by training, replay, model initialization, and eval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeedConfig {
    /// Caller-provided root seed.
    pub root: u64,

    /// Environment reset seed stream.
    pub env_reset: u64,

    /// Exploration/action sampling seed stream.
    pub action: u64,

    /// Replay sampling seed stream.
    pub replay: u64,

    /// Model initialization seed stream.
    pub model: u64,

    /// Deterministic evaluation seed stream.
    pub eval: u64,
}

impl SeedConfig {
    /// Derive all trainer seed streams from a single root seed.
    #[must_use]
    pub const fn from_root(root: u64) -> Self {
        Self {
            root,
            env_reset: splitmix64(root ^ 0x6576_6e5f_7265_7365),
            action: splitmix64(root ^ 0x6163_7469_6f6e_0001),
            replay: splitmix64(root ^ 0x7265_706c_6179_0002),
            model: splitmix64(root ^ 0x6d6f_6465_6c00_0003),
            eval: splitmix64(root ^ 0x6576_616c_0000_0004),
        }
    }
}

/// Mix one seed value with `SplitMix64`.
const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_config_is_deterministic_and_distinct() {
        let first = SeedConfig::from_root(7);
        let second = SeedConfig::from_root(7);
        let third = SeedConfig::from_root(8);

        assert_eq!(first, second);
        assert_ne!(first, third);
        assert_ne!(first.env_reset, first.action);
        assert_ne!(first.action, first.replay);
        assert_ne!(first.replay, first.model);
        assert_ne!(first.model, first.eval);
    }
}
