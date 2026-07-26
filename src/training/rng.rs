//! Deterministic seed stream configuration.

/// Independent seed streams used by training, selection, proof, and demos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeedConfig {
    /// Caller-provided root seed.
    pub root: u64,

    /// Environment-construction seed stream.
    pub env_construction: u64,

    /// Environment reset seed stream.
    pub env_reset: u64,

    /// Exploration/action sampling seed stream.
    pub action: u64,

    /// Replay sampling seed stream.
    pub replay: u64,

    /// On-policy rollout sampling and minibatch seed stream.
    pub rollout: u64,

    /// Model initialization seed stream.
    pub model: u64,

    /// Fixed checkpoint-selection validation seed stream.
    pub validation: u64,

    /// Disjoint final test evaluation seed stream.
    pub test: u64,

    /// Fixed checkpoint-replay demo seed stream.
    pub demo: u64,
}

impl SeedConfig {
    /// Derive all trainer seed streams from a single root seed.
    #[must_use]
    pub const fn from_root(root: u64) -> Self {
        Self {
            root,
            env_construction: splitmix64(root ^ 0x656e_765f_636f_6e73),
            env_reset: splitmix64(root ^ 0x6576_6e5f_7265_7365),
            action: splitmix64(root ^ 0x6163_7469_6f6e_0001),
            replay: splitmix64(root ^ 0x7265_706c_6179_0002),
            rollout: splitmix64(root ^ 0x726f_6c6c_6f75_0003),
            model: splitmix64(root ^ 0x6d6f_6465_6c00_0004),
            validation: splitmix64(root ^ 0x7661_6c69_6461_0005),
            test: splitmix64(root ^ 0x7465_7374_0000_0006),
            demo: splitmix64(root ^ 0x6465_6d6f_0000_0007),
        }
    }

    /// Derive a deterministic reset seed for one environment and episode.
    ///
    /// This keeps parallel environment streams independent without relying on
    /// scheduling or completion order.
    #[must_use]
    pub const fn environment_episode(self, env_id: usize, episode: u64) -> u64 {
        let env_lane = (env_id as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
        let episode_lane = episode.wrapping_mul(0xa076_1d64_78bd_642f);
        splitmix64(self.env_reset ^ env_lane ^ episode_lane)
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
        assert_ne!(first.env_construction, first.env_reset);
        assert_ne!(first.env_reset, first.action);
        assert_ne!(first.action, first.replay);
        assert_ne!(first.replay, first.rollout);
        assert_ne!(first.rollout, first.model);
        assert_ne!(first.model, first.validation);
        assert_ne!(first.validation, first.test);
        assert_ne!(first.test, first.demo);
    }

    #[test]
    fn environment_episode_seeds_are_stable_and_partitioned() {
        let seeds = SeedConfig::from_root(42);

        assert_eq!(
            seeds.environment_episode(3, 9),
            seeds.environment_episode(3, 9)
        );
        assert_ne!(
            seeds.environment_episode(3, 9),
            seeds.environment_episode(4, 9)
        );
        assert_ne!(
            seeds.environment_episode(3, 9),
            seeds.environment_episode(3, 10)
        );
        assert_ne!(seeds.environment_episode(0, 0), seeds.validation);
    }
}
