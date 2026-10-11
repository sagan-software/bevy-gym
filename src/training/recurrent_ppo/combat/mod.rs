//! Joint recurrent continuous and categorical PPO.
mod action;
mod actor;
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
mod agent;
mod checkpoint;
mod policy;
mod sequence;
mod signal;
mod update;
pub use action::RecurrentCombatAction;
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
pub use agent::RecurrentCombatAgent;
pub use policy::RecurrentCombatPolicy;
pub use sequence::RecurrentCombatSequence;
pub use signal::CombatSignal;
pub use update::RecurrentCombatUpdate;

/// Retain a positive shape as a validated domain value after configuration validation.
fn positive_width(value: usize) -> Result<std::num::NonZeroUsize, super::RecurrentPpoError> {
    std::num::NonZeroUsize::try_from(value).map_err(|_zero_width| {
        super::RecurrentPpoError::invalid_config("dimensions", "network widths must be positive")
    })
}

#[cfg(test)]
mod tests {
    /// Zero cannot remain a stored shape; the minimum positive width is accepted.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn width_boundary() {
        assert!(matches!(
            super::positive_width(0),
            Err(super::super::RecurrentPpoError::InvalidConfig {
                field: "dimensions",
                ..
            })
        ));
        assert_eq!(super::positive_width(1).expect("positive").get(), 1);
    }
}
