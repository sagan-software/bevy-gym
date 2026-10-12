//! Frozen arrival architecture for heading-preserving shared-world balance.
use bevy_gym::training::{RecurrentPpoError, RecurrentPpoPolicy};

/// Restore the independently qualified fifteen-feature arrival checkpoint profile.
pub(super) fn load_policy(bytes: Vec<u8>) -> Result<RecurrentPpoPolicy, RecurrentPpoError> {
    RecurrentPpoPolicy::load_bytes(
        bytes,
        15,
        15,
        1,
        &[0.0; 4],
        &[1.0; 4],
        &crate::learning::learning_config(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    /// The shared actor uses the recorded arrival bytes and rejects incompatible data.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn arrival_identity_and_architecture_are_exact() {
        let bytes = super::super::DRONE_REFERENCE;
        let digest = Sha256::digest(bytes);
        assert_eq!(
            format!("{digest:x}"),
            "3e6964d0f1475081b2572387f2193e070fd396800023639074ca073e69b54c88"
        );
        load_policy(bytes.to_vec()).expect("recorded arrival architecture");
        assert!(matches!(
            load_policy(Vec::new()),
            Err(RecurrentPpoError::Checkpoint(_))
        ));
    }
}
