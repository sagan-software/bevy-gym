//! Default Burn backend aliases and device helpers.

use burn::backend::flex::FlexDevice;
use burn::tensor::{backend::Backend, Device};
use std::sync::Mutex;

/// Serializes process-global backend seeding and lazy model materialization.
pub(super) static MODEL_INITIALIZATION_LOCK: Mutex<()> = Mutex::new(());

/// Default inference backend for CPU/headless policy execution.
pub type InferenceBackend = burn::backend::Flex;

/// Default autodiff backend for CPU/headless training.
pub type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;

/// Device type for the default inference backend.
pub type InferenceDevice = Device<InferenceBackend>;

/// Device type for the default training backend.
pub type TrainingDevice = Device<TrainingBackend>;

/// Construct the default inference device.
#[must_use]
pub const fn inference_device() -> InferenceDevice {
    FlexDevice
}

/// Construct the default training device.
#[must_use]
pub const fn training_device() -> TrainingDevice {
    FlexDevice
}

/// Human-readable names for the default backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendNames {
    /// Inference backend alias name.
    pub inference: &'static str,

    /// Training backend alias name.
    pub training: &'static str,
}

/// Return the backend names used by documentation and diagnostics.
#[must_use]
pub const fn backend_names() -> BackendNames {
    BackendNames {
        inference: "burn::backend::Flex",
        training: "burn::backend::Autodiff<burn::backend::Flex>",
    }
}

/// Assert at compile time that the aliases implement Burn's backend trait.
pub const fn assert_default_backends_compile() {
    const fn assert_backend<B: Backend>() {}

    assert_backend::<InferenceBackend>();
    assert_backend::<TrainingBackend>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_names_are_architecture_selected() {
        assert_eq!(backend_names().inference, "burn::backend::Flex");
        assert_eq!(
            backend_names().training,
            "burn::backend::Autodiff<burn::backend::Flex>"
        );
    }

    #[test]
    fn default_devices_are_constructible() {
        let _inference = inference_device();
        let _training = training_device();
        assert_default_backends_compile();
    }
}
