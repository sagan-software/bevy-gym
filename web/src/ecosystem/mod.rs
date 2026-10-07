//! Browser adapter over the library-owned ecosystem simulation facade.

#[cfg(target_arch = "wasm32")]
mod renderer;
mod session;

#[cfg(target_arch = "wasm32")]
pub(crate) use renderer::{draw_world, setup_camera};
#[cfg(target_arch = "wasm32")]
pub(crate) use session::InferenceSession;

/// Current browser simulation, absent while checkpoint assets load.
#[cfg(target_arch = "wasm32")]
pub(crate) struct ActiveSession(pub(crate) Option<InferenceSession>);
