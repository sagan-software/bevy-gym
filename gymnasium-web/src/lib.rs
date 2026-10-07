//! Browser training sessions backed by Bevy Gym environments and learners.
use serde_json as _;
#[cfg(all(test, not(target_arch = "wasm32")))]
use tokio as _;
#[cfg(target_arch = "wasm32")]
use {console_error_panic_hook as _, getrandom as _, js_sys as _, wasm_bindgen as _, web_sys as _};
mod advance_steps;
mod command;
mod environment;
mod observation;
mod session;
mod session_error;
mod snapshot;
mod task;

pub use advance_steps::AdvanceSteps;
pub use command::Command;
pub use session::Session;
pub use session_error::SessionError;
pub use snapshot::{Episode, Snapshot};

pub use task::{InvalidTask, Task};
