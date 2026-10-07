//! Browser entry point for ecosystem checkpoint inference.

#[cfg(target_arch = "wasm32")]
mod app;
#[cfg(any(target_arch = "wasm32", test))]
mod checkpoint;
#[cfg(any(target_arch = "wasm32", test))]
mod checkpoint_observation;
#[cfg(target_arch = "wasm32")]
mod ecosystem;
#[cfg(any(target_arch = "wasm32", test))]
mod manifest;

#[cfg(target_arch = "wasm32")]
use getrandom as _;
#[cfg(not(target_arch = "wasm32"))]
use sha2 as _;
#[cfg(all(not(target_arch = "wasm32"), feature = "native-export"))]
use {bevy_gym as _, clap as _, serde as _, serde_json as _, tokio as _};

#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();
    app::run();
}

#[cfg(not(target_arch = "wasm32"))]
const fn main() {}
