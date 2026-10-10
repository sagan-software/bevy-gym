//! Train or evaluate direct-torque standing policies through the same physical environment.

#[cfg(not(target_arch = "wasm32"))]
#[expect(
    unused_imports,
    dead_code,
    reason = "Standing reuses only the collector from the shared drone tutorial facade."
)]
mod learning;
#[cfg(not(target_arch = "wasm32"))]
mod standing;

/// Dispatch the separate training and frozen-inference commands.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    standing::run()
}

/// Browser applications use the shared model and environment directly, without this native CLI.
#[cfg(target_arch = "wasm32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("droid-standing is a native file-based CLI; its policy and environment are browser-portable".into())
}
