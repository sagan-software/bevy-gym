//! Convert native arguments and browser markup into the same scene vocabulary.

use super::mode::Mode;

/// Native viewer arguments; parsing finishes before any checkpoint loads.
#[cfg(not(target_arch = "wasm32"))]
#[derive(clap::Parser)]
#[command(about = "Inspect frozen RL policy inference")]
pub(crate) struct Options {
    /// Select the existing standing view or the six-robot shared world.
    #[arg(long, default_value_t)]
    pub(crate) scene: Mode,
}

/// Absence preserves the default; explicit strings use the exact closed grammar.
/// `Element.getAttribute` returns null for absence and preserves present text.
/// <https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttribute>.
#[cfg(target_arch = "wasm32")]
pub(crate) fn from_markup(value: Option<&str>) -> Result<Mode, super::mode::InvalidMode> {
    value.map_or_else(|| Ok(Mode::default()), str::parse)
}
