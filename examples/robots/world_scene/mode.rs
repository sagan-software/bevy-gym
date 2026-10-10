//! Select one scene before loading any policy or starting the renderer.

use std::{fmt, str::FromStr};

/// The existing standing view and the six-robot shared-world view.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum Mode {
    /// Preserve the standing viewer when no mode is supplied.
    #[default]
    Standing,
    /// Inspect six frozen policies in the common physical world.
    SharedWorld,
}

/// An explicit scene value is outside the supported closed vocabulary.
#[derive(Debug)]
pub(crate) struct InvalidMode;

impl FromStr for Mode {
    type Err = InvalidMode;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "standing" => Ok(Self::Standing),
            "shared-world" => Ok(Self::SharedWorld),
            _ => Err(InvalidMode),
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Standing => "standing",
            Self::SharedWorld => "shared-world",
        })
    }
}

impl fmt::Display for InvalidMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("scene must be standing or shared-world")
    }
}

impl std::error::Error for InvalidMode {}
