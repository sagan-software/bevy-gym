//! Explicit viewer modes reject unsupported values before checkpoint loading.
#![cfg(all(
    feature = "robots",
    any(not(target_arch = "wasm32"), feature = "browser")
))]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// The entry point shares one closed selector with native CLI and browser markup.
#[path = "../examples/robots/world_scene/mode.rs"]
mod mode;

/// CLI and markup feed the same typed selector before any renderer starts.
#[path = "../examples/robots/world_scene/selector.rs"]
mod selector;

/// An absent markup attribute uses the default; an explicit invalid value fails.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn markup_mode_rejects_explicit_invalid_values() {
    assert_eq!(
        selector::from_markup(None).expect("absent attribute"),
        mode::Mode::Standing
    );
    assert_eq!(
        selector::from_markup(Some("shared-world")).expect("world attribute"),
        mode::Mode::SharedWorld
    );
    assert_eq!(
        selector::from_markup(Some("standing")).expect("standing attribute"),
        mode::Mode::Standing
    );
    assert!(selector::from_markup(Some("")).is_err());
    assert!(selector::from_markup(Some("world")).is_err());
}

/// Native argument parsing preserves the default and rejects unsupported explicit modes.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_cli_uses_the_closed_mode() {
    use clap::Parser;
    assert_eq!(
        selector::Options::try_parse_from(["viewer"])
            .expect("default arguments")
            .scene,
        mode::Mode::Standing
    );
    assert_eq!(
        selector::Options::try_parse_from(["viewer", "--scene", "shared-world"])
            .expect("world arguments")
            .scene,
        mode::Mode::SharedWorld
    );
    assert_eq!(
        selector::Options::try_parse_from(["viewer", "--scene", "standing"])
            .expect("standing arguments")
            .scene,
        mode::Mode::Standing
    );
    for value in ["", "world", "Shared-world", "shared-world "] {
        assert!(selector::Options::try_parse_from(["viewer", "--scene", value]).is_err());
    }
    assert!(selector::Options::try_parse_from(["viewer", "--scene"]).is_err());
    assert!(selector::Options::try_parse_from(["viewer", "--unsupported"]).is_err());
}

/// Both canonical modes parse and display exactly; no aliases or normalization apply.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn closed_modes_preserve_the_standing_default_and_reject_other_spellings() {
    assert_eq!(mode::Mode::default(), mode::Mode::Standing);
    for (text, expected) in [
        ("standing", mode::Mode::Standing),
        ("shared-world", mode::Mode::SharedWorld),
    ] {
        let parsed: mode::Mode = text.parse().expect("canonical scene mode");
        assert_eq!(parsed, expected);
        assert_eq!(parsed.to_string(), text);
    }
    for text in [
        "",
        "world",
        "shared_world",
        "Shared-world",
        "SHARED-WORLD",
        " standing",
        "standing ",
    ] {
        let error = text
            .parse::<mode::Mode>()
            .expect_err("unsupported explicit mode");
        assert_eq!(error.to_string(), "scene must be standing or shared-world");
    }
}
