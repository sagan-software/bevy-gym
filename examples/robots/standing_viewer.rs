//! Inspect frozen PPO inference in standing and shared world scenes.
#[cfg(all(test, target_arch = "wasm32"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
/// Licensed source mesh alignment shared with the single-drone scenes.
mod drone_model;
#[expect(
    dead_code,
    reason = "The scene loads frozen weights; trainer commands are separate examples."
)]
#[cfg_attr(not(test), path = "learning/inference.rs")]
#[cfg_attr(test, path = "learning/mod.rs")]
#[cfg_attr(
    test,
    expect(
        unused_imports,
        reason = "Shared collector tests need the complete training facade."
    )
)]
mod learning;
/// Explicit scene vocabulary validated before policy loading.
#[path = "world_scene/mode.rs"]
mod mode;
/// Native CLI and browser markup mode boundary.
#[path = "world_scene/selector.rs"]
mod selector;
#[expect(
    dead_code,
    reason = "Frozen playback shares the standing model and encoding with native training."
)]
mod standing;
mod standing_scene;
/// Six physical robots with independently owned frozen-policy memory.
#[path = "world_scene/viewer.rs"]
mod world_viewer;

/// Render physical observations; authored animations never choose living motion.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use clap::Parser;
    render(selector::Options::parse().scene);
}

/// wasm-bindgen initializes the browser through the fallible start function.
#[cfg(target_arch = "wasm32")]
fn main() {}

/// Reject initialization before policy loading when browser markup is invalid.
/// wasm-bindgen 0.2.121 accepts `Result<(), JsValue>` on its private start function.
/// <https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.121/guide/src/reference/attributes/on-rust-exports/start.md>.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start, private)]
fn start() -> Result<(), wasm_bindgen::JsValue> {
    render(selected_mode()?);
    Ok(())
}

/// Dispatch a validated scene without changing either scene's actor path.
fn render(mode: mode::Mode) {
    match mode {
        mode::Mode::Standing => standing_scene::run(),
        mode::Mode::SharedWorld => world_viewer::run(),
    }
}

/// Reject missing browser surfaces and unsupported explicit scene attributes.
#[cfg(target_arch = "wasm32")]
fn selected_mode() -> Result<mode::Mode, wasm_bindgen::JsValue> {
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("viewer document unavailable"))?;
    let canvas = document
        .get_element_by_id("drone-canvas")
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("viewer canvas unavailable"))?;
    let value = canvas.get_attribute("data-scene");
    selector::from_markup(value.as_deref())
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use wasm_bindgen_test::wasm_bindgen_test;

    /// Missing canvas errors cross the JavaScript initialization boundary.
    #[wasm_bindgen_test]
    fn missing_canvas_has_a_javascript_error() {
        let error = super::selected_mode().expect_err("the test page has no viewer canvas");
        assert_eq!(
            error.as_string().as_deref(),
            Some("viewer canvas unavailable")
        );
    }

    /// Present browser markup retains absence defaults and rejects explicit invalid modes.
    #[wasm_bindgen_test]
    fn browser_canvas_validates_scene_before_loading() {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document.create_element("canvas").unwrap();
        canvas.set_id("drone-canvas");
        document
            .document_element()
            .unwrap()
            .append_child(&canvas)
            .unwrap();
        assert_eq!(super::selected_mode().unwrap(), super::mode::Mode::Standing);
        canvas.set_attribute("data-scene", "shared-world").unwrap();
        assert_eq!(
            super::selected_mode().unwrap(),
            super::mode::Mode::SharedWorld
        );
        canvas.set_attribute("data-scene", "unsupported").unwrap();
        let error = super::selected_mode().expect_err("unsupported scene must stop startup");
        canvas.remove();
        assert_eq!(
            error.as_string().as_deref(),
            Some("scene must be standing or shared-world")
        );
    }
}
