//! Browser worker isolates simulation and CPU autodiff from rendering and controls.
use bevy_gym as _;
use serde as _;
#[cfg(all(test, not(target_arch = "wasm32")))]
use tokio as _;
#[cfg(not(target_arch = "wasm32"))]
use {bevy_gym_browser as _, serde_json as _};

#[cfg(not(target_arch = "wasm32"))]
const fn main() {}

#[cfg(target_arch = "wasm32")]
fn main() {
    use bevy_gym_browser::{AdvanceSteps, Command, Session};
    use getrandom as _;
    use wasm_bindgen::prelude::*;
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

    console_error_panic_hook::set_once();
    let scope: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let responder = scope.clone();
    let mut session: Option<Session> = None;
    let listener = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
        let response = (|| -> Result<serde_json::Value, String> {
            let text = event
                .data()
                .as_string()
                .ok_or("worker requires a JSON string")?;
            // A full DQN policy is below 32 KiB; cap untrusted uploads before deserialization.
            if text.len() > 1_048_576 {
                return Err("worker message exceeds 1 MiB".into());
            }
            let command: Command =
                serde_json::from_str(&text).map_err(|error| error.to_string())?;
            match command {
                Command::StartTraining { seed } => {
                    session =
                        Some(Session::train(u64::from(seed)).map_err(|error| error.to_string())?);
                    Ok(serde_json::json!({"event": "started"}))
                }
                Command::StartInference { bytes, seed } => {
                    session = Some(
                        Session::inference(bytes, u64::from(seed))
                            .map_err(|error| error.to_string())?,
                    );
                    Ok(serde_json::json!({"event": "started"}))
                }
                Command::Advance { steps } => {
                    let steps = AdvanceSteps::try_from(steps).map_err(|error| error.to_string())?;
                    let snapshot = session
                        .as_mut()
                        .ok_or("start a session before advancing")?
                        .advance(steps)
                        .map_err(|error| error.to_string())?;
                    Ok(serde_json::json!({"event": "snapshot", "snapshot": snapshot}))
                }
                Command::Export => {
                    let bytes = session
                        .as_ref()
                        .ok_or("start a session before exporting")?
                        .export_policy()
                        .map_err(|error| error.to_string())?;
                    Ok(serde_json::json!({"event": "policy", "bytes": bytes}))
                }
            }
        })()
        .unwrap_or_else(|error| serde_json::json!({"event": "error", "message": error}));
        if let Err(error) = responder.post_message(&JsValue::from_str(&response.to_string())) {
            wasm_bindgen::throw_val(error);
        }
    });
    scope.set_onmessage(Some(listener.as_ref().unchecked_ref()));
    listener.forget();
    // The host waits for this message before sending work; WASM initialization is asynchronous.
    if let Err(error) = scope.post_message(&JsValue::from_str(r#"{"event":"ready"}"#)) {
        wasm_bindgen::throw_val(error);
    };
}
