//! Browser worker isolates simulation and CPU autodiff from rendering and controls.
use bevy_gym as _;
#[cfg(any(target_arch = "wasm32", test))]
use bevy_gym_browser::{AdvanceSteps, Command, Session};
use serde as _;
#[cfg(all(test, not(target_arch = "wasm32")))]
use tokio as _;
#[cfg(not(target_arch = "wasm32"))]
use {bevy_gym_browser as _, serde_json as _};

#[cfg(not(target_arch = "wasm32"))]
const fn main() {}

#[cfg(target_arch = "wasm32")]
fn main() {
    use getrandom as _;
    use wasm_bindgen::prelude::*;
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

    console_error_panic_hook::set_once();
    let scope: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let responder = scope.clone();
    let mut session: Option<Session> = None;
    let listener = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
        let response = event.data().as_string().map_or_else(
            || serde_json::json!({"event": "error", "message": "worker requires a JSON string"}),
            |text| respond(&mut session, &text),
        );
        if let Err(error) = responder.post_message(&JsValue::from_str(&response.to_string())) {
            wasm_bindgen::throw_val(error);
        }
    });
    scope.set_onmessage(Some(listener.as_ref().unchecked_ref()));
    listener.forget();
    // The host waits for this message before sending work; WASM initialization is asynchronous.
    let ready = JsValue::from_str(r#"{"event":"ready","protocol":1}"#);
    let sent = scope.post_message(&ready);
    if let Err(error) = sent {
        wasm_bindgen::throw_val(error);
    }
}

/// Parse a bounded request and encode either its successful response or diagnostic.
#[cfg(any(target_arch = "wasm32", test))]
fn respond(session: &mut Option<Session>, text: &str) -> serde_json::Value {
    // A full DQN policy is below 32 KiB; reject oversized uploads before decoding.
    if text.len() > 1_048_576 {
        return serde_json::json!({"event":"error", "message":"worker message exceeds 1 MiB"});
    }
    serde_json::from_str::<Command>(text)
        .map_err(|error| error.to_string())
        .and_then(|command| handle_command(session, command))
        .unwrap_or_else(|error| serde_json::json!({"event":"error", "message":error}))
}

/// Execute one FIFO command, replacing a session only after construction succeeds.
#[cfg(any(target_arch = "wasm32", test))]
fn handle_command(
    session: &mut Option<Session>,
    command: Command,
) -> Result<serde_json::Value, String> {
    match command {
        Command::StartTraining { seed } => {
            *session = Some(Session::train(u64::from(seed)).map_err(|error| error.to_string())?);
            Ok(serde_json::json!({"event":"started"}))
        }
        Command::StartInference { bytes, seed } => {
            *session = Some(
                Session::inference(bytes, u64::from(seed)).map_err(|error| error.to_string())?,
            );
            Ok(serde_json::json!({"event":"started"}))
        }
        Command::Advance { steps } => {
            let steps = AdvanceSteps::try_from(steps).map_err(|error| error.to_string())?;
            let snapshot = session
                .as_mut()
                .ok_or("start a session before advancing")?
                .advance(steps)
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({"event":"snapshot", "snapshot":snapshot}))
        }
        Command::Export => {
            let bytes = session
                .as_ref()
                .ok_or("start a session before exporting")?
                .export_policy()
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({"event":"policy", "bytes":bytes}))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::respond;

    #[test]
    fn invalid_messages_and_unstarted_commands_return_errors_without_state() {
        let mut session = None;
        for message in [
            "not JSON",
            r#"{"command":"advance","steps":0}"#,
            r#"{"command":"advance","steps":1}"#,
            r#"{"command":"export"}"#,
        ] {
            assert_eq!(respond(&mut session, message)["event"], "error");
            assert!(session.is_none());
        }
        let oversized = " ".repeat(1_048_577);
        assert_eq!(
            respond(&mut session, &oversized)["message"],
            "worker message exceeds 1 MiB"
        );
    }

    #[test]
    fn worker_commands_round_trip_policy_and_preserve_state_on_rejected_import() {
        let mut session = None;
        assert_eq!(
            respond(&mut session, r#"{"command":"start_training","seed":42}"#)["event"],
            "started"
        );
        let first = respond(&mut session, r#"{"command":"advance","steps":1}"#);
        assert_eq!(first["snapshot"]["transitions"], 1);
        let exported = respond(&mut session, r#"{"command":"export"}"#);
        assert_eq!(exported["event"], "policy");
        let input =
            serde_json::json!({"command":"start_inference", "seed":42, "bytes":exported["bytes"]});
        assert_eq!(
            respond(&mut session, &input.to_string())["event"],
            "started"
        );
        assert_eq!(
            respond(
                &mut session,
                r#"{"command":"start_inference","seed":42,"bytes":[0]}"#
            )["event"],
            "error"
        );
        let result = respond(&mut session, r#"{"command":"advance","steps":1}"#);
        assert_eq!(result["snapshot"]["transitions"], 1);
        assert_eq!(result["snapshot"]["optimizer_steps"], 0);
    }
}
