//! Dedicated worker for the recovery lesson's existing CPU trainer.

#[cfg(target_arch = "wasm32")]
mod curriculum;
#[cfg(target_arch = "wasm32")]
#[path = "../../examples/robots/learning/mod.rs"]
mod learning;
#[cfg(target_arch = "wasm32")]
#[path = "../../examples/robots/curriculum/lesson.rs"]
mod lesson;
#[cfg(target_arch = "wasm32")]
mod protocol;
#[cfg(target_arch = "wasm32")]
mod session;

/// Native behavior is exercised through the portable protocol integration test.
#[cfg(not(target_arch = "wasm32"))]
const fn main() {}

/// Install the listener before announcing readiness to the browser host.
#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::prelude::*;
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

    console_error_panic_hook::set_once();
    let scope: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let responder = scope.clone();
    let mut session = session::Session::default();
    let listener = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
        let response = event.data().as_string().map_or_else(
            || {
                protocol::failure(session::Failure::new(
                    session::FailureKind::InvalidRequest,
                    "Worker requires a JSON string.",
                ))
            },
            |text| protocol::respond(&mut session, &text),
        );
        if let Err(error) = responder.post_message(&JsValue::from_str(&response.to_string())) {
            wasm_bindgen::throw_val(error);
        }
    });
    scope.set_onmessage(Some(listener.as_ref().unchecked_ref()));
    // The worker owns its callback until the browser host terminates it.
    listener.forget();
    let sent = scope.post_message(&JsValue::from_str(r#"{"event":"ready","protocol":1}"#));
    if let Err(error) = sent {
        wasm_bindgen::throw_val(error);
    }
}
