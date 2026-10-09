//! Exercise JSON requests at the same boundary used by the browser worker.

#[path = "../src/curriculum.rs"]
mod curriculum;
#[path = "../../examples/robots/learning/mod.rs"]
mod learning;
#[path = "../../examples/robots/curriculum/lesson.rs"]
mod lesson;
#[path = "../src/protocol.rs"]
mod protocol;
#[path = "../src/session.rs"]
mod session;

use protocol::respond;
use session::Session;

#[test]
fn requests_train_one_batch_and_export_reloadable_weights() {
    let mut session = Session::default();
    assert_eq!(
        respond(&mut session, r#"{"command":"advance"}"#)["code"],
        "not_started"
    );
    let started = respond(&mut session, r#"{"command":"start","seed":7}"#);
    assert_eq!(
        started,
        serde_json::json!({"event":"started", "seed":7,
        "updates":0,"transitions":0,"update_limit":260,
        "actor_learning_rate":0.0003,"critic_learning_rate":0.001})
    );
    let initial = respond(&mut session, r#"{"command":"export"}"#);
    let progress = respond(&mut session, r#"{"command":"advance"}"#);
    assert_eq!(progress["event"], "progress");
    assert_eq!(progress["updates"], 1);
    assert_eq!(progress["transitions"], 512);
    assert_eq!(progress["status"], "training");
    assert!(
        progress["optimizer_steps"]
            .as_u64()
            .expect("optimizer count")
            > 0
    );
    assert_eq!(progress["actor_learning_rate"], 0.0003);
    assert_eq!(progress["critic_learning_rate"], 0.001);
    let exported = respond(&mut session, r#"{"command":"export"}"#);
    assert_ne!(initial["bytes"], exported["bytes"]);
    let bytes = serde_json::from_value(exported["bytes"].clone()).expect("byte array");
    learning::load_policy(bytes).expect("viewer-compatible checkpoint");
    let evaluation = respond(
        &mut session,
        &serde_json::json!({
            "command": "evaluate", "bytes": exported["bytes"]
        })
        .to_string(),
    );
    assert_eq!(evaluation["event"], "evaluation");
    assert_eq!(evaluation["episodes"].as_array().expect("scores").len(), 5);
    assert_eq!(respond(&mut session, r#"{"command":"export"}"#), exported);
}

#[test]
fn malformed_requests_preserve_an_existing_run() {
    let mut session = Session::default();
    assert_eq!(
        respond(&mut session, r#"{"command":"export"}"#)["code"],
        "not_started"
    );
    respond(&mut session, r#"{"command":"start","seed":7}"#);
    let before = respond(&mut session, r#"{"command":"export"}"#);
    for request in [
        "",
        "{",
        "null",
        "[]",
        "{}",
        r#"{"command":"unknown"}"#,
        r#"{"command":"advance","extra":1}"#,
        r#"{"command":"export","extra":1}"#,
        r#"{"command":"start"}"#,
        r#"{"command":"start","seed":null}"#,
        r#"{"command":"start","seed":-1}"#,
        r#"{"command":"start","seed":4294967296}"#,
        r#"{"command":"start","seed":1.5}"#,
        r#"{"command":"start","seed":"7"}"#,
        r#"{"command":"start","seed":7,"extra":0}"#,
        r#"{"command":"start","seed":7,"seed":8}"#,
        r#"{"command":"export","command":"advance"}"#,
        r#"{"command":"evaluate"}"#,
        r#"{"command":"evaluate","bytes":null}"#,
        r#"{"command":"evaluate","bytes":[-1]}"#,
        r#"{"command":"evaluate","bytes":[256]}"#,
        r#"{"command":"evaluate","bytes":[1.5]}"#,
        r#"{"command":"evaluate","bytes":["0"]}"#,
        r#"{"command":"evaluate","bytes":[],"bytes":[]}"#,
        r#"{"command":"evaluate","bytes":[],"extra":0}"#,
    ] {
        assert_eq!(
            respond(&mut session, request)["code"],
            "invalid_request",
            "{request}"
        );
        assert_eq!(respond(&mut session, r#"{"command":"export"}"#), before);
    }
    for bytes in [vec![], vec![0], vec![193], vec![255]] {
        let request = serde_json::json!({"command": "evaluate", "bytes": bytes}).to_string();
        assert_eq!(respond(&mut session, &request)["code"], "checkpoint");
        assert_eq!(respond(&mut session, r#"{"command":"export"}"#), before);
    }
    assert_eq!(
        respond(&mut session, r#"{"command":"advance"}"#)["updates"],
        1
    );
}

#[test]
fn malformed_curriculum_requests_preserve_an_existing_run() {
    let mut session = Session::default();
    respond(&mut session, r#"{"command":"start","seed":7}"#);
    let before = respond(&mut session, r#"{"command":"export"}"#);
    for request in [
        r#"{"command":"start_curriculum"}"#,
        r#"{"command":"start_curriculum","seed":null}"#,
        r#"{"command":"start_curriculum","seed":-1}"#,
        r#"{"command":"start_curriculum","seed":4294967296}"#,
        r#"{"command":"start_curriculum","seed":1.5}"#,
        r#"{"command":"start_curriculum","seed":"7"}"#,
        r#"{"command":"start_curriculum","seed":7,"extra":0}"#,
        r#"{"command":"start_curriculum","seed":7,"seed":8}"#,
    ] {
        assert_eq!(
            respond(&mut session, request)["code"],
            "invalid_request",
            "{request}"
        );
        assert_eq!(respond(&mut session, r#"{"command":"export"}"#), before);
    }
    assert_eq!(
        respond(&mut session, r#"{"command":"advance"}"#)["updates"],
        1
    );
}

#[test]
fn message_size_checks_utf8_bytes_at_the_inclusive_boundary() {
    let mut session = Session::default();
    let mut request = String::from(r#"{"command":"export"}"#);
    request.push_str(&" ".repeat(1_048_576 - request.len()));
    assert_eq!(respond(&mut session, &request)["code"], "not_started");
    request.push(' ');
    assert_eq!(
        respond(&mut session, &request)["message"],
        "Worker message exceeds 1 MiB."
    );
    let multibyte = "é".repeat(524_289);
    assert_eq!(
        respond(&mut session, &multibyte)["message"],
        "Worker message exceeds 1 MiB."
    );
    assert_eq!(
        respond(&mut session, r#"{"command":"export"}"#)["code"],
        "not_started"
    );
}

#[test]
fn seed_boundaries_restart_the_count_and_evaluation_preserves_idle_state() {
    let mut session = Session::default();
    let bytes = include_bytes!("../../assets/robots/recovery.mpk");
    let request = serde_json::json!({"command": "evaluate", "bytes": bytes.as_slice()}).to_string();
    let result = respond(&mut session, &request);
    assert_eq!(result["event"], "evaluation");
    assert_eq!(result["episodes"][4]["seed"], "18446744073709551615");
    assert!(result["episodes"]
        .as_array()
        .expect("episodes")
        .iter()
        .all(|episode| episode["survived"] == true));
    assert_eq!(
        respond(&mut session, r#"{"command":"advance"}"#)["code"],
        "not_started"
    );
    for seed in [0, u32::MAX] {
        let request = serde_json::json!({"command": "start", "seed": seed}).to_string();
        let started = respond(&mut session, &request);
        assert_eq!(started["event"], "started");
        assert_eq!(started["seed"], seed);
        assert_eq!(started["update_limit"], 260);
        assert_eq!(started["updates"], 0);
        assert_eq!(
            respond(&mut session, r#"{"command":"advance"}"#)["updates"],
            1
        );
    }
}

#[test]
fn curriculum_starts_calm_and_preserves_the_direct_recovery_recipe() {
    let mut session = Session::default();
    let started = respond(&mut session, r#"{"command":"start_curriculum","seed":7}"#);
    assert_eq!(started["event"], "started");
    assert_eq!(started["update_limit"], 1200);
    assert_eq!(started["curriculum"]["lesson"], "hover");
    assert_eq!(started["curriculum"]["lesson_updates"], 0);
    let progress = respond(&mut session, r#"{"command":"advance"}"#);
    assert_eq!(progress["updates"], 1);
    assert_eq!(progress["curriculum"]["lesson_updates"], 1);
    assert_eq!(progress["curriculum"]["lesson_limit"], 600);
    assert!(progress["curriculum"]["evaluation"].is_null());
    assert_eq!(progress["status"], "training");
    let direct = respond(&mut session, r#"{"command":"start","seed":7}"#);
    assert_eq!(direct["update_limit"], 260);
    assert!(direct.get("curriculum").is_none());
    assert!(respond(&mut session, r#"{"command":"advance"}"#)
        .get("curriculum")
        .is_none());
}
