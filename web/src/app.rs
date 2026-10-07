//! Bevy application lifecycle and browser control wiring.

use std::str::FromStr;
use std::sync::mpsc::{self, Receiver, Sender};

use bevy::prelude::*;
use bevy::window::{PresentMode, WindowResolution};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::{Document, Event, HtmlInputElement, HtmlSelectElement};

use crate::checkpoint::{
    fetch_asset, read_upload, read_upload_sidecar, CheckpointError, UploadedRecord,
};
use crate::ecosystem::{draw_world, setup_camera, ActiveSession, InferenceSession};
use crate::manifest::{
    CheckpointAsset, CheckpointManifest, CheckpointSidecar, PolicyRole, Stage, StageManifest,
};

/// Manifest embedded into the hashed WASM asset.
const CHECKPOINT_MANIFEST: &str = include_str!("../checkpoints/manifest.json");

/// Successfully fetched curated policy pair.
struct CheckpointSet {
    /// Validated bunny record bytes.
    bunny: Vec<u8>,
    /// Validated fox record bytes where required.
    fox: Option<Vec<u8>>,
}

/// Browser callback result delivered at the Bevy update boundary.
enum BrowserEvent {
    /// Curated assets completed for one route generation.
    DefaultsLoaded {
        generation: u64,
        stage: Stage,
        result: Result<CheckpointSet, CheckpointError>,
    },
    /// One local file completed its asynchronous read.
    UploadLoaded {
        route: Box<str>,
        role: PolicyRole,
        result: Result<UploadedRecord, CheckpointError>,
    },
    /// One local configuration sidecar completed its asynchronous read.
    SidecarLoaded {
        route: Box<str>,
        role: PolicyRole,
        result: Result<CheckpointSidecar, CheckpointError>,
    },
    /// Restore the curated policy for the selected route.
    ResetDefaults,
    /// End the current episode and clear recurrent state.
    Restart,
    /// Toggle simulation advancement.
    TogglePause,
    /// Change the simulation playback multiplier.
    SetSpeed(f32),
}

/// Browser-local record and optional provenance sidecar for one policy role.
#[derive(Default)]
struct PendingUpload {
    /// Decoded file bytes and untrusted local identity.
    record: Option<UploadedRecord>,
    /// Typed sidecar awaiting exact route validation.
    sidecar: Option<CheckpointSidecar>,
}

/// Repeatable button action mapped into one browser event.
#[derive(Clone, Copy)]
enum ButtonAction {
    /// Restore curated checkpoints.
    ResetDefaults,
    /// Start a fresh deterministic episode.
    Restart,
    /// Toggle playback advancement.
    TogglePause,
}

impl ButtonAction {
    /// Create a fresh event for one click.
    const fn event(self) -> BrowserEvent {
        match self {
            Self::ResetDefaults => BrowserEvent::ResetDefaults,
            Self::Restart => BrowserEvent::Restart,
            Self::TogglePause => BrowserEvent::TogglePause,
        }
    }
}

/// Browser-side lifecycle state retained outside Bevy's Send resource graph.
struct DemoRuntime {
    /// Validated static manifest.
    manifest: CheckpointManifest,
    /// Current fragment route.
    stage: Stage,
    /// Monotonic route or reset generation.
    generation: u64,
    /// Browser callback sender.
    sender: Sender<BrowserEvent>,
    /// Browser callback receiver.
    receiver: Receiver<BrowserEvent>,
    /// Browser-local bunny record and optional sidecar.
    pending_bunny: PendingUpload,
    /// Browser-local fox record and optional sidecar.
    pending_fox: PendingUpload,
    /// Whether fixed-step simulation advancement is paused.
    paused: bool,
    /// Playback multiplier applied to wall-clock delta.
    speed: f32,
    /// Current short policy-source label.
    policy_source: Box<str>,
    /// Route-selection time used for first-frame evidence.
    route_started_ms: f64,
    /// Whether the first active post-render frame was recorded.
    first_render_recorded: bool,
    /// Previous active post-render timestamp.
    last_frame_ms: Option<f64>,
    /// Accumulated active frame duration in milliseconds.
    frame_time_sum_ms: f64,
    /// Number of active frame intervals in the accumulated duration.
    frame_time_samples: u32,
}

impl DemoRuntime {
    /// Construct the initial route and start its curated fetch.
    fn new(manifest: CheckpointManifest) -> Result<Self, Box<str>> {
        let stage = route_stage();
        let (sender, receiver) = mpsc::channel();
        let runtime = Self {
            manifest,
            stage,
            generation: 1,
            sender,
            receiver,
            pending_bunny: PendingUpload::default(),
            pending_fox: PendingUpload::default(),
            paused: false,
            speed: 1.0,
            policy_source: "curated default".into(),
            route_started_ms: performance_now().unwrap_or(0.0),
            first_render_recorded: false,
            last_frame_ms: None,
            frame_time_sum_ms: 0.0,
            frame_time_samples: 0,
        };
        runtime.load_defaults()?;
        Ok(runtime)
    }

    /// Start an integrity-checked fetch for the selected route.
    fn load_defaults(&self) -> Result<(), Box<str>> {
        let entry = self
            .manifest
            .stage(self.stage)
            .map_err(|error| error.to_string().into_boxed_str())?
            .clone();
        show_loading(&entry);
        fetch_defaults(self.generation, entry, self.sender.clone());
        Ok(())
    }

    /// Select a new fragment route and clear route-specific upload state.
    fn select_stage(&mut self, stage: Stage) -> Result<(), Box<str>> {
        self.stage = stage;
        self.generation = self.generation.saturating_add(1);
        self.pending_bunny = PendingUpload::default();
        self.pending_fox = PendingUpload::default();
        self.paused = false;
        self.policy_source = "curated default".into();
        self.route_started_ms = performance_now().unwrap_or(0.0);
        self.first_render_recorded = false;
        self.last_frame_ms = None;
        self.frame_time_sum_ms = 0.0;
        self.frame_time_samples = 0;
        self.load_defaults()
    }
}

/// Start the browser-owned Bevy renderer.
pub(super) fn run() {
    let result = CheckpointManifest::from_json(CHECKPOINT_MANIFEST)
        .map_err(|error| error.to_string().into_boxed_str())
        .and_then(DemoRuntime::new);
    let runtime = match result {
        Ok(runtime) => runtime,
        Err(error) => {
            show_failure(&error);
            return;
        }
    };
    install_controls(runtime.sender.clone());

    App::new()
        .insert_resource(ClearColor(Color::srgb_u8(12, 17, 14)))
        .insert_non_send_resource(runtime)
        .insert_non_send_resource(ActiveSession(None))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                canvas: Some("#ecosystem-canvas".into()),
                fit_canvas_to_parent: true,
                present_mode: PresentMode::AutoVsync,
                resolution: WindowResolution::new(1280, 720),
                title: "bevy-gym ecosystem inference".into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup_camera)
        .add_systems(
            Update,
            (process_browser_state, advance_inference, update_metrics).chain(),
        )
        .add_systems(PostUpdate, (draw_world, record_render_metrics).chain())
        .run();
}

/// Poll route changes and apply completed browser events.
fn process_browser_state(
    mut runtime: NonSendMut<'_, DemoRuntime>,
    mut active: NonSendMut<'_, ActiveSession>,
) {
    let route = route_stage();
    if route != runtime.stage {
        active.0 = None;
        if let Err(error) = runtime.select_stage(route) {
            show_failure(&error);
        }
    }

    loop {
        let event_result = runtime.receiver.try_recv();
        let Ok(event) = event_result else {
            break;
        };
        match event {
            BrowserEvent::DefaultsLoaded {
                generation,
                stage,
                result,
            } if generation == runtime.generation && stage == runtime.stage => {
                apply_defaults(&mut runtime, &mut active, result);
            }
            BrowserEvent::DefaultsLoaded { .. } => {}
            BrowserEvent::UploadLoaded {
                route,
                role,
                result,
            } if route.as_ref() == runtime.stage.as_key() => {
                apply_upload(&mut runtime, &mut active, role, result);
            }
            BrowserEvent::UploadLoaded { .. } => {}
            BrowserEvent::SidecarLoaded {
                route,
                role,
                result,
            } if route.as_ref() == runtime.stage.as_key() => {
                apply_sidecar(&mut runtime, &mut active, role, result);
            }
            BrowserEvent::SidecarLoaded { .. } => {}
            BrowserEvent::ResetDefaults => {
                active.0 = None;
                runtime.generation = runtime.generation.saturating_add(1);
                runtime.pending_bunny = PendingUpload::default();
                runtime.pending_fox = PendingUpload::default();
                runtime.policy_source = "curated default".into();
                if let Err(error) = runtime.load_defaults() {
                    show_failure(&error);
                }
            }
            BrowserEvent::Restart => {
                if let Some(session) = active.0.as_mut() {
                    if let Err(error) = session.restart() {
                        show_failure(&error.to_string());
                    }
                }
            }
            BrowserEvent::TogglePause => {
                runtime.paused = !runtime.paused;
                set_text(
                    "pause-playback",
                    if runtime.paused { "Resume" } else { "Pause" },
                );
            }
            BrowserEvent::SetSpeed(speed) => runtime.speed = speed.clamp(0.25, 4.0),
        }
    }
}

/// Advance the active simulation without allowing a large frame-time burst.
fn advance_inference(
    runtime: NonSend<'_, DemoRuntime>,
    mut active: NonSendMut<'_, ActiveSession>,
    time: Res<'_, Time>,
) {
    if runtime.paused {
        return;
    }
    let Some(session) = active.0.as_mut() else {
        return;
    };
    if let Err(error) = session.advance(time.delta_secs() * runtime.speed) {
        show_failure(&error.to_string());
        active.0 = None;
    }
}

/// Update lightweight text metrics after the simulation step.
fn update_metrics(runtime: NonSend<'_, DemoRuntime>, active: NonSend<'_, ActiveSession>) {
    let Some(session) = active.0.as_ref() else {
        return;
    };
    set_text("metric-step", &session.step().to_string());
    set_text("metric-living", &session.living_agents().to_string());
    set_text("metric-episode", &session.episode().to_string());
    set_text("metric-policy", &runtime.policy_source);
    set_text(
        "runtime-status",
        if runtime.paused {
            "Inference paused"
        } else {
            "WASM inference running"
        },
    );
}

/// Activate one fully validated curated policy set.
fn apply_defaults(
    runtime: &mut DemoRuntime,
    active: &mut ActiveSession,
    result: Result<CheckpointSet, CheckpointError>,
) {
    let decode_started_ms = performance_now();
    let decoded = result.and_then(|set| {
        InferenceSession::from_checkpoint_bytes(runtime.stage, set.bunny, set.fox)
            .map_err(|error| CheckpointError::Decode(error.to_string().into_boxed_str()))
    });
    if let (Some(started_ms), Some(finished_ms)) = (decode_started_ms, performance_now()) {
        set_performance_attribute("data-checkpoint-decode-ms", finished_ms - started_ms);
    }
    match decoded {
        Ok(session) => {
            active.0 = Some(session);
            runtime.policy_source = "curated default".into();
            show_ready(runtime);
        }
        Err(error) => show_failure(&error.to_string()),
    }
}

/// Validate and activate a local upload at an explicit episode boundary.
fn apply_upload(
    runtime: &mut DemoRuntime,
    active: &mut ActiveSession,
    role: PolicyRole,
    result: Result<UploadedRecord, CheckpointError>,
) {
    let record = match result {
        Ok(record) => record,
        Err(error) => {
            show_upload_rejection(&error.to_string());
            return;
        }
    };
    if !runtime.stage.requires_fox() && role == PolicyRole::Fox {
        show_upload_rejection("this environment does not use a fox policy");
        return;
    }
    match role {
        PolicyRole::Bunny => {
            runtime.pending_bunny.record = Some(record);
            if runtime.stage.requires_fox() && runtime.pending_fox.record.is_none() {
                set_text(
                    "bunny-upload-state",
                    "Validated bunny; waiting for fox policy.",
                );
            }
        }
        PolicyRole::Fox => {
            runtime.pending_fox.record = Some(record);
            if runtime.pending_bunny.record.is_none() {
                set_text(
                    "fox-upload-state",
                    "Validated fox; waiting for bunny policy.",
                );
            }
        }
    }
    try_activate_uploaded(runtime, active);
}

/// Validate one optional sidecar while preserving the last working policy.
fn apply_sidecar(
    runtime: &mut DemoRuntime,
    active: &mut ActiveSession,
    role: PolicyRole,
    result: Result<CheckpointSidecar, CheckpointError>,
) {
    let sidecar = match result {
        Ok(sidecar) => sidecar,
        Err(error) => {
            show_upload_rejection(&error.to_string());
            return;
        }
    };
    let Ok(entry) = runtime.manifest.stage(runtime.stage) else {
        show_upload_rejection("selected route is missing from the manifest");
        return;
    };
    if let Err(error) = sidecar.validate_for(entry) {
        show_upload_rejection(&error.to_string());
        return;
    }
    match role {
        PolicyRole::Bunny => runtime.pending_bunny.sidecar = Some(sidecar),
        PolicyRole::Fox if runtime.stage.requires_fox() => {
            runtime.pending_fox.sidecar = Some(sidecar);
        }
        PolicyRole::Fox => {
            show_upload_rejection("this environment does not use a fox policy");
            return;
        }
    }
    try_activate_uploaded(runtime, active);
}

/// Decode a complete upload set before committing either actor to the session.
fn try_activate_uploaded(runtime: &mut DemoRuntime, active: &mut ActiveSession) {
    let Some(session) = active.0.as_mut() else {
        show_upload_rejection("wait for the curated checkpoint before uploading");
        return;
    };
    let Some(bunny) = runtime.pending_bunny.record.as_ref() else {
        return;
    };
    let fox = if runtime.stage.requires_fox() {
        let Some(fox) = runtime.pending_fox.record.as_ref() else {
            return;
        };
        Some(fox)
    } else {
        None
    };
    let sidecars_verified = runtime.pending_bunny.sidecar.is_some()
        && (!runtime.stage.requires_fox() || runtime.pending_fox.sidecar.is_some());
    let bunny_bytes = bunny.bytes.clone();
    let fox_bytes = fox.map(|record| record.bytes.clone());
    let record_label = upload_record_label(bunny, fox);
    match session.replace_checkpoint_bytes(bunny_bytes, fox_bytes) {
        Ok(()) => {
            runtime.policy_source = if sidecars_verified {
                "uploaded, sidecars verified".into()
            } else {
                "uploaded, metadata unverified".into()
            };
            set_text("checkpoint-status", "Ready");
            set_text("checkpoint-source", "Local browser upload");
            set_text(
                "checkpoint-qualification",
                if sidecars_verified {
                    "local sidecars verified"
                } else {
                    "unverified"
                },
            );
            set_text("checkpoint-record", &record_label);
            set_text("bunny-upload-state", "Uploaded policy is active.");
            if runtime.stage.requires_fox() {
                set_text("fox-upload-state", "Uploaded policy is active.");
            }
            clear_upload_rejection();
        }
        Err(error) => show_upload_rejection(&error.to_string()),
    }
}

/// Format untrusted local filenames with their computed digest prefixes.
fn upload_record_label(bunny: &UploadedRecord, fox: Option<&UploadedRecord>) -> String {
    let bunny_prefix = bunny.sha256.chars().take(12).collect::<String>();
    let mut label = format!("{} @ {bunny_prefix}", bunny.name);
    if let Some(fox) = fox {
        let fox_prefix = fox.sha256.chars().take(12).collect::<String>();
        label.push_str(&format!(" / {} @ {fox_prefix}", fox.name));
    }
    label
}

/// Fetch both required policies and emit one atomic completion event.
fn fetch_defaults(generation: u64, entry: StageManifest, sender: Sender<BrowserEvent>) {
    spawn_local(async move {
        let stage = entry.stage;
        let result = async {
            let bunny = fetch_asset(entry.bunny).await?;
            let fox = match entry.fox {
                Some(asset) => Some(fetch_asset(asset).await?),
                None => None,
            };
            Ok(CheckpointSet { bunny, fox })
        }
        .await;
        drop(sender.send(BrowserEvent::DefaultsLoaded {
            generation,
            stage,
            result,
        }));
    });
}

/// Install browser controls for the lifetime of this page.
fn install_controls(sender: Sender<BrowserEvent>) {
    install_button("reset-checkpoints", ButtonAction::ResetDefaults, &sender);
    install_button("restart-environment", ButtonAction::Restart, &sender);
    install_button("pause-playback", ButtonAction::TogglePause, &sender);
    install_upload("bunny-checkpoint", PolicyRole::Bunny, &sender);
    install_upload("fox-checkpoint", PolicyRole::Fox, &sender);
    install_sidecar_upload("bunny-checkpoint-config", PolicyRole::Bunny, &sender);
    install_sidecar_upload("fox-checkpoint-config", PolicyRole::Fox, &sender);

    let Some(document) = document() else {
        return;
    };
    let Some(select) = document
        .get_element_by_id("playback-speed")
        .and_then(|element| element.dyn_into::<HtmlSelectElement>().ok())
    else {
        return;
    };
    let speed_sender = sender.clone();
    let select_for_callback = select.clone();
    let callback = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Ok(speed) = select_for_callback.value().parse::<f32>() {
            drop(speed_sender.send(BrowserEvent::SetSpeed(speed)));
        }
    });
    drop(select.add_event_listener_with_callback("change", callback.as_ref().unchecked_ref()));
    // These callbacks live for the full page lifetime.
    callback.forget();
}

/// Install one page-lifetime button event.
fn install_button(id: &str, action: ButtonAction, sender: &Sender<BrowserEvent>) {
    let Some(element) = document().and_then(|document| document.get_element_by_id(id)) else {
        return;
    };
    let sender = sender.clone();
    let callback = Closure::<dyn FnMut(Event)>::new(move |_| {
        drop(sender.send(action.event()));
    });
    drop(element.add_event_listener_with_callback("click", callback.as_ref().unchecked_ref()));
    // These callbacks live for the full page lifetime.
    callback.forget();
}

/// Install one page-lifetime file input event.
fn install_upload(id: &str, role: PolicyRole, sender: &Sender<BrowserEvent>) {
    let Some(input) = document()
        .and_then(|document| document.get_element_by_id(id))
        .and_then(|element| element.dyn_into::<HtmlInputElement>().ok())
    else {
        return;
    };
    let sender = sender.clone();
    let input_for_callback = input.clone();
    let callback = Closure::<dyn FnMut(Event)>::new(move |_| {
        let Some(file) = input_for_callback.files().and_then(|files| files.get(0)) else {
            return;
        };
        let route: Box<str> = route_stage().as_key().into();
        let sender = sender.clone();
        spawn_local(async move {
            let result = read_upload(file).await;
            drop(sender.send(BrowserEvent::UploadLoaded {
                route,
                role,
                result,
            }));
        });
    });
    drop(input.add_event_listener_with_callback("change", callback.as_ref().unchecked_ref()));
    // These callbacks live for the full page lifetime.
    callback.forget();
}

/// Install one page-lifetime configuration sidecar input event.
fn install_sidecar_upload(id: &str, role: PolicyRole, sender: &Sender<BrowserEvent>) {
    let Some(input) = document()
        .and_then(|document| document.get_element_by_id(id))
        .and_then(|element| element.dyn_into::<HtmlInputElement>().ok())
    else {
        return;
    };
    let sender = sender.clone();
    let input_for_callback = input.clone();
    let callback = Closure::<dyn FnMut(Event)>::new(move |_| {
        let Some(file) = input_for_callback.files().and_then(|files| files.get(0)) else {
            return;
        };
        let route: Box<str> = route_stage().as_key().into();
        let sender = sender.clone();
        spawn_local(async move {
            let result = read_upload_sidecar(file).await;
            drop(sender.send(BrowserEvent::SidecarLoaded {
                route,
                role,
                result,
            }));
        });
    });
    drop(input.add_event_listener_with_callback("change", callback.as_ref().unchecked_ref()));
    // These callbacks live for the full page lifetime.
    callback.forget();
}

/// Parse the fragment route with forage as the deterministic fallback.
fn route_stage() -> Stage {
    let route = web_sys::window()
        .and_then(|window| window.location().hash().ok())
        .and_then(|hash| hash.strip_prefix("#/ecosystem/").map(str::to_owned));
    route
        .as_deref()
        .and_then(|key| Stage::from_str(key).ok())
        .unwrap_or(Stage::Survival)
}

/// Display a route's manifest evidence while its assets load.
fn show_loading(entry: &StageManifest) {
    set_route_labels(entry.stage);
    set_text("runtime-status", "Loading checkpoint assets");
    set_text("checkpoint-status", "Fetching and validating");
    set_text("checkpoint-qualification", &entry.qualification.to_string());
    set_text("checkpoint-source", &entry.bunny.source_run);
    set_text("checkpoint-record", &curated_record_label(entry));
    set_text("metric-policy", "loading");
    set_text("bunny-upload-state", "Loading curated bunny policy.");
    if entry.stage.requires_fox() {
        set_text("fox-upload-state", "Loading curated fox policy.");
    }
    clear_upload_rejection();
    set_hidden("canvas-message", false);
    set_text("canvas-message", "Fetching the curated checkpoint…");
}

/// Display a ready curated session and its evidence classification.
fn show_ready(runtime: &DemoRuntime) {
    let Ok(entry) = runtime.manifest.stage(runtime.stage) else {
        return;
    };
    set_text("runtime-status", "WASM inference running");
    set_text("checkpoint-status", "Ready");
    set_text("checkpoint-qualification", &entry.qualification.to_string());
    set_text("checkpoint-source", &entry.bunny.source_run);
    set_text("checkpoint-record", &curated_record_label(entry));
    set_text("bunny-upload-state", "Curated bunny policy ready.");
    if runtime.stage.requires_fox() {
        set_text("fox-upload-state", "Curated fox policy ready.");
    }
    clear_upload_rejection();
    set_hidden("canvas-message", true);
}

/// Format curated content-addressed records for the visible provenance surface.
fn curated_record_label(entry: &StageManifest) -> String {
    let bunny = asset_record_label(&entry.bunny);
    entry.fox.as_ref().map_or(bunny.clone(), |fox| {
        format!("{bunny} / {}", asset_record_label(fox))
    })
}

/// Format one static record filename with its manifest digest prefix.
fn asset_record_label(asset: &CheckpointAsset) -> String {
    let filename = asset
        .path
        .as_ref()
        .rsplit('/')
        .next()
        .unwrap_or(asset.path.as_ref());
    let prefix = asset.sha256.as_ref().chars().take(12).collect::<String>();
    format!("{filename} @ {prefix}")
}

/// Display one visible terminal load or inference failure.
fn show_failure(message: &str) {
    set_text("runtime-status", "Inference failed");
    set_text("checkpoint-status", "Failed");
    set_text("canvas-message", message);
    set_hidden("canvas-message", false);
}

/// Report a recoverable local-file rejection without stopping inference.
fn show_upload_rejection(message: &str) {
    set_text("checkpoint-status", "Upload rejected");
    set_text("upload-error", message);
    set_hidden("upload-error", false);
}

/// Clear the recoverable upload diagnostic after a successful state change.
fn clear_upload_rejection() {
    set_text("upload-error", "");
    set_hidden("upload-error", true);
}

/// Update stage navigation, labels, and predator-only controls.
fn set_route_labels(stage: Stage) {
    set_text("environment-title", stage.title());
    let position = stage.position();
    set_text(
        "environment-index",
        &format!("Environment {position:02} / 06"),
    );
    set_hidden("fox-upload-group", !stage.requires_fox());
    for item in Stage::ALL {
        let selector = format!("a[data-stage=\"{}\"]", item.as_key());
        if let Some(element) =
            document().and_then(|document| document.query_selector(&selector).ok().flatten())
        {
            if item == stage {
                drop(element.set_attribute("aria-current", "page"));
            } else {
                drop(element.remove_attribute("aria-current"));
            }
        }
    }
}

/// Return the current browser document.
fn document() -> Option<Document> {
    web_sys::window().and_then(|window| window.document())
}

/// Return the browser monotonic clock in milliseconds when available.
fn performance_now() -> Option<f64> {
    web_sys::window()
        .and_then(|window| window.performance())
        .map(|performance| performance.now())
}

/// Store one measured browser value on the root element for automated evidence.
fn set_performance_attribute(name: &str, value_ms: f64) {
    if !value_ms.is_finite() || value_ms < 0.0 {
        return;
    }
    if let Some(root) = document().and_then(|document| document.document_element()) {
        drop(root.set_attribute(name, &format!("{value_ms:.3}")));
    }
}

/// Record first-render latency and a steady active-frame mean after drawing.
fn record_render_metrics(
    mut runtime: NonSendMut<'_, DemoRuntime>,
    active: NonSend<'_, ActiveSession>,
) {
    if active.0.is_none() || runtime.paused {
        runtime.last_frame_ms = None;
        return;
    }
    let Some(now_ms) = performance_now() else {
        return;
    };
    if !runtime.first_render_recorded {
        set_performance_attribute("data-first-render-ms", now_ms - runtime.route_started_ms);
        runtime.first_render_recorded = true;
        runtime.last_frame_ms = Some(now_ms);
        return;
    }
    if let Some(previous_ms) = runtime.last_frame_ms {
        let elapsed_ms = now_ms - previous_ms;
        if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
            runtime.frame_time_sum_ms += elapsed_ms;
            runtime.frame_time_samples = runtime.frame_time_samples.saturating_add(1);
            let mean_ms = runtime.frame_time_sum_ms / f64::from(runtime.frame_time_samples);
            set_performance_attribute("data-steady-frame-ms", mean_ms);
        }
    }
    runtime.last_frame_ms = Some(now_ms);
}

/// Replace one text node when the static shell contains it.
fn set_text(id: &str, value: &str) {
    if let Some(element) = document().and_then(|document| document.get_element_by_id(id)) {
        element.set_text_content(Some(value));
    }
}

/// Apply the semantic hidden attribute when the static shell contains an element.
fn set_hidden(id: &str, hidden: bool) {
    if let Some(element) = document().and_then(|document| document.get_element_by_id(id)) {
        if hidden {
            drop(element.set_attribute("hidden", ""));
        } else {
            drop(element.remove_attribute("hidden"));
        }
    }
}
