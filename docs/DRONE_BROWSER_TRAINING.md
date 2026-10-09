# Drone browser training

Status: browser training and checkpoint playback verified, 2026-10-08.
A seed-7 browser run completed 260 updates and passed the frozen 32-episode
qualification on both native and browser targets.

## Contract and acceptance

The worker reuses the native recovery recipe and its eight 64-action lanes.
Each `advance` collects 512 transitions and performs one recurrent PPO update.
A run stops after 260 updates. This budget does not guarantee a learned policy.
Model, optimizer, rollout memory, and environments belong to one worker.
Retained memory is bounded by that fixed state and one batch.

The browser uses a dedicated module worker and waits for its readiness message.
The host sends at most one outstanding request. Pausing stops requests after the
current update; terminating the worker discards the run immediately.
The [Web Workers documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers)
describes message copying, separate execution, and termination.
The [wasm-bindgen guide](https://rustwasm.github.io/docs/wasm-bindgen/examples/without-a-bundler.html)
describes asynchronous initialization of the generated ES module.
This package uses the repository's pinned wasm-bindgen 0.2.121.

Legal states are idle, a valid run, and failed. Completion is derived from the
valid run's successful-update count. A completed run can export its weights.
An update failure discards the run because collection or optimization may have
partially mutated it. Non-finite metrics and a batch count other than 512 also fail. A failed run cannot advance or export. Start replaces any
state only after initialization succeeds. Evaluation is stateless.

Protocol 1 accepts JSON strings of at most 1,048,576 UTF-8 bytes. All command objects
reject unknown and duplicate fields. Missing fields and null values are errors.
Commands are closed objects with the following exact members:

- `{"command":"start","seed":7}` starts fresh. Seed is an integer in `[0, 4294967295]`.
- `{"command":"advance"}` performs one update.
- `{"command":"export"}` exports the current MessagePack checkpoint as a byte array.
- `{"command":"evaluate","bytes":[...]}` evaluates an independent checkpoint.
  Each array element is an integer in `[0, 255]`.

The worker first emits `{"event":"ready","protocol":1}`. The host must reject an
unsupported protocol before sending a command. Input errors emit `event: error`,
a machine-facing `code`, and diagnostic `message`. Categories are `invalid_request`,
`not_started`, `complete`, `failed`, `training`, and `checkpoint`.
Malformed requests do not change the current run. Incompatible checkpoints do not
change it either. Panics and worker transport failures require host-side failure
handling; they cannot be reported as successful protocol responses.

A `started` response contains seed, zero updates and transitions, update_limit,
and both learning rates. A `progress` response contains updates, transitions,
status `training` or `complete`, cumulative optimizer_steps, actor_loss,
critic_loss, entropy, approximate_kl, and both learning rates.
Transitions equal successful updates multiplied by 512. A `policy` response
contains `bytes`. An `evaluation` response contains `episodes` and `baseline`.
Each ordered score has decimal-string seed, steps, reward, final_distance in
metres, and survived. Decimal strings preserve the full 64-bit evaluation seeds.
Only the five selection seeds are evaluated; final qualification remains separate.

Required checks for this checkpoint:

- Record a failing external JSON-boundary test before implementation.
- Prove real optimization changes reloadable weights and reports learning rates.
- Reject malformed, oversized, duplicate, unknown, and out-of-range input before mutation.
- Test completion, failed updates, restart, independent scoring, and export purity.
- Run formatting, native tests, exact root Clippy, worker native/WASM Clippy,
  personal lints, coverage, and an actual browser worker exchange.
- Exercise start, pause, resume, discard, checkpoint download, and explicit playback.
- Reject stale replies after discard, including already-resolved promise continuations.

A stale response from a terminated worker must never replace the current run or
its viewer policy. The host controller tests discard run A, start run B, complete
B, then resolve A last. They also cover discard between export and evaluation.

## Browser controls

1. Select a seed and press Start training.
2. Press Pause training to stop after the current update.
3. Press Watch checkpoint to evaluate five episodes and select those frozen weights.
4. Press Resume training to continue the same optimizer and rollout state.

Download checkpoint saves the current weights after the same independent scoring.
Discard run terminates the worker and allows a fresh seed. It preserves the
selected scene policy. Training never changes the scene policy automatically.

The host states are idle, starting, running, pausing, paused, inspecting, complete,
and failed. Only paused or complete runs with at least one update can export.
Loading, pausing, and evaluation disable conflicting actions. Failure allows retry.
Worker identity guards every asynchronous continuation and pending response.

Watch checkpoint validates at most 1 MiB of model bytes before replacing the
pending scene selection. The next Bevy update starts disturbed seed 42 with fresh
recurrent memory. Rejected bytes preserve the previous selection. Bundled policy
selects the original qualified weights. The scene labels these sources separately.

On narrow screens, Watch checkpoint focuses and scrolls to the scene. Keyboard
input in the training panel does not activate the scene shortcuts. The build keeps
JavaScript, CSS, and matching WASM files in one content-addressed directory.

## Verification

All twelve native protocol and private invariant tests pass. The first test failed
because the worker session did not exist. The final-update test starts its private
counter at 259; it verifies the transition to completion without claiming that it
trained 260 batches.

Root tests, root strict Clippy, worker native/WASM strict Clippy, formatting,
workflow validation, and shell checks pass. The selected worker package passes
strict personal Clippy and Dylint with no raw diagnostics. The separate WASM
worker check passes the same personal Clippy flags with `--no-deps`.
The root package's existing personal-lint backlog remains unresolved.

[Coverage](progress/drone-worker-coverage.json) records 61/61 protocol lines and
163/165 session lines, including tests, with all ten measured branch outcomes hit.
The two unexecuted lines convert initialization and serialization failures that
this fixed recipe cannot induce. The WASM adapter is exercised in the browser;
it has no native coverage measurement. Transport-send errors remain unforced.

[Browser evidence](progress/drone-worker-browser.json) records two real optimizer
updates, a changed 46,343-byte checkpoint, successful reload for independent
scoring, and rejection of non-string input. A page timer fired 288 times during
the second update's 2.88 seconds. This proves concurrent main-thread progress;
it does not measure rendering frame rate or qualify training speed generally.
The first update's policy failed all five selection episodes. That is expected
at this early stage and is not a qualified recovery policy.

The full seed-7 browser run completed 260 updates and 133,120 transitions.
Its checkpoint survived all 32 held-out episodes, with mean return 469.4906 and
mean final distance 0.2131 m. The pre-existing gates require at least 30 survivors,
return at least 400, and distance at most 0.5 m. The original bundled model remains
unchanged. [The training record](progress/drone-browser-training.json) includes
selection milestones, qualification episodes, and the checkpoint hash.

All eleven learning tests passed in the actual browser, including qualification
of both frozen models. An extra update after completion returned `complete` and
left exported weights unchanged. The training worker was then terminated.

The host's fifteen Node tests pass. All 28 viewer tests pass. The new inbox has
full native line and branch coverage. [The coverage record](progress/drone-training-ui-coverage.json)
records startup, WASM adapter, DOM, and defensive-branch measurement gaps. The
root personal-lint gate still fails on existing library findings. A separate
warning-enabled discovery run reached the viewer and found no new diagnostics.

The [desktop screenshot](progress/drone-training-desktop.png),
[narrow screenshot](progress/drone-training-narrow.png), and
[28-second recording](progress/drone-training-controls.mp4) show the first real UI
update, pause, checkpoint playback, resume, and discard. That early checkpoint
fails all five selection episodes. It is different from the qualified final model.
Download was clicked; the operating-system download receipt was not inspected.
Native window interaction and physical mobile input remain unverified.

The final optimized build passed a separate desktop and narrow-screen check.
[Its browser record](progress/drone-training-ui-browser.json) retains runtime hashes,
seed validation, keyboard isolation, and checkpoint focus/scroll results. Empty
and oversized checkpoints also returned errors through the actual WASM entry point.

The [23-second recovery recording](progress/drone-browser-trained-flight.mp4),
[screenshot](progress/drone-browser-trained-flight.png), and
[contact sheet](progress/drone-browser-trained-flight-contact-sheet.png) show the
saved update-260 browser model reaching action 500. This recording loads the saved
model through the checkpoint API; its training panel remains idle. The
[narrow playback screenshot](progress/drone-training-watch-narrow.png) instead
shows the freshly trained update-three model after Watch checkpoint returned focus
to the scene. These checks do not establish touch-device compatibility.
