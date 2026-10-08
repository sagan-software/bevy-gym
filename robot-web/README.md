# Drone hover viewer

Run the native example from the repository root:

```sh
nix develop --command cargo run --features robots --example drone-flight
```

Build and serve the browser version:

```sh
nix develop --command scripts/build_drone_viewer.sh --release
python3 -m http.server 8082 --bind 127.0.0.1 --directory robot-web/dist-release
```

Open `http://127.0.0.1:8082/`. Omit `--release` to build into `robot-web/dist`
without WebAssembly size optimization. The public deployment copies the release
bundle to `/bevy-gym/robots/hover/`. Its Examples link points to the deployed
gallery; that sibling route is absent from the standalone local server.

The scene starts paused. Run and Pause toggle continuous stepping. Step advances
one 20 ms action while paused. Calm start and Disturbed start select the initial
conditions and begin a paused episode. Learned policy loads the bundled recovery
model and begins a paused episode. Reset retains the start choice and restores
seed 42. It preserves learned weights and clears recurrent memory. Manual reset
restores the hover command.
The four presets apply power-off, balanced hover, climb, or asymmetric thrust.
Keyboard shortcuts appear on the buttons. Ground contact, leaving the flight
region, or 500 actions ends the episode; Reset starts another.

The viewer compares learned and manual control of the
[drone environment](../docs/ROBOT_ENVIRONMENT.md).
The 28-line [headless guide](../examples/robots/hover.rs) introduces its API.
The 25-line [recovery guide](../examples/robots/recovery.rs) uses
`DroneHover::disturbed()` to add initial tilt and velocity.
The [visual guide](../examples/robots/flight.rs) connects the environment to Bevy.
Its private modules contain the session, policy, scene, and controls. Rendering reads the
physics observation and never writes the authoritative pose. Rotor animation
illustrates the command; the environment models force rather than rotor RPM.

The [recording](../docs/progress/drone-inference.mp4) compares learned recovery with
constant half-thrust from the same disturbed start. The
[inference guide](../docs/DRONE_INFERENCE.md) records controls and failure behavior.
Browser training controls remain unfinished. Browser execution, focused tests, and the exact
remaining coverage gaps are recorded in [the status document](../docs/EXAMPLE_STATUS.md).
Native window interaction and mobile device input remain unverified.

The build includes the [model attribution and license](../assets/robots/README.md)
and the [font license](../assets/fonts/README.md). Preserve them when distributing it.
