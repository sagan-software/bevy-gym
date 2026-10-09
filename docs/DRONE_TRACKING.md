# Fly to a waypoint

Run the [32-line guide](../examples/robots/track.rs):

```sh
nix develop --command cargo run --no-default-features --features robots --example drone-track
```

The drone starts with random tilt, velocity, and heading. It flies eight metres
east and faces east. After ten simulated seconds, the guide prints its position.
The run uses seed 42 and opens no window.

Change the position and heading in `FlightGoal::try_from` to request another goal.
Positions use world metres, with Y pointing up. A `Dir2` heading maps its X and Y
components to world X and Z. `Dir2::X` faces east; `Dir2::NEG_Y` faces north.

The private example helper validates finite positions inside X/Z (-10, 10) and
Y (0, 10) metres. These are open bounds. A valid goal does not guarantee safe
flight, collision avoidance, or recovery after damage.

## Browser flight

Open the drone viewer using the [browser build guide](../robot-web/README.md).
Select Fly east, then Run. The drone flies toward the ring, two metres above
the ground. Ground ticks mark each metre. The camera frames both route ends
and moves closer as the drone arrives. The HUD reports distance to the goal.

The keyboard shortcuts are T for Fly east, Space for Run/Pause, N for Step,
and R for Reset. Reset retains the waypoint pilot, repairs motors, and restores
seed 42. Calm start and Disturbed start retain the goal. Bundled policy returns
to recovery inference. The training panel still trains recovery only.

Fail front left disables one motor and triggers the existing destruction effects.
This pilot was trained with all four motors working. A crash after damage is an
expected demonstration of that limit, not evidence of damage recovery.

The [flight recording](progress/drone-tracking-flight.mp4) shows the desktop run.
The [damage recording](progress/drone-tracking-damage.mp4) shows smoke and debris
after a failed motor. The [visual record](progress/drone-tracking-viewer.json)
includes narrow-layout playback, screenshots, and source/runtime hashes.
Both inspected healthy browser flights ended at 500 actions with displayed
distance 0.01 metres. Physical mobile touch remains unverified.

The controls activate on primary pointer press. Bevy 0.18.1 sends `Click` to the
[previous frame's hovered target](https://docs.rs/bevy_picking/0.18.1/bevy_picking/events/fn.pointer_events.html).
A rapid move, press, and release previously missed the new button. A regression
now passes those inputs through Bevy's picking pipeline and asserts one action.

## Controller contract

`FlightPilot::bundled()` loads the frozen imitation-trained controller.
`pilot.action(observation, goal)` returns four validated motor fractions.
The guide does not need network configuration, feature encoding, or memory handling.
This helper remains private to examples; it does not expand the library API.

The pilot limits requested displacement to three metres on every action.
It then rotates displacement into body coordinates and divides by two metres.
The remaining existing flight features retain their ordering and units.
The thirteenth feature is signed horizontal heading error divided by pi.

The actor has 64 hidden units. Every inference uses zero cell and hidden state;
returned recurrent memory is discarded. Calls cannot change another call's
history. The helper exposes no mutable network or memory accessor.
Checkpoint errors, inference errors, and invalid motor outputs retain their
original error sources.

Training used an original deterministic flight controller for demonstrations,
including labels at states reached by the learner. It used no PPO updates.
The [research record](progress/drone-heading-candidate.json) preserves the recipe,
sources, failed experiments, and selection results. No teacher executes during
inference. This is low-level flight control, not learned pursuit or search.

The runtime weights in `assets/robots/tracking.mpk` exactly match the frozen
research artifact. Their SHA-256 is
`4e9f539f54b261bdaf67ab3636700a76b8c7fbe28d31d6579202b500c1d577e1`.
Both use the repository's MIT OR Apache-2.0 license.

## Qualification

The nine [integration and shared unit tests](../tests/drone_tracking.rs) pass
natively and in Chrome/WASM. They include two fixed qualification sets:

- 128 heading cases use 32 held-out seeds and four cardinal headings. Every case
  survives 500 actions, returns at least 400, ends within 0.5 metres, and holds
  position and heading within the target band for the final 100 actions.
- 80 waypoint cases use five selection seeds, four cardinal headings, and targets
  zero, two, four, and eight metres away. Every case survives 500 actions and holds
  the same final band. The hover-return threshold does not apply to travel.

The target band is position error below 0.5 metres and heading error below
15 degrees. It does not include a velocity threshold. The native research run's
minimum heading return was 418.62; its maximum final position error was
5.362 millimetres. The committed regression tests enforce the stated bounds,
not those incidental measurements.

Run the focused native check and the full browser robot checks:

```sh
nix develop --command cargo test --features robots --test drone_tracking
nix run .#drone-browser-check
```

Both sets passed with the helper's three-metre command limit. Without that limit,
the earlier research run failed nine of twenty eight-metre cases.
Other tests cover every coordinate's non-finite inputs, each box face and nearby
interior values, corrupt and incompatible checkpoints, feature scaling, error
sources, and repeated commands after an intervening goal.

[Coverage](progress/drone-tracking-coverage.json) records hits for every added
instrumented line and both outcomes of every added production condition.
The guide also ran under coverage. Invalid network output cannot be produced by
the bundled finite four-output policy; its error variant and the shared decoder
have direct tests.

The playable arena still uses its previous hover controller. Moving goals,
arena obstacles, failed motors, and navigation from filtered senses remain
unqualified. The waypoint viewer is a separate flight lesson.
