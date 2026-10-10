# Frozen RL skill scenes

Build and serve the independent lessons and the shared world diagnostic:

```sh
nix develop --command scripts/build_drone_skills.sh
python3 -m http.server 8000 --directory robot-web/skills-dist
```

Open [hover](http://localhost:8000/hover/) or
[disturbed recovery](http://localhost:8000/recovery/), or the
[unqualified travel trial](http://localhost:8000/travel/), or the
[unqualified standing trial](http://localhost:8000/standing/), or the
[shared world diagnostic](http://localhost:8000/world/). Select Run to start an episode.

Space toggles playback and N applies one policy action.
R resets seed 42 in the four standalone lessons; it restores the shared world placements
in the diagnostic scene.
The buttons provide the same actions. Reset clears recurrent memory. Standing and the shared world also
provide V to cycle one, four and sixteen actions per presentation tick.

Hover and recovery embed `docs/progress/drone-curriculum.mpk`, SHA-256
`8b94182a1368f5449a52db5c0859160fc5819d6772c0547f31d240df7a0a465a`.
The checkpoint was trained with PPO through hover and disturbed recovery.
Both lessons pass their 32-seed held-out evaluation in native and browser tests.

Travel embeds the failed RL candidate `docs/progress/drone-travel-trial.mpk`,
SHA-256 `0f5a36039389e68c66281029d77eb2d7397bba8483e4a1f366092f512397d5f7`.
It has no travel qualification. The [travel guide](../../docs/DRONE_TRAVEL.md)
records its training provenance, contracts, failed results, and evaluation commands.

Standing embeds `docs/progress/standing-seed17-update7920/policy.mpk`, PPO seed 17
update 7,920. Its SHA-256 is
`bd0d16693de7ce4db52c2bceef9d50569a558efdd3efc9a34c3562e672e042a2`.
Its digest-bound sidecar is `policy.json`.
The candidate remains unqualified. The [standing guide](../../docs/DROID_STANDING.md)
records the articulated physical model, 26 torque outputs and qualification gates.

The viewer projects physical states onto the licensed mannequin without authored
animation playback. It displays `Unqualified RL` and the checkpoint identity.

These scenes do not train or load arbitrary checkpoints. The separate
[hover](../../docs/DRONE_HOVER.md) and [recovery](../../docs/DRONE_RECOVERY.md)
guides provide training and native checkpoint-selection commands.

The actor chooses every motor value from its observation and episode memory.
Physics supplies pose and termination. Rendering reads the resulting observation;
playback controls never provide motor commands. Inference errors remain visible
and prevent further actions. There is no fallback controller.

For deployment, build with `scripts/build_drone_skills.sh --release` inside
`nix develop`. All five pages go to `robot-web/skills-dist-release`.
GitHub Pages places these pages under `/bevy-gym/robots/skills/`.
The gallery links the drone lessons and identifies the old mixed-controller viewer as
historical. Bindings and WASM use one content-hashed runtime directory per lesson.

The drone scenes reuse the licensed [drone model](../../assets/robots/README.md) and
bundle its attribution and license. Standing bundles the CC0
[mannequin](../../assets/robots/survival/README.md) and its provenance. Its
[scene evidence](../../docs/progress/droid-standing-scene.json) records policy parity,
the unresolved physical trajectory difference and browser playback.
The [validation record][evidence] includes
browser recordings, inspected contact sheets, source hashes, and coverage gaps.
Mobile evidence uses a 390-pixel browser viewport, not a physical mobile device.
The native scene builds and tests pass; native window interaction is unverified.

[evidence]: ../../docs/progress/drone-skill-scenes.json

The [shared world guide](../../docs/ROBOT_WORLD.md) describes the fifth page.
Three drones and three droids run six independent recurrent memories in one solver.
Every actuator request comes from a frozen PPO actor. The drones use hover seed 7
update 280, SHA-256
`5aa47c6941b1aa243c4eafcb0fafaf8c6fa9ee816944dc5a5ca0d3f95712403b`.
The droids use standing seed 17 update 22,940, SHA-256
`9c9e4c5bc8f7734da9835161476997e6d39d2aba28b24439e932a1f9949a2fdb`.
The standalone standing page retains its separate update-7,920 candidate.

The shared world starts paused and waits for six complete licensed models.
Space runs or pauses; N steps once; R resets; V changes speed.

C cycles the overview and six follow targets. F selects free camera.
WASD, Q/E and arrow keys move or turn only the free camera.
The pointer controls provide playback, speed and camera selection.
The complete clip stops after 1,000 common frames, or 20 physical seconds.

Shared competition and its qualification remain unfinished.
The [scene record](../../docs/progress/robot-world-scene.json) preserves complete
browser recordings, source hashes, validation and measured coverage gaps.
