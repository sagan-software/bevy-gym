# Frozen RL skill scenes

Build and serve the independent hover, disturbed recovery, travel and standing lessons:

```sh
nix develop --command scripts/build_drone_skills.sh
python3 -m http.server 8000 --directory robot-web/skills-dist
```

Open [hover](http://localhost:8000/hover/) or
[disturbed recovery](http://localhost:8000/recovery/), or the
[unqualified travel trial](http://localhost:8000/travel/), or the
[unqualified standing trial](http://localhost:8000/standing/). Select Run to start an episode.
Space toggles playback, N applies one policy action, and R resets seed 42.
The buttons provide the same actions. Reset clears recurrent memory. Standing also
provides V to cycle one, four and sixteen actions per presentation tick.

Hover and recovery embed `docs/progress/drone-curriculum.mpk`, SHA-256
`8b94182a1368f5449a52db5c0859160fc5819d6772c0547f31d240df7a0a465a`.
The checkpoint was trained with PPO through hover and disturbed recovery.
Both lessons pass their 32-seed held-out evaluation in native and browser tests.

Travel embeds the failed RL candidate `docs/progress/drone-travel-trial.mpk`,
SHA-256 `0f5a36039389e68c66281029d77eb2d7397bba8483e4a1f366092f512397d5f7`.
It has no travel qualification. The [travel guide](../../docs/DRONE_TRAVEL.md)
records its training provenance, contracts, failed results, and evaluation commands.

Standing embeds `docs/progress/droid-standing-trial.mpk`, PPO seed 7 update 100,
SHA-256 `bd7ee38c4a7e44cfc615c366cafcd8b48e9bc0ba57522fd7fec33ee1fded3024`.
Its selection failed all five cases. The [standing guide](../../docs/DROID_STANDING.md)
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
`nix develop`. All four pages go to `robot-web/skills-dist-release`.
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
