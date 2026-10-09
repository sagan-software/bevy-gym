# Frozen RL skill scenes

Build and serve the independent hover and disturbed recovery lessons:

```sh
nix develop --command scripts/build_drone_skills.sh
python3 -m http.server 8000 --directory robot-web/skills-dist
```

Open [hover](http://localhost:8000/hover/) or
[disturbed recovery](http://localhost:8000/recovery/). Select Run to start an episode.
Space toggles playback, N applies one policy action, and R resets seed 42.
The buttons provide the same actions. Reset clears recurrent memory.

Each executable embeds `docs/progress/drone-curriculum.mpk`, SHA-256
`8b94182a1368f5449a52db5c0859160fc5819d6772c0547f31d240df7a0a465a`.
The checkpoint was trained with PPO through hover and disturbed recovery.
Both lessons pass their 32-seed held-out evaluation in native and browser tests.
These scenes do not train or load arbitrary checkpoints. The separate
[hover](../../docs/DRONE_HOVER.md) and [recovery](../../docs/DRONE_RECOVERY.md)
guides provide training and native checkpoint-selection commands.

The actor chooses every motor value from its observation and episode memory.
Physics supplies pose and termination. Rendering reads the resulting observation;
playback controls never provide motor commands. Inference errors remain visible
and prevent further actions. There is no fallback controller.

For deployment, build with `scripts/build_drone_skills.sh --release` inside
`nix develop`. Output goes to `robot-web/skills-dist-release`.
GitHub Pages places these pages under `/bevy-gym/robots/skills/`.
The gallery links each lesson and identifies the old mixed-controller viewer as
historical. Bindings and WASM use one content-hashed runtime directory per lesson.

The scenes reuse the licensed [drone model](../../assets/robots/README.md) and
bundle its attribution and license. The [validation record][evidence] includes
browser recordings, inspected contact sheets, source hashes, and coverage gaps.
Mobile evidence uses a 390-pixel browser viewport, not a physical mobile device.
The native scene builds and tests pass; native window interaction is unverified.

[evidence]: ../../docs/progress/drone-skill-scenes.json
