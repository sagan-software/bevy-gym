#!/usr/bin/env bash
set -euo pipefail

# Run inside nix develop; each lesson has its own executable and deployable page.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
case "$#:${1:-}" in
0:)
    profile=debug
    flags=()
    output="$project_root/robot-web/skills-dist"
    ;;
1:--release)
    profile=release
    flags=(--release)
    output="$project_root/robot-web/skills-dist-release"
    ;;
*)
    printf 'Usage: %s [--release]\n' "$0" >&2
    exit 2
    ;;
esac
skill_target="$(cargo metadata --locked --no-deps --format-version 1 |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
cargo build --locked --no-default-features --target wasm32-unknown-unknown \
    --features robots,browser,render-core,bevy/webgl2 \
    --example drone-hover-scene --example drone-recovery-scene --example drone-travel-scene --example droid-standing-scene "${flags[@]}"
for lesson in hover recovery travel standing; do
    destination="$output/$lesson"
    mkdir -p -- "$destination/assets/robots" "$destination/assets/fonts" "$destination/LICENSES"
    example="drone-$lesson-scene"
    if [[ "$lesson" == standing ]]; then example=droid-standing-scene; fi
    wasm-bindgen --target web --out-name lesson --out-dir "$destination" \
        "$skill_target/wasm32-unknown-unknown/$profile/examples/$example.wasm"
    cp -- robot-web/skills/{index.html,start.js,styles.css} "$destination/"
    cp -- assets/robots/{drone.glb,README.md} "$destination/assets/robots/"
    cp -- assets/fonts/{MonaSans-VariableFont.ttf,OFL.txt} "$destination/assets/fonts/"
    cp -- LICENSES/DRONE-CC-BY-3.0.txt "$destination/LICENSES/"
    if [[ "$lesson" == standing ]]; then
        mkdir -p -- "$destination/assets/robots/survival"
        cp -- assets/robots/survival/{mannequin.glb,README.md,manifest.json} "$destination/assets/robots/survival/"
        cp -- assets/robots/survival/LICENSE-ANIMATIONS.txt "$destination/assets/robots/survival/"
        sed -i 's|assets/robots/README.md|assets/robots/survival/README.md|' "$destination/index.html"
    fi
    case "$lesson" in
    hover)
        title=Hover
        guide=DRONE_HOVER
        ;;
    recovery)
        title='Disturbed recovery'
        guide=DRONE_RECOVERY
        ;;
    travel)
        title='Travel trial'
        guide=DRONE_TRAVEL
        ;;
    standing)
        title='Standing trial'
        guide=DROID_STANDING
        ;;
    esac
    sed -i -e "s/LESSON_TITLE/$title/g" -e "s/LESSON_GUIDE/$guide/g" "$destination/index.html"
    if [[ "$lesson" == standing ]]; then sed -i 's/R resets./R resets. V changes speed./' "$destination/index.html"; fi
    # Pin the bindings and binary to one URL so reloads cannot mix checkpoint builds.
    runtime_hash="$(cd -- "$destination" && sha256sum lesson.js lesson_bg.wasm start.js | sha256sum | cut -d ' ' -f 1)"
    runtime_directory="build-$runtime_hash"
    mkdir -p -- "$destination/$runtime_directory"
    mv -- "$destination/lesson.js" "$destination/lesson_bg.wasm" "$destination/start.js" \
        "$destination/$runtime_directory/"
    sed -i "s|src=\"start.js\"|src=\"$runtime_directory/start.js\"|" "$destination/index.html"
done
