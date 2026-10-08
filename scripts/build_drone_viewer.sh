#!/usr/bin/env bash
set -euo pipefail

# Run inside `nix develop`. Keep the output separate from existing browser examples.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
# The release bundle is separate so a development server keeps serving complete files.
case "$#:${1:-}" in
  0:)
    viewer_profile="debug"
    viewer_flags=()
    viewer_dist="$project_root/robot-web/dist"
    ;;
  1:--release)
    viewer_profile="release"
    viewer_flags=(--release)
    viewer_dist="$project_root/robot-web/dist-release"
    ;;
  *)
    printf 'Usage: %s [--release]\n' "$0" >&2
    exit 2
    ;;
esac
# Cargo resolves any workspace or user build-cache configuration.
viewer_target="$(cargo metadata --locked --no-deps --format-version 1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"

cargo build --locked --no-default-features --target wasm32-unknown-unknown --example drone-flight \
  --features robots,browser,render-core,bevy/webgl2 "${viewer_flags[@]}"
mkdir -p -- "$viewer_dist/assets/robots" "$viewer_dist/LICENSES" "$viewer_dist/assets/fonts"
wasm-bindgen --target web --out-name drone-flight --out-dir "$viewer_dist" \
  "$viewer_target/wasm32-unknown-unknown/$viewer_profile/examples/drone-flight.wasm"
if [[ "$viewer_profile" == "release" ]]; then
  # Keep optimization out of development builds and strip browser debug metadata.
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-reference-types \
    --strip-debug --strip-producers "$viewer_dist/drone-flight_bg.wasm" \
    -o "$viewer_dist/drone-flight_optimized.wasm"
  mv -- "$viewer_dist/drone-flight_optimized.wasm" "$viewer_dist/drone-flight_bg.wasm"
fi
cp -- robot-web/index.html robot-web/styles.css robot-web/start.js "$viewer_dist/"
cp -- assets/fonts/MonaSans-VariableFont.ttf assets/fonts/OFL.txt "$viewer_dist/assets/fonts/"
cp -- assets/robots/drone.glb assets/robots/README.md "$viewer_dist/assets/robots/"
cp -- LICENSES/DRONE-CC-BY-3.0.txt "$viewer_dist/LICENSES/DRONE-CC-BY-3.0.txt"
