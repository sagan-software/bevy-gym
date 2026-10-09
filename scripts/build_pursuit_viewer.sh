#!/usr/bin/env bash
set -euo pipefail

# Run inside `nix develop`; this scene shares the native example's movement model.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
case "$#:${1:-}" in
  0:)
    pursuit_profile="debug"
    pursuit_flags=()
    pursuit_dist="$project_root/robot-web/pursuit-dist"
    ;;
  1:--release)
    pursuit_profile="release"
    pursuit_flags=(--release)
    pursuit_dist="$project_root/robot-web/pursuit-dist-release"
    ;;
  *)
    printf 'Usage: %s [--release]\n' "$0" >&2
    exit 2
    ;;
esac
pursuit_target="$(cargo metadata --locked --no-deps --format-version 1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
cargo build --locked --no-default-features --target wasm32-unknown-unknown --example drone-pursuit \
  --features robots,browser,render-core,bevy/webgl2 "${pursuit_flags[@]}"
mkdir -p -- "$pursuit_dist/assets/fonts" "$pursuit_dist/assets/robots" "$pursuit_dist/LICENSES"
wasm-bindgen --target web --out-name drone-pursuit --out-dir "$pursuit_dist" \
  "$pursuit_target/wasm32-unknown-unknown/$pursuit_profile/examples/drone-pursuit.wasm"
if [[ "$pursuit_profile" == "release" ]]; then
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-reference-types \
    --strip-debug --strip-producers "$pursuit_dist/drone-pursuit_bg.wasm" \
    -o "$pursuit_dist/drone-pursuit_optimized.wasm"
  mv -- "$pursuit_dist/drone-pursuit_optimized.wasm" "$pursuit_dist/drone-pursuit_bg.wasm"
fi
cp -- robot-web/pursuit/index.html robot-web/pursuit/styles.css robot-web/pursuit/start.js "$pursuit_dist/"
cp -- assets/fonts/MonaSans-VariableFont.ttf assets/fonts/OFL.txt "$pursuit_dist/assets/fonts/"
cp -- assets/robots/drone.glb assets/robots/README.md "$pursuit_dist/assets/robots/"
cp -- LICENSES/DRONE-CC-BY-3.0.txt "$pursuit_dist/LICENSES/"
# Keep the entry module and WASM bindings together across cached deployments.
runtime_files=(drone-pursuit.js drone-pursuit_bg.wasm start.js styles.css)
runtime_hash="$(cd -- "$pursuit_dist" && sha256sum "${runtime_files[@]}" | sha256sum | cut -d ' ' -f 1)"
runtime_directory="build-$runtime_hash"
mkdir -p -- "$pursuit_dist/$runtime_directory"
for runtime_file in "${runtime_files[@]}"; do
  mv -- "$pursuit_dist/$runtime_file" "$pursuit_dist/$runtime_directory/$runtime_file"
done
sed -i \
  -e "s|src=\"start.js\"|src=\"$runtime_directory/start.js\"|" \
  -e "s|href=\"styles.css\"|href=\"$runtime_directory/styles.css\"|" "$pursuit_dist/index.html"
