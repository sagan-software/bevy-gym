#!/usr/bin/env bash
set -euo pipefail

# Build the CPU trainer separately so its optimizer never runs on the render thread.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
case "$#:${1:-}" in
  0:)
    worker_profile="debug"
    worker_flags=()
    worker_dist="$project_root/robot-web/dist"
    ;;
  1:--release)
    worker_profile="release"
    worker_flags=(--release)
    worker_dist="$project_root/robot-web/dist-release"
    ;;
  *)
    printf 'Usage: %s [--release]\n' "$0" >&2
    exit 2
    ;;
esac
worker_target="$(cargo metadata --locked --no-deps --format-version 1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
cargo build --locked -p bevy-gym-drone-worker --target wasm32-unknown-unknown "${worker_flags[@]}"
mkdir -p -- "$worker_dist"
wasm-bindgen --target web --out-name drone-worker --out-dir "$worker_dist" \
  "$worker_target/wasm32-unknown-unknown/$worker_profile/drone-worker.wasm"
if [[ "$worker_profile" == "release" ]]; then
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-reference-types \
    --strip-debug --strip-producers "$worker_dist/drone-worker_bg.wasm" \
    -o "$worker_dist/drone-worker_optimized.wasm"
  mv -- "$worker_dist/drone-worker_optimized.wasm" "$worker_dist/drone-worker_bg.wasm"
fi
cp -- robot-web/worker.js "$worker_dist/"
