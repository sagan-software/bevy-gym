set -eu
p=/home/sagan/.cache/bevy-gym-quality-validation/standing-pgpe-adaptive-probe-20261011
r=/nix/store/jwvci799mqqb4kd5w6y8va8iw6mipc7h-rust-default-1.98.0-nightly-2026-06-15
mkdir -p "$p/coverage"
"$r/bin/rustc" --edition=2021 --test -C instrument-coverage -Z coverage-options=branch "$p/adaptive-coverage.rs" -o "$p/coverage/gradient-tests"
LLVM_PROFILE_FILE="$p/coverage/gradient.profraw" "$p/coverage/gradient-tests"
"$r/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata" merge -sparse "$p/coverage/gradient.profraw" -o "$p/coverage/gradient.profdata"
"$r/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-cov" export "$p/coverage/gradient-tests" -instr-profile="$p/coverage/gradient.profdata" > "$p/coverage/gradient.json"
