#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root inside the Nix development shell. Each example
# selects first, 33 percent, 66 percent, and best checkpoints from its run.
record() {
  local example="$1"
  local run="$2"
  local output="$3"
  if [[ -f "docs/videos/$output.mp4" && "${FORCE_RECORDINGS:-0}" != "1" ]]; then
    return
  fi
  cargo run --example "$example" -- video \
    --checkpoint "$run" \
    --output "docs/videos/$output.mp4"
}

record_mujoco() {
  local example="$1"
  local run="$2"
  local output="$3"
  if [[ -f "docs/videos/$output.mp4" && "${FORCE_RECORDINGS:-0}" != "1" ]]; then
    return
  fi
  cargo run --features mujoco --example "$example" -- video \
    --checkpoint "$run" \
    --output "docs/videos/$output.mp4"
}

record_best_gif() {
  local video="$1"
  local stem
  stem="$(basename "${video%.mp4}")"
  if [[ -f "docs/images/$stem.gif" && "${FORCE_IMAGES:-0}" != "1" ]]; then
    return
  fi
  ffmpeg -nostdin -y -loglevel error -ss 0 -t 5 -i "$video" \
    -filter_complex \
    "[0:v]fps=20,split[base][palette_source];[palette_source]palettegen=max_colors=256[palette];[base][palette]paletteuse=dither=sierra2_4a" \
    "docs/images/$stem.gif"
}

record acrobot runs/acrobot-dqn/tuned-lr3e4-scale10-seed907-20260730 acrobot
record mountain-car runs/mountain-car-dqn/sweep-potential25-lr1e3-seed42-20260730 mountain-car
record mountain-car-continuous runs/mountain-car-continuous-ppo/tuned-potential25-actor3e3-critic1e3-seed157-20260730 mountain-car-continuous
record pendulum runs/pendulum-ppo/tuned-actor3e3-critic1e3-scale01-seed42-20260730 pendulum
record blackjack runs/blackjack-tabular-q/tuned-lr001-seed157-20260729 blackjack
record cliff-walking runs/cliff-walking-tabular-q/progress-dense-seed42-20260730 cliff-walking
record frozen-lake runs/frozen-lake-tabular-q/progress-seed42-20260730 frozen-lake
record taxi runs/taxi-tabular-q/tuned-lr020-seed907-20260729 taxi
record bipedal-walker runs/bipedal-walker-ppo/progress-reset-seed42-20260730 bipedal-walker
record car-racing runs/car-racing-ppo/progress-reset-seed42-20260730 car-racing
record lunar-lander runs/lunar-lander-dqn/guided50pct-full-lr3e4-seed157-20260730 lunar-lander
record ant runs/ant-ppo/progress-reset-seed42-20260730 ant
record half-cheetah runs/half-cheetah-ppo/progress-reset-seed42-20260730 half-cheetah
record hopper runs/hopper-ppo/progress-reset-seed42-20260730 hopper
record humanoid runs/humanoid-ppo/progress-reset-seed42-20260730 humanoid
record humanoid-standup runs/humanoid-standup-ppo/progress-reset-seed42-20260730 humanoid-standup
record inverted-double-pendulum runs/inverted-double-pendulum-ppo/progress-encoded-seed42-20260730 inverted-double-pendulum
record inverted-pendulum runs/inverted-pendulum-ppo/progress-dense-seed907-20260730 inverted-pendulum
record pusher runs/pusher-ppo/progress-reset-seed42-20260730 pusher
record reacher runs/reacher-ppo/progress-reset-seed42-20260730 reacher
record swimmer runs/swimmer-ppo/progress-bounded-seed42-20260730 swimmer
record walker2d runs/walker2d-ppo/progress-reset-seed42-20260730 walker2d

if [[ ! -f docs/videos/cartpole.mp4 || "${FORCE_RECORDINGS:-0}" == "1" ]]; then
  cargo run --example cartpole -- video \
    --checkpoint runs/cartpole-dqn/final-cartpole-control-20260729a \
    --output docs/videos/cartpole.mp4
fi

record_mujoco mujoco-inverted-pendulum \
  runs/mujoco-inverted-pendulum-ppo/lqr-progress-seed42-20260730 \
  mujoco-inverted-pendulum
record_mujoco mujoco-inverted-double-pendulum \
  runs/mujoco-inverted-double-pendulum-ppo/lqr-rollout-progress-seed42-20260730 \
  mujoco-inverted-double-pendulum
record_mujoco mujoco-reacher \
  runs/mujoco-reacher-ppo/tuned-heldout-progress-seed42-20260730 \
  mujoco-reacher

while IFS= read -r video; do
  record_best_gif "$video"
done < <(find docs/videos -maxdepth 1 -type f -name '*.mp4' -print | sort)
