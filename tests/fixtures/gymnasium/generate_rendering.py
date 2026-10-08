"""Regenerate browser image fixtures with the pinned Gymnasium renderer."""

import subprocess
from pathlib import Path

import numpy as np
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.classic_control.continuous_mountain_car import (
    Continuous_MountainCarEnv,
)
from PIL import Image

REVISION = "7a1191388aa4aa973d3a5e4b039899cd99cc991f"
ROOT = Path(__file__).resolve().parents[3]
actual_revision = subprocess.run(
    ["git", "-C", str(ROOT / "ref/gymnasium"), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
    timeout=10,
).stdout.strip()
if actual_revision != REVISION:
    raise RuntimeError(f"Expected Gymnasium {REVISION}; got {actual_revision}")

output = ROOT / "gymnasium-web/tests/fixtures"
output.mkdir(parents=True, exist_ok=True)
for name, environment, state in [
    ("cartpole-upright", CartPoleEnv, [0, 0, 0, 0]),
    ("cartpole-tilted", CartPoleEnv, [-1.25, 0, 0.2, 0]),
    ("mountain-car-valley", MountainCarEnv, [-0.5, 0]),
    ("mountain-car-slope", MountainCarEnv, [0.45, 0.03]),
    ("mountain-car-continuous-valley", Continuous_MountainCarEnv, [-0.5, 0]),
    ("mountain-car-continuous-slope", Continuous_MountainCarEnv, [0.45, 0.03]),
    ("acrobot-down", AcrobotEnv, [0, 0, 0, 0]),
    ("acrobot-upright", AcrobotEnv, [np.pi, 0, 0, 0]),
    ("acrobot-bent", AcrobotEnv, [0.8, -1.4, 2, -3]),
    ("acrobot-goal", AcrobotEnv, [-2.6, 0.6, -1, 4]),
]:
    env = environment(render_mode="rgb_array")
    env.state = np.array(state, dtype=np.float64)
    Image.fromarray(env.render()).save(output / f"{name}.png")
    env.close()

for name, state, torque in [
    ("pendulum-upright", [0, 0], None),
    ("pendulum-down", [np.pi, 0], 0.0),
    ("pendulum-positive-torque", [-0.8, 2.0], 2.0),
    ("pendulum-negative-torque", [1.4, -3.0], -1.0),
]:
    env = PendulumEnv(render_mode="rgb_array")
    env.state = np.array(state, dtype=np.float64)
    env.last_u = torque
    Image.fromarray(env.render()).save(output / f"{name}.png")
    env.close()
