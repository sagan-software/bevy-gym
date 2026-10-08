"""Generate float32-action trajectories from the pinned Gymnasium source."""

import json
import subprocess
from pathlib import Path

import numpy as np
from gymnasium.envs.classic_control.continuous_mountain_car import (
    Continuous_MountainCarEnv,
)

assert np.__version__ == "2.4.4", "Review promotion before changing NumPy."
ROOT = Path(__file__).resolve().parents[3]
REVISION = "7a1191388aa4aa973d3a5e4b039899cd99cc991f"
ORACLE = subprocess.run(
    ["git", "-C", str(ROOT / "ref/gymnasium"), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True, timeout=10,
)
assert ORACLE.stdout.strip() == REVISION, "Use the pinned Gymnasium checkout."
CASES = [
    ("valley", [-0.5, 0.0], [1.0] * 25 + [-1.0] * 25 + [0.0, 0.5, -0.75]),
    ("clipped_force", [-0.5, 0.0], [2.0, -2.0, 1.000000119, -1.000000119]),
    ("left_wall", [-1.199, -0.07], [-1.0, -1.0, 0.0, 1.0]),
    ("stepped_left_wall", [-1.1, -0.06], [-1.0, -1.0, -1.0]),
    ("upper_caps", [0.59, 0.5], [2.0, 0.0]),
    ("lower_speed", [-0.5, -0.5], [-2.0, 0.0]),
    ("goal_forward", [0.45, 0.001], [0.0, 0.0]),
    ("goal_backward", [0.5, -0.03], [0.0, 0.0]),
    ("large_finite_state", [1e200, 1e200], [2.0]),
    ("mixed", [-0.55, 0.0], np.random.default_rng(17).uniform(-2, 2, 128)),
]
trajectories = []
for name, state, actions in CASES:
    env = Continuous_MountainCarEnv()
    env.state = np.array(state, dtype=np.float64)
    steps = []
    for action in actions:
        raw = np.float32(action)
        result = env.step(np.array([raw]))
        observation, reward, terminated, truncated, _ = result
        steps.append({
            "action": float(raw),
            "observation_bits": observation.view(np.uint32).tolist(),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
        })
    trajectories.append({"name": name, "initial_state": state, "steps": steps})
Path(__file__).with_name("continuous_mountain_car_sequences.json").write_text(
    json.dumps({
        "gymnasium_commit": REVISION,
        "numpy_version": np.__version__,
        "action_dtype": "float32",
        "initial_state_dtype": "float64",
        "subsequent_state_dtype": "float32",
        "cases": trajectories,
    }, indent=2) + "\n"
)
