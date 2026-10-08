"""Generate Pendulum trajectories from the pinned Gymnasium and NumPy profile."""

import json
import subprocess
from pathlib import Path

import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv

assert np.__version__ == "2.4.4", "Review scalar promotion before changing NumPy."
ROOT = Path(__file__).resolve().parents[3]
REVISION = "7a1191388aa4aa973d3a5e4b039899cd99cc991f"
actual = subprocess.run(
    ["git", "-C", str(ROOT / "ref/gymnasium"), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True, timeout=10,
).stdout.strip()
assert actual == REVISION, "Use the pinned Gymnasium checkout."

CASES = [
    ("upright", [0.0, 0.0], [0.0, 2.0, -2.0, 0.1]),
    ("reward_before_step", [0.5, -0.25], [0.0, 0.123456789, -0.75]),
    ("positive_pi", [np.pi, 1.0], [5.0, -5.0, 2.000000238, -2.000000238]),
    ("negative_pi", [-np.pi, -1.0], [0.0, 1.0, -1.0]),
    ("upper_velocity_cap", [np.pi / 2, 8.0], [2.0, 2.0, -2.0]),
    ("lower_velocity_cap", [-np.pi / 2, -8.0], [-2.0, -2.0, 2.0]),
    ("unwrapped_angle", [17.0 * np.pi, 0.2], [0.25] * 200),
    ("mixed", [-2.13, 0.713], np.random.default_rng(17).uniform(-5, 5, 200)),
]
cases = []
for name, state, actions in CASES:
    env = PendulumEnv()
    env.state = np.array(state, dtype=np.float64)
    steps = []
    for action in actions:
        raw = np.float32(action)
        observation, reward, terminated, truncated, _ = env.step(np.array([raw]))
        steps.append({
            "action": float(raw),
            "state": env.state.tolist(),
            "observation_bits": observation.view(np.uint32).tolist(),
            "reward": float(reward),
            "last_torque": float(env.last_u),
            "terminated": terminated,
            "truncated": truncated,
        })
    cases.append({"name": name, "initial_state": state, "steps": steps})
Path(__file__).with_name("pendulum_sequences.json").write_text(
    json.dumps({
        "gymnasium_commit": REVISION,
        "numpy_version": np.__version__,
        "action_dtype": "float32",
        "state_dtype": "float64",
        "observation_dtype": "float32",
        "cases": cases,
    }, indent=2) + "\n"
)
