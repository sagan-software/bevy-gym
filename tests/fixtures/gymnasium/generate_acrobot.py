"""Generate Acrobot trajectories from pinned Gymnasium and NumPy sources."""

import json
import math
import subprocess
from pathlib import Path

import numpy as np
from gymnasium.envs.classic_control.acrobot import AcrobotEnv

ROOT = Path(__file__).resolve().parents[3]
REVISION = "7a1191388aa4aa973d3a5e4b039899cd99cc991f"
assert np.__version__ == "2.4.4", "Review precision before changing NumPy."
actual = subprocess.run(
    ["git", "-C", str(ROOT / "ref/gymnasium"), "rev-parse", "HEAD"],
    capture_output=True, check=True, text=True, timeout=10,
).stdout.strip()
assert actual == REVISION, "Use the pinned Gymnasium checkout."


def local_reset(seed):
    """Inject the SplitMix64 reset stream into the Python observer."""
    samples = []
    mask = (1 << 64) - 1
    for _ in range(4):
        seed = (seed + 0x9E3779B97F4A7C15) & mask
        value = seed
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        uniform = (value >> 11) / (1 << 53)
        samples.append(np.float32(math.fma(uniform, 0.2, -0.1)))
    return np.asarray(samples, dtype=np.float32)


cases = [
    ("downward", [0.0, 0.0, 0.0, 0.0], [1, 0, 2, 1]),
    ("upright", [np.pi, 0.0, 0.0, 0.0], [0, 1, 2]),
    ("positive_pi", [np.pi, np.pi, 0.0, 0.0], [0, 1, 2]),
    ("negative_pi", [-np.pi, -np.pi, 0.0, 0.0], [2, 1, 0]),
]
for first in [-1, 1]:
    for second in [-1, 1]:
        state = [0.73, -2.15, first * 4 * np.pi, second * 9 * np.pi]
        name = f"velocity_caps_{first}_{second}"
        cases.append((name, state, [0, 1, 2] * 10))
for seed in [0, 17, 42, 71]:
    env = AcrobotEnv()
    env.reset(seed=seed)
    actions = np.random.default_rng(seed + 900).integers(0, 3, 500).tolist()
    cases.append((f"reset_{seed}", env.state.tolist(), actions))

records = []
for name, initial, actions in cases:
    env = AcrobotEnv()
    env.state = np.asarray(initial, dtype=np.float64)
    steps = []
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        assert info == {}
        steps.append({
            "action": action, "state": env.state.tolist(),
            "observation_bits": observation.view(np.uint32).tolist(),
            "reward": reward, "terminated": terminated, "truncated": truncated,
        })
    records.append({"name": name, "initial_state": initial, "steps": steps})

resets = []
for seed in [*range(32), (1 << 64) - 1]:
    env = AcrobotEnv()
    env.state = local_reset(seed)
    resets.append({
        "seed": seed, "state_bits": env.state.view(np.uint32).tolist(),
        "observation_bits": env._get_ob().view(np.uint32).tolist(),
    })
output = Path(__file__).with_name("acrobot_sequences.json")
output.write_text(json.dumps({
    "gymnasium_commit": REVISION, "numpy_version": np.__version__,
    "reset_stream": "SplitMix64", "state_dtype_after_step": "float64",
    "cases": records, "resets": resets,
}, indent=2) + "\n")
transition_count = sum(len(case["steps"]) for case in records)
print(f"{transition_count} transitions; {len(resets)} resets")
