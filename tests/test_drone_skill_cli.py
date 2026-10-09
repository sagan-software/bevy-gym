"""Exercise frozen-policy lesson entry points through their executable interface."""

import json
import os
from pathlib import Path
import subprocess
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))


class DroneSkillCli(unittest.TestCase):
    """Missing inference weights must stop each lesson visibly."""

    def test_missing_checkpoint_fails(self):
        for lesson in ("hover", "recovery"):
            with self.subTest(lesson=lesson):
                result = subprocess.run(
                    [str(TARGET / "debug/examples" / f"drone-{lesson}"),
                     "--checkpoint", "/nonexistent/bevy-gym-policy.mpk"],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No such file", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_unreadable_corrupt_and_incompatible_checkpoints_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.mpk"
            corrupt.write_bytes(b"invalid checkpoint")
            for checkpoint in (ROOT, corrupt, ROOT / "assets/robots/tracking.mpk"):
                for lesson in ("hover", "recovery"):
                    with self.subTest(checkpoint=checkpoint, lesson=lesson):
                        result = subprocess.run(
                            [str(TARGET / "debug/examples" / f"drone-{lesson}"),
                             "--checkpoint", str(checkpoint)],
                            cwd=ROOT, capture_output=True, text=True, check=False,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertTrue(result.stderr)
                        self.assertEqual(result.stdout, "")

    def test_qualified_policy_survives_both_profiles(self):
        episodes = []
        for lesson in ("hover", "recovery"):
            with self.subTest(lesson=lesson):
                command = [str(TARGET / "debug/examples" / f"drone-{lesson}")]
                result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                        text=True, check=True)
                record = json.loads(result.stdout)
                self.assertEqual(record["mode"], "frozen-policy-inference")
                self.assertEqual(record["checkpoint"], "docs/progress/drone-curriculum.mpk")
                episode, = record["episodes"]
                self.assertEqual(episode["seed"], 42)
                self.assertEqual(episode["steps"], 500)
                self.assertTrue(episode["survived"])
                self.assertGreaterEqual(episode["reward"], 400)
                self.assertLessEqual(episode["final_distance"], 0.5)
                repeated = subprocess.run(command, cwd=ROOT, capture_output=True,
                                          text=True, check=True)
                self.assertEqual(result.stdout, repeated.stdout)
                episodes.append(episode)
        self.assertNotEqual(*episodes)


if __name__ == "__main__":
    unittest.main()
