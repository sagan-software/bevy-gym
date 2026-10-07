"""Keep Gymnasium builds independent of the optional ecosystem physics engine."""

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GymnasiumDependencies(unittest.TestCase):
    """Check Cargo's resolved graph and example registration contracts."""

    def test_default_and_headless_graphs_exclude_avian(self):
        """Include dev dependencies so example and test builds cannot reintroduce Avian."""
        for flags in ([], ["--no-default-features"], ["--features", "mujoco"]):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [
                        "cargo", "tree", "--locked", "-p", "bevy-gym",
                        "--edges", "normal,build,dev", "--prefix", "none", *flags,
                    ],
                    cwd=ROOT, check=True, capture_output=True, text=True,
                )
                self.assertFalse(
                    any(line.startswith("avian") for line in result.stdout.splitlines()),
                    result.stdout,
                )

    def test_ecosystem_examples_require_explicit_physics(self):
        """Retain the ecosystem examples behind their opt-in feature."""
        result = subprocess.run(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        package = next(
            item for item in json.loads(result.stdout)["packages"]
            if item["name"] == "bevy-gym"
        )
        ecosystem = [
            target for target in package["targets"]
            if target["name"].startswith("ecosystem-")
        ]
        self.assertTrue(ecosystem)
        for target in ecosystem:
            with self.subTest(target=target["name"]):
                self.assertIn("ecosystem-inference", target["required-features"])
        for target in package["targets"]:
            self.assertNotIn("/examples/avian", target["src_path"])

    def test_ecosystem_keeps_opt_in_native_physics(self):
        """Native ecosystem builds retain Avian2D while Avian3D stays removed."""
        result = subprocess.run(
            [
                "cargo", "tree", "--locked", "-p", "bevy-gym",
                "--no-default-features", "--features", "ecosystem-inference",
                "--edges", "normal,build,dev", "--prefix", "none",
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        lines = result.stdout.splitlines()
        self.assertTrue(any(line.startswith("avian2d ") for line in lines))
        self.assertFalse(any(line.startswith("avian3d ") for line in lines))


if __name__ == "__main__":
    unittest.main()
