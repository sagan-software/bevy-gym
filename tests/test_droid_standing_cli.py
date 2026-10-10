"""Exercise standing training and frozen inference through the CLI."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class StandingCliTests(unittest.TestCase):
    """Check saved RL candidates and explicit failure without fallback."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train one isolated candidate for the executable contract checks."""
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.executable = str(Path(os.environ["DROID_STANDING_BIN"]).resolve())
        cls.output = cls.root / "trial"
        cls.training = subprocess.run(
            [
                cls.executable, "train", "--updates", "1",
                "--output", str(cls.output),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        cls.checkpoint = cls.output / "standing-1.mpk"

    def run_command(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run one bounded CLI check without inheriting interactive input."""
        return subprocess.run(
            [self.executable, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def test_budget_exhaustion_preserves_candidate(self) -> None:
        """A failed trial preserves weights, provenance and evaluation."""
        self.assertEqual(self.training.returncode, 1)
        self.assertIn("exhausted its update budget", self.training.stderr)
        metadata = self.checkpoint.with_suffix(".json")
        provenance = json.loads(metadata.read_text())
        self.assertEqual(provenance["algorithm"], "PPO")
        self.assertEqual(provenance["transitions"], 512)
        self.assertEqual(
            provenance["sha256"],
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
        )
        evaluation_path = self.output / "standing-1.evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        self.assertFalse(evaluation["passed"])
        self.assertEqual(len(evaluation["episodes"]), 5)
        rows = (self.output / "optimization.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["valid_samples"], 512)

    def test_frozen_inference_preserves_files(self) -> None:
        """Single and held-out inference preserve checkpoint files."""
        before = {
            path.name: path.read_bytes() for path in self.output.iterdir()
        }
        for options, suite, count in [
            ((), "held-out", 32),
            (("--seed", "42"), "single-episode", 1),
        ]:
            with self.subTest(suite=suite):
                result = self.run_command(
                    "evaluate", "--checkpoint", str(self.checkpoint), *options
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("Standing evaluation failed", result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["mode"], "frozen-inference")
                self.assertEqual(report["suite"], suite)
                self.assertEqual(len(report["episodes"]), count)
                self.assertFalse(report["passed"])
        self.assertEqual(
            before,
            {path.name: path.read_bytes() for path in self.output.iterdir()},
        )

    def test_existing_output_and_conflicting_modes_are_rejected(self) -> None:
        """Reject existing trials and mixed training/evaluation options."""
        before = {
            path.name: path.read_bytes() for path in self.output.iterdir()
        }
        existing = self.run_command(
            "train", "--updates", "1", "--output", str(self.output)
        )
        self.assertEqual(existing.returncode, 1)
        self.assertEqual(existing.stdout, "")
        for arguments in [
            ("train", "--updates", "0"),
            (
                "evaluate", "--checkpoint", str(self.checkpoint),
                "--updates", "1",
            ),
            ("train", "--checkpoint", str(self.checkpoint)),
            ("evaluate",),
        ]:
            self.assertEqual(self.run_command(*arguments).returncode, 2)
        self.assertEqual(
            before,
            {path.name: path.read_bytes() for path in self.output.iterdir()},
        )

    def test_two_updates_record_unselected_progress(self) -> None:
        """Record updates between selections in the flushed journal."""
        output = self.root / "two-updates"
        result = self.run_command(
            "train", "--updates", "2", "--output", str(output)
        )
        self.assertEqual(result.returncode, 1)
        journal = output / "optimization.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual([row["update"] for row in rows], [1, 2])
        self.assertFalse((output / "standing-1.mpk").exists())
        provenance = json.loads((output / "standing-2.json").read_text())
        self.assertEqual(provenance["transitions"], 1024)

    def test_missing_or_corrupt_inference_never_emits_an_episode(self) -> None:
        """Reject missing records and changed weights before inference."""
        missing = self.run_command(
            "evaluate", "--checkpoint", str(self.root / "missing.mpk")
        )
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(missing.stdout, "")
        copied = self.root / "copied.mpk"
        copied.write_bytes(self.checkpoint.read_bytes())
        record_missing = self.run_command(
            "evaluate", "--checkpoint", str(copied)
        )
        self.assertEqual(record_missing.returncode, 1)
        self.assertEqual(record_missing.stdout, "")
        copied.with_suffix(".json").write_bytes(
            self.checkpoint.with_suffix(".json").read_bytes()
        )
        copied.write_bytes(b"corrupt weights")
        corrupt = self.run_command("evaluate", "--checkpoint", str(copied))
        self.assertEqual(corrupt.returncode, 1)
        self.assertIn("digest mismatch", corrupt.stderr)
        self.assertEqual(corrupt.stdout, "")


if __name__ == "__main__":
    unittest.main()
