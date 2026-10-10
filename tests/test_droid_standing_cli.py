"""Exercise standing training and frozen inference through the CLI."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
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

    def test_checkpoint_start_preserves_source_and_records_fresh_state(
        self,
    ) -> None:
        """Retain source identity and start fresh optimization state."""
        source_bytes = self.checkpoint.read_bytes()
        metadata = self.checkpoint.with_suffix(".json")
        source_record = json.loads(metadata.read_text())
        output = self.root / "checkpoint-start"
        result = self.run_command(
            "warm-start", "--checkpoint", str(self.checkpoint),
            "--seed", "13", "--updates", "1", "--output", str(output),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("exhausted its update budget", result.stderr)
        self.assertEqual(self.checkpoint.read_bytes(), source_bytes)
        self.assertEqual((output / "initial.mpk").read_bytes(), source_bytes)
        self.assertEqual(
            json.loads((output / "initial.json").read_text()), source_record
        )
        self.assertEqual(
            json.loads((output / "warm-start.json").read_text()),
            {
                "schema": "droid-standing-warm-start-v1",
                "seed": 13,
                "source": source_record,
            },
        )
        record = json.loads((output / "standing-1.json").read_text())
        self.assertEqual(record["seed"], 13)
        self.assertEqual(record["update"], 1)
        self.assertEqual(record["transitions"], 512)
        journal = json.loads((output / "optimization.jsonl").read_text())
        self.assertGreater(journal["optimizer_updates"], 0)
        self.assertEqual(journal["valid_samples"], 512)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        rejected = self.run_command(
            "warm-start", "--checkpoint", str(self.checkpoint),
            "--updates", "1", "--output", str(output),
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertEqual(
            before,
            {path.name: path.read_bytes() for path in output.iterdir()},
        )

    def test_invalid_checkpoint_start_cannot_create_output(self) -> None:
        """Reject invalid input before creating an output directory."""
        absent = self.root / "absent.mpk"
        unidentified = self.root / "unidentified.mpk"
        unidentified.write_bytes(self.checkpoint.read_bytes())
        corrupt = self.root / "bad-digest.mpk"
        corrupt.write_bytes(b"corrupt weights")
        metadata = self.checkpoint.with_suffix(".json")
        source_record = json.loads(metadata.read_text())
        corrupt.with_suffix(".json").write_text(json.dumps(source_record))
        malformed = self.root / "malformed.mpk"
        malformed.write_bytes(b"not a network")
        digest = hashlib.sha256(malformed.read_bytes()).hexdigest()
        source_record["sha256"] = digest
        malformed.with_suffix(".json").write_text(json.dumps(source_record))
        invalid = (absent, unidentified, corrupt, malformed)
        for index, checkpoint in enumerate(invalid):
            with self.subTest(checkpoint=checkpoint.name):
                output = self.root / f"rejected-{index}"
                result = self.run_command(
                    "warm-start", "--checkpoint",
                    str(checkpoint),
                    "--updates", "1", "--output", str(output),
                )
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertFalse(output.exists())
        self.assertEqual(self.run_command("warm-start").returncode, 2)
        self.assertEqual(self.run_command(
            "warm-start", "--checkpoint", str(self.checkpoint),
            "--updates", "0",
        ).returncode, 2)

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

    def test_trace_preserves_weights_and_matches_evaluation(self) -> None:
        """Record every frozen action and its authoritative physical state."""
        before = {
            path.name: path.read_bytes() for path in self.output.iterdir()
        }
        result = self.run_command(
            "trace", "--checkpoint", str(self.checkpoint), "--seed", "42"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        origin, *frames = rows
        self.assertEqual(list(origin), [
            "kind", "schema", "seed", "source", "horizon",
            "policy_interval_ms",
        ])
        self.assertEqual(origin["kind"], "origin")
        self.assertEqual(origin["schema"], "droid-standing-trace-v1")
        self.assertEqual(origin["seed"], 42)
        self.assertEqual(origin["horizon"], 1000)
        self.assertEqual(origin["policy_interval_ms"], 20)
        self.assertEqual(origin["source"], json.loads(
            self.checkpoint.with_suffix(".json").read_text()
        ))
        self.assertIsNone(frames[0]["torques"])
        self.assertEqual(frames[0]["status"], "continuing")
        for index, frame in enumerate(frames):
            self.assertEqual(list(frame), [
                "kind", "step", "status", "torques", "observation",
                "segments",
            ])
            self.assertEqual(frame["kind"], "frame")
            self.assertEqual(frame["step"], index)
            self.assertEqual(len(frame["observation"]), 204)
            self.assertTrue(all(
                math.isfinite(value) for value in frame["observation"]
            ))
            self.assertEqual(len(frame["segments"]), 13)
            if index:
                self.assertEqual(len(frame["torques"]), 26)
                self.assertTrue(all(
                    -1 <= value <= 1 for value in frame["torques"]
                ))
            for segment in frame["segments"]:
                self.assertEqual(list(segment), [
                    "position", "orientation", "linear_velocity",
                    "angular_velocity", "floor_contact",
                ])
                self.assertEqual(len(segment["position"]), 3)
                self.assertEqual(len(segment["orientation"]), 4)
                self.assertEqual(len(segment["linear_velocity"]), 3)
                self.assertEqual(len(segment["angular_velocity"]), 3)
                self.assertIsInstance(segment["floor_contact"], bool)
                self.assertTrue(all(
                    math.isfinite(value)
                    for field in (
                        "position", "orientation", "linear_velocity",
                        "angular_velocity",
                    )
                    for value in segment[field]
                ))
        final = frames[-1]
        self.assertEqual(final["status"], "terminated")
        evaluation = self.run_command(
            "evaluate", "--checkpoint", str(self.checkpoint),
            "--seed", "42",
        )
        score = json.loads(evaluation.stdout)["episodes"][0]
        self.assertEqual(final["step"], score["steps"])
        pelvis = final["segments"][0]
        self.assertEqual(
            struct.pack("f", pelvis["position"][1]),
            struct.pack("f", score["final_height"]),
        )
        repeat = self.run_command(
            "trace", "--checkpoint", str(self.checkpoint)
        )
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        self.assertEqual(result.stdout, repeat.stdout)
        self.assertEqual(before, {
            path.name: path.read_bytes() for path in self.output.iterdir()
        })

    def test_trace_rejects_invalid_weights_before_any_record(self) -> None:
        """Invalid identity and architecture cannot emit frozen evidence."""
        missing = self.root / "trace-missing.mpk"
        no_record = self.root / "trace-no-record.mpk"
        no_record.write_bytes(self.checkpoint.read_bytes())
        wrong_digest = self.root / "trace-wrong-digest.mpk"
        wrong_digest.write_bytes(b"corrupt weights")
        record = json.loads(self.checkpoint.with_suffix(".json").read_text())
        wrong_digest.with_suffix(".json").write_text(json.dumps(record))
        wrong_model = self.root / "trace-wrong-model.mpk"
        wrong_model.write_bytes(b"not a network")
        record["sha256"] = hashlib.sha256(wrong_model.read_bytes()).hexdigest()
        wrong_model.with_suffix(".json").write_text(json.dumps(record))
        for path in [missing, no_record, wrong_digest, wrong_model]:
            result = self.run_command("trace", "--checkpoint", str(path))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
        self.assertEqual(self.run_command("trace").returncode, 2)
        self.assertEqual(self.run_command(
            "trace", "--checkpoint", str(self.checkpoint), "--seed", "-1"
        ).returncode, 2)
        for invalid in ["18446744073709551616", "not-a-number"]:
            self.assertEqual(self.run_command(
                "trace", "--checkpoint", str(self.checkpoint),
                "--seed", invalid,
            ).returncode, 2)


if __name__ == "__main__":
    unittest.main()
