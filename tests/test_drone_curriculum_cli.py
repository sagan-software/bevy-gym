"""Run individual curriculum lessons through the native training command."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))


class DroneCurriculumCli(unittest.TestCase):
    """A bounded standalone lesson must save its own failed evaluation."""

    def test_selected_lesson_exhausts_without_training_another_lesson(self):
        for lesson in ("hover", "recovery"):
            with self.subTest(lesson=lesson), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"),
                     "--lesson", lesson, "--updates", "1", "--output", directory],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"Lesson {lesson} exhausted", result.stderr)
                self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                                 [f"{lesson}-1.json", f"{lesson}-1.mpk"])
                record = json.loads((Path(directory) / f"{lesson}-1.json").read_text())
                self.assertEqual(record["lesson"], lesson)
                self.assertEqual(record["update"], 1)
                self.assertFalse(record["passed"])
                self.assertGreater(record["optimizer_steps"], 0)
                self.assertEqual(len(record["episodes"]), 5)

    def test_travel_starts_with_rl_transfer_and_preserves_failed_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel", "--updates", "1", "--seed", "11",
                 "--output", directory],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("Lesson travel-endurance exhausted", result.stderr)
            record = json.loads((Path(directory) / "travel-endurance-1.json").read_text())
            self.assertFalse(record["passed"])
            self.assertEqual(record["update"], 1)
            self.assertGreater(record["optimizer_steps"], 0)
            self.assertEqual(len(record["episodes"]), 5)
            transfer = json.loads((Path(directory) / "transfer.json").read_text())
            self.assertEqual(transfer["source"], "qualified-recovery")
            self.assertEqual(transfer["destination"], "travel-endurance")
            self.assertEqual(transfer["optimizer"], "fresh")
            self.assertEqual(transfer["actor_observations"], 13)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["transfer.json", "travel-endurance-1.json", "travel-endurance-1.mpk"])
            checkpoint = Path(directory) / "travel-endurance-1.mpk"
            before = checkpoint.read_bytes()
            inference = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel", "--evaluate-checkpoint", str(checkpoint),
                 "--travel-stage", "travel-endurance", "--seed", "42",
                 "--output", str(Path(directory) / "inference-must-not-write")],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(inference.returncode, 0, inference.stderr)
            score = json.loads(inference.stdout)
            self.assertEqual(score["mode"], "frozen-inference")
            self.assertEqual(score["qualification"], "not established by this episode")
            self.assertEqual(len(score["episodes"]), 1)
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertFalse((Path(directory) / "inference-must-not-write").exists())

    def test_held_out_evaluation_binds_all_stages_to_unchanged_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            training = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"), "--lesson", "travel",
                 "--updates", "1", "--seed", "11", "--output", directory],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(training.returncode, 1, training.stderr)
            checkpoint = output / "travel-endurance-1.mpk"
            before = checkpoint.read_bytes()
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"), "--lesson", "travel",
                 "--evaluate-held-out", str(checkpoint), "--output", str(output / "unused")],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["mode"], "held-out-evaluation")
            self.assertEqual(report["suite"], "travel-v1")
            self.assertEqual(report["checkpoint_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(report["evaluation"], "failed")
            self.assertEqual([row["stage"] for row in report["stages"]],
                             ["travel-endurance", "travel-near", "travel-far", "travel-fast"])
            for row in report["stages"]:
                self.assertEqual([episode["seed"] for episode in row["episodes"]],
                                 [(1 << 64) - 1 - offset for offset in range(1, 33)])
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertFalse((output / "unused").exists())

    def test_travel_inference_rejects_missing_and_incompatible_checkpoints(self):
        for checkpoint in (ROOT / "absent-travel-checkpoint.mpk",
                           ROOT / "docs/progress/drone-recovery-transfer.mpk"):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "must-not-exist"
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"),
                     "--lesson", "travel", "--evaluate-checkpoint", str(checkpoint),
                     "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertTrue(result.stderr)
                self.assertFalse(output.exists())

    def test_held_out_evaluation_rejects_missing_incompatible_and_conflicting_inputs(self):
        cases = [
            (["--evaluate-held-out", "missing"], 2),
            (["--lesson", "hover", "--evaluate-held-out", "missing"], 1),
            (["--lesson", "travel", "--evaluate-held-out", "missing"], 1),
            (["--lesson", "travel", "--evaluate-held-out",
              str(ROOT / "docs/progress/drone-recovery-transfer.mpk")], 1),
        ]
        for extra in (["--updates", "1"], ["--seed", "11"],
                      ["--initialize-from", "qualified-hover"],
                      ["--evaluate-checkpoint", "missing"],
                      ["--travel-stage", "travel-near"]):
            cases.append((["--lesson", "travel", "--evaluate-held-out", "missing", *extra], 2))
        for arguments, status in cases:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "must-not-exist"
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"), *arguments,
                     "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertFalse(result.stdout)
                self.assertTrue(result.stderr)
                self.assertFalse(output.exists())

    def test_travel_inference_rejects_conflicting_modes_before_output(self):
        cases = [
            (["--evaluate-checkpoint", "missing"], 2),
            (["--lesson", "hover", "--evaluate-checkpoint", "missing"], 1),
            (["--lesson", "travel", "--evaluate-checkpoint", "missing", "--updates", "1"], 2),
            (["--lesson", "travel", "--travel-stage", "travel-far"], 2),
            (["--lesson", "travel", "--evaluate-checkpoint", "missing", "--travel-stage", "unknown"], 2),
            (["--lesson", "travel", "--evaluate-checkpoint", "missing",
              "--initialize-from", "qualified-hover"], 2),
            (["--lesson", "travel", "--initialize-from", "qualified-hover"], 1),
        ]
        for arguments, status in cases:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "must-not-exist"
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"), *arguments,
                     "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertTrue(result.stderr)
                self.assertFalse(output.exists())

    def test_default_curriculum_starts_with_hover(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--updates", "1", "--output", directory],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Lesson hover exhausted", result.stderr)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["hover-1.json", "hover-1.mpk"])

    def test_standalone_recovery_can_transfer_the_qualified_hover_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "recovery", "--initialize-from", "qualified-hover",
                 "--updates", "1", "--output", directory],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertIn(result.returncode, (0, 1), result.stderr)
            transfer = json.loads((Path(directory) / "transfer.json").read_text())
            self.assertEqual(transfer["source"], "qualified-hover")
            self.assertEqual(transfer["destination"], "recovery")
            self.assertEqual(transfer["optimizer"], "fresh")
            self.assertEqual(transfer["optimizer_steps"], 0)
            reference = ROOT / "docs/progress/drone-hover.mpk"
            self.assertEqual(transfer["qualified_record_sha256"],
                             hashlib.sha256(reference.read_bytes()).hexdigest())
            self.assertEqual(transfer["seed"], 7)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["recovery-1.json", "recovery-1.mpk", "transfer.json"])
            record = json.loads((Path(directory) / "recovery-1.json").read_text())
            self.assertGreater(record["optimizer_steps"], 0)
            self.assertEqual(result.returncode, 0 if record["passed"] else 1)
            self.assertEqual(json.loads(result.stdout.splitlines()[0]), transfer)

    def test_transfer_rejects_unsupported_destination_before_output_creation(self):
        for lesson_args in ([], ["--lesson", "hover"]):
            with self.subTest(args=lesson_args), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"), *lesson_args,
                     "--initialize-from", "qualified-hover", "--updates", "1",
                     "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("qualified-hover initialization requires --lesson recovery", result.stderr)
                self.assertFalse(output.exists())

    def test_unknown_initialization_fails_before_creating_output(self):
        for source in ("unknown", "QualifiedHover", ""):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"),
                     "--lesson", "recovery", "--initialize-from", source,
                     "--updates", "1", "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertFalse(output.exists())

    def test_unknown_lesson_fails_before_creating_output(self):
        for value, error in (("clearance", "invalid value"),
                             ("Hover", "invalid value"),
                             ("", "a value is required")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"),
                     "--lesson", value, "--updates", "1", "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
