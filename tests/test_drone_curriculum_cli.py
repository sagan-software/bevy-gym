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
        for value, error in (("travel", "invalid value"),
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
