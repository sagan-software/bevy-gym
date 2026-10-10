"""Run individual curriculum lessons through the native training command."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))


class DroneCurriculumCli(unittest.TestCase):
    """A bounded standalone lesson must save its own failed evaluation."""

    def test_rehearsal_preserves_endurance(self):
        """The opt-in recipe preserves the first stage."""
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for mode in ([], ["--rehearse-prerequisites"]):
                name = "rehearsal" if mode else "original"
                output = Path(directory) / name
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"),
                     "--lesson", "travel", "--updates", "1", "--seed", "11",
                     "--output", str(output), *mode],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(
                    "Lesson travel-endurance exhausted", result.stderr,
                )
                outputs.append(output)
            original, rehearsal = outputs
            for name in ("travel-endurance-1.json", "transfer.json",
                         "optimization.jsonl"):
                self.assertEqual((original / name).read_bytes(),
                                 (rehearsal / name).read_bytes(), name)
            # Burn assigns fresh parameter IDs. Compare frozen inference.
            scores = []
            for output in outputs:
                inference = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"),
                     "--lesson", "travel", "--evaluate-checkpoint",
                     str(output / "travel-endurance-1.mpk"), "--travel-stage",
                     "travel-endurance", "--seed", "42"],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(inference.returncode, 0, inference.stderr)
                scores.append(json.loads(inference.stdout)["episodes"])
            self.assertEqual(*scores)
            self.assertFalse((original / "training-recipe.json").exists())
            recipe_path = rehearsal / "training-recipe.json"
            recipe = json.loads(recipe_path.read_text())
            self.assertEqual(
                recipe["recipe"], "travel-prerequisite-rehearsal-v1",
            )
            self.assertEqual(recipe["task_stream_channel"], 2)
            self.assertEqual(recipe["evaluation"], "original-single-stage")
            self.assertEqual(recipe["current_stage_fraction"], 0.5)
            self.assertEqual(recipe["seed"], 11)
            self.assertEqual(
                recipe["prerequisites"],
                "remaining half split among earlier stages; "
                "endurance unchanged",
            )
            self.assertEqual(set(recipe), {
                "recipe", "task_stream_channel", "current_stage_fraction",
                "prerequisites", "evaluation", "seed",
            })

    def test_rehearsal_rejects_other_lessons_and_all_inference_modes(self):
        """Unsupported modes fail before checkpoint loading or output."""
        cases = [
            (["--lesson", "hover"], 1),
            (["--lesson", "recovery"], 1),
            ([], 2),
            (["--lesson", "travel", "--evaluate-checkpoint", "missing"], 2),
            (["--lesson", "travel", "--evaluate-held-out", "missing"], 2),
            (["--lesson", "travel", "--evaluate-promotion", "missing",
              "--travel-stage", "travel-near"], 2),
            (["--lesson", "travel", "--initialize-from",
              "qualified-hover"], 2),
            (["--lesson", "travel", "false"], 2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            for arguments, status in cases:
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [str(TARGET / "debug/examples/drone-curriculum"),
                         "--rehearse-prerequisites", "--output", str(output),
                         *arguments],
                        cwd=ROOT, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, status, result.stderr)
                    if "false" in arguments:
                        self.assertIn(
                            "unexpected argument 'false'", result.stderr,
                        )
                    else:
                        self.assertIn("rehearse-prerequisites", result.stderr)
                    self.assertFalse(
                        output.exists(),
                    )

    def test_rehearsal_record_failure_stops_before_rollout(self):
        """Blocking the recipe artifact prevents unrecorded training."""
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "training-recipe.json").mkdir()
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel", "--rehearse-prerequisites",
                 "--updates", "1", "--output", directory],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertFalse((Path(directory) / "optimization.jsonl").exists())
            self.assertFalse((Path(directory) / "transfer.json").exists())

    def test_rehearsal_output_failures_stop_before_promotion(self):
        """Blocked artifact paths stop training before promotion."""
        for name in ("optimization.jsonl", "transfer.json",
                     "travel-endurance-1.mpk", "travel-endurance-1.json"):
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as path,
            ):
                output = Path(path)
                (output / name).mkdir()
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"),
                     "--lesson", "travel", "--rehearse-prerequisites",
                     "--updates", "1", "--output", path],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("Is a directory", result.stderr)
                self.assertFalse(list(output.glob("*.promotion.json")))
                self.assertFalse(list(output.glob("travel-near-*")))
                if name in ("optimization.jsonl", "transfer.json"):
                    self.assertEqual(result.stdout, "")
        with tempfile.TemporaryDirectory() as path:
            output = Path(path) / "existing-file"
            output.write_text("preserved")
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel", "--rehearse-prerequisites",
                 "--updates", "1", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(output.read_text(), "preserved")

    def test_rehearsal_failed_selection_continues_to_budget(self):
        """Failed selection keeps collecting until the final update."""
        with tempfile.TemporaryDirectory() as path:
            output = Path(path)
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel", "--rehearse-prerequisites",
                 "--updates", "21", "--seed", "29", "--output", path],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            for update in (20, 21):
                record = json.loads(
                    (output / f"travel-endurance-{update}.json").read_text(),
                )
                self.assertFalse(record["passed"])
                self.assertEqual(record["update"], update)
            rows = (output / "optimization.jsonl").read_text().splitlines()
            progress = [json.loads(line) for line in rows]
            self.assertEqual(len(progress), 21)
            self.assertEqual(sum(row["valid_samples"] for row in progress),
                             10_752)
            self.assertGreater(progress[-1]["optimizer_steps"],
                               progress[-2]["optimizer_steps"])
            self.assertFalse(list(output.glob("travel-near-*")))
            self.assertFalse(list(output.glob("*.promotion.json")))

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
            progress_path = Path(directory) / "optimization.jsonl"
            progress = [json.loads(line)
                        for line in progress_path.read_text().splitlines()]
            self.assertEqual(len(progress), 1)
            self.assertEqual(progress[0]["lesson"], "travel-endurance")
            self.assertEqual(progress[0]["update"], 1)
            self.assertEqual(progress[0]["valid_samples"], 512)
            self.assertEqual(progress[0]["optimizer_steps"],
                             record["optimizer_steps"])
            self.assertGreater(progress[0]["optimizer_updates"], 0)
            for field in ("actor_loss", "critic_loss", "entropy",
                          "approximate_kl", "actor_learning_rate",
                          "critic_learning_rate"):
                self.assertIsInstance(progress[0][field], float)
                self.assertTrue(math.isfinite(progress[0][field]), field)
            transfer = json.loads((Path(directory) / "transfer.json").read_text())
            self.assertEqual(transfer["source"], "qualified-recovery")
            self.assertEqual(transfer["destination"], "travel-endurance")
            self.assertEqual(transfer["optimizer"], "fresh")
            self.assertEqual(transfer["actor_observations"], 13)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["optimization.jsonl", "transfer.json",
                              "travel-endurance-1.json",
                              "travel-endurance-1.mpk"])
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

    def test_promotion_replays_prerequisites_on_validation_roots(self):
        checkpoint = ROOT / "docs/progress/drone-travel-trial.mpk"
        before = checkpoint.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel",
                 "--evaluate-promotion", str(checkpoint),
                 "--travel-stage", "travel-near",
                 "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["schema"], "travel-promotion-v1")
            digest = hashlib.sha256(before).hexdigest()
            self.assertEqual(report["checkpoint_sha256"], digest)
            self.assertEqual([row["stage"] for row in report["stages"]],
                             ["travel-endurance", "travel-near"])
            for row in report["stages"]:
                roots = [(1 << 64) - 1 - offset
                         for offset in range(1024, 1056)]
                self.assertEqual(
                    [episode["seed"] for episode in row["episodes"]], roots)
            self.assertIn("Promotion validation failed", result.stderr)
            self.assertFalse(output.exists())
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_promotion_rejects_invalid_modes_before_creating_output(self):
        base = ["--lesson", "travel", "--evaluate-promotion", "missing",
                "--travel-stage", "travel-endurance"]
        cases = [
            (["--evaluate-promotion", "missing",
              "--travel-stage", "travel-endurance"], 2),
            (["--lesson", "hover", "--evaluate-promotion", "missing",
              "--travel-stage", "travel-endurance"], 1),
            (["--lesson", "travel", "--evaluate-promotion", "missing"], 2),
            (base, 1),
            (["--lesson", "travel", "--evaluate-promotion",
              str(ROOT / "docs/progress/drone-recovery-transfer.mpk"),
              "--travel-stage", "travel-endurance"], 1),
            (["--lesson", "travel", "--evaluate-promotion", "missing",
              "--travel-stage", "unknown"], 2),
        ]
        for extra in (["--updates", "1"], ["--seed", "11"],
                      ["--initialize-from", "qualified-hover"],
                      ["--evaluate-checkpoint", "missing"],
                      ["--evaluate-held-out", "missing"]):
            cases.append(([*base, *extra], 2))
        for arguments, status in cases:
            with (
                self.subTest(arguments=arguments),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "must-not-exist"
                result = subprocess.run(
                    [str(TARGET / "debug/examples/drone-curriculum"),
                     *arguments,
                     "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertFalse(result.stdout)
                self.assertTrue(result.stderr)
                self.assertFalse(output.exists())

    def test_travel_refuses_an_unwritable_progress_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "optimization.jsonl"
            destination.mkdir()
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel", "--updates", "1",
                 "--output", directory],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue(result.stderr)
            self.assertFalse(result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [destination])
            self.assertTrue(destination.is_dir())

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_travel_stops_when_progress_flush_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "optimization.jsonl"
            destination.symlink_to("/dev/full")
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel", "--updates", "2",
                 "--output", directory],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue(result.stderr)
            self.assertNotIn("exhausted", result.stderr)
            self.assertEqual(json.loads(result.stdout)["event"],
                             "checkpoint-transfer")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["optimization.jsonl", "transfer.json"])
            self.assertTrue(destination.is_symlink())

    def test_travel_progress_includes_updates_between_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(TARGET / "debug/examples/drone-curriculum"),
                 "--lesson", "travel", "--updates", "2", "--seed", "11",
                 "--output", directory],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            progress_path = Path(directory) / "optimization.jsonl"
            rows = [json.loads(line)
                    for line in progress_path.read_text().splitlines()]
            self.assertEqual([row["update"] for row in rows], [1, 2])
            self.assertEqual([row["valid_samples"] for row in rows],
                             [512, 512])
            self.assertLess(rows[0]["optimizer_steps"],
                            rows[1]["optimizer_steps"])
            first = Path(directory) / "travel-endurance-1.json"
            second = Path(directory) / "travel-endurance-2.json"
            self.assertFalse(first.exists())
            self.assertTrue(second.is_file())

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
