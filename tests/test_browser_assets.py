"""Check deterministic asset versioning and cache invalidation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gymnasium-web/version_assets.py"
SPEC = importlib.util.spec_from_file_location("version_assets", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BrowserAssets(unittest.TestCase):
    """All runtime files must move with their importing application."""

    def build(
        self, root: Path, change: str = "", is_reverse: bool = False
    ) -> str:
        """Create Trunk output with models and nested images."""
        files = {
            "app.js": "worker and model imports",
            "renderers.js": "renderer",
            "gymnasium-worker_loader.js": "loader",
            "gymnasium-worker.js": "bindings",
            "gymnasium-worker_bg.wasm": "binary",
            "models/task.json": "metadata",
            "models/task.mpk": "policy",
            "assets/pendulum/arrow.png": "image",
        }
        for name in sorted(files, reverse=is_reverse):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            suffix = "changed" if name == change else ""
            path.write_text(files[name] + suffix)
        (root / "index.html").write_text(
            '<script type="module" src="app.js"></script>'
        )
        (root / "styles-123.css").write_text("hashed style")
        return MODULE.version_assets(root)

    def test_moves_the_complete_runtime_and_preserves_hashed_css(self):
        """HTML selects one build with relative imports."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.build(root)
            self.assertRegex(directory, r"^build-[0-9a-f]{64}$")
            policy = root / directory / "models/task.mpk"
            self.assertEqual(policy.read_text(), "policy")
            self.assertEqual(
                (root / "styles-123.css").read_text(), "hashed style"
            )
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                [directory, "index.html", "styles-123.css"],
            )
            self.assertIn(
                f'src="{directory}/app.js"', (root / "index.html").read_text()
            )

    def test_stable_order_and_invalidation_for_every_runtime_asset(self):
        """Each stale runtime asset has a different build URL."""
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            original = self.build(Path(first))
            self.assertEqual(
                original, self.build(Path(second), is_reverse=True)
            )
        for name in [
            "app.js", "renderers.js", "gymnasium-worker_loader.js",
            "gymnasium-worker.js", "gymnasium-worker_bg.wasm",
            "models/task.json", "models/task.mpk", "assets/pendulum/arrow.png",
        ]:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                self.assertNotEqual(
                    original, self.build(Path(temporary), change=name)
                )

    def test_missing_or_duplicate_entrypoints_fail_before_moving_files(self):
        """Reject changed HTML before rewriting staging."""
        for html in ["", '<script src="app.js"></script>' * 2]:
            with (
                self.subTest(html=html),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                (root / "index.html").write_text(html)
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    MODULE.version_assets(root)
                self.assertEqual(list(root.iterdir()), [root / "index.html"])


if __name__ == "__main__":
    unittest.main()
