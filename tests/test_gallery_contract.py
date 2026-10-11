"""Keep the examples index tied to real robot lessons and motion previews."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import unittest


class Catalog(HTMLParser):
    """Collect example links and their preview media."""

    def __init__(self) -> None:
        """Start with no example links or media."""
        super().__init__()
        self.links = []
        self.videos = []

    def handle_starttag(self, tag, attrs):
        """Retain the attributes that define routes and preview playback."""
        attributes = dict(attrs)
        if tag == "a" and attributes.get("class") == "example":
            self.links.append(attributes["href"])
        if tag == "video":
            self.videos.append(attributes)


class GalleryContractTests(unittest.TestCase):
    """Expose droid prerequisites and honest six-agent playback."""

    def test_routes_and_looping_previews_match_the_available_lessons(self):
        """Each lesson has a preview and a visible route."""
        root = Path(__file__).resolve().parents[1]
        html = (root / "gallery/index.html").read_text()
        records = json.loads((root / "gallery/recordings.json").read_text())
        catalog = Catalog()
        catalog.feed(html)
        self.assertIn("Drones versus droids", html)
        self.assertIn("../robots/skills/world/", catalog.links)
        self.assertIn("../robots/skills/standing/", catalog.links)
        self.assertNotIn("../robots/pursuit/", catalog.links)
        self.assertEqual(len(catalog.videos), len(catalog.links))
        self.assertEqual(len({v["data-src"] for v in catalog.videos}),
                         len(catalog.videos))
        for video in catalog.videos:
            self.assertIn("loop", video)
            self.assertIn("muted", video)
            self.assertIn("playsinline", video)
            self.assertNotIn("autoplay", video)
            self.assertTrue((root / "gallery" / video["poster"]).is_file())
            path = root / "gallery" / video["data-src"]
            self.assertTrue(path.is_file())
            record = records[path.stem]
            self.assertEqual(record["duration_seconds"], 10.0)
            self.assertEqual(record["frames"], 240)
            self.assertEqual(record["sha256"],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            poster = root / "gallery" / video["poster"]
            self.assertEqual(record["poster_sha256"],
                             hashlib.sha256(poster.read_bytes()).hexdigest())
        self.assertIn("Walking", html)
        self.assertIn("Not implemented", html)

    def test_published_legacy_routes_open_policy_only_scenes(self):
        """Existing robot links must open the replacement PPO viewers."""
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/pages.yml").read_text()
        self.assertNotIn("scripts/build_pursuit_viewer.sh", workflow)
        self.assertNotIn("scripts/build_drone_viewer.sh", workflow)
        self.assertIn(
            "cp -a robot-web/skills-dist-release/world/. site/robots/pursuit/",
            workflow,
        )
        self.assertIn(
            "cp -a robot-web/skills-dist-release/hover/. site/robots/hover/",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
