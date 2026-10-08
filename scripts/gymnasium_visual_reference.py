"""Download official environment GIFs and make six-frame reference sheets."""

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7a1191388aa4aa973d3a5e4b039899cd99cc991f"
actual_revision = subprocess.run(
    ["git", "-C", str(ROOT / "ref/gymnasium"), "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True, timeout=10,
).stdout.strip()
if actual_revision != REVISION:
    raise RuntimeError(f"Expected Gymnasium {REVISION}; got {actual_revision}")
SOURCE = ROOT / "ref/gymnasium/docs/_static/videos"
DOWNLOADS = ROOT / "runs/visual-references"
OUTPUT = ROOT / "docs/visual-comparisons/references"
DOWNLOADS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
records = []
for family in ("classic_control", "toy_text", "box2d", "mujoco"):
    for original in sorted((SOURCE / family).glob("*.gif")):
        name = original.stem
        url = f"https://gymnasium.farama.org/_images/{name}.gif"
        with urllib.request.urlopen(url, timeout=30) as response:
            contents = response.read()
        if not contents.startswith((b"GIF87a", b"GIF89a")):
            raise ValueError(f"Expected a GIF from {url}")
        downloaded = DOWNLOADS / original.name
        downloaded.write_bytes(contents)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(downloaded)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        sheet = OUTPUT / f"{name}.png"
        filters = (
            f"fps=6/{duration},"
            "scale=300:200:force_original_aspect_ratio=decrease,"
            "pad=300:200:(ow-iw)/2:(oh-ih)/2:white,"
            "tile=3x2:padding=4:margin=4:color=white"
        )
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(downloaded), "-vf", filters, "-frames:v", "1",
             str(sheet)],
            check=True, timeout=30,
        )
        records.append({
            "environment": name,
            "family": family,
            "url": url,
            "sha256": hashlib.sha256(contents).hexdigest(),
            "duration_seconds": duration,
            "matches_pinned_repository": contents == original.read_bytes(),
            "contact_sheet": sheet.name,
        })
        print(f"{family}/{name}: {duration}s", flush=True)
(OUTPUT / "manifest.json").write_text(json.dumps({
    "source_revision": REVISION,
    "frames_per_sheet": 6,
    "sampling": "FFmpeg fps=6/duration, chronological left to right",
    "references": records,
}, indent=2) + "\n")
