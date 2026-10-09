"""Validate the licensed humanoid and pistol used by the survival scene."""

import hashlib
import json
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/robots/survival"


def read_glb(path):
    """Read the JSON chunk of a glTF 2 binary without importing a renderer."""
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    if (magic, version, length) != (0x46546C67, 2, len(data)):
        raise ValueError("Invalid glTF binary header")
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:
        raise ValueError("Expected glTF JSON chunk")
    return json.loads(data[20:20 + chunk_length])


class SurvivalAssets(unittest.TestCase):
    """Check actual skeleton, animation, and external dependency contracts."""

    def test_mannequin_has_authored_locomotion_and_weapon_clips(self):
        """Animation names and the hand socket must match the renderer."""
        model = read_glb(ASSETS / "mannequin.glb")
        clips = {clip["name"] for clip in model["animations"]}
        self.assertTrue({
            "Idle_Loop", "Jog_Fwd_Loop", "Sprint_Loop", "Death01",
            "Pistol_Idle_Loop", "Pistol_Shoot", "Pistol_Reload",
            "Pistol_Aim_Up", "Pistol_Aim_Down", "Pistol_Aim_Neutral",
        }.issubset(clips))
        hands = sum(n.get("name") == "hand_r" for n in model["nodes"])
        self.assertEqual(hands, 1)
        self.assertTrue(model["skins"])
        self.assertTrue(all("uri" not in b for b in model["buffers"]))

    def test_pistol_dependencies_are_bundled(self):
        """Every pistol dependency must resolve within its asset directory."""
        model = json.loads((ASSETS / "pistol/Gun_Pistol.gltf").read_text())
        for dependency in model["buffers"] + model["images"]:
            path = ASSETS / "pistol" / dependency["uri"]
            self.assertTrue(path.is_file(), dependency["uri"])
            self.assertTrue(path.read_bytes())

    def test_cover_dependencies_are_bundled(self):
        """The crate must load with its original buffer and textures."""
        path = ASSETS / "cover/Prop_Crate_Large.gltf"
        model = json.loads(path.read_text())
        for dependency in model["buffers"] + model["images"]:
            self.assertTrue((ASSETS / "cover" / dependency["uri"]).is_file())

    def test_asset_provenance_matches_committed_bytes(self):
        """Every asset has a source, license, and verified hash."""
        manifest = json.loads((ASSETS / "manifest.json").read_text())
        for asset in manifest["files"]:
            self.assertEqual(asset["license"], "CC0-1.0")
            self.assertTrue(asset["source"].startswith("https://"))
            data = (ASSETS / asset["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), asset["sha256"])
        self.assertIn("CC0", (ASSETS / "LICENSE-ANIMATIONS.txt").read_text())
        self.assertIn("CC0", (ASSETS / "LICENSE-ESSENTIALS.txt").read_text())


if __name__ == "__main__":
    unittest.main()
