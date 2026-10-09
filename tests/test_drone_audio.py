"""Verify that the original weapon audio is reproducible and finite."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "drone_audio", ROOT / "scripts/generate_drone_audio.py"
)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class DroneAudio(unittest.TestCase):
    """Compare generated PCM with committed assets without an audio device."""

    def test_generation_matches_committed_mono_wave_files(self):
        """Both profiles must retain their exact sample count and encoding."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            GENERATOR.generate(output)
            profiles = [
                ("drone-charge.wav", 17_640),
                ("drone-shot.wav", 2_646),
            ]
            for filename, frames in profiles:
                generated = output / filename
                committed = ROOT / "assets/robots" / filename
                self.assertEqual(
                    generated.read_bytes(), committed.read_bytes()
                )
                with wave.open(str(generated), "rb") as sound:
                    self.assertEqual(sound.getnchannels(), 1)
                    self.assertEqual(sound.getsampwidth(), 2)
                    self.assertEqual(sound.getframerate(), 22_050)
                    self.assertEqual(sound.getnframes(), frames)


if __name__ == "__main__":
    unittest.main()
