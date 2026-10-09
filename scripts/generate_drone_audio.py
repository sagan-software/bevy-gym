"""Create original pursuit sounds as mono 22,050 Hz PCM WAV files."""

import math
from pathlib import Path
import random
import struct
import wave


RATE = 22_050
OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "robots"


def generate(output_directory: Path = OUTPUT) -> None:
    """Write a rising 800 ms warning and a 120 ms mechanical shot."""
    noise = random.Random(42)
    profiles = [("drone-charge.wav", 0.8), ("drone-shot.wav", 0.12)]
    for filename, duration in profiles:
        samples = []
        for index in range(round(RATE * duration)):
            seconds = index / RATE
            attack = min(seconds / 0.005, 1.0)
            release = min((duration - seconds) / 0.01, 1.0)
            if filename == "drone-charge.wav":
                # Integrate a linear frequency ramp from 180 Hz to 750 Hz.
                phase = math.tau * (180.0 * seconds + 570.0 * seconds**2 / 1.6)
                value = 0.16 * (math.sin(phase) + 0.2 * math.sin(2.0 * phase))
            else:
                body = math.sin(math.tau * 110.0 * seconds)
                transient = 0.5 * body + 0.5 * noise.uniform(-1.0, 1.0)
                value = 0.24 * math.exp(-32.0 * seconds) * transient
            sample = round(value * attack * release * 32_767)
            samples.append(struct.pack("<h", sample))
        with wave.open(str(output_directory / filename), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(RATE)
            output.writeframes(b"".join(samples))


if __name__ == "__main__":
    generate()
