"""Create a finite Burn 0.21 record whose recurrent products overflow."""

import math
from pathlib import Path
import struct

import msgpack

ROOT = Path(__file__).resolve().parents[2]
MAXIMUM = float.fromhex("0x1.fffffep127")


def fill(parameter, value, shape=None):
    """Replace one tensor with repeated little-endian float32 values."""
    data = parameter["param"]
    if shape is not None:
        data["shape"] = shape
    data["bytes"] = struct.pack("<f", value) * math.prod(data["shape"])


def main():
    """Construct an adversarial actor in the committed record envelope."""
    source = ROOT / "models/mountain-car-continuous.mpk"
    record = msgpack.unpackb(source.read_bytes())
    actor = record["item"]["actor"]
    for name in ("input_gate", "forget_gate", "output_gate", "cell_gate"):
        gate = actor["memory"][name]
        transform = gate["input_transform"]
        fill(transform["weight"], 0.0, [3, 32])
        # Positive cosine and the maximum bias overflow on the first action.
        transform["weight"]["param"]["bytes"] = (
            struct.pack("<f", MAXIMUM) * 32 + bytes(2 * 32 * 4)
        )
        fill(transform["bias"], MAXIMUM)
        # Each gate adds positive and negative infinity on the second action.
        fill(gate["hidden_transform"]["weight"], -MAXIMUM)
        fill(gate["hidden_transform"]["bias"], 0.0)
    fill(actor["mean_head"]["weight"], 1.0)
    fill(actor["mean_head"]["bias"], 0.0)
    fill(actor["log_std"], -0.5)
    fill(record["item"]["critic"]["layers"][0]["weight"], 0.0, [3, 64])
    output = Path(__file__).with_name("pendulum-overflow.mpk")
    output.write_bytes(msgpack.packb(record, use_bin_type=True))


if __name__ == "__main__":
    main()
