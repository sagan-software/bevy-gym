#!/usr/bin/env python3
"""Render ecosystem JSONL training metrics without third-party Python packages."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable

WIDTH = 1200
HEIGHT = 900
MARGIN_X = 70
MARGIN_Y = 70
PANEL_GAP = 42
PANEL_WIDTH = (WIDTH - MARGIN_X * 2 - PANEL_GAP) / 2
PANEL_HEIGHT = (HEIGHT - MARGIN_Y * 2 - PANEL_GAP * 2) / 3
BACKGROUND = "#10141d"
PANEL = "#171d29"
GRID = "#30394b"
TEXT = "#e6edf7"
MUTED = "#8f9db3"
COLORS = ["#ffb454", "#5cc8ff", "#7ee787", "#d2a8ff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state", type=Path)
    return parser.parse_args()


def finite_number(record: dict[str, object], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def points(records: Iterable[dict[str, object]], key: str) -> list[tuple[float, float]]:
    result = []
    for record in records:
        x = finite_number(record, "global_step")
        y = finite_number(record, key)
        if x is not None and y is not None:
            result.append((x, y))
    return result


def scaled_points(
    records: Iterable[dict[str, object]], key: str, scale: float
) -> list[tuple[float, float]]:
    return [(x, y * scale) for x, y in points(records, key)]


def xml_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line_path(
    samples: list[tuple[float, float]],
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-12)
    commands = []
    for index, (x, y) in enumerate(samples):
        px = x0 + (x - x_min) / x_span * width
        py = y0 + height - (y - y_min) / y_span * height
        commands.append(f"{'M' if index == 0 else 'L'} {px:.2f} {py:.2f}")
    return " ".join(commands)


def panel_svg(
    row: int,
    column: int,
    title: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    x_max: float,
) -> str:
    x0 = MARGIN_X + column * (PANEL_WIDTH + PANEL_GAP)
    y0 = MARGIN_Y + row * (PANEL_HEIGHT + PANEL_GAP)
    plot_x = x0 + 54
    plot_y = y0 + 34
    plot_width = PANEL_WIDTH - 76
    plot_height = PANEL_HEIGHT - 70
    values = [value for _, samples in series for _, value in samples]
    if values:
        y_min = min(values)
        y_max = max(values)
        padding = max((y_max - y_min) * 0.12, abs(y_max) * 0.03, 1e-9)
        y_min -= padding
        y_max += padding
    else:
        y_min, y_max = 0.0, 1.0

    svg = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{PANEL_WIDTH:.1f}" height="{PANEL_HEIGHT:.1f}" rx="10" fill="{PANEL}"/>',
        f'<text x="{x0 + 16:.1f}" y="{y0 + 23:.1f}" fill="{TEXT}" font-size="16" font-weight="600">{xml_escape(title)}</text>',
    ]
    for grid_index in range(5):
        gy = plot_y + plot_height * grid_index / 4
        value = y_max - (y_max - y_min) * grid_index / 4
        svg.append(f'<line x1="{plot_x:.1f}" y1="{gy:.1f}" x2="{plot_x + plot_width:.1f}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{plot_x - 7:.1f}" y="{gy + 4:.1f}" text-anchor="end" fill="{MUTED}" font-size="10">{value:.3g}</text>')
    for index, (label, samples) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        if samples:
            path = line_path(samples, plot_x, plot_y, plot_width, plot_height, 0.0, x_max, y_min, y_max)
            svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
            x_span = max(x_max, 1.0)
            y_span = max(y_max - y_min, 1e-12)
            for sample_x, sample_y in samples:
                point_x = plot_x + sample_x / x_span * plot_width
                point_y = plot_y + plot_height - (sample_y - y_min) / y_span * plot_height
                svg.append(
                    f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="3.2" '
                    f'fill="{color}" stroke="{PANEL}" stroke-width="1"/>'
                )
        legend_spacing = plot_width / max(len(series), 1)
        legend_x = plot_x + index * legend_spacing
        svg.append(f'<line x1="{legend_x:.1f}" y1="{y0 + PANEL_HEIGHT - 14:.1f}" x2="{legend_x + 20:.1f}" y2="{y0 + PANEL_HEIGHT - 14:.1f}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 26:.1f}" y="{y0 + PANEL_HEIGHT - 10:.1f}" fill="{MUTED}" font-size="11">{xml_escape(label)}</text>')
    svg.append(f'<text x="{plot_x + plot_width / 2:.1f}" y="{y0 + PANEL_HEIGHT + 15:.1f}" text-anchor="middle" fill="{MUTED}" font-size="11">environment steps</text>')
    return "\n".join(svg)


def render(records: list[dict[str, object]], output: Path) -> int:
    training = [record for record in records if "train/iteration" in record]
    if not training:
        return 0
    iteration = int(finite_number(training[-1], "train/iteration") or 0)
    x_max = max(finite_number(record, "global_step") or 0.0 for record in records)
    x_max = max(x_max, 1.0)
    eval_records = [record for record in records if "eval/bunny_mean_lifetime" in record]

    panels = [
        ("Actor loss", [("actor", points(training, "train/actor_loss"))]),
        ("Critic loss", [("critic", points(training, "train/critic_loss"))]),
        ("Policy entropy", [("entropy", points(training, "train/entropy"))]),
        ("Approximate KL", [("KL", points(training, "train/approximate_kl"))]),
        (
            "Bunny lifetime",
            [
                ("train mean", points(training, "train/bunny_mean_lifetime")),
                ("eval mean", points(eval_records, "eval/bunny_mean_lifetime")),
                ("eval CI low", points(eval_records, "eval/bunny_lifetime_ci95_lower")),
                ("eval CI high", points(eval_records, "eval/bunny_lifetime_ci95_upper")),
            ],
        ),
        (
            "Optimizer work",
            [
                ("valid samples", points(training, "train/valid_samples")),
                ("updates ×80", scaled_points(training, "train/optimizer_updates", 80.0)),
            ],
        ),
    ]
    body = [panel_svg(index // 2, index % 2, title, series, x_max) for index, (title, series) in enumerate(panels)]
    subtitle = f"sparse-food survival · iteration {iteration} · {int(x_max):,} environment steps"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<rect width="100%" height="100%" fill="{BACKGROUND}"/>
<text x="{MARGIN_X}" y="34" fill="{TEXT}" font-size="24" font-weight="700">Ecosystem Survival Training Progress</text>
<text x="{MARGIN_X}" y="55" fill="{MUTED}" font-size="13">{xml_escape(subtitle)}</text>
{''.join(body)}
</svg>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output.with_suffix(".svg")
    svg_path.write_text(svg, encoding="utf-8")
    subprocess.run(
        ["magick", "-background", BACKGROUND, svg_path, output],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return iteration


def main() -> None:
    args = parse_args()
    records = read_records(args.metrics)
    training_count = sum("train/iteration" in record for record in records)
    if training_count == 0:
        return
    if args.state and args.state.exists():
        try:
            if int(args.state.read_text(encoding="utf-8").strip()) >= training_count:
                return
        except ValueError:
            pass
    iteration = render(records, args.output)
    if args.state:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(str(training_count), encoding="utf-8")
    print(f"Sparse-food survival training progress: iteration {iteration}")
    print(f"MEDIA:{args.output.resolve()}")


if __name__ == "__main__":
    main()
