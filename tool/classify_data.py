"""Split a line-based TSP dataset into files grouped by instance size."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TextIO


BUCKETS = ("small", "medium", "large", "ultra_large")


def classify_size(size: int) -> str:
    """Return the bucket for a TSP instance containing ``size`` points."""
    if size < 100:
        return "small"
    if size < 1_000:
        return "medium"
    if size <= 10_000:
        return "large"
    return "ultra_large"


def point_count(line: str, *, line_number: int) -> int | None:
    """Count coordinate pairs in a data line; ignore blank and comment lines."""
    content = line.split("#", 1)[0].strip()
    if not content:
        return None

    tokens = content.split()
    marker = next(
        (index for index, token in enumerate(tokens) if token.lower() == "output"),
        len(tokens),
    )
    coordinate_count = marker
    if coordinate_count == 0 or coordinate_count % 2:
        raise ValueError(
            f"line {line_number}: coordinate section must contain pairs of numbers"
        )
    return coordinate_count // 2


def split_dataset(
    input_path: Path, output_dir: Path, output_stem: str | None = None
) -> dict[str, int]:
    """Write each non-comment instance line to its size bucket."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in BUCKETS}
    output_stem = output_stem or input_path.stem
    outputs: dict[str, TextIO] = {}
    try:
        with input_path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                size = point_count(line, line_number=line_number)
                if size is None:
                    continue
                bucket = classify_size(size)
                if bucket not in outputs:
                    outputs[bucket] = (
                        output_dir / f"{output_stem}_{bucket}.txt"
                    ).open("w", encoding="utf-8", newline="\n")
                outputs[bucket].write(line.rstrip("\r\n") + "\n")
                counts[bucket] += 1
    finally:
        for output in outputs.values():
            output.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a line-based TSP text file into files grouped by point count."
    )
    parser.add_argument("input", type=Path, help="input TSP .txt file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="directory for output files (default: input file directory)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    output_dir = args.output_dir or args.input.parent
    try:
        counts = split_dataset(args.input, output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    for bucket in BUCKETS:
        if counts[bucket]:
            print(f"{args.input.stem}_{bucket}.txt: {counts[bucket]} instances")


if __name__ == "__main__":
    main()
