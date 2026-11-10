"""Convert TSPLIB95 coordinate instances to this project's line-based format.

The converter supports plain ``.tsp`` / ``.opt.tour`` files and their gzip
variants. It only accepts symmetric ``EUC_2D`` coordinate instances, because
the project currently computes plain Euclidean distances from coordinates.

By default, coordinates are translated and divided by one common scale so the
point set fits in the unit square without changing its shape. Use
``--no-normalize`` to write the original TSPLIB coordinates instead.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path


def open_text(path: Path):
    """Open a plain or gzip-compressed text file in ASCII mode."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="ascii")
    return path.open("r", encoding="utf-8-sig")


def spec_value(text: str, key: str) -> str:
    match = re.search(rf"^{key}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing specification field: {key}")
    return match.group(1)


def parse_coordinates(path: Path) -> tuple[str, int, list[tuple[float, float]]]:
    with open_text(path) as source:
        lines = source.read().splitlines()

    header = "\n".join(lines)
    problem_type = spec_value(header, "TYPE").upper()
    if problem_type != "TSP":
        raise ValueError(f"{path}: unsupported TYPE {problem_type!r}, expected TSP")

    weight_type = spec_value(header, "EDGE_WEIGHT_TYPE").upper()
    if weight_type != "EUC_2D":
        raise ValueError(
            f"{path}: unsupported EDGE_WEIGHT_TYPE {weight_type!r}, expected EUC_2D"
        )

    dimension = int(spec_value(header, "DIMENSION"))
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if line.strip().upper() == "NODE_COORD_SECTION"
        ) + 1
    except StopIteration as exc:
        raise ValueError(f"{path}: missing NODE_COORD_SECTION") from exc

    points: dict[int, tuple[float, float]] = {}
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper == "EOF" or upper.endswith("_SECTION"):
            break
        parts = stripped.split()
        if len(parts) != 3:
            raise ValueError(f"{path}: invalid coordinate line: {stripped!r}")
        city_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
        if city_id in points:
            raise ValueError(f"{path}: duplicate city id {city_id}")
        points[city_id] = (x, y)

    expected = set(range(1, dimension + 1))
    if set(points) != expected:
        raise ValueError(
            f"{path}: expected {dimension} city ids from 1 to {dimension}, "
            f"got {len(points)}"
        )
    name = spec_value(header, "NAME")
    return name, dimension, [points[city_id] for city_id in range(1, dimension + 1)]


def normalize_coordinates(
    coordinates: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], float, float, float]:
    """Translate and uniformly scale coordinates into the unit square.

    One shared scale is used for both axes, so distances, route order, and
    relative optimality gaps are preserved under the project's Euclidean metric.
    """
    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    if scale <= 0:
        raise ValueError("cannot normalize coordinates with zero width and height")

    normalized = [
        ((x - min_x) / scale, (y - min_y) / scale) for x, y in coordinates
    ]
    return normalized, min_x, min_y, scale


def parse_tour(path: Path, dimension: int) -> list[int]:
    with open_text(path) as source:
        lines = source.read().splitlines()

    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if line.strip().upper() == "TOUR_SECTION"
        ) + 1
    except StopIteration as exc:
        raise ValueError(f"{path}: missing TOUR_SECTION") from exc

    route: list[int] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.upper() == "EOF":
            continue
        for token in stripped.split():
            city_id = int(token)
            if city_id < 0:
                break
            route.append(city_id)
        else:
            continue
        break

    expected = set(range(1, dimension + 1))
    if len(route) != dimension or set(route) != expected:
        raise ValueError(
            f"{path}: tour must contain each city from 1 to {dimension} exactly once"
        )
    return [*route, route[0]]


def convert_file(
    input_path: Path, output_dir: Path, *, normalize: bool = True
) -> tuple[Path, int, bool, str]:
    name, dimension, coordinates = parse_coordinates(input_path)
    tour_path = input_path.with_name(f"{name}.opt.tour")
    compressed_tour_path = input_path.with_name(f"{name}.opt.tour.gz")
    if tour_path.is_file():
        tour_source = tour_path
    elif compressed_tour_path.is_file():
        tour_source = compressed_tour_path
    else:
        tour_source = None

    if normalize:
        coordinates, min_x, min_y, scale = normalize_coordinates(coordinates)
        normalization_note = (
            "normalization: uniform; "
            f"offset: ({format(min_x, '.17g')}, {format(min_y, '.17g')}); "
            f"scale: {format(scale, '.17g')}"
        )
    else:
        normalization_note = "normalization: none"

    route = parse_tour(tour_source, dimension) if tour_source else None
    coordinate_tokens = [
        format(value, ".17g") for point in coordinates for value in point
    ]
    line_tokens = [*coordinate_tokens]
    if route is not None:
        line_tokens.extend(["output", *(str(city_id) for city_id in route)])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.txt"
    comment = (
        f"# source: TSPLIB95 {name}; DIMENSION: {dimension}; "
        f"EDGE_WEIGHT_TYPE: EUC_2D; {normalization_note}"
    )
    output_path.write_text(f"{comment}\n{' '.join(line_tokens)}\n", encoding="utf-8")
    return output_path, dimension, route is not None, normalization_note


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TSPLIB95 EUC_2D .tsp files to this project's text format."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="TSPLIB .tsp or .tsp.gz files")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="output directory (default: each input file's directory)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="keep the original TSPLIB coordinates instead of uniform normalization",
    )
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    for input_path in args.inputs:
        if not input_path.is_file():
            parser.error(f"input file does not exist: {input_path}")
        output_dir = args.output_dir or input_path.parent
        try:
            output_path, dimension, has_route, normalization_note = convert_file(
                input_path, output_dir, normalize=args.normalize
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        suffix = " with optimal route" if has_route else ""
        print(f"{output_path}: {dimension} cities{suffix}; {normalization_note}")


if __name__ == "__main__":
    main()
