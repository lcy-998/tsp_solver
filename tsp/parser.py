from collections.abc import Iterator
from pathlib import Path

from .models import TSPInstance


def parse_line(
    line: str, *, name: str | None = None, distance_matrix_threshold: int | None = None
) -> TSPInstance:
    content = line.split("#", 1)[0].strip()
    if not content:
        raise ValueError("cannot parse an empty TSP line")
    parts = content.split()
    try:
        marker = next(i for i, part in enumerate(parts) if part.lower() == "output")
    except StopIteration:
        marker = len(parts)
    coordinate_tokens = parts[:marker]
    if len(coordinate_tokens) == 0 or len(coordinate_tokens) % 2:
        raise ValueError("coordinate section must contain pairs of numbers")
    try:
        values = [float(token) for token in coordinate_tokens]
    except ValueError as exc:
        raise ValueError("coordinate section contains a non-number") from exc
    coordinates = list(zip(values[::2], values[1::2]))
    reference_route = None
    if marker < len(parts):
        if marker == len(parts) - 1:
            raise ValueError("output marker must be followed by a route")
        try:
            reference_route = [int(token) for token in parts[marker + 1 :]]
        except ValueError as exc:
            raise ValueError("reference route contains a non-integer") from exc
        _validate_route(reference_route, len(coordinates))
    return TSPInstance(coordinates, reference_route, name, distance_matrix_threshold)


def load_instances(
    path: str | Path, *, distance_matrix_threshold: int | None = None
) -> list[TSPInstance]:
    return list(iter_instances(path, distance_matrix_threshold=distance_matrix_threshold))


def iter_instances(
    path: str | Path, *, distance_matrix_threshold: int | None = None
) -> Iterator[TSPInstance]:
    """Yield instances one at a time for large datasets."""
    instances_seen = 0
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.split("#", 1)[0].strip():
                continue
            try:
                instance = parse_line(
                    raw_line,
                    name=f"instance-{instances_seen + 1}",
                    distance_matrix_threshold=distance_matrix_threshold,
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            instances_seen += 1
            yield instance


def _validate_route(route: list[int], size: int) -> None:
    if len(route) < size + 1 or route[0] != route[-1]:
        raise ValueError("route must visit all cities and return to its start")
    if set(route[:-1]) != set(range(1, size + 1)):
        raise ValueError("route must contain each city exactly once before returning")
