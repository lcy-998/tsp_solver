"""Near-neighbour graphs for instances where a dense distance matrix is too big.

LKH-style heuristics only ever look at a handful of edges per city, so there is
no reason to materialise all n^2 distances just to find them.  This module
builds a symmetric k-nearest-neighbour graph directly from coordinates:

* a uniform grid gives an exact k-nearest-neighbour graph in O(nk) time and
  O(nk) memory for reasonably spread-out points;
* a blocked numpy fallback keeps the answer exact when points are so clustered
  that the grid would degenerate into one cell.

The result is deliberately a *graph*, not a ranked candidate list: callers use
it both to restrict the 1-tree relaxation and to order alpha-nearness values.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

try:  # pragma: no cover - numpy is optional
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

Point = tuple[float, float]


def _squared_distance(points: Sequence[Point], a: int, b: int) -> float:
    xa, ya = points[a]
    xb, yb = points[b]
    return (xa - xb) ** 2 + (ya - yb) ** 2


def _symmetrize(points: Sequence[Point], rows: list[list[int]]) -> list[list[int]]:
    """Union each row with its reverse edges, then order by distance."""
    n = len(points)
    neighbours: list[set[int]] = [set(row) for row in rows]
    for city, row in enumerate(rows):
        for other in row:
            neighbours[other].add(city)
    result: list[list[int]] = []
    for city in range(n):
        choices = [other for other in neighbours[city] if other != city]
        choices.sort(key=lambda other: (_squared_distance(points, city, other), other))
        result.append(choices)
    return result


def _blocked_neighbours(points: Sequence[Point], k: int, symmetric: bool,
                        chunk_size: int) -> list[list[int]]:
    """Exact k-nearest neighbours, one row block at a time (O(n) memory)."""
    n = len(points)
    if _np is None:
        rows: list[list[int]] = []
        for city, (x, y) in enumerate(points):
            closest = heapq.nsmallest(
                k + 1, range(n),
                key=lambda other: _squared_distance(points, city, other),
            )
            rows.append([other for other in closest if other != city][:k])
        return _symmetrize(points, rows) if symmetric else rows

    xs = _np.fromiter((point[0] for point in points), dtype=_np.float64, count=n)
    ys = _np.fromiter((point[1] for point in points), dtype=_np.float64, count=n)
    width = min(k + 1, n)
    rows = []
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        block = _np.hypot(xs[start:stop, None] - xs[None, :],
                          ys[start:stop, None] - ys[None, :])
        # Degenerate inputs have many exactly tied distances (e.g. points on a
        # line); a stable full sort makes tie breaking match the grid path.
        part = _np.argsort(block, axis=1, kind="stable")[:, :width]
        for local, candidates in enumerate(part):
            city = start + local
            rows.append([
                int(other) for other in candidates if int(other) != city
            ][:k])
    return _symmetrize(points, rows) if symmetric else rows


def _grid_neighbours(points: Sequence[Point], k: int, symmetric: bool) -> list[list[int]]:
    """Exact k-nearest neighbours using expanding rings of a uniform grid."""
    n = len(points)
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span_x, span_y = max_x - min_x, max_y - min_y
    axis = max(1, int(math.sqrt(n / 4.0)))
    cell_x = (span_x or 1.0) / axis
    cell_y = (span_y or 1.0) / axis
    scale = min(cell_x, cell_y)

    cells: dict[tuple[int, int], list[int]] = {}
    cell_of: list[tuple[int, int]] = []
    for city, (x, y) in enumerate(points):
        cx = min(axis - 1, int((x - min_x) / cell_x))
        cy = min(axis - 1, int((y - min_y) / cell_y))
        cell_of.append((cx, cy))
        cells.setdefault((cx, cy), []).append(city)

    rows: list[list[int]] = []
    for city, (cx, cy) in enumerate(cell_of):
        heap: list[tuple[float, int]] = []
        radius = 0
        while True:
            for gx in range(cx - radius, cx + radius + 1):
                for gy in range(cy - radius, cy + radius + 1):
                    if max(abs(gx - cx), abs(gy - cy)) != radius:
                        continue
                    for other in cells.get((gx, gy), ()):
                        if other == city:
                            continue
                        distance = _squared_distance(points, city, other)
                        if len(heap) < k:
                            heapq.heappush(heap, (-distance, other))
                        elif (-distance, other) > heap[0]:
                            heapq.heapreplace(heap, (-distance, other))
            # Any cell outside the searched rings is at least radius full cell
            # widths away, so the k-th neighbour found so far cannot be beaten.
            if len(heap) == k and -heap[0][0] <= (radius * scale) ** 2 + 1e-15:
                break
            if radius >= 2 * axis + 1:
                break
            radius += 1
        ordered = sorted(((-neg, other) for neg, other in heap),
                         key=lambda item: (item[0], item[1]))
        rows.append([other for _, other in ordered])
    return _symmetrize(points, rows) if symmetric else rows


def nearest_neighbour_graph(
    points: Sequence[Point],
    k: int,
    *,
    symmetric: bool = True,
    chunk_size: int = 256,
) -> list[list[int]]:
    """Return the k nearest neighbours of each city as a symmetric graph.

    Rows are ordered by Euclidean distance.  Because reverse edges are added,
    a row may contain up to ``2k`` entries; callers that need exactly k should
    take the first k.  The graph never stores an n-by-n distance matrix.
    """
    n = len(points)
    if n <= 1:
        return [[] for _ in range(n)]
    k = max(1, min(int(k), n - 1))
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span_x, span_y = max_x - min_x, max_y - min_y
    degenerate = span_x <= 1e-12 or span_y <= 1e-12
    if degenerate:
        return _blocked_neighbours(points, k, symmetric, chunk_size)
    return _grid_neighbours(points, k, symmetric)
