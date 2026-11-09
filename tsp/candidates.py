"""K-nearest-neighbour candidate lists.

Candidate lists are the single most important structural optimisation for
local search on the TSP: they turn an O(n^2) neighbourhood scan into O(nK)
per pass, at a quality cost that is negligible for K around 8-12
(Johnson & McGeoch, 1997).

The returned lists are sorted by increasing distance and never contain the
city itself, which lets callers prune the inner loop with a single
``if distance >= best_so_far: break``.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is optional
    _np = None


_ARRAY_FALLBACK_LIMIT = 3000


def nearest_neighbours(distances: Sequence[Sequence[float]], k: int) -> list[list[int]]:
    """Return, for every city, its ``k`` nearest neighbours sorted by distance."""
    n = len(distances)
    if n <= 1:
        return [[] for _ in range(n)]
    k = max(1, min(int(k), n - 1))
    if _np is not None and (isinstance(distances, _np.ndarray) or n <= _ARRAY_FALLBACK_LIMIT):
        try:
            return _nearest_numpy(distances, k)
        except (TypeError, ValueError):
            pass
    return _nearest_python(distances, k)


def _nearest_numpy(distances, k: int) -> list[list[int]]:
    matrix = _np.asarray(distances, dtype=_np.float64)
    n = matrix.shape[0]
    width = min(k + 1, n)
    # argpartition is O(n) per row, so the whole build is O(n^2) with a tiny constant.
    partitioned = _np.argpartition(matrix, width - 1, axis=1)[:, :width]
    rows = _np.arange(n)[:, None]
    keep = partitioned != rows
    result: list[list[int]] = []
    for city in range(n):
        candidates = partitioned[city][keep[city]][:k]
        order = _np.argsort(matrix[city, candidates], kind="stable")
        result.append(candidates[order].tolist())
    return result


def _nearest_python(distances, k: int) -> list[list[int]]:
    result: list[list[int]] = []
    for city, row in enumerate(distances):
        closest = heapq.nsmallest(k + 1, range(len(row)), key=row.__getitem__)
        result.append([other for other in closest if other != city][:k])
    return result
