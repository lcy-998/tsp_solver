"""Reusable candidate lists for TSP construction and local search.

``nearest_neighbours`` orders edges by ordinary distance.  The alpha-nearness
routines instead use a minimum 1-tree, optionally improved by Held--Karp node
penalties, to estimate how likely an edge is to belong to an optimal tour.
Every returned candidate list excludes the city itself.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

from .one_tree import OneTree, minimum_one_tree

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


def _edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def alpha_values_for_city(
    city: int,
    distances: Sequence[Sequence[float]],
    penalties: Sequence[float] | None = None,
    one_tree: OneTree | None = None,
) -> list[float]:
    """Return alpha-nearness from ``city`` to every other city.

    For a non-tree edge, adding it to the MST creates one cycle.  Forcing the
    edge into a minimum 1-tree therefore costs the new edge minus the most
    expensive edge on that cycle.  At the special node, one of its two edges
    is replaced instead.  Tree edges have alpha zero by definition.
    """
    n = len(distances)
    tree = minimum_one_tree(distances, penalties) if one_tree is None else one_tree
    pi = tree.penalties if penalties is None else penalties
    values = [0.0] * n
    if n <= 2:
        return values

    def cost(a: int, b: int) -> float:
        return distances[a][b] + pi[a] + pi[b]

    if city == 0:
        replaced = max(cost(0, other) for _, other in tree.special_edges)
        for other in range(1, n):
            if _edge(city, other) not in tree.edges:
                values[other] = cost(city, other) - replaced
        return values

    # One DFS computes the maximum edge on the path from city to every node.
    # Doing this for each root gives all alpha values in O(n^2), rather than
    # climbing a path for every pair (which can be O(n^3)).
    maximum = [0.0] * n
    seen = bytearray(n)
    seen[city] = 1
    stack = [city]
    while stack:
        current = stack.pop()
        for other in tree.mst_adjacency[current]:
            if seen[other]:
                continue
            seen[other] = 1
            maximum[other] = max(maximum[current], cost(current, other))
            stack.append(other)

    for other in range(1, n):
        if other == city or _edge(city, other) in tree.edges:
            continue
        values[other] = cost(city, other) - maximum[other]
    # Alpha for a non-special edge incident to zero uses the same special-edge
    # exchange as above; the MST itself remains unchanged.
    if _edge(city, 0) not in tree.edges:
        replaced = max(cost(0, other) for _, other in tree.special_edges)
        values[0] = cost(city, 0) - replaced
    return values


def alpha_candidate_sets(
    distances: Sequence[Sequence[float]],
    k: int,
    penalties: Sequence[float] | None = None,
    one_tree: OneTree | None = None,
) -> list[list[int]]:
    """Return the k alpha-nearest neighbours of every city.

    The list is sorted by alpha-nearness, then by original distance to break
    ties deterministically.  Penalties and a precomputed 1-tree may be supplied
    together; omitting penalties uses those stored in the tree.
    """
    n = len(distances)
    if n <= 1:
        return [[] for _ in range(n)]
    k = max(1, min(int(k), n - 1))
    tree = minimum_one_tree(distances, penalties) if one_tree is None else one_tree
    result: list[list[int]] = []
    for city in range(n):
        alpha = alpha_values_for_city(city, distances, penalties, tree)
        choices = [other for other in range(n) if other != city]
        choices.sort(key=lambda other: (alpha[other], distances[city][other], other))
        result.append(choices[:k])
    return result
