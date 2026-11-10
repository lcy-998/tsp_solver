"""Reusable candidate lists for TSP construction and local search.

``nearest_neighbours`` orders edges by ordinary distance.  The alpha-nearness
routines instead use a minimum 1-tree, optionally improved by Held--Karp node
penalties, to estimate how likely an edge is to belong to an optimal tour.
Every returned candidate list excludes the city itself.

Passing a ``candidates`` near-neighbour graph to the alpha routines restricts
1-trees and alpha values to O(nK) edges (using a binary-lifting path maximum
instead of one DFS per city), which is what makes 10000+ city instances
tractable.  Without it the full O(n^2) relaxation is used.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from math import inf

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
    candidates: Sequence[Sequence[int]] | None = None,
) -> list[float]:
    """Return alpha-nearness from ``city`` to every other city.

    For a non-tree edge, adding it to the MST creates one cycle.  Forcing the
    edge into a minimum 1-tree therefore costs the new edge minus the most
    expensive edge on that cycle.  At the special node, one of its two edges
    is replaced instead.  Tree edges have alpha zero by definition.
    """
    n = len(distances)
    if candidates is not None:
        return _sparse_alpha_values(city, distances, penalties, one_tree, candidates)
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
    candidates: Sequence[Sequence[int]] | None = None,
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
    if candidates is not None:
        return _sparse_candidate_sets(distances, k, penalties, one_tree, candidates)
    tree = minimum_one_tree(distances, penalties) if one_tree is None else one_tree
    result: list[list[int]] = []
    for city in range(n):
        alpha = alpha_values_for_city(city, distances, penalties, tree)
        choices = [other for other in range(n) if other != city]
        choices.sort(key=lambda other: (alpha[other], distances[city][other], other))
        result.append(choices[:k])
    return result


class _PathMaximum:
    """Max edge cost on MST paths, via binary lifting (O(n log n) setup)."""

    def __init__(self, tree: OneTree, distances, pi: Sequence[float]) -> None:
        n = len(distances)
        self.log = max(1, n.bit_length())
        self.up = [[0] * n for _ in range(self.log)]
        self.mx = [[0.0] * n for _ in range(self.log)]
        self.depth = [0] * n
        self.up[0][1] = 1
        seen = bytearray(n)
        seen[1] = 1
        stack = [1]
        while stack:
            current = stack.pop()
            for other in tree.mst_adjacency[current]:
                if other == 0 or seen[other]:
                    continue
                seen[other] = 1
                self.up[0][other] = current
                self.mx[0][other] = distances[current][other] + pi[current] + pi[other]
                self.depth[other] = self.depth[current] + 1
                stack.append(other)
        for level in range(1, self.log):
            previous_up, previous_mx = self.up[level - 1], self.mx[level - 1]
            up, mx = self.up[level], self.mx[level]
            for city in range(n):
                middle = previous_up[city]
                up[city] = previous_up[middle]
                mx[city] = max(previous_mx[city], previous_mx[middle])

    def path_max(self, a: int, b: int) -> float:
        result = 0.0
        if self.depth[a] < self.depth[b]:
            a, b = b, a
        difference = self.depth[a] - self.depth[b]
        level = 0
        while difference:
            if difference & 1:
                result = max(result, self.mx[level][a])
                a = self.up[level][a]
            difference >>= 1
            level += 1
        if a == b:
            return result
        for level in range(self.log - 1, -1, -1):
            if self.up[level][a] != self.up[level][b]:
                result = max(result, self.mx[level][a], self.mx[level][b])
                a = self.up[level][a]
                b = self.up[level][b]
        return max(result, self.mx[0][a], self.mx[0][b])


def _prepare_sparse(distances, penalties, one_tree, candidates):
    tree = minimum_one_tree(distances, penalties, candidates) if one_tree is None else one_tree
    pi = tree.penalties if penalties is None else penalties
    return tree, pi, _PathMaximum(tree, distances, pi)


def _sparse_alpha(city: int, other: int, tree: OneTree, distances,
                  pi: Sequence[float], helper: _PathMaximum) -> float:
    if _edge(city, other) in tree.edges:
        return 0.0
    cost = distances[city][other] + pi[city] + pi[other]
    if city == 0 or other == 0:
        replaced = max(
            distances[0][s] + pi[0] + pi[s] for _, s in tree.special_edges
        )
        return max(0.0, cost - replaced)
    return max(0.0, cost - helper.path_max(city, other))


def _sparse_alpha_values(city, distances, penalties, one_tree, candidates):
    tree, pi, helper = _prepare_sparse(distances, penalties, one_tree, candidates)
    values = [inf] * len(distances)
    values[city] = 0.0
    for other in candidates[city]:
        if other == city or not 0 <= other < len(distances):
            continue
        values[other] = _sparse_alpha(city, other, tree, distances, pi, helper)
    return values


def _sparse_candidate_sets(distances, k, penalties, one_tree, candidates):
    """Alpha-order candidates restricted to a near-neighbour graph (O(nK log K))."""
    tree, pi, helper = _prepare_sparse(distances, penalties, one_tree, candidates)
    result: list[list[int]] = []
    for city in range(len(distances)):
        scored = []
        for other in candidates[city]:
            if other == city or not 0 <= other < len(distances):
                continue
            alpha = _sparse_alpha(city, other, tree, distances, pi, helper)
            scored.append((alpha, distances[city][other], other))
        scored.sort()
        result.append([item[2] for item in scored[:k]])
    return result
