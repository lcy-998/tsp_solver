"""Minimum 1-trees and Held--Karp ascent for symmetric TSP relaxations.

The routines in this module use zero-based city ids and symmetric distance
matrices.  Node zero is the special node of the 1-tree.  Node penalties only
change the costs used by the Held--Karp relaxation; they never change the
actual length of a tour.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from math import inf
from typing import Sequence

Edge = tuple[int, int]


def _edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


@dataclass
class OneTree:
    """A minimum 1-tree under node-penalised edge costs."""

    edges: frozenset[Edge]
    tree_edges: frozenset[Edge]
    special_edges: tuple[Edge, ...]
    adjacency: list[list[int]]
    mst_adjacency: list[list[int]]
    degrees: list[int]
    modified_cost: float
    lower_bound: float
    penalties: list[float]
    # "complete" uses every edge; "candidate" restricts the 1-tree to a
    # near-neighbour graph, in which case the bound is a bound on that graph's
    # relaxation and may overestimate the true TSP lower bound.
    scope: str = "complete"


@dataclass
class AscentResult:
    """Best lower bound and penalties found by Held--Karp ascent."""

    penalties: list[float]
    one_tree: OneTree
    lower_bound: float
    iterations: int


def minimum_one_tree(
    distances: Sequence[Sequence[float]],
    penalties: Sequence[float] | None = None,
    candidates: Sequence[Sequence[int]] | None = None,
) -> OneTree:
    """Compute a minimum 1-tree using Prim's algorithm.

    The spanning tree covers cities 1..n-1.  The two cheapest penalised edges
    incident to city zero complete the 1-tree.  Deterministic tie breaking is
    important because the resulting alpha candidate lists must be reproducible.

    Without ``candidates`` every edge is considered, which costs O(n^2).  With
    a near-neighbour graph only O(nK) edges are examined; the resulting tree is
    minimum *within that graph* (which is what LKH does on large instances).
    """
    n = len(distances)
    if n == 0:
        raise ValueError("a 1-tree needs at least one city")
    if candidates is not None:
        return _sparse_minimum_one_tree(distances, penalties, candidates)
    pi = [0.0] * n if penalties is None else list(penalties)
    if len(pi) != n:
        raise ValueError("there must be one penalty per city")
    if n == 1:
        return OneTree(frozenset(), frozenset(), (), [[]], [[]], [0], 0.0, 0.0, pi)
    if n == 2:
        # The 2-city 1-tree is a multigraph with two parallel edges.
        edge = (0, 1)
        cost = 2.0 * (distances[0][1] + pi[0] + pi[1])
        return OneTree(
            frozenset((edge,)), frozenset(), (edge, edge), [[1], [0]], [[], []],
            [2, 2], cost, cost - 2.0 * sum(pi), pi,
        )

    key = [inf] * n
    parent = [-1] * n
    used = bytearray(n)
    key[1] = 0.0
    tree_edges: set[Edge] = set()
    mst_adjacency = [[] for _ in range(n)]
    tree_cost = 0.0

    for _ in range(n - 1):
        city = min(
            (other for other in range(1, n) if not used[other]),
            key=lambda other: (key[other], other),
        )
        used[city] = 1
        if parent[city] >= 0:
            previous = parent[city]
            edge = _edge(previous, city)
            tree_edges.add(edge)
            mst_adjacency[previous].append(city)
            mst_adjacency[city].append(previous)
            tree_cost += key[city]
        row = distances[city]
        for other in range(1, n):
            if used[other]:
                continue
            cost = row[other] + pi[city] + pi[other]
            if cost < key[other] or (cost == key[other] and city < parent[other]):
                key[other] = cost
                parent[other] = city

    special = sorted(
        ((distances[0][other] + pi[0] + pi[other], other) for other in range(1, n)),
        key=lambda item: (item[0], item[1]),
    )[:2]
    special_edges = tuple(_edge(0, other) for _, other in special)
    edges = tree_edges | set(special_edges)
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for neighbours in adjacency:
        neighbours.sort()
    for neighbours in mst_adjacency:
        neighbours.sort()
    modified_cost = tree_cost + special[0][0] + special[1][0]
    return OneTree(
        frozenset(edges),
        frozenset(tree_edges),
        special_edges,
        adjacency,
        mst_adjacency,
        [len(neighbours) for neighbours in adjacency],
        modified_cost,
        modified_cost - 2.0 * sum(pi),
        pi,
    )


def _sparse_minimum_one_tree(
    distances: Sequence[Sequence[float]],
    penalties: Sequence[float] | None,
    candidates: Sequence[Sequence[int]],
) -> OneTree:
    """Minimum 1-tree restricted to a symmetric near-neighbour graph."""
    n = len(distances)
    pi = [0.0] * n if penalties is None else list(penalties)
    if len(pi) != n:
        raise ValueError("there must be one penalty per city")
    if n <= 2:
        return replace(minimum_one_tree(distances, pi), scope="candidate")

    key = [inf] * n
    parent = [-1] * n
    used = bytearray(n)
    key[1] = 0.0
    heap = [(0.0, 1)]
    tree_edges: set[Edge] = set()
    mst_adjacency: list[list[int]] = [[] for _ in range(n)]
    tree_cost = 0.0

    while heap:
        value, city = heapq.heappop(heap)
        if used[city] or value > key[city] + 1e-12:
            continue
        used[city] = 1
        if parent[city] >= 0:
            previous = parent[city]
            tree_edges.add(_edge(previous, city))
            mst_adjacency[previous].append(city)
            mst_adjacency[city].append(previous)
            tree_cost += value
        for other in candidates[city]:
            if other <= 0 or other >= n or other == city or used[other]:
                continue
            cost = distances[city][other] + pi[city] + pi[other]
            if (cost < key[other] - 1e-12
                    or (abs(cost - key[other]) <= 1e-12 and city < parent[other])):
                key[other] = cost
                parent[other] = city
                heapq.heappush(heap, (cost, other))

    if sum(used[1:]) != n - 1:
        # A disconnected candidate graph cannot contain a spanning tree.  Fall
        # back to the complete relaxation rather than silently binding a
        # forest; with a k-nearest-neighbour graph this is not normally hit.
        return replace(minimum_one_tree(distances, pi), scope="candidate")

    pool = [other for other in candidates[0] if 0 < other < n]
    if len(pool) < 2:
        pool = list(range(1, n))
    special = sorted(
        ((distances[0][other] + pi[0] + pi[other], other) for other in pool),
        key=lambda item: (item[0], item[1]),
    )[:2]
    special_edges = tuple(_edge(0, other) for _, other in special)
    edges = tree_edges | set(special_edges)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for neighbours in adjacency:
        neighbours.sort()
    for neighbours in mst_adjacency:
        neighbours.sort()
    modified_cost = tree_cost + special[0][0] + special[1][0]
    return OneTree(
        frozenset(edges),
        frozenset(tree_edges),
        special_edges,
        adjacency,
        mst_adjacency,
        [len(neighbours) for neighbours in adjacency],
        modified_cost,
        modified_cost - 2.0 * sum(pi),
        pi,
        "candidate",
    )


def held_karp_ascent(
    distances: Sequence[Sequence[float]],
    upper_bound: float,
    max_iterations: int = 50,
    candidates: Sequence[Sequence[int]] | None = None,
) -> AscentResult:
    """Maximise the minimum 1-tree lower bound by subgradient ascent.

    With penalties pi, edge costs become d[i][j] + pi[i] + pi[j], while the
    corresponding TSP lower bound is L(T_pi) - 2*sum(pi).  The subgradient at
    city i is degree(i)-2.  A Polyak-style step uses a valid tour length as
    the upper bound and is halved after each period.  When ``candidates`` is
    given, every 1-tree is built on that graph, so the resulting bound applies
    to the restricted relaxation (see ``OneTree.scope``).
    """
    n = len(distances)
    if n == 0:
        raise ValueError("cannot run ascent on an empty instance")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    penalties = [0.0] * n
    best_tree = minimum_one_tree(distances, penalties, candidates)
    best_penalties = penalties[:]
    best_bound = best_tree.lower_bound
    if n <= 2 or max_iterations == 0:
        return AscentResult(best_penalties, best_tree, best_bound, 0)

    step_scale = 1.0
    period = max(10, min(50, n))
    iterations = 0
    for iteration in range(max_iterations):
        tree = minimum_one_tree(distances, penalties, candidates)
        if tree.lower_bound > best_bound:
            best_bound = tree.lower_bound
            best_penalties = penalties[:]
            best_tree = tree
        subgradient = [degree - 2 for degree in tree.degrees]
        denominator = sum(value * value for value in subgradient)
        if denominator == 0:
            # A minimum 1-tree that is already a tour proves optimality.
            best_tree = tree
            best_penalties = penalties[:]
            best_bound = tree.lower_bound
            iterations = iteration + 1
            break
        gap = max(0.0, upper_bound - tree.lower_bound)
        if gap <= 1e-15:
            iterations = iteration + 1
            break
        step = step_scale * gap / denominator
        if step <= 1e-15:
            iterations = iteration + 1
            break
        for city, value in enumerate(subgradient):
            penalties[city] += step * value
        iterations = iteration + 1
        if (iteration + 1) % period == 0:
            step_scale *= 0.5

    return AscentResult(best_penalties, best_tree, best_bound, iterations)
