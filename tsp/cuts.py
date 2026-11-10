"""Separation for the DFJ relaxation: subtour and comb inequalities.

Subtour constraints ``x(delta(S)) >= 2`` are separated exactly:

* integral solutions  - every connected component of the support graph whose
  size is smaller than ``n`` yields a violated cut;
* fractional solutions - the global minimum cut of the support graph weighted
  by ``x`` is the cheapest way to find a violated ``delta(S)``, computed with
  Stoer & Wagner's (1997) contraction algorithm.

Combs are the next rung up the ladder: for a handle ``H`` and an odd number
``k >= 3`` of pairwise disjoint teeth ``T_i`` that each straddle ``H``,

    x(delta(H)) + sum_i x(delta(T_i)) >= 3k + 1

holds for every tour.  The separator below is a heuristic: handles come from
the components of the ``x = 1`` support and from a minimum cut, teeth are grown
greedily out of the fractional boundary edges.  It never emits an inequality
that is not violated by the current ``x``, so a bad heuristic costs time, never
correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

try:  # pragma: no cover - numpy is optional
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

TOL = 1e-6


def connected_components(n: int, weights: Sequence[Sequence[float]], eps: float = TOL) -> list[list[int]]:
    """Components of the graph keeping only pairs with ``weights[i][j] > eps``."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        row = weights[i]
        for j in range(i + 1, n):
            if row[j] > eps:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    buckets: dict[int, list[int]] = {}
    for city in range(n):
        buckets.setdefault(find(city), []).append(city)
    return list(buckets.values())


def global_min_cut(n: int, weights: Sequence[Sequence[float]]) -> tuple[float, frozenset[int]]:
    """Minimum cut value of an undirected weighted graph and one side of it."""
    if n < 2:
        return 0.0, frozenset(range(n))
    if _np is not None:
        return _min_cut_numpy(n, weights)
    return _min_cut_python(n, weights)


def canonical_side(side: frozenset[int], n: int) -> frozenset[int]:
    """Represent ``S`` and its complement by one canonical set, for deduplication."""
    complement = frozenset(range(n)) - side
    left, right = sorted(side), sorted(complement)
    if len(left) != len(right):
        return side if len(left) < len(right) else complement
    return side if left < right else complement


def _min_cut_numpy(n: int, weights: Sequence[Sequence[float]]) -> tuple[float, frozenset[int]]:
    mat = _np.array(weights, dtype=float)
    groups = [frozenset((i,)) for i in range(n)]
    active = list(range(n))
    best_value = float("inf")
    best_side = frozenset(range(n))

    while len(active) > 1:
        size = len(active)
        sub = mat[_np.ix_(active, active)]
        added = _np.zeros(size, dtype=bool)
        scores = _np.zeros(size, dtype=float)
        previous = -1
        for step in range(size):
            candidate = _np.where(added, -_np.inf, scores)
            selected = int(candidate.argmax())
            added[selected] = True
            if step == size - 1:
                value = float(scores[selected])
                if value < best_value - 1e-12:
                    best_value = value
                    best_side = groups[active[selected]]
                if previous >= 0:
                    _merge(mat, active[previous], active[selected], groups)
                    active.pop(selected)
            else:
                previous = selected
                scores += sub[selected]
                scores[added] = 0.0

    return best_value, best_side


def _merge(mat, keep: int, drop: int, groups) -> None:
    mat[keep, :] += mat[drop, :]
    mat[:, keep] += mat[:, drop]
    mat[drop, :] = 0.0
    mat[:, drop] = 0.0
    groups[keep] = groups[keep] | groups[drop]


def _min_cut_python(n: int, weights: Sequence[Sequence[float]]) -> tuple[float, frozenset[int]]:
    mat = [list(row) for row in weights]
    groups = [frozenset((i,)) for i in range(n)]
    active = list(range(n))
    best_value = float("inf")
    best_side = frozenset(range(n))

    while len(active) > 1:
        size = len(active)
        added = [False] * size
        scores = [0.0] * size
        previous = -1
        for step in range(size):
            selected = -1
            for index in range(size):
                if added[index]:
                    continue
                if selected < 0 or scores[index] > scores[selected]:
                    selected = index
            added[selected] = True
            if step == size - 1:
                value = scores[selected]
                if value < best_value - 1e-12:
                    best_value = value
                    best_side = groups[active[selected]]
                if previous >= 0:
                    keep, drop = active[previous], active[selected]
                    for j in range(n):
                        mat[keep][j] += mat[drop][j]
                        mat[j][keep] += mat[j][drop]
                    for j in range(n):
                        mat[drop][j] = 0.0
                        mat[j][drop] = 0.0
                    groups[keep] = groups[keep] | groups[drop]
                    active.pop(selected)
            else:
                previous = selected
                source = mat[active[selected]]
                for index in range(size):
                    if not added[index]:
                        scores[index] += source[active[index]]

    return best_value, best_side


@dataclass(frozen=True)
class Comb:
    """A violated comb: ``x(delta(handle)) + sum_i x(delta(tooth_i)) < 3k + 1``."""

    handle: frozenset[int]
    teeth: tuple[frozenset[int], ...]
    violation: float


def comb_inequalities(
    n: int,
    weights: Sequence[Sequence[float]],
    *,
    max_cuts: int = 8,
    tolerance: float = 1e-6,
    handle_limit: int = 150,
) -> list[Comb]:
    """Heuristically separate violated comb inequalities.

    The inequality is ``x(delta(H)) + sum_i x(delta(T_i)) >= 3k + 1`` for an odd
    number ``k >= 3`` of pairwise disjoint teeth, each meeting both ``H`` and its
    complement, so a comb is violated when that left-hand side falls *below*
    ``3k + 1``.  Because every term is at least two at a feasible LP point, the
    interesting combs are the ones whose teeth are tight, hence the teeth are
    built by descending on the tooth boundary:

    * first the ``x = 1`` components that straddle ``delta(H)`` (the classic
      Padberg--Rinaldi choice, boundary exactly 2 when the component is tight);
    * then two-vertex teeth ``{h, v}`` seeded on an edge leaving the handle and
      grown while the boundary shrinks.

    Handles are the ``x = 1`` components, unions of adjacent components,
    component-plus-neighbour sets, the minimum cut side and the complements of
    all of these.  Every candidate is re-checked against ``3k + 1`` before being
    returned, so a weak heuristic costs time, never correctness.
    """
    if n < 6:
        return []
    adjacency = _adjacency(n, weights)
    components = [frozenset(component) for component in _unit_weight_components(n, weights)]
    handles = _candidate_handles(n, components, adjacency, weights, handle_limit)

    results: list[Comb] = []
    seen: set[tuple] = set()
    for handle in handles:
        for teeth_count in (3, 5):
            comb = _build_comb(adjacency, handle, components, teeth_count)
            if comb is None or comb.violation <= tolerance:
                continue
            key = (comb.handle, comb.teeth)
            if key in seen:
                continue
            seen.add(key)
            results.append(comb)
            break
        if len(results) >= max_cuts:
            break
    return results


def comb_coefficients(
    comb: Comb,
    edges: Sequence[tuple[int, int]],
    incident: Sequence[Sequence[int]],
) -> dict[int, float]:
    """Edge coefficients of the comb: ``delta(handle)`` plus every ``delta(teeth)``."""
    coefficients: dict[int, float] = {}
    for side in (comb.handle, *comb.teeth):
        inside = set(side)
        for city in inside:
            for edge in incident[city]:
                i, j = edges[edge]
                if (i in inside) != (j in inside):
                    coefficients[edge] = coefficients.get(edge, 0.0) + 1.0
    return coefficients


def _adjacency(n: int, weights) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        row = weights[i]
        for j in range(i + 1, n):
            value = float(row[j])
            if value > 0.0:
                adjacency[i].append((j, value))
                adjacency[j].append((i, value))
    return adjacency


def _candidate_handles(n: int, components, adjacency, weights, limit: int) -> list[frozenset[int]]:
    owner: dict[int, int] = {}
    for index, component in enumerate(components):
        for city in component:
            owner[city] = index

    neighbours: dict[int, set[int]] = {index: set() for index in range(len(components))}
    for index, component in enumerate(components):
        for city in component:
            for other, _ in adjacency[city]:
                partner = owner[other]
                if partner != index:
                    neighbours[index].add(partner)

    handles: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()

    def push(candidate: frozenset[int]) -> None:
        if not 1 < len(candidate) < n:
            return
        key = canonical_side(candidate, n)
        if key in seen:
            return
        seen.add(key)
        handles.append(candidate)

    for component in components:
        push(component)
        # A single vertex glued onto a tight component can give delta(H) = 3,
        # the narrowest handle a comb can still improve on.
        for city in component:
            for other, value in adjacency[city]:
                if owner[other] != owner[city] and value < 1.0 - 1e-6:
                    push(component | {other})
    for index in range(len(components)):
        for partner in sorted(neighbours[index]):
            if partner > index:
                push(components[index] | components[partner])

    # Handles made of fractional edges.  A triangle of x = 1/2 edges has
    # delta(H) = 3 when each corner also carries one tight edge, which is exactly
    # the shape a half-integral vertex of the DFJ polytope needs for a comb to
    # bite; whole tight components can never produce it.
    fractional: list[set[int]] = [set() for _ in range(n)]
    for city in range(n):
        for other, value in adjacency[city]:
            if 1e-6 < value < 1.0 - 1e-6:
                fractional[city].add(other)
    for city in range(n):
        for other in sorted(fractional[city]):
            if other <= city:
                continue
            for third in sorted(fractional[city] & fractional[other]):
                if third > other:
                    push(frozenset({city, other, third}))
        star = fractional[city]
        if 2 <= len(star) <= 4:
            push(frozenset({city} | star))

    _, side = global_min_cut(n, weights)
    push(side)

    base = list(handles)
    for candidate in base:
        push(frozenset(set(range(n)) - candidate))

    # Only narrow handles can be violated, so keep the cheapest ones.
    scored = [(_cut_value(set(candidate), adjacency), candidate) for candidate in handles]
    scored.sort(key=lambda item: (item[0], len(item[1])))
    return [candidate for _, candidate in scored[:limit]]


def _build_comb(adjacency, handle: frozenset[int], components, teeth_count: int) -> Comb | None:
    inside = set(handle)
    used: set[int] = set()
    teeth: list[frozenset[int]] = []

    straddling = [
        component
        for component in components
        if component & handle and component - handle
    ]
    straddling.sort(key=lambda side: _cut_value(set(side), adjacency))
    for side in straddling:
        if side & used:
            continue
        teeth.append(side)
        used |= side
        if len(teeth) == teeth_count:
            break

    if len(teeth) < teeth_count:
        ports: list[tuple[float, int, int]] = []
        for city in handle:
            for other, value in adjacency[city]:
                if other not in inside and value > 1e-9:
                    ports.append((4.0 - 2.0 * value, city, other))
        ports.sort()
        for _, inner, outer in ports:
            if len(teeth) == teeth_count:
                break
            if inner in used or outer in used:
                continue
            tooth = _grow_tooth(adjacency, inside, {inner, outer}, used)
            if tooth & used:
                continue
            teeth.append(frozenset(tooth))
            used |= tooth

    if len(teeth) < teeth_count:
        return None

    total = _cut_value(inside, adjacency) + sum(_cut_value(set(tooth), adjacency) for tooth in teeth)
    return Comb(handle, tuple(teeth), (3 * teeth_count + 1) - total)


def _grow_tooth(adjacency, inside, seed: set[int], used) -> set[int]:
    """Extend a tooth while doing so shrinks its boundary.

    ``x(delta(.))`` is at least two at any feasible point, so this descent is
    bounded; stopping early just leaves a weaker (still valid) candidate.
    """
    tooth = set(seed)
    best = _cut_value(tooth, adjacency)
    for _ in range(8):
        best_city, best_value = None, best
        for city in list(tooth):
            for other, _ in adjacency[city]:
                if other in tooth or other in used:
                    continue
                value = _cut_value(tooth | {other}, adjacency)
                if value < best_value - 1e-9:
                    best_city, best_value = other, value
        if best_city is None:
            break
        tooth.add(best_city)
        best = best_value
    return tooth


def _cut_value(inside: set[int], adjacency) -> float:
    total = 0.0
    for city in inside:
        for other, weight in adjacency[city]:
            if other not in inside:
                total += weight
    return total


def _unit_weight_components(n: int, weights, eps: float = 1e-6) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        row = weights[i]
        for j in range(i + 1, n):
            if row[j] >= 1.0 - eps:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    buckets: dict[int, list[int]] = {}
    for city in range(n):
        buckets.setdefault(find(city), []).append(city)
    return list(buckets.values())
