"""Candidate-list 2-opt with don't-look bits.

The implementation follows the classical engineering recipe for local search
on the TSP (Bentley 1992; Johnson & McGeoch 1997):

* only edges into a city's K nearest neighbours are examined,
* a queue of "dirty" cities replaces the O(n) restart-after-every-move scan,
* a position array makes edge lookups O(1) and tour reversal O(length).

Together these turn a naive O(n^2) scan per improvement (O(n^3) overall) into
roughly O(nK) per improvement, which is what makes n ~ 1000 tractable.

Both orientations of the two cut edges are examined. A 2-opt move is only a
single cycle when the two edges are oriented the same way along the tour
(both ``(v, succ(v))`` or both ``(pred(v), v)``); restricting the scan to the
successor orientation silently drops about half of the improving moves.
"""

from __future__ import annotations

from collections.abc import Sequence


def two_opt(
    tour: list[int],
    distances: Sequence[Sequence[float]],
    candidates: Sequence[Sequence[int]],
    *,
    epsilon: float = 1e-12,
) -> list[int]:
    """Improve ``tour`` in place until no candidate 2-opt move is improving.

    ``tour`` is a list of city ids in visiting order; the closing edge from the
    last city back to the first is implicit.
    """
    n = len(tour)
    if n < 4:
        return tour

    position = [0] * n
    for index, city in enumerate(tour):
        position[city] = index

    # 1 = queued (its neighbourhood may have changed), 0 = settled.
    queued = bytearray(b"\x01") * n
    pending = list(range(n))

    while pending:
        city_a = pending.pop()
        queued[city_a] = 0
        index_a = position[city_a]

        for a_is_tail in (True, False):
            city_b = tour[(index_a + 1) % n] if a_is_tail else tour[index_a - 1]
            applied = False

            for city_c in candidates[city_a]:
                if city_c == city_b:
                    continue
                index_c = position[city_c]
                # Both cut edges must be oriented the same way to stay a single cycle.
                city_e = tour[(index_c + 1) % n] if a_is_tail else tour[index_c - 1]
                if city_e == city_a or city_e == city_b or city_c == city_a:
                    continue

                if a_is_tail:
                    u1, v1, u2, v2 = city_a, city_b, city_c, city_e
                else:
                    u1, v1, u2, v2 = city_b, city_a, city_e, city_c

                if (
                    distances[u1][v1] + distances[u2][v2]
                    <= distances[u1][u2] + distances[v1][v2] + epsilon
                ):
                    continue

                start, stop, length = _reversal_window(position, n, v1, u2, u1)
                if length < 2 or length > n - 2:
                    continue

                _reverse_segment(tour, position, n, start, stop)
                for city in (city_a, city_b, city_c, city_e):
                    if not queued[city]:
                        queued[city] = 1
                        pending.append(city)
                applied = True
                break
            if applied:
                break


    return tour


def _reversal_window(position: list[int], n: int, v1: int, u2: int, u1: int) -> tuple[int, int, int]:
    """Window to reverse: the path from ``v1`` to ``u2`` that avoids ``u1``."""
    start, stop = position[v1], position[u2]
    length = (stop - start) % n + 1
    if (position[u1] - start) % n < length:
        start, stop = (stop + 1) % n, (start - 1) % n
        length = n - length
    return start, stop, length


def _reverse_segment(tour: list[int], position: list[int], n: int, start: int, stop: int) -> None:
    """Reverse the cyclic window ``tour[start:stop]`` (inclusive) in place."""
    length = (stop - start) % n + 1
    if length <= 1 or length >= n:
        return

    end = start + length
    if end <= n:
        tour[start:end] = tour[start:end][::-1]
        for index in range(start, end):
            position[tour[index]] = index
        return

    window = tour[start:] + tour[: end - n]
    window.reverse()
    cut = n - start
    tour[start:] = window[:cut]
    tour[: end - n] = window[cut:]
    for index, city in enumerate(tour):
        position[city] = index


def tour_length(tour: Sequence[int], distances: Sequence[Sequence[float]]) -> float:
    """Length of the closed tour, without allocating a copy."""
    total = 0.0
    previous = tour[-1]
    for city in tour:
        total += distances[previous][city]
        previous = city
    return total
