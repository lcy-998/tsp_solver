"""Perturbation operators shared by iterated TSP heuristics."""

from random import Random


def double_bridge(route: list[int], rng: Random) -> list[int]:
    """Return a double-bridge kick without modifying ``route``.

    Four non-empty path segments are reconnected in a different order. Unlike
    a random permutation, this changes only four edges and therefore preserves
    most of the local structure found by local search. Tours with fewer than
    four cities have no non-trivial double-bridge move and are simply copied.
    """
    n = len(route)
    if n < 4:
        return route[:]
    a, b, c = sorted(rng.sample(range(1, n), 3))
    return route[:a] + route[c:] + route[b:c] + route[a:b]
