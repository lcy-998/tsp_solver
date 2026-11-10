"""Reusable construction heuristics for zero-based, open TSP tours."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random


def nearest_neighbor_route(
    distances: Sequence[Sequence[float]],
    *,
    start: int = 0,
    rng: Random | None = None,
    rcl_size: int = 1,
    candidates: Sequence[Sequence[int]] | None = None,
) -> list[int]:
    """Build a tour by repeatedly visiting a nearby unvisited city.

    ``rcl_size`` controls the restricted candidate list. With a random number
    generator and a value greater than one, the next city is sampled from the
    closest unvisited cities; otherwise the nearest city is always selected.
    Optional candidate lists avoid sorting all remaining cities on every step.
    """
    n = len(distances)
    if n == 0:
        return []
    if type(start) is not int or not 0 <= start < n:
        raise ValueError("start must be a valid city index")
    if type(rcl_size) is not int or rcl_size < 1:
        raise ValueError("rcl_size must be a positive integer")

    route = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        current = route[-1]
        if candidates is None:
            choices = list(unvisited)
        else:
            choices = [city for city in candidates[current] if city in unvisited]
            if not choices:
                choices = list(unvisited)
        choices.sort(key=lambda city: (distances[current][city], city))
        restricted = choices[: min(rcl_size, len(choices))]
        next_city = rng.choice(restricted) if rng is not None and rcl_size > 1 else restricted[0]
        route.append(next_city)
        unvisited.remove(next_city)
    return route
