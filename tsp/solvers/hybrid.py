from __future__ import annotations

from random import Random
from typing import Sequence

from ..candidates import nearest_neighbours
from ..construction import nearest_neighbor_route
from ..local_search import tour_length, two_opt
from ..models import TSPInstance, TSPSolution
from ..perturbations import double_bridge
from ..solver import TSPSolver


class HybridTSPSolver(TSPSolver):

    name = "hybrid"

    def __init__(
        self,
        starts: int = 12,
        perturbations: int = 2,
        three_opt: bool = True,
        candidate_count: int = 15,
        seed: int | None = None,
    ) -> None:
        if starts < 1 or perturbations < 0:
            raise ValueError("starts must be at least 1 and perturbations non-negative")
        if candidate_count < 1:
            raise ValueError("candidate_count must be at least 1")
        self.starts = starts
        self.perturbations = perturbations
        self.three_opt = three_opt
        self.candidate_count = candidate_count
        self.seed = seed

    def solve(self, instance: TSPInstance) -> TSPSolution:
        n = instance.size
        if n == 0:
            raise ValueError("cannot solve an empty instance")
        if n == 1:
            return TSPSolution(self.name, [1, 1], 0.0, 0.0)

        distances = instance.distance_matrix
        candidates = nearest_neighbours(distances, self.candidate_count)
        rng = Random(self.seed)
        best: list[int] | None = None
        best_length = float("inf")
        history: list[float] = []
        for start in range(self.starts):
            if start % 2 == 0:
                route = self._nearest_neighbor(n, distances, rng, randomized=start > 1)
            else:
                route = self._savings_route(n, distances, rng)
            route = self._improve(route, distances, candidates)
            for _ in range(self.perturbations):
                kicked = self._double_bridge(route, rng)
                kicked = self._improve(kicked, distances, candidates)
                if self._length(kicked, distances) < self._length(route, distances):
                    route = kicked
            length = self._length(route, distances)
            if length < best_length:
                best, best_length = route[:], length
            history.append(best_length)

        assert best is not None
        closed = [city + 1 for city in best]
        closed.append(closed[0])
        return TSPSolution(
            self.name,
            closed,
            best_length,
            0.0,
            metadata={
                "history": history,
                "history_label": "best length",
                "parameters": {
                    "starts": self.starts,
                    "perturbations": self.perturbations,
                    "three_opt": self.three_opt,
                    "candidate_count": self.candidate_count,
                    "seed": self.seed,
                },
                "starts": self.starts,
                "perturbations": self.perturbations,
                "three_opt": self.three_opt,
                "seed": self.seed,
            },
        )

    @staticmethod
    def _nearest_neighbor(n: int, d: Sequence[Sequence[float]], rng: Random, randomized: bool) -> list[int]:
        return nearest_neighbor_route(
            d,
            start=rng.randrange(n) if randomized else 0,
            rng=rng,
            rcl_size=3 if randomized else 1,
        )

    @staticmethod
    def _savings_route(n: int, d: Sequence[Sequence[float]], rng: Random) -> list[int]:
        """Clarke-Wright route merging with city 0 as the depot."""
        if n <= 2:
            return list(range(n))
        routes = {city: [0, city] for city in range(1, n)}
        owner = {city: city for city in range(1, n)}
        savings = sorted(
            ((d[0][i] + d[0][j] - d[i][j], i, j) for i in range(1, n) for j in range(i + 1, n)),
            reverse=True,
        )
        for saving, i, j in savings:
            if saving < 0:
                break
            ri, rj = owner[i], owner[j]
            if ri not in routes or rj not in routes:
                continue
            if ri == rj:
                continue
            left, right = routes[ri], routes[rj]
            if left[-1] == i and right[1] == j:
                merged = left + right[1:]
            elif left[1] == i and right[-1] == j:
                merged = right + left[1:]
            elif left[-1] == i and right[-1] == j:
                merged = left + list(reversed(right[:-1]))
            elif left[1] == i and right[1] == j:
                merged = list(reversed(left)) + right[1:]
            else:
                continue
            pivot = merged.index(0)
            merged = merged[pivot:] + merged[:pivot]
            del routes[ri], routes[rj]
            key = merged[1]
            routes[key] = merged
            for city in merged[1:]:
                owner[city] = key
        route = next(iter(routes.values()))
        # A merge can orient the depot in the middle of the path. Rotate it
        # back to the front before inserting any still-unmerged routes.
        pivot = route.index(0)
        route = route[pivot:] + route[:pivot]
        remaining = set(range(1, n)) - set(route[1:])
        while remaining:
            city = rng.choice(tuple(remaining))
            best_pos = min(
                range(1, len(route)),
                key=lambda pos: d[route[pos - 1]][city] + d[city][route[pos]] - d[route[pos - 1]][route[pos]],
            )
            route.insert(best_pos, city)
            remaining.remove(city)
        # Defensive normalization after route merges: every city must occur
        # exactly once before the caller closes the tour.
        normalized = [0]
        seen = {0}
        for city in route:
            if city not in seen and 0 <= city < n:
                normalized.append(city)
                seen.add(city)
        normalized.extend(city for city in range(1, n) if city not in seen)
        return normalized

    def _improve(self, route: list[int], d: Sequence[Sequence[float]], candidates) -> list[int]:
        # Candidate-list 2-opt with don't-look bits: O(nK) per pass instead of
        # the O(n^2) full scan that used to dominate the runtime.
        route = two_opt(route, d, candidates)
        # The exhaustive three-cut neighbourhood grows cubically (and each
        # candidate needs a tour-length scan), so use it only on small tours.
        if self.three_opt and 6 <= len(route) <= 80:
            route = self._three_opt(route, d)
        return route

    @staticmethod
    def _three_opt(route: list[int], d: Sequence[Sequence[float]]) -> list[int]:
        n = len(route)
        while True:
            old = HybridTSPSolver._length(route, d)
            for i in range(n - 5):
                for j in range(i + 2, n - 3):
                    for k in range(j + 2, n):
                        a, b = route[i], route[i + 1]
                        c, e = route[j], route[j + 1]
                        f, g = route[k], route[(k + 1) % n]
                        candidates = (
                            route[: i + 1] + route[i + 1 : j + 1][::-1] + route[j + 1 : k + 1][::-1] + route[k + 1 :],
                            route[: i + 1] + route[i + 1 : j + 1] + route[j + 1 : k + 1][::-1] + route[k + 1 :],
                            route[: i + 1] + route[i + 1 : j + 1][::-1] + route[j + 1 : k + 1] + route[k + 1 :],
                        )
                        candidate = min(candidates, key=lambda r: HybridTSPSolver._length(r, d))
                        if HybridTSPSolver._length(candidate, d) + 1e-12 < old:
                            route = candidate
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                return route

    @staticmethod
    def _double_bridge(route: list[int], rng: Random) -> list[int]:
        return double_bridge(route, rng)

    @staticmethod
    def _length(route: list[int], d: Sequence[Sequence[float]]) -> float:
        return tour_length(route, d)


# Short alias for callers who prefer the generic name used in experiment code.
HybridSolver = HybridTSPSolver
HybridHeuristicSolver = HybridTSPSolver
