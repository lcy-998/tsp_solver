from collections.abc import Sequence
from math import inf
from random import Random

from ..construction import nearest_neighbor_route
from ..local_search import tour_length
from ..models import TSPInstance, TSPSolution
from ..solver import TSPSolver


class AntColonySolver(TSPSolver):
    """Approximate a symmetric TSP tour with ant colony optimization.

    City numbers in the returned route are 1-based, matching the rest of the
    package. City 1 is used as the common starting city to make seeded runs
    reproducible and to avoid reporting equivalent rotations of a tour.
    """

    name = "ant_colony"

    def __init__(
        self,
        n_ants: int | None = None,
        n_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.5,
        q: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_ants is not None and (type(n_ants) is not int or n_ants < 1):
            raise ValueError("n_ants must be a positive integer or None")
        if type(n_iterations) is not int or n_iterations < 1:
            raise ValueError("n_iterations must be a positive integer")
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if not 0 < evaporation_rate <= 1:
            raise ValueError("evaporation_rate must be between 0 and 1")
        if q <= 0:
            raise ValueError("q must be positive")

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.seed = seed

    def solve(self, instance: TSPInstance) -> TSPSolution:
        size = instance.size
        if size == 0:
            raise ValueError("cannot solve an empty instance")
        if size == 1:
            return TSPSolution(self.name, [1, 1], 0.0, 0.0)

        distances = instance.distance_matrix
        ants = self.n_ants if self.n_ants is not None else size
        rng = Random(self.seed)

        # A nearest-neighbor tour provides a sensible initial pheromone scale.
        initial_route = self._nearest_neighbor(distances, size)
        best_length = self._length(initial_route, distances)
        best_route = initial_route[:]
        initial_pheromone = 1.0 if best_length == 0 else 1.0 / (size * best_length)
        pheromone = [[initial_pheromone for _ in range(size)] for _ in range(size)]
        history = [best_length]

        for _ in range(self.n_iterations):
            routes: list[tuple[list[int], float]] = []
            for _ in range(ants):
                route = self._construct_route(distances, pheromone, size, rng)
                length = self._length(route, distances)
                routes.append((route, length))
                if length < best_length:
                    best_route = route[:]
                    best_length = length

            retain = 1.0 - self.evaporation_rate
            for row in pheromone:
                for city in range(size):
                    row[city] *= retain
            for route, length in routes:
                if length == 0:
                    amount = self.q
                else:
                    amount = self.q / length
                for start, end in zip(route, route[1:] + route[:1]):
                    pheromone[start][end] += amount
                    pheromone[end][start] += amount
            history.append(best_length)

        closed_route = [city + 1 for city in best_route]
        closed_route.append(1)
        return TSPSolution(
            self.name,
            closed_route,
            best_length,
            0.0,
            metadata={"history": history, "history_label": "best length"},
        )

    def _construct_route(
        self,
        distances: Sequence[Sequence[float]],
        pheromone: list[list[float]],
        size: int,
        rng: Random,
    ) -> list[int]:
        route = [0]
        unvisited = set(range(1, size))
        while unvisited:
            current = route[-1]
            candidates = list(unvisited)
            weights = [
                pheromone[current][city] ** self.alpha
                * self._heuristic(distances[current][city]) ** self.beta
                for city in candidates
            ]
            total = sum(weights)
            if total <= 0 or not total < inf:
                next_city = rng.choice(candidates)
            else:
                threshold = rng.random() * total
                next_city = candidates[-1]
                for city, weight in zip(candidates, weights):
                    threshold -= weight
                    if threshold <= 0:
                        next_city = city
                        break
            route.append(next_city)
            unvisited.remove(next_city)
        return route

    @staticmethod
    def _heuristic(distance: float) -> float:
        # The epsilon also makes coincident cities selectable without division
        # by zero while preserving their preference over distant cities.
        return 1.0 / max(distance, 1e-12)

    @staticmethod
    def _nearest_neighbor(
        distances: Sequence[Sequence[float]], size: int
    ) -> list[int]:
        return nearest_neighbor_route(distances)

    @staticmethod
    def _length(route: list[int], distances: Sequence[Sequence[float]]) -> float:
        return tour_length(route, distances)
