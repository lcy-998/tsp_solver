from collections.abc import Sequence
from math import exp
from random import Random

from ..construction import nearest_neighbor_route
from ..local_search import tour_length
from ..models import TSPInstance, TSPSolution
from ..solver import TSPSolver


class SimulatedAnnealingSolver(TSPSolver):
    """Approximate a TSP tour with simulated annealing and 2-opt moves."""

    name = "simulated_annealing"

    def __init__(
        self,
        initial_temperature: float = 1000.0,
        cooling_rate: float = 0.995,
        min_temperature: float = 1e-3,
        iterations_per_temperature: int = 100,
        seed: int | None = None,
    ) -> None:
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if not 0 < cooling_rate < 1:
            raise ValueError("cooling_rate must be between 0 and 1")
        if min_temperature <= 0 or min_temperature >= initial_temperature:
            raise ValueError("min_temperature must be positive and below initial_temperature")
        if iterations_per_temperature < 1:
            raise ValueError("iterations_per_temperature must be at least 1")

        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temperature = iterations_per_temperature
        self.seed = seed

    def solve(self, instance: TSPInstance) -> TSPSolution:
        if not instance.coordinates:
            raise ValueError("cannot solve an empty instance")

        size = instance.size
        if size == 1:
            return TSPSolution(self.name, [1, 1], 0.0, 0.0)

        rng = Random(self.seed)
        distances = instance.distance_matrix

        # Nearest-neighbor initialization gives annealing a useful starting tour.
        route = nearest_neighbor_route(distances)
        current_length = self._length(route, distances)
        best_route = route[:]
        best_length = current_length
        history = [best_length]
        temperature = self.initial_temperature

        while temperature > self.min_temperature:
            for _ in range(self.iterations_per_temperature):
                left, right = sorted(rng.sample(range(size), 2))
                candidate = route[:]
                candidate[left : right + 1] = reversed(candidate[left : right + 1])
                candidate_length = self._length(candidate, distances)
                delta = candidate_length - current_length

                if delta < 0 or rng.random() < exp(-delta / temperature):
                    route = candidate
                    current_length = candidate_length
                    if current_length < best_length:
                        best_route = route[:]
                        best_length = current_length
                history.append(best_length)

            temperature *= self.cooling_rate

        closed_route = [city + 1 for city in best_route]
        closed_route.append(closed_route[0])
        return TSPSolution(
            self.name,
            closed_route,
            best_length,
            0.0,
            metadata={
                "history": history,
                "history_label": "best length",
                "parameters": {
                    "initial_temperature": self.initial_temperature,
                    "cooling_rate": self.cooling_rate,
                    "min_temperature": self.min_temperature,
                    "iterations_per_temperature": self.iterations_per_temperature,
                    "seed": self.seed,
                },
            },
        )

    @staticmethod
    def _length(route: list[int], distances: Sequence[Sequence[float]]) -> float:
        return tour_length(route, distances)
