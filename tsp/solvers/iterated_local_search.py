from __future__ import annotations

from random import Random
from typing import Literal

from ..candidates import nearest_neighbours
from ..construction import nearest_neighbor_route
from ..local_search import tour_length, two_opt
from ..models import TSPInstance, TSPSolution
from ..perturbations import double_bridge
from ..solver import TSPSolver


AcceptanceCriterion = Literal["better", "always"]


class IteratedLocalSearchSolver(TSPSolver):
    """Iterated local search using double-bridge kicks and candidate 2-opt.

    The incumbent is perturbed and improved at every iteration. ``better``
    accepts improving or equal-length candidates, while ``always`` performs a
    random walk over local optima. The best tour is tracked separately, so an
    accepted worse candidate can never degrade the returned solution.
    """

    name = "ils"

    def __init__(
        self,
        iterations: int = 100,
        candidate_count: int = 15,
        perturbation_strength: int = 1,
        acceptance: AcceptanceCriterion = "better",
        seed: int | None = None,
    ) -> None:
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if type(candidate_count) is not int or candidate_count < 1:
            raise ValueError("candidate_count must be a positive integer")
        if type(perturbation_strength) is not int or perturbation_strength < 1:
            raise ValueError("perturbation_strength must be a positive integer")
        if acceptance not in {"better", "always"}:
            raise ValueError("acceptance must be 'better' or 'always'")

        self.iterations = iterations
        self.candidate_count = candidate_count
        self.perturbation_strength = perturbation_strength
        self.acceptance = acceptance
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

        current = nearest_neighbor_route(distances)
        current = two_opt(current, distances, candidates)
        current_length = tour_length(current, distances)
        best = current[:]
        best_length = current_length
        history = [best_length]
        accepted_iterations = 0

        for _ in range(self.iterations):
            candidate = current
            for _ in range(self.perturbation_strength):
                candidate = double_bridge(candidate, rng)
            candidate = two_opt(candidate, distances, candidates)
            candidate_length = tour_length(candidate, distances)

            if candidate_length < best_length - 1e-12:
                best = candidate[:]
                best_length = candidate_length

            if self.acceptance == "always" or candidate_length <= current_length + 1e-12:
                current = candidate
                current_length = candidate_length
                accepted_iterations += 1
            history.append(best_length)

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
                    "iterations": self.iterations,
                    "candidate_count": self.candidate_count,
                    "perturbation_strength": self.perturbation_strength,
                    "acceptance": self.acceptance,
                    "seed": self.seed,
                },
                "accepted_iterations": accepted_iterations,
            },
        )


ILSSolver = IteratedLocalSearchSolver
