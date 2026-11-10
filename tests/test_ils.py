import random

import pytest

from tsp.construction import nearest_neighbor_route
from tsp.metrics import route_length
from tsp.models import TSPInstance
from tsp.perturbations import double_bridge
from tsp.solvers import ILSSolver, IteratedLocalSearchSolver


def make_instance(n=40, seed=0):
    rng = random.Random(seed)
    return TSPInstance([(rng.random(), rng.random()) for _ in range(n)])


def test_ils_is_reproducible_and_tracks_best_solution():
    instance = make_instance()
    solver = ILSSolver(iterations=12, candidate_count=10, seed=42)

    first = solver.solve(instance)
    second = solver.solve(instance)

    assert first.route == second.route
    assert first.length == pytest.approx(route_length(instance, first.route))
    assert first.metadata["history"] == second.metadata["history"]
    history = first.metadata["history"]
    assert len(history) == 13
    assert history[-1] == pytest.approx(first.length)
    assert all(later <= earlier + 1e-12 for earlier, later in zip(history, history[1:]))
    assert 0 <= first.metadata["accepted_iterations"] <= 12


@pytest.mark.parametrize("acceptance", ["better", "always"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 10, 40])
def test_ils_returns_valid_routes_for_all_supported_sizes(n, acceptance):
    instance = make_instance(n)
    solver = IteratedLocalSearchSolver(
        iterations=4,
        perturbation_strength=2,
        acceptance=acceptance,
        seed=7,
    )

    solution = solver.solve(instance)

    assert solution.length == pytest.approx(route_length(instance, solution.route))


def test_ils_supports_zero_iterations():
    instance = make_instance(20)
    solution = ILSSolver(iterations=0, seed=1).solve(instance)
    assert len(solution.metadata["history"]) == 1
    assert solution.metadata["accepted_iterations"] == 0
    assert solution.length == pytest.approx(route_length(instance, solution.route))


@pytest.mark.parametrize(
    "parameters",
    [
        {"iterations": -1},
        {"iterations": 1.5},
        {"candidate_count": 0},
        {"perturbation_strength": 0},
        {"acceptance": "unknown"},
    ],
)
def test_ils_rejects_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        ILSSolver(**parameters)


def test_ils_rejects_empty_instance():
    with pytest.raises(ValueError, match="empty"):
        ILSSolver().solve(TSPInstance([]))


def test_shared_double_bridge_preserves_input_and_cities():
    route = list(range(10))
    kicked = double_bridge(route, random.Random(3))
    assert route == list(range(10))
    assert kicked != route
    assert sorted(kicked) == route
    assert sorted(double_bridge([0, 1, 2, 3], random.Random(3))) == [0, 1, 2, 3]


def test_shared_nearest_neighbor_supports_candidate_lists_and_randomization():
    instance = make_instance(30)
    distances = instance.distance_matrix
    deterministic = nearest_neighbor_route(distances)
    randomized = nearest_neighbor_route(distances, rng=random.Random(5), rcl_size=3)

    assert sorted(deterministic) == list(range(30))
    assert sorted(randomized) == list(range(30))
    assert deterministic[0] == randomized[0] == 0
