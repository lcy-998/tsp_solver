import itertools
import math
import random

import pytest

from tsp.candidates import _nearest_numpy, _nearest_python, nearest_neighbours
from tsp.local_search import tour_length, two_opt
from tsp.metrics import route_length
from tsp.models import TSPInstance, _dense_distance_matrix
from tsp.solvers import HybridSolver, LKHSolver


def make_points(n, seed=0):
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n)]


def test_dense_matrix_matches_scalar_hypot():
    points = make_points(40)
    matrix = _dense_distance_matrix(points)
    assert isinstance(matrix, list)
    for i in (0, 7, 39):
        for j in (0, 3, 39):
            assert matrix[i][j] == pytest.approx(math.hypot(points[i][0] - points[j][0],
                                                            points[i][1] - points[j][1]))
            assert matrix[i][j] == matrix[j][i]


def test_candidate_lists_are_sorted_and_exclude_self():
    points = make_points(60)
    d = _dense_distance_matrix(points)
    candidates = nearest_neighbours(d, 8)
    assert len(candidates) == 60
    for city, row in enumerate(candidates):
        assert len(row) == 8
        assert city not in row
        distances = [d[city][other] for other in row]
        assert distances == sorted(distances)


def test_candidate_k_is_capped_by_instance_size():
    d = _dense_distance_matrix(make_points(5))
    assert all(len(row) == 4 for row in nearest_neighbours(d, 10))


def test_numpy_and_python_candidate_builders_agree():
    d = _dense_distance_matrix(make_points(80))
    assert _nearest_numpy(d, 6) == _nearest_python(d, 6)


@pytest.mark.parametrize("n", [3, 4, 5, 60, 200])
def test_two_opt_returns_a_shorter_valid_tour(n):
    points = make_points(max(n, 200))
    d = _dense_distance_matrix(points[:n])
    candidates = nearest_neighbours(d, 10)
    tour = list(range(n))
    random.Random(n).shuffle(tour)
    before = tour_length(tour, d)
    two_opt(tour, d, candidates)
    assert sorted(tour) == list(range(n))
    assert tour_length(tour, d) <= before + 1e-9
    assert tour_length(tour, d) == pytest.approx(
        sum(d[tour[k]][tour[(k + 1) % n]] for k in range(n)))


def test_two_opt_is_near_optimal_on_small_instances():
    n = 8
    points = make_points(n, seed=3)
    d = _dense_distance_matrix(points)
    best = min(
        sum(d[route[k]][route[(k + 1) % n]] for k in range(n))
        for route in ([0] + list(rest) for rest in itertools.permutations(range(1, n)))
    )
    tour = list(range(n))
    random.Random(3).shuffle(tour)
    two_opt(tour, d, nearest_neighbours(d, 5))
    assert tour_length(tour, d) <= best * 1.05


@pytest.mark.parametrize("solver", [HybridSolver(seed=1), LKHSolver(seed=1, runs=1, kick_count=0)])
def test_solvers_return_valid_routes(solver):
    points = make_points(40)
    instance = TSPInstance(points)
    solution = solver.solve(instance)
    assert route_length(instance, solution.route) > 0
    assert solution.length == pytest.approx(route_length(instance, solution.route))
