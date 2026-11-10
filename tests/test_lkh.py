import itertools
import math
import random

import pytest

from tsp.candidates import alpha_candidate_sets, alpha_values_for_city
from tsp.local_search import tour_length, two_opt
from tsp.metrics import route_length
from tsp.models import TSPInstance, _OnDemandDistanceMatrix, _dense_distance_matrix
from tsp.one_tree import held_karp_ascent, minimum_one_tree
from tsp.solvers import LKHSolver
from tsp.spatial import nearest_neighbour_graph


def make_points(n, seed=0):
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n)]


def brute_one_tree_cost(d, pi, required=None):
    """Exhaustively find a minimum 1-tree, optionally forcing one edge."""
    n = len(d)
    special = 0
    others = list(range(1, n))
    mst_edges = list(itertools.combinations(others, 2))
    best = math.inf
    for edges in itertools.combinations(mst_edges, n - 2):
        if required is not None and required[0] != 0 and required not in edges:
            continue
        adjacency = {city: [] for city in others}
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        seen = {others[0]}
        stack = [others[0]]
        while stack:
            for other in adjacency[stack.pop()]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        if len(seen) != n - 1:
            continue
        tree_cost = sum(d[a][b] + pi[a] + pi[b] for a, b in edges)
        for a, b in itertools.combinations(others, 2):
            special_edges = tuple(sorted(((special, a), (special, b))))
            if required is not None:
                if required[0] == 0 and required not in special_edges:
                    continue
                if required[0] != 0 and required in special_edges:
                    continue
            cost = tree_cost + sum(d[special][city] + pi[special] + pi[city] for city in (a, b))
            best = min(best, cost)
    return best


def optimal_length(d):
    n = len(d)
    return min(
        tour_length([0] + list(rest), d)
        for rest in itertools.permutations(range(1, n))
    )


def test_minimum_one_tree_matches_exhaustive_search_with_penalties():
    d = _dense_distance_matrix(make_points(6, seed=4))
    pi = [0.1, -0.3, 0.5, 0.2, -0.1, 0.4]
    tree = minimum_one_tree(d, pi)
    assert len(tree.edges) == 6
    assert len(tree.tree_edges) == 4
    assert len(tree.special_edges) == 2
    assert tree.degrees[0] == 2
    assert tree.modified_cost == pytest.approx(brute_one_tree_cost(d, pi))
    assert tree.lower_bound == pytest.approx(tree.modified_cost - 2 * sum(pi))


def test_alpha_nearness_matches_forced_minimum_one_tree_for_every_edge():
    d = _dense_distance_matrix(make_points(5, seed=7))
    pi = [0.2, -0.1, 0.3, 0.0, 0.4]
    tree = minimum_one_tree(d, pi)
    base = brute_one_tree_cost(d, pi)
    for city in range(5):
        alpha = alpha_values_for_city(city, d, pi, tree)
        for other in range(5):
            if city == other:
                continue
            edge = tuple(sorted((city, other)))
            forced = base if edge in tree.edges else brute_one_tree_cost(d, pi, edge)
            assert alpha[other] == pytest.approx(forced - base, abs=1e-9)
            assert alpha[other] >= -1e-9


def test_alpha_candidates_are_sorted_by_alpha_then_distance():
    d = _dense_distance_matrix(make_points(12, seed=3))
    pi = [0.0] * 12
    tree = minimum_one_tree(d, pi)
    candidates = alpha_candidate_sets(d, 4, pi, tree)
    for city, row in enumerate(candidates):
        alpha = alpha_values_for_city(city, d, pi, tree)
        expected = [other for other in range(12) if other != city]
        expected.sort(key=lambda other: (alpha[other], d[city][other], other))
        assert row == expected[:4]


def test_ascent_improves_valid_lower_bound_and_respects_zero_budget():
    d = _dense_distance_matrix(make_points(7, seed=11))
    initial = minimum_one_tree(d)
    optimum = optimal_length(d)
    ascent = held_karp_ascent(d, tour_length(list(range(7)), d), 40)
    assert ascent.iterations > 0
    assert ascent.lower_bound >= initial.lower_bound - 1e-9
    assert ascent.lower_bound <= optimum + 1e-9
    assert minimum_one_tree(d, ascent.penalties).lower_bound >= ascent.lower_bound - 1e-9
    zero = held_karp_ascent(d, optimum, 0)
    assert zero.iterations == 0
    assert zero.lower_bound == pytest.approx(initial.lower_bound)


def test_lk_search_improves_without_changing_tour_membership():
    d = _dense_distance_matrix(make_points(30, seed=19))
    candidates = alpha_candidate_sets(d, 6)
    route = list(range(30))
    random.Random(5).shuffle(route)
    route = two_opt(route, d, candidates)
    before = tour_length(route, d)
    improved = LKHSolver(max_depth=5)._lk_search(route[:], candidates, d)
    assert sorted(improved) == list(range(30))
    assert tour_length(improved, d) <= before + 1e-9
    assert tour_length(improved, d) == pytest.approx(
        sum(d[improved[i]][improved[(i + 1) % 30]] for i in range(30))
    )


def test_lk_search_escapes_a_two_opt_local_optimum():
    points = [
        (0.5219839097124932, 0.8066907771186791),
        (0.9604947743238768, 0.2896253777644655),
        (0.7661074377979527, 0.7042198668434126),
        (0.6613830572238304, 0.11016204891721182),
        (0.026936778790526805, 0.3841711045442975),
        (0.7463951961663844, 0.252383767513222),
        (0.5016026705680338, 0.3174350905457497),
        (0.8463278091284241, 0.9451939145048893),
        (0.4015909369106695, 0.998447005087021),
        (0.0621927912524467, 0.8126090004416155),
    ]
    route = [2, 0, 6, 5, 1, 3, 4, 9, 8, 7]
    d = _dense_distance_matrix(points)
    candidates = alpha_candidate_sets(d, 6)
    before = tour_length(route, d)
    assert tour_length(two_opt(route[:], d, candidates), d) == pytest.approx(before)
    improved = LKHSolver(max_depth=5)._lk_search(route, candidates, d)
    assert tour_length(improved, d) < before - 0.5
    assert tour_length(improved, d) == pytest.approx(optimal_length(d))


def test_legacy_constructor_and_solution_format_remain_compatible():
    solver = LKHSolver(
        candidate_count=6, runs=2, max_depth=4, seed=42,
        kick_count=2, max_trials=3,
    )
    instance = TSPInstance(make_points(20, seed=2))
    first = solver.solve(instance)
    second = solver.solve(instance)
    assert first == second
    assert first.route[0] == first.route[-1]
    assert sorted(first.route[:-1]) == list(range(1, 21))
    assert first.length == pytest.approx(route_length(instance, first.route))
    assert first.metadata["candidate_type"] == "alpha"
    assert first.metadata["lower_bound"] <= first.length + 1e-9
    assert first.metadata["trials"] <= 6
    assert first.metadata["parameters"]["max_depth"] == 4


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_lkh_handles_tiny_instances(n):
    instance = TSPInstance(make_points(n, seed=8))
    solution = LKHSolver(runs=1, kick_count=0, ascent_iterations=3).solve(instance)
    assert solution.route[0] == solution.route[-1]
    assert sorted(solution.route[:-1]) == list(range(1, n + 1))
    assert solution.length == pytest.approx(route_length(instance, solution.route))


def test_lkh_rejects_empty_instance_and_invalid_ascent_budget():
    with pytest.raises(ValueError):
        LKHSolver().solve(TSPInstance([]))
    with pytest.raises(ValueError):
        LKHSolver(ascent_iterations=-1)


def _brute_neighbours(points, k):
    return [
        sorted(
            (other for other in range(len(points)) if other != city),
            key=lambda other: (
                (points[city][0] - points[other][0]) ** 2
                + (points[city][1] - points[other][1]) ** 2,
                other,
            ),
        )[:k]
        for city in range(len(points))
    ]


def test_near_neighbour_graph_is_exact_and_symmetric():
    rng = random.Random(21)
    datasets = [
        [(rng.random(), rng.random()) for _ in range(60)],
        [(rng.gauss(0, 0.01) + i * 0.001, rng.gauss(0, 0.01)) for i in range(60)],
        [(i / 40, 0.0) for i in range(40)],
    ]
    for points in datasets:
        graph = nearest_neighbour_graph(points, 5)
        assert [row[:5] for row in graph] == _brute_neighbours(points, 5)
        assert all(city in graph[other] for city, row in enumerate(graph) for other in row)


def test_sparse_one_tree_and_alpha_match_complete_graph_when_graph_is_complete():
    d = _dense_distance_matrix(make_points(8, seed=5))
    full = [sorted(other for other in range(8) if other != city) for city in range(8)]
    pi = [0.1, -0.2, 0.3, 0.0, 0.15, -0.05, 0.2, -0.1]
    complete = minimum_one_tree(d, pi)
    sparse = minimum_one_tree(d, pi, full)
    assert sparse.scope == "candidate"
    assert sparse.edges == complete.edges
    assert sparse.modified_cost == pytest.approx(complete.modified_cost)
    assert alpha_candidate_sets(d, 3, pi, complete) == alpha_candidate_sets(d, 3, pi, sparse, full)
    assert alpha_values_for_city(3, d, pi, sparse, full) == pytest.approx(
        alpha_values_for_city(3, d, pi, complete)
    )
    assert held_karp_ascent(d, 4.0, 5).lower_bound == pytest.approx(
        held_karp_ascent(d, 4.0, 5, full).lower_bound
    )


def test_sparse_one_tree_on_narrow_graph_is_still_a_valid_tree():
    points = make_points(14, seed=6)
    d = _dense_distance_matrix(points)
    graph = nearest_neighbour_graph(points, 5)
    tree = minimum_one_tree(d, [0.0] * 14, graph)
    assert tree.scope == "candidate"
    assert len(tree.tree_edges) == 12
    assert tree.degrees[0] == 2
    assert len(tree.edges) == 14


def test_large_scale_mode_solves_1000_plus_cities_with_on_demand_distances():
    points = make_points(1200, seed=9)
    instance = TSPInstance(points, distance_matrix_threshold=1000)
    assert isinstance(instance.distance_matrix, _OnDemandDistanceMatrix)
    solver = LKHSolver(runs=1, kick_count=0, ascent_iterations=3, sparse=True, seed=2)
    first = solver.solve(instance)
    second = solver.solve(instance)
    assert first == second
    assert first.route[0] == first.route[-1]
    assert sorted(first.route[:-1]) == list(range(1, 1201))
    assert first.length == pytest.approx(route_length(instance, first.route))
    assert first.metadata["mode"] == "sparse"
    assert first.metadata["bound_scope"] == "candidate"
    assert first.metadata["candidate_graph_width"] >= 5
    assert math.isfinite(first.metadata["lower_bound"])


def test_mode_switch_and_large_scale_parameter_validation():
    small = TSPInstance(make_points(20, seed=1))
    normal = LKHSolver(runs=1, kick_count=0, ascent_iterations=1).solve(small)
    assert normal.metadata["mode"] == "complete"
    assert normal.metadata["bound_scope"] == "complete"
    forced = LKHSolver(runs=1, kick_count=0, ascent_iterations=1, large_scale_threshold=10)
    assert forced.solve(small).metadata["mode"] == "sparse"
    boundary = LKHSolver(runs=1, kick_count=0, ascent_iterations=0, large_scale_threshold=20)
    assert boundary.solve(small).metadata["mode"] == "sparse"
    overridden = LKHSolver(runs=1, kick_count=0, ascent_iterations=0,
                           large_scale_threshold=20, sparse=False)
    assert overridden.solve(small).metadata["mode"] == "complete"
    with pytest.raises(ValueError):
        LKHSolver(large_scale_threshold=-1)
    with pytest.raises(ValueError):
        LKHSolver(sparse="yes")
    with pytest.raises(ValueError):
        LKHSolver(candidate_graph_width=0)
