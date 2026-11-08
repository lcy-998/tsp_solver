from tsp.compare import compare, format_results
from tsp.metrics import route_length
from tsp.models import TSPInstance
from tsp.solvers import NearestNeighborSolver, SimulatedAnnealingSolver


def test_square_route_length():
    instance = TSPInstance([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert instance.distance_matrix[0][2] == 2**0.5
    assert route_length(instance, [1, 2, 3, 4, 1]) == 4


def test_compare_records_solution():
    instance = TSPInstance([(0, 0), (1, 0), (1, 1)])
    result = compare(instance, [NearestNeighborSolver()])[0]
    assert result.route == [1, 2, 3, 1]
    assert result.length > 0


def test_compare_numbers_repeated_algorithm_names():
    instance = TSPInstance([(0, 0), (1, 0), (1, 1)])
    solvers = [
        SimulatedAnnealingSolver(
            initial_temperature=2, min_temperature=1, cooling_rate=0.5,
            iterations_per_temperature=1, seed=1,
        ),
        SimulatedAnnealingSolver(
            initial_temperature=2, min_temperature=1, cooling_rate=0.5,
            iterations_per_temperature=2, seed=1,
        ),
    ]

    results = compare(instance, [NearestNeighborSolver(), *solvers])

    assert [result.algorithm for result in results] == [
        "nearest_neighbor",
        "simulated_annealing_1",
        "simulated_annealing_2",
    ]
    assert [solver.name for solver in solvers] == ["simulated_annealing"] * 2
    output = format_results(instance, results)
    assert "simulated_annealing_1" in output
    assert "simulated_annealing_2" in output


def test_compare_numbering_avoids_existing_algorithm_names():
    instance = TSPInstance([(0, 0), (1, 0), (1, 1)])
    solvers = [NearestNeighborSolver() for _ in range(3)]
    solvers[2].name = "nearest_neighbor_1"

    results = compare(instance, solvers)

    assert [result.algorithm for result in results] == [
        "nearest_neighbor_2",
        "nearest_neighbor_3",
        "nearest_neighbor_1",
    ]



