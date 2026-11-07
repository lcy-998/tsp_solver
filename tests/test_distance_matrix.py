import pytest

from tsp.metrics import route_length
from tsp.models import TSPInstance
from tsp.parser import iter_instances, load_instances, parse_line
from tsp.solvers import DynamicProgrammingSolver, NearestNeighborSolver, SimulatedAnnealingSolver


COORDINATES = [(0, 0), (1, 0), (1, 1), (0, 1)]


def test_threshold_switches_only_above_boundary():
    default = TSPInstance(COORDINATES)
    at_threshold = TSPInstance(COORDINATES, distance_matrix_threshold=4)
    above_threshold = TSPInstance(COORDINATES, distance_matrix_threshold=3)

    assert isinstance(default.distance_matrix, list)
    assert isinstance(at_threshold.distance_matrix, list)
    assert not isinstance(above_threshold.distance_matrix, list)
    assert len(above_threshold.distance_matrix) == 4
    assert len(above_threshold.distance_matrix[0]) == 4
    assert above_threshold.distance_matrix[0][2] == default.distance_matrix[0][2]
    assert above_threshold.distance_matrix[-1][-2] == default.distance_matrix[-1][-2]
    assert above_threshold.distance_matrix[0][1:3] == [1, 2**0.5]
    assert route_length(above_threshold, [1, 2, 3, 4, 1]) == 4


@pytest.mark.parametrize("threshold", [-1, 1.5, True])
def test_reject_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="distance_matrix_threshold"):
        TSPInstance(COORDINATES, distance_matrix_threshold=threshold)


def test_parser_entry_points_pass_threshold(tmp_path):
    line = "0 0 1 0 1 1 0 1"
    path = tmp_path / "instances.txt"
    path.write_text(line + "\n", encoding="utf-8")

    instances = [
        parse_line(line, distance_matrix_threshold=3),
        *iter_instances(path, distance_matrix_threshold=3),
        *load_instances(path, distance_matrix_threshold=3),
    ]
    assert all(not isinstance(instance.distance_matrix, list) for instance in instances)
    assert all(instance.distance_matrix_threshold == 3 for instance in instances)


@pytest.mark.parametrize(
    "solver",
    [
        NearestNeighborSolver(),
        DynamicProgrammingSolver(),
        SimulatedAnnealingSolver(
            initial_temperature=2,
            min_temperature=1,
            cooling_rate=0.5,
            iterations_per_temperature=2,
            seed=1,
        ),
    ],
)
def test_solvers_use_on_demand_distances(solver):
    eager = TSPInstance(COORDINATES)
    on_demand = TSPInstance(COORDINATES, distance_matrix_threshold=3)

    eager_solution = solver.solve(eager)
    on_demand_solution = solver.solve(on_demand)

    assert on_demand_solution.route == eager_solution.route
    assert route_length(on_demand, on_demand_solution.route) == route_length(
        eager, eager_solution.route
    )
