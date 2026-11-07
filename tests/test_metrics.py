from tsp.compare import compare
from tsp.metrics import route_length
from tsp.models import TSPInstance
from tsp.solvers import NearestNeighborSolver


def test_square_route_length():
    instance = TSPInstance([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert instance.distance_matrix[0][2] == 2**0.5
    assert route_length(instance, [1, 2, 3, 4, 1]) == 4


def test_compare_records_solution():
    instance = TSPInstance([(0, 0), (1, 0), (1, 1)])
    result = compare(instance, [NearestNeighborSolver()])[0]
    assert result.route == [1, 2, 3, 1]
    assert result.length > 0



