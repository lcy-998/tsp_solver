from ..construction import nearest_neighbor_route
from ..models import TSPInstance, TSPSolution
from ..solver import TSPSolver


class NearestNeighborSolver(TSPSolver):
    name = "nearest_neighbor"

    def solve(self, instance: TSPInstance) -> TSPSolution:
        if not instance.coordinates:
            raise ValueError("cannot solve an empty instance")
        route = nearest_neighbor_route(instance.distance_matrix)
        closed_route = [city + 1 for city in route]
        closed_route.append(closed_route[0])
        return TSPSolution(self.name, closed_route, 0.0, 0.0)

