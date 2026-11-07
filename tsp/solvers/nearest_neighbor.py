from ..models import TSPInstance, TSPSolution
from ..solver import TSPSolver


class NearestNeighborSolver(TSPSolver):
    name = "nearest_neighbor"

    def solve(self, instance: TSPInstance) -> TSPSolution:
        if not instance.coordinates:
            raise ValueError("cannot solve an empty instance")
        unvisited = set(range(2, instance.size + 1))
        route = [1]
        while unvisited:
            current = route[-1] - 1
            next_city = min(
                unvisited,
                key=lambda city: instance.distance_matrix[current][city - 1],
            )
            route.append(next_city)
            unvisited.remove(next_city)
        route.append(1)
        return TSPSolution(self.name, route, 0.0, 0.0)

