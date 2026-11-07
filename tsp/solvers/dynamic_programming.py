from ..models import TSPInstance, TSPSolution
from ..solver import TSPSolver


class DynamicProgrammingSolver(TSPSolver):
    """Solve the TSP exactly with the Held-Karp subset dynamic program."""

    name = "dynamic_programming"

    def solve(self, instance: TSPInstance) -> TSPSolution:
        size = instance.size
        if size == 0:
            raise ValueError("cannot solve an empty instance")
        if size == 1:
            return TSPSolution(self.name, [1, 1], 0.0, 0.0)

        distances = instance.distance_matrix
        # Bit positions represent cities 2..n; city 1 is the fixed start.
        states: dict[tuple[int, int], tuple[float, int]] = {}
        for city in range(1, size):
            states[(1 << (city - 1), city)] = (distances[0][city], 0)

        for mask in range(1, 1 << (size - 1)):
            for last in range(1, size):
                if not mask & (1 << (last - 1)) or mask == 1 << (last - 1):
                    continue
                previous_mask = mask ^ (1 << (last - 1))
                best_cost = float("inf")
                best_previous = -1
                for previous in range(1, size):
                    if previous_mask & (1 << (previous - 1)):
                        cost = states[(previous_mask, previous)][0] + distances[previous][last]
                        if cost < best_cost:
                            best_cost = cost
                            best_previous = previous
                states[(mask, last)] = (best_cost, best_previous)

        full_mask = (1 << (size - 1)) - 1
        best_length = float("inf")
        last_city = -1
        for last in range(1, size):
            cost = states[(full_mask, last)][0] + distances[last][0]
            if cost < best_length:
                best_length = cost
                last_city = last

        route = [last_city]
        mask = full_mask
        while mask:
            _, previous = states[(mask, route[-1])]
            mask ^= 1 << (route[-1] - 1)
            if previous == 0:
                break
            route.append(previous)

        route.reverse()
        closed_route = [1, *(city + 1 for city in route), 1]
        return TSPSolution(self.name, closed_route, best_length, 0.0)
