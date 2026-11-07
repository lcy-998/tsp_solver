from .models import TSPInstance


def validate_route(route: list[int], size: int) -> None:
    if len(route) != size + 1 or route[0] != route[-1]:
        raise ValueError("route must contain every city once and close the tour")
    if set(route[:-1]) != set(range(1, size + 1)):
        raise ValueError("route contains missing or repeated cities")


def route_length(instance: TSPInstance, route: list[int]) -> float:
    validate_route(route, instance.size)
    return sum(instance.distance_matrix[start - 1][end - 1] for start, end in zip(route, route[1:]))

