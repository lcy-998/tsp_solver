from time import perf_counter

from .metrics import route_length
from .models import TSPInstance, TSPSolution
from .solver import TSPSolver


def compare(instance: TSPInstance, solvers: list[TSPSolver]) -> list[TSPSolution]:
    results: list[TSPSolution] = []
    for solver in solvers:
        started = perf_counter()
        result = solver.solve(instance)
        elapsed = perf_counter() - started
        result.length = route_length(instance, result.route)
        result.runtime_seconds = elapsed
        results.append(result)
    return results


def format_results(instance: TSPInstance, results: list[TSPSolution]) -> str:
    reference = route_length(instance, instance.reference_route) if instance.reference_route else None
    algorithm_width = max([len("algorithm"), *(len(result.algorithm) for result in results)], default=len("algorithm"))
    lines = [
        f"{'algorithm':<{algorithm_width}}  {'length':>12}  {'runtime_seconds':>16}  {'gap':>8}",
    ]
    for result in results:
        gap = "-" if reference is None else f"{(result.length / reference - 1) * 100:.2f}%"
        lines.append(
            f"{result.algorithm:<{algorithm_width}}  {result.length:>12.6f}  "
            f"{result.runtime_seconds:>16.6f}  {gap:>8}"
        )
    return "\n".join(lines)

