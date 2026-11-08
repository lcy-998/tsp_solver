from collections import Counter
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

    counts = Counter(result.algorithm for result in results)
    seen: Counter[str] = Counter()
    used_names = set(counts)
    for result in results:
        name = result.algorithm
        if counts[name] > 1:
            while True:
                seen[name] += 1
                numbered_name = f"{name}_{seen[name]}"
                if numbered_name not in used_names:
                    break
            result.algorithm = numbered_name
            used_names.add(numbered_name)
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

