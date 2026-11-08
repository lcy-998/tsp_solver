from collections.abc import Sequence

from .models import TSPInstance, TSPSolution


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib; install requirements.txt") from exc
    return plt


def plot_solution(instance: TSPInstance, solution: TSPSolution, ax=None):
    plt = _import_pyplot()
    if ax is None:
        _, ax = plt.subplots()
    xs = [instance.coordinates[city - 1][0] for city in solution.route]
    ys = [instance.coordinates[city - 1][1] for city in solution.route]
    ax.plot(xs, ys, marker="o", label=f"{solution.algorithm}: {solution.length:.4f}")
    for city, (x, y) in enumerate(instance.coordinates, 1):
        ax.annotate(str(city), (x, y), xytext=(4, 4), textcoords="offset points")
    ax.set_title("TSP solution")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    return ax


def plot_comparison(instance: TSPInstance, solutions: list[TSPSolution]):
    plt = _import_pyplot()
    fig, axes = plt.subplots(1, len(solutions), squeeze=False)
    for ax, solution in zip(axes[0], solutions):
        plot_solution(instance, solution, ax=ax)
    fig.tight_layout()
    return fig, axes[0]


def plot_performance_comparison(
    instances: Sequence[TSPInstance],
    results: Sequence[Sequence[TSPSolution]],
):
    """Compare route length and runtime for each algorithm across instances.

    ``results[i]`` must contain the solutions for ``instances[i]``. The two
    panels use the same instance order, making algorithm trade-offs easy to
    compare without drawing all tours on top of one another.
    """
    plt = _import_pyplot()
    if len(instances) != len(results):
        raise ValueError("instances and results must have the same length")
    if not instances:
        raise ValueError("at least one instance is required")

    algorithms = list(dict.fromkeys(solution.algorithm for group in results for solution in group))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    x = list(range(1, len(instances) + 1))
    for algorithm in algorithms:
        lengths = []
        runtimes = []
        for group in results:
            solution = next((item for item in group if item.algorithm == algorithm), None)
            lengths.append(solution.length if solution else float("nan"))
            runtimes.append(solution.runtime_seconds if solution else float("nan"))
        axes[0].plot(x, lengths, marker="o", label=algorithm)
        axes[1].plot(x, runtimes, marker="o", label=algorithm)
    axes[0].set_title("Route length")
    axes[1].set_title("Runtime")
    axes[0].set_ylabel("length")
    axes[1].set_ylabel("seconds")
    axes[0].set_xlabel("instance")
    axes[1].set_xlabel("instance")
    axes[1].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    return fig, axes


def plot_scale_statistics(
    instances: Sequence[TSPInstance],
    results: Sequence[Sequence[TSPSolution]],
):
    """Show algorithm quality and runtime grouped by instance size."""
    plt = _import_pyplot()
    if len(instances) != len(results):
        raise ValueError("instances and results must have the same length")
    if not instances:
        raise ValueError("at least one instance is required")

    algorithms = list(dict.fromkeys(solution.algorithm for group in results for solution in group))
    sizes = [instance.size for instance in instances]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for algorithm in algorithms:
        lengths = []
        runtimes = []
        for size, group in zip(sizes, results):
            solution = next((item for item in group if item.algorithm == algorithm), None)
            if solution is not None:
                lengths.append((size, solution.length))
                runtimes.append((size, solution.runtime_seconds))
        if lengths:
            axes[0].scatter([item[0] for item in lengths], [item[1] for item in lengths], label=algorithm)
            axes[1].scatter([item[0] for item in runtimes], [item[1] for item in runtimes], label=algorithm)
    axes[0].set_title("Length by instance size")
    axes[1].set_title("Runtime by instance size")
    axes[0].set_xlabel("cities")
    axes[1].set_xlabel("cities")
    axes[0].set_ylabel("length")
    axes[1].set_ylabel("seconds")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    return fig, axes


def plot_iteration_progress(solutions: Sequence[TSPSolution]):
    """Plot convergence histories stored in ``solution.metadata['history']``."""
    plt = _import_pyplot()
    histories = [solution for solution in solutions if solution.metadata.get("history")]
    if not histories:
        raise ValueError("solutions do not contain iteration histories")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for solution in histories:
        history = solution.metadata["history"]
        ax.plot(range(len(history)), history, label=solution.algorithm)
    ax.set_title("Iteration progress")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best route length")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax

