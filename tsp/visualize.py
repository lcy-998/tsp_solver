from .models import TSPInstance, TSPSolution


def plot_solution(instance: TSPInstance, solution: TSPSolution, ax=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib; install requirements.txt") from exc
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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib; install requirements.txt") from exc
    fig, axes = plt.subplots(1, len(solutions), squeeze=False)
    for ax, solution in zip(axes[0], solutions):
        plot_solution(instance, solution, ax=ax)
    fig.tight_layout()
    return fig, axes[0]

