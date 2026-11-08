from collections.abc import Sequence
from textwrap import wrap

from .metrics import route_length
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


def plot_runtime_vs_gap(
    instances: Sequence[TSPInstance],
    results: Sequence[Sequence[TSPSolution]],
):
    """Plot runtime against gap to the reference route for every result.

    Each instance/algorithm result is represented by one point. Only
    instances with a reference route contribute points.
    """
    plt = _import_pyplot()
    if len(instances) != len(results):
        raise ValueError("instances and results must have the same length")
    if not instances:
        raise ValueError("at least one instance is required")

    grouped: dict[str, list[tuple[float, float, str]]] = {}
    for instance, instance_results in zip(instances, results):
        if instance.reference_route is None:
            continue
        reference_length = route_length(instance, instance.reference_route)
        if reference_length <= 0:
            raise ValueError("reference routes must have a positive length")
        for solution in instance_results:
            gap = (float(solution.length) / reference_length - 1) * 100
            grouped.setdefault(solution.algorithm, []).append(
                (float(solution.runtime_seconds), gap, instance.name or "instance")
            )

    if not grouped:
        raise ValueError("results do not contain reference routes and solutions")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for algorithm, values in grouped.items():
        ax.scatter(
            [runtime for runtime, _, _ in values],
            [gap for _, gap, _ in values],
            s=55,
            label=algorithm,
        )
        for runtime, gap, instance_name in values:
            ax.annotate(instance_name, (runtime, gap), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
    ax.set_title("Runtime vs gap to reference route")
    ax.set_xlabel("runtime (seconds)")
    ax.set_ylabel("gap to reference route (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig, ax
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
    """Plot best-so-far histories for iterative solvers."""
    plt = _import_pyplot()
    histories = [solution for solution in solutions if solution.metadata.get("history")]
    if not histories:
        raise ValueError("solutions do not contain iteration histories")
    fig_height = max(5, 2.5 + 0.65 * len(histories))
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    for solution in histories:
        history = solution.metadata["history"]
        parameters = solution.metadata.get("parameters", {})
        label = _iteration_label(solution.algorithm, parameters)
        ax.plot(range(len(history)), history, label=label, linewidth=2)
    ax.set_title("Iteration progress")
    ax.set_xlabel("iteration")
    ax.set_ylabel("route length")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    return fig, ax


def plot_aggregate_iteration_progress(
    instances: Sequence[TSPInstance],
    results: Sequence[Sequence[TSPSolution]],
    *,
    points: int = 101,
):
    """Aggregate histories using common elapsed-time and reference-gap axes."""
    plt = _import_pyplot()
    if len(instances) != len(results):
        raise ValueError("instances and results must have the same length")
    if not instances:
        raise ValueError("at least one instance is required")
    if points < 2:
        raise ValueError("points must be at least 2")

    grouped: dict[str, list[list[float]]] = {}
    for instance, instance_results in zip(instances, results):
        if instance.reference_route is None:
            continue
        reference_length = route_length(instance, instance.reference_route)
        max_runtime = max(
            (float(solution.runtime_seconds) for solution in instance_results),
            default=0.0,
        )
        target_times = [max_runtime * index / (points - 1) for index in range(points)]
        for solution in instance_results:
            history = solution.metadata.get("history")
            if not history:
                continue
            runtime = max(float(solution.runtime_seconds), 0.0)
            step = runtime / max(len(history) - 1, 1)
            elapsed = [index * step for index in range(len(history))]
            label = _iteration_label(
                solution.algorithm, solution.metadata.get("parameters", {})
            )
            sampled = _resample_by_time(history, elapsed, target_times)
            if reference_length == 0:
                gaps = [float(value) - reference_length for value in sampled]
            else:
                gaps = [(float(value) / reference_length - 1) * 100 for value in sampled]
            grouped.setdefault(label, []).append(gaps)
    if not grouped:
        raise ValueError("results do not contain reference routes and iteration histories")

    progress = [index * 100 / (points - 1) for index in range(points)]
    fig_height = max(5, 2.5 + 0.65 * len(grouped))
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    for label, histories in grouped.items():
        median = [_quantile([row[index] for row in histories], 0.5) for index in range(points)]
        lower = [_quantile([row[index] for row in histories], 0.25) for index in range(points)]
        upper = [_quantile([row[index] for row in histories], 0.75) for index in range(points)]
        line = ax.plot(progress, median, linewidth=2, label=label)[0]
        ax.fill_between(progress, lower, upper, color=line.get_color(), alpha=0.18)

    ax.set_title("Aggregate convergence against reference routes")
    ax.set_xlabel("elapsed time / slowest method time (%)")
    ax.set_ylabel("gap to reference route (%)")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    return fig, ax


def plot_iteration_summary(
    instances: Sequence[TSPInstance],
    results: Sequence[Sequence[TSPSolution]],
    *,
    points: int = 101,
):
    """Compare convergence speed, final precision, and runtime together.

    The three panels use the same configuration colors. Convergence is shown
    as relative gap to the best result observed for each instance; precision
    and runtime use one box per configuration across all instances.
    """
    plt = _import_pyplot()
    from matplotlib.lines import Line2D

    if len(instances) != len(results):
        raise ValueError("instances and results must have the same length")
    if not instances:
        raise ValueError("at least one instance is required")
    if points < 2:
        raise ValueError("points must be at least 2")

    grouped: dict[str, dict[str, list]] = {}
    for instance_results in results:
        available = [solution for solution in instance_results if solution.metadata.get("history")]
        if not available:
            continue
        best_length = min(solution.length for solution in instance_results)
        for solution in available:
            label = _iteration_label(solution.algorithm, solution.metadata.get("parameters", {}))
            entry = grouped.setdefault(label, {"histories": [], "gaps": [], "runtimes": []})
            history = [float(value) for value in solution.metadata["history"]]
            sampled = _resample_history(history, points)
            entry["histories"].append([(value / best_length - 1) * 100 for value in sampled])
            entry["gaps"].append((solution.length / best_length - 1) * 100)
            entry["runtimes"].append(float(solution.runtime_seconds))

    if not grouped:
        raise ValueError("results do not contain iteration histories")

    labels = list(grouped)
    identifiers = [f"P{index}" for index in range(1, len(labels) + 1)]
    colors = plt.get_cmap("tab10").colors
    progress = [index * 100 / (points - 1) for index in range(points)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

    for index, label in enumerate(labels):
        entry = grouped[label]
        histories = entry["histories"]
        median = [_quantile([row[column] for row in histories], 0.5) for column in range(points)]
        lower = [_quantile([row[column] for row in histories], 0.25) for column in range(points)]
        upper = [_quantile([row[column] for row in histories], 0.75) for column in range(points)]
        color = colors[index % len(colors)]
        axes[0].plot(progress, median, color=color, linewidth=2, label=identifiers[index])
        axes[0].fill_between(progress, lower, upper, color=color, alpha=0.16)

    axes[0].set_title("Convergence speed")
    axes[0].set_xlabel("normalized iteration progress (%)")
    axes[0].set_ylabel("gap to best observed result (%)")
    axes[0].set_xlim(0, 100)

    precision = axes[1].boxplot(
        [grouped[label]["gaps"] for label in labels],
        tick_labels=identifiers,
        patch_artist=True,
    )
    runtime = axes[2].boxplot(
        [grouped[label]["runtimes"] for label in labels],
        tick_labels=identifiers,
        patch_artist=True,
    )
    for boxes in (precision["boxes"], runtime["boxes"]):
        for index, box in enumerate(boxes):
            box.set_facecolor(colors[index % len(colors)])
            box.set_alpha(0.65)
    axes[1].set_title("Final precision")
    axes[1].set_ylabel("gap to best observed result (%)")
    axes[2].set_title("Total runtime")
    axes[2].set_ylabel("seconds")
    axes[2].set_yscale("log")

    legend_handles = [
        Line2D([0], [0], color=colors[index % len(colors)], linewidth=3, label=f"{identifiers[index]}\n{label}")
        for index, label in enumerate(labels)
    ]
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    return fig, axes


def _resample_history(history: Sequence[float], points: int) -> list[float]:
    if len(history) == 1:
        return [history[0]] * points
    scale = (len(history) - 1) / (points - 1)
    sampled = []
    for index in range(points):
        position = index * scale
        left = min(int(position), len(history) - 2)
        fraction = position - left
        sampled.append(history[left] + fraction * (history[left + 1] - history[left]))
    return sampled


def _resample_by_time(
    history: Sequence[float], elapsed: Sequence[float], target_times: Sequence[float]
) -> list[float]:
    if len(history) != len(elapsed) or not history:
        raise ValueError("history and elapsed must have the same non-zero length")
    if len(history) == 1:
        return [float(history[0])] * len(target_times)

    sampled = []
    for target in target_times:
        if target <= elapsed[0]:
            sampled.append(float(history[0]))
            continue
        if target >= elapsed[-1]:
            sampled.append(float(history[-1]))
            continue
        right = next(index for index, value in enumerate(elapsed) if value >= target)
        left = right - 1
        duration = elapsed[right] - elapsed[left]
        fraction = 0.0 if duration <= 0 else (target - elapsed[left]) / duration
        sampled.append(
            float(history[left])
            + fraction * (float(history[right]) - float(history[left]))
        )
    return sampled


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _iteration_label(algorithm: str, parameters: object) -> str:
    """Build a readable, complete legend label for any iterative solver."""
    if not isinstance(parameters, dict) or not parameters:
        return algorithm

    parameter_text = ", ".join(
        f"{name}={value!r}" for name, value in parameters.items()
    )
    wrapped = wrap(parameter_text, width=38) or [parameter_text]
    parameter_lines = [f"({wrapped[0]}"]
    parameter_lines.extend(wrapped[1:])
    parameter_lines[-1] = f"{parameter_lines[-1]})"
    return "\n".join([algorithm, *parameter_lines])

