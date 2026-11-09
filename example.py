import matplotlib.pyplot as plt

from tsp.compare import compare, format_results
from tsp.parser import iter_instances
from tsp.solvers import NearestNeighborSolver, SimulatedAnnealingSolver, \
                        DynamicProgrammingSolver, LKHSolver, AntColonySolver,\
                        HybridSolver
from tsp.visualize import (
    plot_comparison,
    plot_iteration_progress,
    plot_performance_comparison,
    plot_scale_statistics,
    plot_aggregate_iteration_progress,
    plot_runtime_vs_gap
)


solvers = [
    HybridSolver(starts=10, perturbations=16, three_opt=True, seed=42),
    LKHSolver(candidate_count=15, runs=4, max_depth=4, kick_count=1, max_trials=400, seed=42),
    SimulatedAnnealingSolver(initial_temperature=5, cooling_rate=0.996, iterations_per_temperature=125, seed=42),
    # AntColonySolver(n_ants=30, n_iterations=50, alpha=1.0, beta=4.0, evaporation_rate=0.3, seed=42),
]
instances = []
all_results = []

for instance in iter_instances("data/tsp1000_2.txt"):
    results = compare(instance, solvers)
    instances.append(instance)
    all_results.append(results)
    print(f"{instance.name} : {instance.size}")
    print(format_results(instance, results))
    # plot_comparison(instance, results)
    # plot_iteration_progress(results)

# plot_performance_comparison(instances, all_results)
# plot_scale_statistics(instances, all_results)
plot_aggregate_iteration_progress(instances, all_results)
plot_runtime_vs_gap(instances, all_results)  

plt.show()

