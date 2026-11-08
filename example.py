import matplotlib.pyplot as plt

from tsp.compare import compare, format_results
from tsp.parser import iter_instances
from tsp.solvers import NearestNeighborSolver, SimulatedAnnealingSolver, \
                        DynamicProgrammingSolver
from tsp.visualize import (
    plot_comparison,
    plot_iteration_progress,
    plot_performance_comparison,
    plot_scale_statistics,
)


solvers = [NearestNeighborSolver(), SimulatedAnnealingSolver(seed=7), DynamicProgrammingSolver()]
instances = []
all_results = []

for instance in iter_instances("data/sample.txt"):
    results = compare(instance, solvers)
    instances.append(instance)
    all_results.append(results)
    print(f"{instance.name} : {instance.size}")
    print(format_results(instance, results))
    # plot_comparison(instance, results)

    # plot_iteration_progress(results)

# plot_performance_comparison(instances, all_results)
# plot_scale_statistics(instances, all_results)

plt.show()

