import matplotlib.pyplot as plt

from tsp.compare import compare, format_results
from tsp.parser import iter_instances
from tsp.solvers import NearestNeighborSolver, SimulatedAnnealingSolver, \
                        DynamicProgrammingSolver
from tsp.visualize import plot_comparison


solvers = [NearestNeighborSolver(), SimulatedAnnealingSolver(), DynamicProgrammingSolver()]

for instance in iter_instances("data/sample.txt"):
    results = compare(instance, solvers)
    print(f"{instance.name} : {instance.size}")
    print(format_results(instance, results))
    plot_comparison(instance, results)

plt.show()

