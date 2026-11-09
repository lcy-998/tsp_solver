"""Before/after benchmark for the three performance fixes.

Run:  .venv-plot/Scripts/python.exe tool/bench_after.py

The legacy routine below is the previous implementation (full neighbourhood,
restart from the first edge after every improving move). It is kept only to
measure the speed-up; the solvers now use tsp.local_search.two_opt.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsp.candidates import nearest_neighbours
from tsp.local_search import tour_length, two_opt
from tsp.metrics import route_length
from tsp.parser import iter_instances
from tsp.solvers.hybrid import HybridTSPSolver
from tsp.solvers.lkh import LKHSolver


def legacy_two_opt(tour, d, budget=None, on_improve=None):
    """Previous implementation: O(n^2) sweep, restarted after every move."""
    n = len(tour)
    deadline = None if budget is None else time.perf_counter() + budget
    moves = 0
    while True:
        changed = False
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, e = tour[j], tour[(j + 1) % n]
                if d[a][c] + d[b][e] + 1e-12 < d[a][b] + d[c][e]:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    moves += 1
                    changed = True
                    if on_improve is not None:
                        on_improve(moves, tour_length(tour, d))
                    if deadline is not None and time.perf_counter() > deadline:
                        return moves
                    break
            if changed:
                break
        if not changed:
            return moves


def nn_tour(d, n):
    tour, left = [0], set(range(1, n))
    while left:
        current = tour[-1]
        nxt = min(left, key=lambda c: d[current][c])
        tour.append(nxt)
        left.remove(nxt)
    return tour


def main():
    instance = next(iter_instances("data/tsp1000_2.txt"))
    d, n = instance.distance_matrix, instance.size
    reference = route_length(instance, instance.reference_route) if instance.reference_route else None
    print(f"instance: n={n}  reference tour: {reference:.3f}" if reference else f"instance: n={n}")

    start = time.perf_counter()
    candidates = nearest_neighbours(d, 15)
    print(f"candidates k=15 built in {(time.perf_counter() - start) * 1000:.0f} ms")

    tour = nn_tour(d, n)
    nn_length = tour_length(tour, d)
    print(f"nearest-neighbour start: {nn_length:.3f}")

    print("\n== new: candidate-list 2-opt, scaling ==")
    scaling = []
    for size in (100, 200, 400, n):
        dn = d if size == n else [row[:size] for row in d[:size]]
        cn = candidates if size == n else nearest_neighbours(dn, 15)
        sub = tour[:] if size == n else nn_tour(dn, size)
        begin = time.perf_counter()
        two_opt(sub, dn, cn)
        elapsed = time.perf_counter() - begin
        length = tour_length(sub, dn)
        scaling.append((size, elapsed, length))
        print(f"  n={size:5d}  {elapsed:8.4f}s  len={length:9.3f}")

    new_length = scaling[-1][2]

    print("\n== legacy: full-neighbourhood 2-opt, scaling ==")
    legacy_scaling = []
    for size in (100, 200, 400):
        dn = [row[:size] for row in d[:size]]
        sub = nn_tour(dn, size)
        begin = time.perf_counter()
        legacy_two_opt(sub, dn)
        elapsed = time.perf_counter() - begin
        legacy_scaling.append((size, elapsed))
        print(f"  n={size:5d}  {elapsed:8.3f}s  len={tour_length(sub, dn):9.3f}")

    print("\n== legacy on the full instance, 60 s budget ==")
    legacy = tour[:]
    samples = [(0.0, nn_length)]
    begin = time.perf_counter()

    def record(moves, length):
        samples.append((time.perf_counter() - begin, length))

    moves = legacy_two_opt(legacy, d, budget=60.0, on_improve=record)
    legacy_time = time.perf_counter() - begin
    legacy_length = tour_length(legacy, d)
    print(f"  {moves} moves in {legacy_time:.1f}s -> len={legacy_length:.3f}")

    print("\n== full solvers ==")
    rows = []
    for label, solver in (
        ("hybrid", HybridTSPSolver(starts=6, perturbations=10, seed=42)),
        ("lkh", LKHSolver(candidate_count=15, runs=3, max_depth=4, kick_count=1,
                          max_trials=400, seed=42)),
    ):
        begin = time.perf_counter()
        solution = solver.solve(instance)
        elapsed = time.perf_counter() - begin
        rows.append((label, elapsed, solution.length))
        print(f"  {label:8s} {elapsed:7.2f}s  len={solution.length:9.3f}")

    baseline = reference if reference else min(
        [legacy_length, new_length] + [length for _, _, length in rows])
    print(f"\n== gap vs reference ({baseline:.3f}) ==")
    print(f"  candidate-list 2-opt   {100 * (new_length / baseline - 1):+6.2f}%  {scaling[-1][1] * 1000:8.1f} ms")
    print(f"  legacy 2-opt           {100 * (legacy_length / baseline - 1):+6.2f}%  {legacy_time:8.1f} s")
    for label, elapsed, length in rows:
        print(f"  {label:22s} {100 * (length / baseline - 1):+6.2f}%  {elapsed:8.2f} s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.step([t for t, _ in samples], [v for _, v in samples], where="post",
             color="#E24B4A", label="legacy 2-opt (60 s budget)")
    ax1.axhline(new_length, color="#378ADD", linewidth=1.8,
                label=f"candidate-list 2-opt ({scaling[-1][1] * 1000:.0f} ms)")
    ax1.axhline(baseline, color="#3B6D11", linestyle=":", linewidth=1.2, label="reference tour")
    ax1.set_xscale("log")
    ax1.set_xlabel("seconds")
    ax1.set_ylabel("tour length")
    ax1.set_title(f"Anytime progress, n = {n}", fontsize=11)
    values = [v for _, v in samples] + [new_length, baseline, nn_length]
    ax1.set_ylim(min(values) * 0.99, max(values) * 1.01)
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(fontsize=8, frameon=False)

    ax2.plot([s for s, _ in legacy_scaling], [t for _, t in legacy_scaling], "o-",
             color="#E24B4A", label="legacy 2-opt")
    ax2.plot([s for s, _, _ in scaling], [t for _, t, _ in scaling], "s-",
             color="#378ADD", label="candidate-list 2-opt")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("cities n")
    ax2.set_ylabel("seconds to local optimum")
    ax2.set_title("Scaling", fontsize=11)
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    output = Path("output")
    output.mkdir(exist_ok=True)
    fig.savefig(output / "before_after.png", dpi=160)
    print(f"\nchart written to {output / 'before_after.png'}")


if __name__ == "__main__":
    main()
