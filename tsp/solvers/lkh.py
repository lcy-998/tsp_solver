from __future__ import annotations

from random import Random
from typing import Iterable

from ..candidates import alpha_candidate_sets, nearest_neighbours
from ..construction import nearest_neighbor_route
from ..local_search import tour_length, two_opt
from ..models import TSPInstance, TSPSolution
from ..one_tree import held_karp_ascent
from ..perturbations import double_bridge
from ..solver import TSPSolver


class LKHSolver(TSPSolver):
    """Pure-Python LKH-2 core for symmetric travelling salesman problems.

    This is not a wrapper around the official executable.  Candidate edges are
    ordered by alpha-nearness after Held--Karp ascent, and improvement uses
    backtracking Lin--Kernighan sequential k-opt exchanges followed by LKH-style
    trials and kicks.  It does not implement LKH-3's other problem classes or
    the official solver's full set of non-sequential and large-tour techniques.
    """

    name = "lkh"

    def __init__(self, candidate_count: int = 5, runs: int = 5,
                 max_depth: int = 5, seed: int | None = None,
                 kick_count: int = 1, max_trials: int = 800,
                 ascent_iterations: int | None = 50) -> None:
        if candidate_count < 1 or runs < 1 or max_depth < 2 or kick_count < 0 or max_trials < 1:
            raise ValueError("invalid LKH parameters")
        if ascent_iterations is not None and ascent_iterations < 0:
            raise ValueError("ascent_iterations must be non-negative or None")
        self.candidate_count, self.runs = candidate_count, runs
        self.max_depth, self.seed = max_depth, seed
        self.kick_count, self.max_trials = kick_count, max_trials
        self.ascent_iterations = ascent_iterations

    def solve(self, instance: TSPInstance) -> TSPSolution:
        n = instance.size
        if n == 0:
            raise ValueError("cannot solve an empty instance")
        if n == 1:
            return TSPSolution(self.name, [1, 1], 0.0, 0.0)
        d = instance.distance_matrix

        # Ordinary neighbours are used only to construct tours and to supply a
        # valid upper bound for ascent.  The LK neighbourhood itself is alpha
        # based, as in LKH, rather than an ordinary nearest-neighbour list.
        construction_candidates = nearest_neighbours(d, self.candidate_count)
        initial = nearest_neighbor_route(d, start=0, candidates=construction_candidates)
        ascent_budget = self.ascent_iterations
        if ascent_budget is None:
            ascent_budget = min(n, 50)
        ascent = held_karp_ascent(d, tour_length(initial, d), ascent_budget)
        candidates = alpha_candidate_sets(
            d, self.candidate_count, ascent.penalties, ascent.one_tree,
        )

        rng = Random(self.seed)
        best, best_cost = None, float("inf")
        history: list[float] = []
        total_trials = 0
        for run in range(self.runs):
            route = self._construct(run % n if run < n else rng.randrange(n), candidates, d, rng, run > 0)
            route = two_opt(route, d, candidates)
            route = self._lk_search(route, candidates, d)
            run_best = route[:]
            run_best_cost = self._length(run_best, d)
            trials = 1
            stale_kicks = 0
            if run_best_cost < best_cost:
                best, best_cost = run_best[:], run_best_cost

            # A trial consists of one kick followed by LK convergence.  The
            # legacy kick_count remains useful as a limit on consecutive kicks
            # that fail to improve this run's incumbent.
            while trials < self.max_trials and stale_kicks < self.kick_count:
                kicked = self._double_bridge(run_best, rng)
                kicked = two_opt(kicked, d, candidates)
                kicked = self._lk_search(kicked, candidates, d)
                trials += 1
                kicked_cost = self._length(kicked, d)
                if kicked_cost + 1e-12 < run_best_cost:
                    run_best, run_best_cost = kicked, kicked_cost
                    stale_kicks = 0
                else:
                    stale_kicks += 1
                if run_best_cost < best_cost:
                    best, best_cost = run_best[:], run_best_cost

            total_trials += trials
            history.append(best_cost)

        assert best is not None
        closed = [x + 1 for x in best] + [best[0] + 1]
        return TSPSolution(self.name, closed, best_cost, 0.0,
                           metadata={
                               "history": history,
                               "history_label": "best length",
                               "parameters": {
                                   "candidate_count": self.candidate_count,
                                   "runs": self.runs,
                                   "max_depth": self.max_depth,
                                   "seed": self.seed,
                                   "kick_count": self.kick_count,
                                   "max_trials": self.max_trials,
                                   "ascent_iterations": self.ascent_iterations,
                               },
                               "runs": self.runs,
                               "max_depth": self.max_depth,
                               "trials": total_trials,
                               "candidate_type": "alpha",
                               "lower_bound": ascent.lower_bound,
                               "ascent_iterations": ascent.iterations,
                           })

    @staticmethod
    def _construct(start: int, candidates, d, rng: Random, randomized: bool) -> list[int]:
        return nearest_neighbor_route(
            d,
            start=start,
            rng=rng,
            rcl_size=4 if randomized else 1,
            candidates=candidates,
        )

    def _lk_search(self, route: list[int], candidates, d) -> list[int]:
        """Apply first-improving sequential LK moves until local convergence."""
        n = len(route)
        if n < 4:
            return route
        while True:
            position = [0] * n
            for index, city in enumerate(route):
                position[city] = index
            tour_edges = {
                (min(route[i], route[(i + 1) % n]), max(route[i], route[(i + 1) % n]))
                for i in range(n)
            }
            base_adjacency = [()] * n
            for index, city in enumerate(route):
                base_adjacency[city] = (route[index - 1], route[(index + 1) % n])
            move = None
            for i, t1 in enumerate(route):
                # Both orientations are necessary: reversing t1 and t2 changes
                # the feasible sequential exchanges, not just the tour direction.
                for t2 in (route[(i + 1) % n], route[i - 1]):
                    x1 = (min(t1, t2), max(t1, t2))
                    move = self._search_exchange(
                        route, position, tour_edges, base_adjacency, candidates, d, t1, t2,
                        {x1}, set(), d[t1][t2], [t1, t2], 1,
                    )
                    if move is not None:
                        break
                if move is not None:
                    break
            if move is None:
                return route
            route = move[0]

    def _search_exchange(self, route, position, tour_edges, base_adjacency, candidates, d,
                         t1, t2, removed, added, gain, seq, depth):
        """Backtracking LK search from the last removed edge ``(t1, t2)``.

        ``gain`` is removed cost minus added cost after the latest x-edge.  A
        y-edge is considered only while the cumulative gain remains positive.
        Closing is always allowed to use an edge outside the candidate set.
        """
        if depth >= self.max_depth:
            return None
        n = len(route)
        for t3 in candidates[t2]:
            y = (min(t2, t3), max(t2, t3))
            if t3 in seq or y in tour_edges or y in removed or y in added:
                continue
            gain_after_y = gain - d[t2][t3]
            if gain_after_y <= 1e-12:
                continue

            p = position[t3]
            for t4 in (route[(p + 1) % n], route[p - 1]):
                x = (min(t3, t4), max(t3, t4))
                if t4 in seq or x in removed or x in added:
                    continue
                next_gain = gain_after_y + d[t3][t4]
                close = (min(t4, t1), max(t4, t1))
                if close not in tour_edges and close not in removed and close not in added:
                    total_gain = next_gain - d[t4][t1]
                    if total_gain > 1e-12:
                        candidate = self._apply_exchange(
                            route, removed | {x}, added | {y, close}, base_adjacency,
                        )
                        if candidate is not None:
                            return candidate, total_gain

                result = self._search_exchange(
                    route, position, tour_edges, base_adjacency, candidates, d, t1, t4,
                    removed | {x}, added | {y}, next_gain,
                    seq + [t3, t4], depth + 1,
                )
                if result is not None:
                    return result
        return None

    @staticmethod
    def _apply_exchange(route, removed, added, base_adjacency=None):
        """Return the single tour induced by an edge exchange, if one exists.

        Only endpoints of changed edges need new adjacency lists.  The walk
        rejects a subtour as soon as it returns to its start, avoiding the old
        O(n) graph rebuild for every economically promising but infeasible move.
        """
        n = len(route)
        if base_adjacency is None:
            base_adjacency = [()] * n
            for index, city in enumerate(route):
                base_adjacency[city] = (route[index - 1], route[(index + 1) % n])
        overrides = {}

        def neighbours(city):
            if city not in overrides:
                overrides[city] = list(base_adjacency[city])
            return overrides[city]

        for a, b in removed:
            for city, other in ((a, b), (b, a)):
                adjacent = neighbours(city)
                if other not in adjacent:
                    return None
                adjacent.remove(other)
        for a, b in added:
            if a == b:
                return None
            for city, other in ((a, b), (b, a)):
                adjacent = neighbours(city)
                if other in adjacent:
                    return None
                adjacent.append(other)
        if any(len(adjacent) != 2 for adjacent in overrides.values()):
            return None

        start = route[0]
        first = overrides.get(start, base_adjacency[start])
        previous, current = start, min(first)
        result = [start]
        while current != start:
            result.append(current)
            if len(result) > n:
                return None
            adjacent = overrides.get(current, base_adjacency[current])
            following = adjacent[0] if adjacent[0] != previous else adjacent[1]
            previous, current = current, following
        return result if len(result) == n else None

    @staticmethod
    def _double_bridge(route, rng):
        return double_bridge(route, rng)

    @staticmethod
    def _length(route: Iterable[int], d) -> float:
        return tour_length(list(route), d)
