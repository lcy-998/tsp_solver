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
from ..spatial import nearest_neighbour_graph


class _TourLinks:
    """Undirected tour adjacency, updated incrementally across LK moves.

    Reversing a segment does not change any edge of the tour, only the order in
    which the two neighbours of each city are visited.  So the adjacency arrays
    only need touching at the few endpoints of an exchange, while a full walk
    stays a tight loop over two flat lists.
    """

    __slots__ = ("adj0", "adj1", "marked", "ov0", "ov1")

    def __init__(self, route: list[int]) -> None:
        n = len(route)
        self.adj0 = [0] * n
        self.adj1 = [0] * n
        for index, city in enumerate(route):
            self.adj0[city] = route[index - 1]
            self.adj1[city] = route[(index + 1) % n]
        self.marked = bytearray(n)
        self.ov0 = [0] * n
        self.ov1 = [0] * n


class LKHSolver(TSPSolver):
    """Pure-Python LKH-2 core for symmetric travelling salesman problems.

    This is not a wrapper around the official executable.  Candidate edges are
    ordered by alpha-nearness after Held--Karp ascent, and improvement uses
    backtracking Lin--Kernighan sequential k-opt exchanges followed by LKH-style
    trials and kicks.  It does not implement LKH-3's other problem classes or
    the official solver's full set of non-sequential and large-tour techniques.

    Instances larger than ``large_scale_threshold`` use a near-neighbour graph:
    the 1-tree relaxation and alpha values are then computed only on O(nK)
    candidate edges instead of all n^2 pairs, which is what makes ten-thousand
    city instances tractable in pure Python.
    """

    name = "lkh"

    def __init__(self, candidate_count: int = 5, runs: int = 5,
                 max_depth: int = 5, seed: int | None = None,
                 kick_count: int = 1, max_trials: int = 800,
                 ascent_iterations: int | None = 50,
                 large_scale_threshold: int = 1000,
                 sparse: bool | None = None,
                 candidate_graph_width: int | None = None) -> None:
        if candidate_count < 1 or runs < 1 or max_depth < 2 or kick_count < 0 or max_trials < 1:
            raise ValueError("invalid LKH parameters")
        if ascent_iterations is not None and ascent_iterations < 0:
            raise ValueError("ascent_iterations must be non-negative or None")
        if large_scale_threshold < 0:
            raise ValueError("large_scale_threshold must be non-negative")
        if sparse is not None and type(sparse) is not bool:
            raise ValueError("sparse must be a bool or None")
        if candidate_graph_width is not None and candidate_graph_width < 1:
            raise ValueError("candidate_graph_width must be positive or None")
        self.candidate_count, self.runs = candidate_count, runs
        self.max_depth, self.seed = max_depth, seed
        self.kick_count, self.max_trials = kick_count, max_trials
        self.ascent_iterations = ascent_iterations
        self.large_scale_threshold = large_scale_threshold
        self.sparse = sparse
        self.candidate_graph_width = candidate_graph_width

    def solve(self, instance: TSPInstance) -> TSPSolution:
        n = instance.size
        if n == 0:
            raise ValueError("cannot solve an empty instance")
        if n == 1:
            return TSPSolution(self.name, [1, 1], 0.0, 0.0)
        d = instance.distance_matrix

        large = self.sparse if self.sparse is not None else n >= self.large_scale_threshold
        graph = None
        width = None
        if large:
            width = self.candidate_graph_width or max(3 * self.candidate_count, 12)
            graph = nearest_neighbour_graph(instance.coordinates, width)
            construction_candidates = graph
        else:
            # Ordinary neighbours are used only to construct tours and to supply
            # a valid upper bound for ascent.  The LK neighbourhood itself is
            # alpha based, as in LKH, rather than an ordinary KNN list.
            construction_candidates = nearest_neighbours(d, self.candidate_count)

        initial = nearest_neighbor_route(d, start=0, candidates=construction_candidates)
        ascent_budget = self.ascent_iterations
        if ascent_budget is None:
            ascent_budget = min(n, 50)
        ascent = held_karp_ascent(d, tour_length(initial, d), ascent_budget, graph)
        candidates = alpha_candidate_sets(
            d, self.candidate_count, ascent.penalties, ascent.one_tree, graph,
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
                                   "large_scale_threshold": self.large_scale_threshold,
                                   "sparse": self.sparse,
                                   "candidate_graph_width": self.candidate_graph_width,
                               },
                               "runs": self.runs,
                               "max_depth": self.max_depth,
                               "trials": total_trials,
                               "candidate_type": "alpha",
                               "mode": "sparse" if large else "complete",
                               "bound_scope": ascent.one_tree.scope,
                               "candidate_graph_width": width,
                               # In sparse mode the 1-tree lives on a restricted
                               # graph, so this value bounds that graph's
                               # relaxation, not necessarily the true TSP.
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
        """Apply first-improving sequential LK moves until local convergence.

        Don't-look bits keep the search linear in the number of moves: after a
        successful exchange only cities incident to changed edges (and their
        tour neighbours) become eligible starting points again.  Tour edges are
        tested from the position array in O(1), so no adjacency table or edge
        set has to be rebuilt after every improvement.
        """
        n = len(route)
        if n < 4:
            return route
        position = [0] * n
        for index, city in enumerate(route):
            position[city] = index
        links = _TourLinks(route)
        active = bytearray(b"\x01") * n
        pending = route[::-1]

        while pending:
            t1 = pending.pop()
            if not active[t1]:
                continue
            active[t1] = 0
            index = position[t1]
            move = None
            # Both orientations are necessary: reversing t1 and t2 changes
            # the feasible sequential exchanges, not just the tour direction.
            for t2 in (route[(index + 1) % n], route[index - 1]):
                x1 = (min(t1, t2), max(t1, t2))
                move = self._search_exchange(
                    route, position, links, candidates, d, t1, t2,
                    {x1}, set(), d[t1][t2], [t1, t2], 1,
                )
                if move is not None:
                    break
            if move is None:
                continue
            route, _, touched = move
            for index, city in enumerate(route):
                position[city] = index
            for city in list(touched):
                at = position[city]
                touched.add(route[at - 1])
                touched.add(route[(at + 1) % n])
            for city in touched:
                if not active[city]:
                    active[city] = 1
                    pending.append(city)
        return route

    @staticmethod
    def _tour_edge(a: int, b: int, position: list[int], n: int) -> bool:
        difference = position[a] - position[b]
        return difference == 1 or difference == -1 or difference == n - 1 or difference == 1 - n

    def _search_exchange(self, route, position, links, candidates, d,
                         t1, t2, removed, added, gain, seq, depth):
        """Backtracking LK search from the last removed edge ``(t1, t2)``.

        ``gain`` is removed cost minus added cost after the latest x-edge.  A
        y-edge is considered only while the cumulative gain remains positive.
        Closing is always allowed to use an edge outside the candidate set.
        Returns ``(tour, gain, touched_cities)`` on success.
        """
        if depth >= self.max_depth:
            return None
        n = len(route)
        for t3 in candidates[t2]:
            y = (min(t2, t3), max(t2, t3))
            if t3 in seq or y in removed or y in added:
                continue
            if self._tour_edge(t2, t3, position, n):
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
                if (not self._tour_edge(t4, t1, position, n)
                        and close not in removed and close not in added):
                    total_gain = next_gain - d[t4][t1]
                    if total_gain > 1e-12:
                        new_removed = removed | {x}
                        new_added = added | {y, close}
                        candidate = self._apply_exchange(
                            route, position, new_removed, new_added, links,
                        )
                        if candidate is not None:
                            touched = {city for edge in new_removed | new_added for city in edge}
                            return candidate, total_gain, touched

                result = self._search_exchange(
                    route, position, links, candidates, d, t1, t4,
                    removed | {x}, added | {y}, next_gain,
                    seq + [t3, t4], depth + 1,
                )
                if result is not None:
                    return result
        return None

    @staticmethod
    def _apply_exchange(route, position, removed, added, links=None):
        """Return the single tour induced by an edge exchange, if one exists.

        Only endpoints of changed edges are overridden.  A subtour is rejected
        as soon as the walk returns to its start, so a promising but infeasible
        exchange costs one walk rather than a full graph rebuild.  On success
        the incremental adjacency arrays are committed; on failure they are
        left exactly as they were.
        """
        n = len(route)
        if links is None:
            links = _TourLinks(route)
        adj0, adj1 = links.adj0, links.adj1
        marked, ov0, ov1 = links.marked, links.ov0, links.ov1
        overrides: dict[int, list[int]] = {}

        def neighbours(city):
            if city not in overrides:
                overrides[city] = (
                    [ov0[city], ov1[city]] if marked[city]
                    else [adj0[city], adj1[city]]
                )
            return overrides[city]

        def rollback():
            for city in overrides:
                marked[city] = 0
            return None

        for a, b in removed:
            for city, other in ((a, b), (b, a)):
                adjacent = neighbours(city)
                if other not in adjacent:
                    return rollback()
                adjacent.remove(other)
        for a, b in added:
            if a == b:
                return rollback()
            for city, other in ((a, b), (b, a)):
                adjacent = neighbours(city)
                if other in adjacent:
                    return rollback()
                adjacent.append(other)
        if any(len(adjacent) != 2 for adjacent in overrides.values()):
            return rollback()

        for city, (first, second) in overrides.items():
            ov0[city], ov1[city] = first, second
            marked[city] = 1

        start = route[0]
        if marked[start]:
            first, second = ov0[start], ov1[start]
        else:
            first, second = adj0[start], adj1[start]
        previous, current = start, min(first, second)
        result = [start]
        while current != start:
            result.append(current)
            if len(result) > n:
                return rollback()
            if marked[current]:
                first, second = ov0[current], ov1[current]
            else:
                first, second = adj0[current], adj1[current]
            following = first if first != previous else second
            previous, current = current, following
        if len(result) != n:
            return rollback()

        for city, (first, second) in overrides.items():
            adj0[city], adj1[city] = first, second
            marked[city] = 0
        return result

    @staticmethod
    def _double_bridge(route, rng):
        return double_bridge(route, rng)

    @staticmethod
    def _length(route: Iterable[int], d) -> float:
        return tour_length(list(route), d)
