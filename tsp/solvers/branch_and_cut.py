"""Exact TSP solver: branch-and-cut over the DFJ relaxation.

This is the algorithm Concorde implements, written from scratch on top of the
framework's own pieces:

* **Relaxation** - the Dantzig-Fulkerson-Johnson LP: degree equalities plus
  subtour elimination constraints ``x(delta(S)) >= 2``.
* **Separation** - violated subtour cuts are found exactly, from connected
  components for integral points and with Stoer-Wagner's global minimum cut
  for fractional ones (see :mod:`tsp.cuts`).
* **Branching** - on a fractional edge, producing ``x_e = 0`` / ``x_e = 1``
  children; nodes are explored best-bound first and pruned against the
  incumbent, so the tour returned is *proven optimal* when the search
  finishes inside the node/time budget.
* **Upper bounds** - nearest-neighbour + candidate 2-opt seeds the incumbent.
* **LP** - solved by :mod:`tsp.lp`, a two-phase simplex shipped with the
  framework, so no third-party solver is required.

Concorde adds far more machinery on top (comb/blossom/domino cuts, pricing,
column generation, a tuned LP); this implementation keeps the same skeleton at
a size a course experiment can read, and is practical for ``n`` up to roughly
30-60 cities.  Above that, use the heuristics.
"""

from __future__ import annotations

from heapq import heappop, heappush
from random import Random
from time import perf_counter
from typing import NamedTuple

from ..candidates import nearest_neighbours
from ..construction import nearest_neighbor_route
from ..cuts import (
    TOL,
    Comb,
    canonical_side,
    comb_coefficients,
    comb_inequalities,
    connected_components,
    global_min_cut,
)
from ..local_search import two_opt
from ..lp import LPModel
from ..models import TSPInstance, TSPSolution
from ..perturbations import double_bridge
from ..solver import TSPSolver

_INFEASIBLE = "infeasible"


class _Cut(NamedTuple):
    """A cut in the ``sum coef_e x_e >= rhs`` family, ready for the LP."""

    key: tuple
    coefs: tuple[tuple[int, float], ...]
    rhs: float
    kind: str = "subtour"


class _NodeLP:
    """One search node's LP: warm starts plus cut-and-column generation.

    Columns start as a candidate set (the nearest neighbours of every city, the
    incumbent tour, and every edge a pooled cut mentions).  After each solve the
    inactive edges are priced against the duals; the ones with a negative
    reduced cost are appended and the LP is re-solved.  If pricing has not
    converged after ``pricing_rounds`` the remaining columns are activated, so
    every bound that leaves this object belongs to the *complete* relaxation -
    a restricted LP bounds the relaxation from above, which would be unsound for
    pruning.
    """

    def __init__(self, solver, fixed: dict[int, int], pool, upper_bounds, *, pricing: bool) -> None:
        self.solver = solver
        self.n = solver._n
        self.edges = solver._edges
        self.lengths = solver._lengths
        self.incident = solver._incident
        self.fixed = fixed
        self.dead = False
        self.pricing = pricing
        self.pricing_rounds = 6
        # A city that already has two forced edges cannot use any other edge, so
        # those variables are pinned at zero.  Dropping them here (instead of
        # letting the degree row ``sum x = 0`` carry the information) keeps the
        # LP small and, more importantly, keeps phase 1 away from a very
        # degenerate vertex: empty-equality rows force hundreds of extra pivots.
        saturated = [0] * self.n
        for edge, state in fixed.items():
            if state == 1:
                i, j = self.edges[edge]
                saturated[i] += 1
                saturated[j] += 1
        self.free = [
            edge
            for edge in range(len(self.edges))
            if edge not in fixed
            and saturated[self.edges[edge][0]] < 2
            and saturated[self.edges[edge][1]] < 2
        ]
        self.constant = sum(self.lengths[edge] for edge, value in fixed.items() if value == 1)
        self.degree_row: dict[int, int] = {}
        self.edge_rows: dict[int, list[tuple[int, float]]] = {}
        self.ub_row: dict[int, int] = {}

        free_set = set(self.free)
        if pricing:
            seed = {edge for edge in solver._seed_edges if edge in free_set}
            seed |= {edge for edge in solver._incumbent_edges if edge in free_set}
        else:
            seed = set(free_set)
        for cut in pool:
            seed.update(edge for edge, _ in cut.coefs if edge in free_set)
        seed.update(edge for edge in upper_bounds if edge in free_set)

        self.active = [edge for edge in self.free if edge in seed]
        self.column = {edge: index for index, edge in enumerate(self.active)}
        if pricing and not self._covers_degrees():
            self.active = list(self.free)
            self.column = {edge: index for index, edge in enumerate(self.active)}
            self.pricing = False

        self.model = LPModel(len(self.active), [self.lengths[edge] for edge in self.active])
        self._build_degree_rows()
        if self.dead:
            return
        for edge in upper_bounds:
            index = self.column.get(edge)
            if index is None:
                continue
            self.ub_row[edge] = self.model.n_rows
            self.model.add_row({index: 1.0}, 1.0, "<=")

    # ------------------------------------------------------------- assembly

    def _covers_degrees(self) -> bool:
        for city in range(self.n):
            available = 0
            for edge in self.incident[city]:
                if edge in self.column:
                    available += 1
                    if available >= 2:
                        break
            if available < 2:
                return False
        return True

    def _build_degree_rows(self) -> None:
        for city in range(self.n):
            row: dict[int, float] = {}
            forced = 0
            for edge in self.incident[city]:
                state = self.fixed.get(edge)
                if state is None:
                    index = self.column.get(edge)
                    if index is not None:
                        row[index] = 1.0
                elif state == 1:
                    forced += 1
            if forced > 2:
                self.dead = True
                return
            rhs = 2 - forced
            if not row:
                if rhs != 0:
                    self.dead = True
                    return
                self.degree_row[city] = -1
                continue
            self.degree_row[city] = self.model.n_rows
            self.model.add_row(row, float(rhs), "==")

    def add_upper_bound(self, edge: int) -> None:
        if edge in self.ub_row:
            return
        index = self.column.get(edge)
        if index is None:
            self._activate([edge])
            index = self.column[edge]
        self.ub_row[edge] = self.model.n_rows
        self.model.add_row({index: 1.0}, 1.0, "<=")

    def add_cut(self, coefficients: dict[int, float], rhs: float) -> bool:
        missing = [edge for edge in coefficients if edge not in self.column]
        if missing:
            self._activate(missing)
        row = {self.column[edge]: value for edge, value in coefficients.items()}
        if not row:
            return False
        index = self.model.n_rows
        self.model.add_row(row, rhs, ">=")
        for edge, value in coefficients.items():
            self.edge_rows.setdefault(edge, []).append((index, value))
        return True

    # ----------------------------------------------------------------- solve

    def solve(self):
        if self.dead:
            return None
        result = self.model.solve()
        if result.status != "optimal":
            if self.pricing and self._activate_remaining():
                result = self.model.solve()
                if result.status != "optimal":
                    return None
                self.pricing = False
            else:
                return None
        if not self.pricing:
            return result
        for _ in range(self.pricing_rounds):
            additions = self._price()
            if not additions:
                return result
            self._activate([edge for _, edge in additions])
            result = self.model.solve()
            if result.status != "optimal":
                return None
        if self._activate_remaining():
            result = self.model.solve()
            if result.status != "optimal":
                return None
        self.pricing = False
        return result

    def full_x(self, x: list[float]) -> list[float]:
        vector = [0.0] * len(self.edges)
        for edge, index in self.column.items():
            vector[edge] = x[index]
        return vector

    # --------------------------------------------------------------- pricing

    def _price(self, limit: int = 60) -> list[tuple[float, int]]:
        duals = self.model.duals
        if not duals:
            return []
        prices = [0.0] * len(self.edges)
        for city, row_index in self.degree_row.items():
            if row_index < 0 or row_index >= len(duals):
                continue
            value = duals[row_index]
            if value:
                for edge in self.incident[city]:
                    prices[edge] += value
        for edge, entries in self.edge_rows.items():
            for row_index, coefficient in entries:
                if row_index < len(duals):
                    value = duals[row_index]
                    if value:
                        prices[edge] += value * coefficient
        for edge, row_index in self.ub_row.items():
            if row_index < len(duals):
                prices[edge] += duals[row_index]

        candidates = []
        for edge in self.free:
            if edge in self.column:
                continue
            reduced = self.lengths[edge] - prices[edge]
            if reduced < -1e-6:
                candidates.append((reduced, edge))
        candidates.sort()
        return candidates[:limit]

    def _activate(self, edges: list[int]) -> None:
        missing = [edge for edge in edges if edge not in self.column]
        if not missing:
            return
        columns = []
        for edge in missing:
            i, j = self.edges[edge]
            entries: dict[int, float] = {}
            for city in (i, j):
                row_index = self.degree_row.get(city, -1)
                if row_index >= 0:
                    entries[row_index] = entries.get(row_index, 0.0) + 1.0
            for row_index, value in self.edge_rows.get(edge, ()):
                entries[row_index] = entries.get(row_index, 0.0) + value
            row_index = self.ub_row.get(edge)
            if row_index is not None:
                entries[row_index] = 1.0
            columns.append((self.lengths[edge], entries))
        for edge, index in zip(missing, self.model.add_columns(columns)):
            self.column[edge] = index
            self.active.append(edge)

    def _activate_remaining(self) -> bool:
        missing = [edge for edge in self.free if edge not in self.column]
        if not missing:
            return False
        self._activate(missing)
        return True


class BranchAndCutSolver(TSPSolver):
    """Exact branch-and-cut solver for the symmetric TSP."""

    name = "branch_and_cut"

    def __init__(
        self,
        *,
        node_limit: int = 2000,
        time_limit: float | None = None,
        cut_rounds: int = 200,
        deep_cut_rounds: int = 3,
        separation_depth: int = 6,
        max_cuts: int = 600,
        candidate_count: int = 12,
        polish_kicks: int = 40,
        comb_cuts: bool = True,
        comb_depth: int = 1,
        comb_rounds: int = 3,
        pricing: bool = True,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if node_limit < 1 or cut_rounds < 1 or deep_cut_rounds < 0 or max_cuts < 0 or separation_depth < 0:
            raise ValueError("invalid branch-and-cut parameters")
        self.node_limit = node_limit
        self.time_limit = time_limit
        self.cut_rounds = cut_rounds
        self.deep_cut_rounds = deep_cut_rounds
        self.separation_depth = separation_depth
        self.max_cuts = max_cuts
        self.candidate_count = candidate_count
        self.polish_kicks = polish_kicks
        self.comb_cuts = comb_cuts
        self.comb_depth = comb_depth
        self.comb_rounds = comb_rounds
        self.pricing = pricing
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------ solve

    def solve(self, instance: TSPInstance) -> TSPSolution:
        n = instance.size
        if n == 0:
            raise ValueError("cannot solve an empty instance")
        if n == 1:
            return TSPSolution(self.name, [1, 1], 0.0, 0.0)
        if n == 2:
            return TSPSolution(self.name, [1, 2, 1], 0.0, 0.0)

        started = perf_counter()
        distances = instance.distance_matrix
        self._prepare(n, distances)

        best_route, best_length = self._initial_tour(distances, instance)
        self._incumbent_edges = self._tour_edges(best_route)
        history: list[float] = [best_length]
        milestone: list[int] = [0]

        pool: list[_Cut] = []
        self._pool_keys: set[tuple] = set()
        self._comb_added = 0
        upper_bounds: list[int] = []

        counter = 0
        queue: list[tuple[float, int, int, dict[int, int]]] = [(float("-inf"), counter, 0, {})]
        nodes = 0
        cuts_added = 0
        lp_solves = 0
        lower_bound = float("-inf")
        stopped_early = False

        while queue:
            if nodes >= self.node_limit or self._out_of_time(started):
                stopped_early = True
                break
            bound, _, depth, fixed = heappop(queue)
            if bound >= best_length - 1e-9:
                continue
            nodes += 1

            outcome, x, value, new_cuts, iterations = self._solve_node(fixed, pool, upper_bounds, depth)
            cuts_added += new_cuts
            lp_solves += iterations
            if outcome == _INFEASIBLE:
                continue
            if value >= best_length - 1e-9:
                continue

            tour = self._integral_tour(x)
            if tour is not None:
                length = self._tour_length(tour)
                if length < best_length - 1e-9:
                    best_route, best_length = tour, length
                    self._incumbent_edges = self._tour_edges(best_route)
                    history.append(best_length)
                    milestone.append(nodes)
                    if self.verbose:
                        print(f"  node {nodes:5d}  incumbent {best_length:.4f}  (LP {value:.4f})")
                continue

            edge = self._branch_edge(x)
            if edge is None:  # numerically integral but structurally invalid
                continue
            for forced_value in (1, 0):
                child = dict(fixed)
                child[edge] = forced_value
                counter += 1
                heappush(queue, (value, counter, depth + 1, child))

        if queue:
            lower_bound = min(item[0] for item in queue)
        else:
            lower_bound = best_length
        optimal = not stopped_early and not queue
        elapsed = perf_counter() - started
        closed = [city + 1 for city in best_route] + [best_route[0] + 1]
        return TSPSolution(
            self.name,
            closed,
            self._tour_length(best_route),
            elapsed,
            metadata={
                "optimal": optimal,
                "lower_bound": max(lower_bound, 0.0) if optimal else lower_bound,
                "nodes": nodes,
                "cuts": cuts_added,
                "comb_cuts": self._comb_added,
                "lp_solves": lp_solves,
                "open_nodes": len(queue),
                "history": history,
                "history_label": "best length",
                "milestones": milestone,
                "parameters": {
                    "node_limit": self.node_limit,
                    "time_limit": self.time_limit,
                    "cut_rounds": self.cut_rounds,
                    "deep_cut_rounds": self.deep_cut_rounds,
                    "separation_depth": self.separation_depth,
                    "max_cuts": self.max_cuts,
                    "polish_kicks": self.polish_kicks,
                    "comb_cuts": self.comb_cuts,
                    "comb_depth": self.comb_depth,
                    "comb_rounds": self.comb_rounds,
                    "pricing": self.pricing,
                },
            },
        )

    # ------------------------------------------------------------- LP plumbing

    def _prepare(self, n: int, distances) -> None:
        self._n = n
        edges: list[tuple[int, int]] = []
        lengths: list[float] = []
        incident: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            row = distances[i]
            for j in range(i + 1, n):
                index = len(edges)
                edges.append((i, j))
                lengths.append(row[j])
                incident[i].append(index)
                incident[j].append(index)
        self._edges = edges
        self._lengths = lengths
        self._incident = incident
        self._distances = distances
        self._edge_id = [[-1] * n for _ in range(n)]
        for index, (i, j) in enumerate(edges):
            self._edge_id[i][j] = index
            self._edge_id[j][i] = index
        k = min(self.candidate_count, max(1, n - 1))
        self._seed_edges = {
            self._edge_id[city][other]
            for city, neighbours in enumerate(nearest_neighbours(distances, k))
            for other in neighbours
        }
        self._incumbent_edges: set[int] = set()

    def _solve_node(self, fixed: dict[int, int], pool: list[_Cut], upper_bounds: list[int], depth: int):
        """Solve one node: warm-started cutting planes, then the integrality test.

        Returns ``(status, x, objective, new_cuts, lp_iterations)`` where
        ``status`` is ``"optimal"`` or ``"infeasible"``.
        """
        pricing = self.pricing and depth == 0
        node = _NodeLP(self, fixed, pool, upper_bounds, pricing=pricing)
        if node.dead:
            return _INFEASIBLE, None, 0.0, 0, 0

        rounds = self.cut_rounds if depth == 0 else (self.deep_cut_rounds if depth <= self.separation_depth else 0)
        comb_left = self.comb_rounds if (self.comb_cuts and depth <= self.comb_depth) else 0
        total_iterations = 0
        new_cuts = 0
        x: list[float] = []
        value = float("inf")
        for round_index in range(max(1, rounds)):
            result = node.solve()
            if result is None:
                return _INFEASIBLE, None, 0.0, new_cuts, total_iterations
            total_iterations += result.iterations
            x = node.full_x(result.x)
            value = result.value

            violated = [edge for edge in node.free if x[edge] > 1 + 1e-6]
            if violated:
                for edge in violated:
                    if edge in node.ub_row:
                        continue
                    node.add_upper_bound(edge)
                    if edge not in upper_bounds:
                        upper_bounds.append(edge)
                continue
            if round_index + 1 >= max(1, rounds):
                break

            separated = self._separate(x, fixed, comb_left > 0)
            added = 0
            for cut in separated:
                if len(pool) >= self.max_cuts:
                    break
                if cut.key in self._pool_keys:
                    continue
                prepared = self._prepare_cut(cut, fixed)
                self._pool_keys.add(cut.key)
                pool.append(cut)
                new_cuts += 1
                if prepared is None:
                    return _INFEASIBLE, None, 0.0, new_cuts, total_iterations
                if prepared is False:
                    continue
                coefficients, rhs = prepared
                if not node.add_cut(coefficients, rhs):
                    return _INFEASIBLE, None, 0.0, new_cuts, total_iterations
                added += 1
                if cut.kind == "comb":
                    comb_left = max(0, comb_left - 1)
                    self._comb_added += 1
            if not added:
                break

        return "optimal", x, value + node.constant, new_cuts, total_iterations

    def _prepare_cut(self, cut: _Cut, fixed) -> tuple[dict[int, float], float] | None | bool:
        """Instantiate a cut at this node; ``None`` infeasible, ``False`` redundant."""
        coefficients: dict[int, float] = {}
        forced = 0.0
        for edge, value in cut.coefs:
            state = fixed.get(edge)
            if state is None:
                coefficients[edge] = coefficients.get(edge, 0.0) + value
            elif state == 1:
                forced += value
        rhs = cut.rhs - forced
        if not coefficients:
            return None if rhs > 1e-9 else False
        if rhs <= 1e-9:
            return False
        return coefficients, rhs

    def _cut_edges(self, side: frozenset[int]) -> frozenset[int]:
        crossing = []
        for edge, (i, j) in enumerate(self._edges):
            if (i in side) != (j in side):
                crossing.append(edge)
        return frozenset(crossing)

    def _subtour_cut(self, side: frozenset[int]) -> _Cut | None:
        crossing = self._cut_edges(side)
        if not crossing:
            return None
        key = ("subtour", canonical_side(side, self._n))
        return _Cut(key, tuple((edge, 1.0) for edge in crossing), 2.0, "subtour")

    def _comb_cut(self, comb: Comb) -> _Cut | None:
        coefficients = comb_coefficients(comb, self._edges, self._incident)
        if not coefficients:
            return None
        key = (
            "comb",
            tuple(sorted(comb.handle)),
            tuple(sorted(tuple(sorted(tooth)) for tooth in comb.teeth)),
        )
        rhs = float(3 * len(comb.teeth) + 1)
        return _Cut(key, tuple(sorted(coefficients.items())), rhs, "comb")

    def _separate(self, x: list[float], fixed, use_combs: bool) -> list[_Cut]:
        """Separate subtours first, combs only once the subtour relaxation is tight."""
        n = self._n
        weights = [[0.0] * n for _ in range(n)]
        for edge, (i, j) in enumerate(self._edges):
            value = x[edge]
            if value > TOL:
                weights[i][j] = value
                weights[j][i] = value
        for edge, value in fixed.items():
            if value == 1:
                i, j = self._edges[edge]
                weights[i][j] = 1.0
                weights[j][i] = 1.0

        components = connected_components(n, weights)
        if len(components) > 1:
            cuts = []
            for component in components:
                if len(component) >= n:
                    continue
                cut = self._subtour_cut(frozenset(component))
                if cut is not None:
                    cuts.append(cut)
            return cuts

        value, side = global_min_cut(n, weights)
        if value < 2 - TOL and 0 < len(side) < n:
            cut = self._subtour_cut(side)
            return [cut] if cut is not None else []
        if not use_combs:
            return []
        cuts = []
        for comb in comb_inequalities(n, weights, max_cuts=8):
            cut = self._comb_cut(comb)
            if cut is not None:
                cuts.append(cut)
        return cuts

    # --------------------------------------------------------- tour handling

    def _initial_tour(self, distances, instance: TSPInstance) -> tuple[list[int], float]:
        """Seed the incumbent: multi-start nearest neighbour, 2-opt, double-bridge kicks.

        A tight upper bound is worth the milliseconds here - every node whose
        bound exceeds the incumbent is pruned immediately.
        """
        n = instance.size
        rng = Random(self.seed if self.seed is not None else 0)
        candidates = nearest_neighbours(distances, min(self.candidate_count, max(1, n - 1)))
        best: list[int] | None = None
        best_length = float("inf")
        starts = min(n, 4 if n <= 120 else 2)
        for index in range(starts):
            tour = nearest_neighbor_route(
                distances,
                start=index % n,
                rng=rng if index else None,
                rcl_size=3 if index else 1,
            )
            tour = two_opt(tour, distances, candidates)
            length = self._tour_length(tour)
            if length < best_length:
                best, best_length = list(tour), length
        assert best is not None
        kicks = self.polish_kicks if n <= 150 else max(2, self.polish_kicks // 8)
        for _ in range(kicks):
            tour = two_opt(double_bridge(best, rng), distances, candidates)
            length = self._tour_length(tour)
            if length < best_length - 1e-9:
                best, best_length = list(tour), length
        return best, best_length

    def _tour_edges(self, tour: list[int]) -> set[int]:
        return {
            self._edge_id[a][b] for a, b in zip(tour, tour[1:] + tour[:1])
        }

    def _tour_length(self, tour: list[int]) -> float:
        distances = self._distances
        return sum(distances[a][b] for a, b in zip(tour, tour[1:] + tour[:1]))

    def _integral_tour(self, x: list[float]) -> list[int] | None:
        """Return the tour if ``x`` is a single Hamiltonian cycle, else ``None``."""
        n = self._n
        neighbours: list[list[int]] = [[] for _ in range(n)]
        for index, value in enumerate(x):
            if value > 0.5:
                i, j = self._edges[index]
                neighbours[i].append(j)
                neighbours[j].append(i)
        if any(len(items) != 2 for items in neighbours):
            return None
        tour = [0]
        previous, current = -1, 0
        while True:
            nxt = neighbours[current][0] if neighbours[current][0] != previous else neighbours[current][1]
            if nxt == 0:
                break
            tour.append(nxt)
            previous, current = current, nxt
            if len(tour) > n:
                return None
        return tour if len(tour) == n else None

    def _branch_edge(self, x: list[float]) -> int | None:
        best_edge, best_gap = None, float("inf")
        for index, value in enumerate(x):
            if 1e-6 < value < 1 - 1e-6:
                gap = abs(value - 0.5)
                if gap < best_gap:
                    best_edge, best_gap = index, gap
        if best_edge is not None:
            return best_edge
        # Numerically integral but not a valid tour: branch on a used edge.
        for index, value in enumerate(x):
            if value > 0.5:
                return index
        return None

    def _out_of_time(self, started: float) -> bool:
        return self.time_limit is not None and perf_counter() - started > self.time_limit
