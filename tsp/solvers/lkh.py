from __future__ import annotations

from random import Random
from typing import Iterable

from ..candidates import nearest_neighbours
from ..local_search import two_opt
from ..models import TSPInstance, TSPSolution
from ..solver import TSPSolver


class LKHSolver(TSPSolver):
    """Lin-Kernighan-Helsgaun style variable-depth TSP heuristic."""

    name = "lkh"

    def __init__(self, candidate_count: int = 15, runs: int = 5,
                 max_depth: int = 4, seed: int | None = None,
                 kick_count: int = 1, max_trials: int = 800) -> None:
        if candidate_count < 1 or runs < 1 or max_depth < 2 or kick_count < 0 or max_trials < 1:
            raise ValueError("invalid LKH parameters")
        self.candidate_count, self.runs = candidate_count, runs
        self.max_depth, self.seed = max_depth, seed
        self.kick_count, self.max_trials = kick_count, max_trials

    def solve(self, instance: TSPInstance) -> TSPSolution:
        n = instance.size
        if n == 0:
            raise ValueError("cannot solve an empty instance")
        if n == 1:
            return TSPSolution(self.name, [1, 1], 0.0, 0.0)
        d = instance.distance_matrix
        candidates = nearest_neighbours(d, self.candidate_count)
        rng = Random(self.seed)
        best, best_cost = None, float("inf")
        history: list[float] = []
        for run in range(self.runs):
            route = self._construct(run % n if run < n else rng.randrange(n), candidates, d, rng, run > 0)
            route = two_opt(route, d, candidates)
            route = self._lk_search(route, candidates, d)
            for _ in range(self.kick_count):
                route = self._double_bridge(route, rng)
                route = two_opt(route, d, candidates)
                route = self._lk_search(route, candidates, d)
            cost = self._length(route, d)
            if cost < best_cost:
                best, best_cost = route[:], cost
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
                               },
                               "runs": self.runs,
                               "max_depth": self.max_depth,
                           })

    @staticmethod
    def _construct(start: int, candidates, d, rng: Random, randomized: bool) -> list[int]:
        route, left = [start], set(range(len(d))) - {start}
        while left:
            choices = [x for x in candidates[route[-1]] if x in left] or list(left)
            choices.sort(key=lambda x: d[route[-1]][x])
            nxt = rng.choice(choices[:min(4, len(choices))]) if randomized else choices[0]
            route.append(nxt); left.remove(nxt)
        return route

    def _lk_search(self, route: list[int], candidates, d) -> list[int]:
        n = len(route)
        while True:
            pos = {city: i for i, city in enumerate(route)}
            best_move, trials = None, 0
            budget = [self.max_trials]
            for i in range(n):
                if trials >= self.max_trials:
                    break
                t1, t2 = route[i], route[(i + 1) % n]
                move = self._search_exchange(route, pos, candidates, d, t1, t2,
                                              {(min(t1, t2), max(t1, t2))}, set(), 0.0, [t1, t2], 1, budget)
                trials += 1
                if move is not None and (best_move is None or move[1] > best_move[1]):
                    best_move = move
            if best_move is None:
                return route
            route = best_move[0]

    def _search_exchange(self, route, pos, candidates, d, t1, t2, removed, added, gain, seq, depth, budget):
        if depth >= self.max_depth or budget[0] <= 0:
            return None
        best = None
        for t3 in candidates[t2]:
            budget[0] -= 1
            if budget[0] < 0:
                break
            y = (min(t2, t3), max(t2, t3))
            if t3 in seq or y in removed or y in added:
                continue
            gain_add = gain + d[seq[-2]][seq[-1]] - d[t2][t3]
            close = (min(t3, t1), max(t3, t1))
            if close not in removed and close not in added:
                total = gain_add
                candidate = self._apply_exchange(route, removed, added | {y, close})
                if total > 1e-12 and candidate is not None:
                    best = (candidate, total)
            p = pos[t3]
            for t4 in (route[(p - 1) % len(route)], route[(p + 1) % len(route)]):
                x = (min(t3, t4), max(t3, t4))
                if x in removed or x in added or t4 in seq:
                    continue
                result = self._search_exchange(route, pos, candidates, d, t1, t4,
                                               removed | {x}, added | {y}, gain_add + d[t3][t4],
                                               seq + [t3, t4], depth + 1, budget)
                if result is not None and (best is None or result[1] > best[1]):
                    best = result
        return best

    @staticmethod
    def _apply_exchange(route, removed, added):
        edges = {(min(route[i], route[(i + 1) % len(route)]), max(route[i], route[(i + 1) % len(route)]))
                 for i in range(len(route))}
        edges.difference_update(removed); edges.update(added)
        graph = {x: [] for x in route}
        for a, b in edges:
            if a == b:
                return None
            graph[a].append(b); graph[b].append(a)
        if any(len(v) != 2 for v in graph.values()):
            return None
        result, seen = [route[0]], {route[0]}
        while len(result) < len(route):
            nxt = next((x for x in graph[result[-1]] if x not in seen), None)
            if nxt is None:
                return None
            result.append(nxt); seen.add(nxt)
        return result if result[0] in graph[result[-1]] else None

    @staticmethod
    def _double_bridge(route, rng):
        if len(route) < 8:
            return route
        a, b, c = sorted(rng.sample(range(1, len(route)), 3))
        return route[:a] + route[c:] + route[b:c] + route[a:b]

    @staticmethod
    def _length(route: Iterable[int], d) -> float:
        route = list(route)
        return sum(d[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))
