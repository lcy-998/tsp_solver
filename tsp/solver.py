from abc import ABC, abstractmethod

from .models import TSPInstance, TSPSolution


class TSPSolver(ABC):
    name = "unnamed"

    @abstractmethod
    def solve(self, instance: TSPInstance) -> TSPSolution:
        raise NotImplementedError

