"""Small, extensible toolkit for TSP course experiments."""

from .models import EpisodeResult, TSPInstance, TSPSolution
from .parser import iter_instances, load_instances, parse_line
from .solver import TSPSolver
from .compare import compare

__all__ = [
    "EpisodeResult",
    "TSPInstance",
    "TSPSolution",
    "TSPSolver",
    "compare",
    "load_instances",
    "iter_instances",
    "parse_line",
]

