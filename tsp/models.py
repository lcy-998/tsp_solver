from collections.abc import Sequence
from dataclasses import dataclass, field
from math import hypot
from typing import Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is optional
    _np = None

# Below this size a list-of-lists is both fast to index and cheap in memory
# (n = 2000 is roughly 130 MB); above it the packed numpy array wins on both.
_LIST_LIMIT = 2000


class _OnDemandDistanceRow(Sequence[float]):
    def __init__(self, point: tuple[float, float], coordinates: list[tuple[float, float]]) -> None:
        self.point = point
        self.coordinates = coordinates

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int | slice) -> float | list[float]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        x1, y1 = self.point
        x2, y2 = self.coordinates[index]
        return hypot(x1 - x2, y1 - y2)


class _OnDemandDistanceMatrix(Sequence[Sequence[float]]):
    def __init__(self, coordinates: list[tuple[float, float]]) -> None:
        self.coordinates = coordinates

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int | slice) -> Sequence[float] | list[Sequence[float]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return _OnDemandDistanceRow(self.coordinates[index], self.coordinates)


def _dense_distance_matrix(coordinates: list[tuple[float, float]]) -> Sequence[Sequence[float]]:
    """Full pairwise distance matrix, vectorised row by row when numpy exists."""
    n = len(coordinates)
    if _np is None or n == 0:
        return [[hypot(x1 - x2, y1 - y2) for x2, y2 in coordinates] for x1, y1 in coordinates]
    xs = _np.fromiter((point[0] for point in coordinates), dtype=_np.float64, count=n)
    ys = _np.fromiter((point[1] for point in coordinates), dtype=_np.float64, count=n)
    matrix = _np.empty((n, n), dtype=_np.float64)
    for row in range(n):
        matrix[row] = _np.hypot(xs - xs[row], ys - ys[row])
    # A list of lists indexes ~2x faster than a 2-D array on scalars, and the
    # solvers do millions of d[i][j] lookups, so keep it where memory allows.
    return matrix.tolist() if n <= _LIST_LIMIT else matrix


@dataclass(frozen=True)
class TSPInstance:
    coordinates: list[tuple[float, float]]
    reference_route: list[int] | None = None
    name: str | None = None
    distance_matrix_threshold: int | None = None
    distance_matrix: Sequence[Sequence[float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        threshold = self.distance_matrix_threshold
        if threshold is not None and (type(threshold) is not int or threshold < 0):
            raise ValueError("distance_matrix_threshold must be a non-negative integer or None")
        if threshold is not None and self.size > threshold:
            object.__setattr__(self, "distance_matrix", _OnDemandDistanceMatrix(self.coordinates))
            return
        object.__setattr__(self, "distance_matrix", _dense_distance_matrix(self.coordinates))

    @property
    def size(self) -> int:
        return len(self.coordinates)


@dataclass
class TSPSolution:
    algorithm: str
    route: list[int]
    length: float
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    episode: int
    total_reward: float
    route: list[int]
    route_length: float
    steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

