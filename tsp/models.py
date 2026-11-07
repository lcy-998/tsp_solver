from collections.abc import Sequence
from dataclasses import dataclass, field
from math import hypot
from typing import Any


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
        matrix = [
            [
                hypot(x1 - x2, y1 - y2)
                for x2, y2 in self.coordinates
            ]
            for x1, y1 in self.coordinates
        ]
        object.__setattr__(self, "distance_matrix", matrix)

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

