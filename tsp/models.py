from dataclasses import dataclass, field
from math import hypot
from typing import Any


@dataclass(frozen=True)
class TSPInstance:
    coordinates: list[tuple[float, float]]
    reference_route: list[int] | None = None
    name: str | None = None
    distance_matrix: list[list[float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
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

