from typing import Any, Protocol


class TSPEnvProtocol(Protocol):
    """Minimal reset/step contract for future RL environments."""

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict]: ...

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]: ...

