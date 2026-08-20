from dataclasses import dataclass


@dataclass(frozen=True)
class GridWorldConfig:
    rows: int
    cols: int
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (0, 0)
    obstacles: frozenset[tuple[int, int]] = frozenset()
    max_steps: int = 100
