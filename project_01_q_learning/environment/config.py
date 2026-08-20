from dataclasses import dataclass


@dataclass(frozen=True)
class GridWorldConfig:
    rows: int
    cols: int
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]
    max_steps: int
