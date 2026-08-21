from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("Grid.width must be greater than 0.")
        if self.height <= 0:
            raise ValueError("Grid.height must be greater than 0.")

    def contains(self, position: np.ndarray) -> bool:
        if position.shape != (2,):
            raise ValueError("Grid.position must have shape (2,)")

        x, y = position
        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )
