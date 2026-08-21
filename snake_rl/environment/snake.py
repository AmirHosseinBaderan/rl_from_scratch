from dataclasses import dataclass

import numpy as np


@dataclass
class Snake:
    body: list[np.ndarray]
    direction: np.ndarray

    def __post_init__(self) -> None:
        if not self.body:
            raise ValueError("Snake body cannot be empty.")

        if self.direction.shape != (2,):
            raise ValueError("Direction must have shape (2,).")

    @property
    def head(self) -> np.ndarray:
        return self.body[0]

    def move(self) -> None:
        new_head = self.head + self.direction
        self.body.insert(0, new_head)
        self.body.pop()

    def change_direction(self, direction: np.ndarray) -> None:
        if direction.shape != (2,):
            raise ValueError("Direction must have shape (2,).")

        self.direction = direction