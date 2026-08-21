from dataclasses import dataclass

import numpy as np

from .direction import Direction


@dataclass
class Snake:
    body: list[np.ndarray]
    direction: Direction

    def __post_init__(self) -> None:
        if not self.body:
            raise ValueError("Snake body cannot be empty.")

    @property
    def head(self) -> np.ndarray:
        return self.body[0]

    def move(self) -> None:
        new_head = self.head + self.direction.vector

        self.body.insert(0, new_head)
        self.body.pop()

    def change_direction(self, direction: Direction) -> None:
        self.direction = direction