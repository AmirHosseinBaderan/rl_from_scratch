from dataclasses import dataclass

import numpy as np

from .direction import Direction
from .food import Food
from .grid import Grid
from .snake import Snake


@dataclass
class SnakeEnvironment:
    width: int
    height: int

    def __post_init__(self) -> None:
        self.grid = Grid(
            width=self.width,
            height=self.height,
        )

        self.reset()

    def reset(self) -> None:
        center = np.array(
            [
                self.width // 2,
                self.height // 2,
            ],
            dtype=np.int64,
        )

        self.snake = Snake(
            body=[
                center.copy(),
                center + np.array([-1, 0]),
                center + np.array([-2, 0]),
            ],
            direction=Direction.RIGHT,
        )

        self.food = Food(
            position=np.array(
                [
                    self.width - 2,
                    self.height // 2,
                ],
                dtype=np.int64,
            )
        )

    def step(self, direction: Direction) -> None:
        self.snake.change_direction(direction)

        next_head = self.snake.head + self.snake.direction.vector

        food_eaten = np.array_equal(
            next_head,
            self.food.position,
        )

        self.snake.move(grow=food_eaten)

    def is_collision(self) -> bool:
        if not self.grid.contains(self.snake.head):
            return True

        return any(
            np.array_equal(
                self.snake.head,
                segment,
            )
            for segment in self.snake.body[1:]
        )

    def is_food_eaten(self) -> bool:
        return np.array_equal(
            self.snake.head,
            self.food.position,
        )