from dataclasses import dataclass

import numpy as np

from .action import to_direction, Action
from .state_builder import StateBuilder
from .state import  SnakeState
from .direction import Direction
from .food import Food
from .grid import Grid
from .snake import Snake
from .reward import Reward


@dataclass
class SnakeEnvironment:
    width: int
    height: int

    def __post_init__(self) -> None:
        self.grid = Grid(
            width=self.width,
            height=self.height,
        )

        self.state_builder = StateBuilder()
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

        self.done = False

    def step(self, action: Action) -> int:
        if self.done:
            return 0

        direction = to_direction(action)

        self.snake.change_direction(direction)

        next_head = self.snake.head + self.snake.direction.vector

        food_eaten = np.array_equal(
            next_head,
            self.food.position,
        )

        self.snake.move(grow=food_eaten)

        if food_eaten:
            self.spawn_food()

        self.done = self.is_collision()

        if self.done:
            return Reward.COLLISION

        if food_eaten:
            return Reward.FOOD

        return Reward.STEP

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

    def spawn_food(self) -> None:
        available_positions = [
            np.array([x, y], dtype=np.int64)
            for x in range(self.width)
            for y in range(self.height)
            if not any(
                np.array_equal(
                    np.array([x, y]),
                    segment,
                )
                for segment in self.snake.body
            )
        ]

        if not available_positions:
            raise RuntimeError("No available position for food.")

        self.food.position = available_positions[0]

    def get_state(self) -> SnakeState:
        return self.state_builder.build(
            snake=self.snake,
            food=self.food,
            grid=self.grid,
        )
