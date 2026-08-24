import numpy as np

from .direction import Direction
from .snake import Snake
from .food import Food
from .grid import Grid
from .state import SnakeState


class StateBuilder:

    def build(
        self,
        snake: Snake,
        food: Food,
        grid: Grid,
    ) -> SnakeState:
        head = snake.head
        direction = snake.direction

        values = np.array(
            [
                float(direction == Direction.UP),
                float(direction == Direction.DOWN),
                float(direction == Direction.LEFT),
                float(direction == Direction.RIGHT),

                float(self._is_danger(head, Direction.UP, snake, grid)),
                float(self._is_danger(head, Direction.DOWN, snake, grid)),
                float(self._is_danger(head, Direction.LEFT, snake, grid)),
                float(self._is_danger(head, Direction.RIGHT, snake, grid)),

                float(food.position[1] < head[1]),
                float(food.position[1] > head[1]),
                float(food.position[0] < head[0]),
                float(food.position[0] > head[0]),
            ],
            dtype=np.float32,
        )

        return SnakeState(values)

    def _is_danger(
        self,
        head: np.ndarray,
        direction: Direction,
        snake: Snake,
        grid: Grid,
    ) -> bool:
        next_position = head + direction.vector

        if not grid.contains(next_position):
            return True

        return any(
            np.array_equal(next_position, segment)
            for segment in snake.body[1:]
        )