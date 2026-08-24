import numpy as np

from environment import (
    Direction,
    Food,
    Grid,
    Snake,
    StateBuilder,
)


def test_state_builder_returns_12_features():
    grid = Grid(width=10, height=10)

    snake = Snake(
        body=[
            np.array([5, 5]),
            np.array([4, 5]),
            np.array([3, 5]),
        ],
        direction=Direction.RIGHT,
    )

    food = Food(
        position=np.array([7, 3])
    )

    state = StateBuilder().build(
        snake=snake,
        food=food,
        grid=grid,
    )

    assert state.values.shape == (12,)