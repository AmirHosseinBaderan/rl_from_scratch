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

def test_state_contains_current_direction():
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

    assert state.values[0] == 0.0
    assert state.values[1] == 0.0
    assert state.values[2] == 0.0
    assert state.values[3] == 1.0

def test_state_detects_wall_danger():
    grid = Grid(width=5, height=5)

    snake = Snake(
        body=[
            np.array([4, 2]),
            np.array([3, 2]),
            np.array([2, 2]),
        ],
        direction=Direction.RIGHT,
    )

    food = Food(
        position=np.array([1, 1])
    )

    state = StateBuilder().build(
        snake=snake,
        food=food,
        grid=grid,
    )

    assert state.values[7] == 1.0