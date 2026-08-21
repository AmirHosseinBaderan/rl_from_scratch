import numpy as np

from environment import Direction, SnakeEnvironment


def test_environment_creation():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    assert environment.grid.width == 10
    assert environment.grid.height == 10

    assert len(environment.snake.body) == 3

    assert np.array_equal(
        environment.food.position,
        np.array([8, 5]),
    )


def test_environment_initializes_snake_in_center():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    assert np.array_equal(
        environment.snake.head,
        np.array([5, 5]),
    )

    assert environment.snake.direction == Direction.RIGHT


def test_environment_step_moves_snake():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.step(Direction.RIGHT)

    assert np.array_equal(
        environment.snake.head,
        np.array([6, 5]),
    )


def test_environment_step_changes_direction():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.step(Direction.UP)

    assert environment.snake.direction == Direction.UP

    assert np.array_equal(
        environment.snake.head,
        np.array([5, 6]),
    )


def test_environment_reset_restores_initial_state():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.step(Direction.UP)
    environment.step(Direction.LEFT)

    environment.reset()

    assert np.array_equal(
        environment.snake.head,
        np.array([5, 5]),
    )

    assert environment.snake.direction == Direction.RIGHT

    assert np.array_equal(
        environment.food.position,
        np.array([8, 5]),
    )