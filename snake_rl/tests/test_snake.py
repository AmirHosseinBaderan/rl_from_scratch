import numpy as np
import pytest

from environment import Snake


def create_snake() -> Snake:
    return Snake(
        body=[
            np.array([5, 5]),
            np.array([4, 5]),
            np.array([3, 5]),
        ],
        direction=np.array([1, 0]),
    )


def test_snake_creation():
    snake = create_snake()

    assert len(snake.body) == 3
    assert np.array_equal(snake.head, np.array([5, 5]))
    assert np.array_equal(snake.direction, np.array([1, 0]))


def test_snake_cannot_have_empty_body():
    with pytest.raises(ValueError):
        Snake(
            body=[],
            direction=np.array([1, 0]),
        )


def test_snake_rejects_invalid_direction_shape():
    with pytest.raises(ValueError):
        Snake(
            body=[np.array([5, 5])],
            direction=np.array([1, 0, 0]),
        )


def test_snake_move():
    snake = create_snake()

    snake.move()

    assert np.array_equal(
        snake.body,
        [
            np.array([6, 5]),
            np.array([5, 5]),
            np.array([4, 5]),
        ],
    )


def test_snake_length_remains_constant_after_move():
    snake = create_snake()

    initial_length = len(snake.body)

    snake.move()

    assert len(snake.body) == initial_length


def test_snake_change_direction():
    snake = create_snake()

    snake.change_direction(np.array([0, 1]))

    assert np.array_equal(
        snake.direction,
        np.array([0, 1]),
    )


def test_snake_rejects_invalid_direction():
    snake = create_snake()

    with pytest.raises(ValueError):
        snake.change_direction(np.array([1, 2, 3]))