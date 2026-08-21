import numpy as np
import pytest

from environment import Direction, Snake


def create_snake() -> Snake:
    return Snake(
        body=[
            np.array([5, 5]),
            np.array([4, 5]),
            np.array([3, 5]),
        ],
        direction=Direction.RIGHT,
    )


def test_snake_creation():
    snake = create_snake()

    assert len(snake.body) == 3
    assert np.array_equal(
        snake.head,
        np.array([5, 5]),
    )
    assert snake.direction == Direction.RIGHT


def test_snake_cannot_have_empty_body():
    with pytest.raises(ValueError):
        Snake(
            body=[],
            direction=Direction.RIGHT,
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


@pytest.mark.parametrize(
    "direction, expected_head",
    [
        (Direction.RIGHT, np.array([6, 5])),
        (Direction.LEFT, np.array([4, 5])),
        (Direction.UP, np.array([5, 6])),
        (Direction.DOWN, np.array([5, 4])),
    ],
)
def test_snake_moves_in_direction(direction, expected_head):
    snake = create_snake()

    snake.change_direction(direction)
    snake.move()

    assert np.array_equal(
        snake.head,
        expected_head,
    )


def test_snake_change_direction():
    snake = create_snake()

    snake.change_direction(Direction.UP)

    assert snake.direction == Direction.UP