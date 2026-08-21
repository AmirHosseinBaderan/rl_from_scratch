import numpy as np

from environment import Direction


def test_up_direction():
    assert np.array_equal(
        Direction.UP.vector,
        np.array([0, 1]),
    )


def test_down_direction():
    assert np.array_equal(
        Direction.DOWN.vector,
        np.array([0, -1]),
    )


def test_left_direction():
    assert np.array_equal(
        Direction.LEFT.vector,
        np.array([-1, 0]),
    )


def test_right_direction():
    assert np.array_equal(
        Direction.RIGHT.vector,
        np.array([1, 0]),
    )


def test_direction_vector_returns_copy():
    vector = Direction.RIGHT.vector

    vector[0] = 999

    assert np.array_equal(
        Direction.RIGHT.vector,
        np.array([1, 0]),
    )