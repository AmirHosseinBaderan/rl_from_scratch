import numpy as np
import pytest

from environment import Grid


def test_grid_creation():
    grid = Grid(width=10, height=8)

    assert grid.width == 10
    assert grid.height == 8


def test_grid_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        Grid(width=0, height=10)

    with pytest.raises(ValueError):
        Grid(width=10, height=0)


@pytest.mark.parametrize(
    "position",
    [
        np.array([0, 0]),
        np.array([5, 5]),
        np.array([9, 7]),
    ],
)
def test_contains_valid_position(position):
    grid = Grid(width=10, height=8)

    assert grid.contains(position)


@pytest.mark.parametrize(
    "position",
    [
        np.array([-1, 0]),
        np.array([0, -1]),
        np.array([10, 0]),
        np.array([0, 8]),
        np.array([10, 8]),
    ],
)
def test_contains_invalid_position(position):
    grid = Grid(width=10, height=8)

    assert not grid.contains(position)


def test_contains_rejects_invalid_position_shape():
    grid = Grid(width=10, height=8)

    with pytest.raises(ValueError):
        grid.contains(np.array([1, 2, 3]))