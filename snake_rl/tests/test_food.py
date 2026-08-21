import numpy as np
import pytest

from environment import Food


def test_food_creation():
    food = Food(
        position=np.array([5, 7])
    )

    assert np.array_equal(
        food.position,
        np.array([5, 7]),
    )


def test_food_rejects_invalid_position_shape():
    with pytest.raises(ValueError):
        Food(
            position=np.array([5, 7, 2])
        )


def test_food_rejects_one_dimensional_invalid_position():
    with pytest.raises(ValueError):
        Food(
            position=np.array([5])
        )


def test_food_position_can_be_zero():
    food = Food(
        position=np.array([0, 0])
    )

    assert np.array_equal(
        food.position,
        np.array([0, 0]),
    )