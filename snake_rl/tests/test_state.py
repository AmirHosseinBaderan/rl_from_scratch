import numpy as np
import pytest

from environment import SnakeState


def test_state_accepts_12_features():
    values = np.zeros(12, dtype=np.float32)

    state = SnakeState(values)

    assert state.values.shape == (12,)


def test_state_rejects_invalid_shape():
    values = np.zeros(10, dtype=np.float32)

    with pytest.raises(ValueError):
        SnakeState(values)


def test_state_as_array_returns_copy():
    values = np.zeros(12, dtype=np.float32)

    state = SnakeState(values)

    result = state.as_array()

    result[0] = 1.0

    assert state.values[0] == 0.0