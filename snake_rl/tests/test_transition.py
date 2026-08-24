import numpy as np

from environment import SnakeState, StepResult


def test_step_result_contains_state_reward_and_done():
    state = SnakeState(
        np.zeros(12, dtype=np.float32)
    )

    result = StepResult(
        state=state,
        reward=10,
        done=False,
    )

    assert result.state is state
    assert result.reward == 10
    assert result.done is False

def test_step_result_is_immutable():
    state = SnakeState(
        np.zeros(12, dtype=np.float32)
    )

    result = StepResult(
        state=state,
        reward=10,
        done=False,
    )

    try:
        result.reward = -10
    except AttributeError:
        pass
    else:
        raise AssertionError("StepResult should be immutable")