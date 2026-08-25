import numpy as np
from agent import Experience
from environment import Action

def test_experience_creation():
    state = np.zeros(12,dtype=np.float32)
    next_state = np.ones(12,dtype=np.float32)

    experience = Experience(
        state=state,
        action=Action.RIGHT,
        reward=-10.0,
        next_state=next_state,
        done=False,
    )

    assert np.array_equal(experience.state, state)
    assert experience.action == Action.RIGHT
    assert experience.reward == -10
    assert  np.array_equal(experience.next_state, next_state)
    assert experience.done is False

def test_experience_is_immutable():
    experience = Experience(
        state=np.zeros(12,dtype=np.float32),
        action=Action.RIGHT,
        reward=-10.0,
        next_state=np.ones(12,dtype=np.float32),
        done=False,
    )

    try:
        experience.reward = -10.0
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "Experience should not be immutable."
        )