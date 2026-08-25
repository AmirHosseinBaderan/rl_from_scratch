import numpy as np
import torch
import pytest

from agent import (
    Experience,
    create_batch,
)
from environment import Action


def create_experience(
    action: Action,
    reward: float,
    done: bool,
) -> Experience:
    return Experience(
        state=np.zeros(12, dtype=np.float32),
        action=action,
        reward=reward,
        next_state=np.ones(12, dtype=np.float32),
        done=done,
    )


def test_create_batch():
    experiences = [
        create_experience(
            Action.UP,
            1.0,
            False,
        ),
        create_experience(
            Action.RIGHT,
            10.0,
            True,
        ),
    ]

    batch = create_batch(experiences)

    assert batch.states.shape == (2, 12)
    assert batch.actions.shape == (2,)
    assert batch.rewards.shape == (2,)
    assert batch.next_states.shape == (2, 12)
    assert batch.dones.shape == (2,)

    assert batch.states.dtype == torch.float32
    assert batch.actions.dtype == torch.long
    assert batch.rewards.dtype == torch.float32
    assert batch.dones.dtype == torch.bool

def test_create_batch_rejects_empty_experiences():
    with pytest.raises(ValueError):
        create_batch([])