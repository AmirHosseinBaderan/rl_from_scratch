import numpy as np
import pytest

from agent import  Experience, ReplayBuffer
from environment import Action


def create_experience() -> Experience:
    return Experience(
        state=np.zeros(12, dtype=np.float32),
        action=Action.RIGHT,
        reward=1.0,
        next_state=np.ones(12, dtype=np.float32),
        done=False,
    )


def test_replay_buffer_starts_empty():
    buffer = ReplayBuffer(capacity=100)

    assert len(buffer) == 0

def test_replay_buffer_add():
    buffer = ReplayBuffer(capacity=100)

    experience = create_experience()

    buffer.add(experience)

    assert len(buffer) == 1

def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=2)

    buffer.add(create_experience())
    buffer.add(create_experience())
    buffer.add(create_experience())

    assert len(buffer) == 2

def test_replay_buffer_sample():
    buffer = ReplayBuffer(capacity=10)

    for _ in range(5):
        buffer.add(create_experience())

    batch = buffer.sample(3)

    assert len(batch) == 3

    for experience in batch:
        assert isinstance(
            experience,
            Experience,
        )

def test_replay_buffer_rejects_large_sample():
    buffer = ReplayBuffer(capacity=10)

    buffer.add(create_experience())

    with pytest.raises(ValueError):
        buffer.sample(2)