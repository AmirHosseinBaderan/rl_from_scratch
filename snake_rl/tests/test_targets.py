import torch
import pytest
from agent import compute_bellman_target
from environment import reward


def test_bellman_target():
    rewards = torch.tensor(
        [10.0, -10.0]
    )

    next_q_values = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    dones = torch.tensor(
        [False, True]
    )

    targets = compute_bellman_target(rewards=rewards, next_q_values=next_q_values, dones=dones, gamma=0.5)
    expected = torch.tensor(
        [
            12.0,
            -10.0
        ]
    )

    assert torch.allclose(targets, expected)

def test_bellman_target_rejects_invalid_gamma():
    rewards = torch.tensor([1.0])

    next_q_values = torch.zeros(1,4)
    dones = torch.tensor([False])

    with pytest.raises(ValueError):
        compute_bellman_target(
            rewards=rewards,
            next_q_values=next_q_values,
            dones=dones,
            gamma=1.5
        )