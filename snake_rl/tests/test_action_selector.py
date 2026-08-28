import pytest
import torch

from agent import  EpsilonGreedySelector
from environment import Action


def test_selector_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        EpsilonGreedySelector(
            epsilon=1.5,
            action_size=4,
        )

def test_selector_rejects_invalid_action_size():
    with pytest.raises(ValueError):
        EpsilonGreedySelector(
            epsilon=0.1,
            action_size=0,
        )

def test_selector_exploits_best_action_when_epsilon_is_zero():
    selector = EpsilonGreedySelector(
        epsilon=0.0,
        action_size=4,
    )

    q_values = torch.tensor(
        [1.0, 5.0, 2.0, 3.0]
    )

    action = selector.select(q_values)

    assert action == Action.DOWN

def test_selector_explores_valid_actions():
    selector = EpsilonGreedySelector(
        epsilon=1.0,
        action_size=4,
    )

    q_values = torch.tensor(
        [1.0, 5.0, 2.0, 3.0]
    )

    for _ in range(100):
        action = selector.select(q_values)

        assert action in Action

def test_selector_rejects_invalid_q_values_shape():
    selector = EpsilonGreedySelector(
        epsilon=0.0,
        action_size=4,
    )

    q_values = torch.zeros(1, 4)

    with pytest.raises(ValueError):
        selector.select(q_values)

def test_epsilon_decay():
    selector = EpsilonGreedySelector(
        epsilon=1.0,
        action_size=4,
    )

    selector.decay(
        amount=0.2,
        minimum=0.1,
    )

    assert selector.epsilon == 0.8

def test_epsilon_decay_does_not_go_below_minimum():
    selector = EpsilonGreedySelector(
        epsilon=0.2,
        action_size=4,
    )

    selector.decay(
        amount=0.2,
        minimum=0.1,
    )

    assert selector.epsilon == 0.1

    selector.decay(
        amount=0.2,
        minimum=0.1,
    )

    assert selector.epsilon == 0.1

def test_epsilon_decay_rejects_negative_amount():
    selector = EpsilonGreedySelector(
        epsilon=1.0,
        action_size=4,
    )

    try:
        selector.decay(
            amount=-0.1,
            minimum=0.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )