import numpy as np
import torch

from agent import Experience, DQNAgent
from environment import Action


def create_experience() -> Experience:
    return Experience(
        state=np.zeros(
            12,
            dtype=np.float32,
        ),
        action=Action.RIGHT,
        reward=1.0,
        next_state=np.ones(
            12,
            dtype=np.float32,
        ),
        done=False,
    )

def test_agent_selects_valid_action():
    agent = DQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.0,
        replay_capacity=100,
    )

    state = torch.zeros(12)

    action = agent.select_action(state)

    assert action in Action

def test_agent_learns_from_replay_buffer():
    agent = DQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.0,
        replay_capacity=100,
    )

    for _ in range(10):
        agent.replay_buffer.add(
            create_experience()
        )

    loss = agent.learn(
        batch_size=4
    )

    assert isinstance(loss, float)
    assert loss >= 0.0

def test_agent_learning_updates_network():
    agent = DQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.0,
        replay_capacity=100,
    )

    for _ in range(10):
        agent.replay_buffer.add(
            create_experience()
        )

    before = [
        parameter.detach().clone()
        for parameter in agent.network.parameters()
    ]

    agent.learn(
        batch_size=4
    )

    after = list(
        agent.network.parameters()
    )

    assert any(
        not torch.equal(
            before_parameter,
            after_parameter,
        )
        for before_parameter, after_parameter
        in zip(before, after)
    )