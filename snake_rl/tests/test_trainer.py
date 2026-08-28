from agent import DQNAgent
from environment import SnakeEnvironment
import pytest

from training import (
    Trainer,
    TrainingResult,
)


def create_agent() -> DQNAgent:
    return DQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.0,
        replay_capacity=100,
    )


def test_trainer_runs_requested_number_of_episodes():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    agent = create_agent()

    trainer = Trainer(
        environment=environment,
        agent=agent,
        episodes=5,
        batch_size=4,
        epsilon_decay=0.1,
        minimum_epsilon=0.1,
    )

    result = trainer.train()

    assert isinstance(
        result,
        TrainingResult,
    )

    assert result.episodes == 5
    assert len(result.rewards) == 5
    assert len(result.steps) == 5

def test_trainer_rejects_invalid_episode_count():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    agent = create_agent()

    try:
        Trainer(
            environment=environment,
            agent=agent,
            episodes=0,
            batch_size=4,
            epsilon_decay=0.1,
            minimum_epsilon=0.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )

def test_trainer_rejects_invalid_batch_size():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    agent = create_agent()

    try:
        Trainer(
            environment=environment,
            agent=agent,
            episodes=5,
            batch_size=0,
            epsilon_decay=0.1,
            minimum_epsilon=0.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )

def test_trainer_decays_epsilon_after_each_episode():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    agent = create_agent()

    agent.selector.epsilon = 1.0

    trainer = Trainer(
        environment=environment,
        agent=agent,
        episodes=3,
        batch_size=4,
        epsilon_decay=0.1,
        minimum_epsilon=0.1,
    )

    trainer.train()

    assert agent.selector.epsilon == pytest.approx(0.7)

def test_trainer_does_not_decay_epsilon_below_minimum():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    agent = create_agent()

    agent.selector.epsilon = 0.2

    trainer = Trainer(
        environment=environment,
        agent=agent,
        episodes=3,
        batch_size=4,
        epsilon_decay=0.2,
        minimum_epsilon=0.1,
    )

    trainer.train()

    assert agent.selector.epsilon == pytest.approx(0.1)