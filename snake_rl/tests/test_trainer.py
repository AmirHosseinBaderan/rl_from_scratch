from agent import DQNAgent
from environment import SnakeEnvironment

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
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )