from agent import DQNAgent
from environment import SnakeEnvironment
from training import run_episode


def test_run_episode_stores_experiences():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    agent = DQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.0,
        replay_capacity=100,
    )

    result = run_episode(
        environment=environment,
        agent=agent,
        batch_size=4,
    )

    assert result.steps > 0
    assert len(agent.replay_buffer) == result.steps

def test_run_episode_learns_when_buffer_is_ready():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    agent = DQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.0,
        replay_capacity=100,
    )

    learn_calls = []

    def fake_learn(batch_size: int) -> float:
        learn_calls.append(batch_size)
        return 0.0

    agent.learn = fake_learn

    result = run_episode(
        environment=environment,
        agent=agent,
        batch_size=4,
    )

    assert result.steps > 0
    assert len(learn_calls) > 0
    assert all(
        batch_size == 4
        for batch_size in learn_calls
    )