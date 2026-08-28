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
    )

    assert result.steps > 0
    assert len(agent.replay_buffer) == result.steps