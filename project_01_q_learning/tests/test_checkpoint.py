import numpy as np

from agent import QLearningAgent
from training import CheckpointManager
from environment import Action
from environment import GridWorldConfig

def test_save_and_load():
    agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    config = GridWorldConfig(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=frozenset({
            (1, 1),
            (2, 2),
        }),
    )

    agent.q_table[0, 0, Action.RIGHT] = 5.5
    agent.epsilon = 0.42

    path = "./checkpoint"
    manager = CheckpointManager(path)

    manager.save_latest(
        agent=agent,
        config=config,
        episode=100,
        best_reward=8.5,
    )

    restored_agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    checkpoint = manager.load_latest(
        restored_agent,
    )

    restored_config = checkpoint["config"]

    assert restored_config.rows == 5
    assert restored_config.cols == 5

    assert restored_config.start == (0, 0)
    assert restored_config.goal == (4, 4)

    assert restored_config.obstacles == frozenset({
        (1, 1),
        (2, 2),
    })

def test_checkpoint_preserves_environment_config(
    tmp_path,
):
    agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    config = GridWorldConfig(
        rows=5,
        cols=5,
        start=(4, 0),
        goal=(0, 4),
        obstacles=frozenset({
            (1, 1),
            (2, 3),
            (3, 2),
        }),
    )

    manager = CheckpointManager(tmp_path)

    manager.save_best(
        agent=agent,
        config=config,
        episode=500,
        best_reward=9.5,
    )

    restored_agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    checkpoint = manager.load_best(
        restored_agent,
    )

    restored_config = checkpoint["config"]

    assert restored_config == config

