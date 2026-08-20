import numpy as np

from ..agent import  QLearningAgent
from ..training import CheckpointManager
from ..environment import Action

def test_save_and_load():
    agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    agent.q_table[0, 0, Action.RIGHT] = 5.5
    agent.epsilon = 0.42

    path = "./checkpoint"
    manager = CheckpointManager(path)

    manager.save_latest(
        agent=agent,
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

    assert checkpoint["episode"] == 100
    assert checkpoint["best_reward"] == 8.5

    assert restored_agent.epsilon == 0.42

    assert np.array_equal(
        restored_agent.q_table,
        agent.q_table,
    )