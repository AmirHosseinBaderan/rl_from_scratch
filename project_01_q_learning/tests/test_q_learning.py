from ..environment import Action
from ..agent import QLearningAgent

import numpy as np



def test_q_table_initialization():
    agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    assert agent.q_table.shape == (5, 5, 4)

    assert np.all(agent.q_table == 0)

def test_terminal_q_update():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        learning_rate=0.1,
    )

    state = (0, 0)
    action = Action.RIGHT

    agent.update(
        state=state,
        action=action,
        reward=10.0,
        next_state=(0, 1),
        done=True,
    )

    assert agent.q_table[0, 0, Action.RIGHT] == 1.0

def test_non_terminal_q_update():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        learning_rate=0.1,
        discount_factor=0.99,
    )

    agent.q_table[0, 1, Action.RIGHT] = 5.0

    agent.update(
        state=(0, 0),
        action=Action.RIGHT,
        reward=-0.1,
        next_state=(0, 1),
        done=False,
    )

    assert np.isclose(
        agent.q_table[0, 0, Action.RIGHT],
        0.485,
    )